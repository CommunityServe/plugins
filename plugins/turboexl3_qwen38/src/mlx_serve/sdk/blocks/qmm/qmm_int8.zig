//! copied_from: src/sdk/blocks/qmm/qmm_int8.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! OPT-IN, LOSSY int8-activation prefill route for 2-bit Prism packs on M5.
//!
//! Default OFF; a model's `int8_prefill` setting turns it on. Unlike every
//! other 2-bit route in this tree it is NOT lossless: the activation is
//! quantized to int8 per 128-group, so committed tokens can differ from the
//! stock path and greedy serial/speculative byte-equality no longer holds.
//!
//! Why it can win at all: at prompt width the packed projections are
//! COMPUTE-bound, and the tensor unit's int8 x int8 rate is about twice its
//! f16 rate on M5.
//!
//! `x[m,k] = as[m,g] * xq[m,k]` (signed) and `w[n,k] = s[n,g]*c[n,k] + b[n,g]`:
//!   y[m,n] = SUM_g as[m,g] * (s[n,g]*C[m,n,g] + b[n,g]*rs[m,g])
//! with C the int32 product of activation codes against weight codes and
//! `rs[m,g] = SUM_k xq[m,k]`.
//!
//! Arrangement follows the Bonsai speedup engine's prompt-width kernel
//! (Layr-Labs/mlxfast-bonsai2-27b-engine, MIT). Theirs feeds the tensor unit
//! `uint2b_format` directly where it exists; this toolchain has no such format,
//! so the 2-bit codes are expanded to int8 straight into the tensor op's right
//! operand in registers, as their register-staged form does.
const std = @import("std");
const env = @import("../../../env.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const log = std.log.scoped(.qmm_int8);

pub const MIN_ROWS: c_int = 64;
pub const GS: c_int = 128;

pub const HEADER = kernelSource("kernels/common/mpp_tensor.metal");

/// Four simdgroups, each one 32x32x128 int8 multiply per 128-group over a
/// 32-row x 32-column tile. The right operand is built in REGISTERS from the
/// row-major 2-bit words: lane l holds columns nl + 8c and, from every word,
/// the plane `(w >> 2kq) & 0x03030303` (codes kq, kq+4, kq+8, kq+12). That is a
/// fixed 4x4 transpose of K inside each 16-block, which the quantizer applies
/// to the activation too, so the integer products are unchanged.
pub const SOURCE = kernelSource("kernels/qmm/qmm_int8.metal");

/// amax, scale, quantize and code-sum for one 128-group in ONE pass. Eight
/// composed MLX ops each re-read and re-wrote the whole activation in f32,
/// which MEASURED at ~2-3 ms/call at prompt width.
pub const QUANT_SOURCE = kernelSource("kernels/qmm/qmm_int8_quant.metal");

/// `H(signs * x)` over one 1024 block (rht.zig's butterfly, rounded through T
/// where the unfused route stores it), then `QUANT_SOURCE` on its eight
/// 128-groups: the int8 route's input without the rotated activation in memory.
pub const ROTQ_SOURCE = kernelSource("kernels/qmm/qmm_int8_rotq.metal");

var quant_kernel: ?mlx.mlx_fast_metal_kernel = null;
const QKey = struct { m: c_int, k: c_int, dt: mlx.mlx_dtype };
var quant_cfg: std.AutoHashMapUnmanaged(QKey, mlx.mlx_fast_metal_kernel_config) = .{};

// ── host side ──

/// `xq = round(x / as)` per 128-group, with `as = amax/127`, plus the
/// per-group code sum the bias term needs. A group that is entirely zero would
/// divide by zero, so the scale is floored at a tiny positive value.
pub const Quantized = struct {
    xq: mlx.mlx_array,
    ascale: mlx.mlx_array, // f32 [M, Kg]
    rsum: mlx.mlx_array, // f32 [M, Kg]

};

/// Quantize launches, for the memo test.
pub var quantize_calls: u64 = 0;

/// Sibling projections (gate/up, q/k/v/z) read ONE rotated activation, so its
/// quantization is kept for the next call on the same underlying array
/// (keyed like `rht.Registry`'s memo: the shape storage a retained handle pins).
var memo_x: mlx.mlx_array = .{ .ctx = null };
var memo_signs: ?*anyopaque = null;
var memo_q: ?Quantized = null;

var rotq_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// The rotation to fuse into the quantizer: `H(signs * x)` per 1024 block.
pub const Rotation = struct { signs: mlx.mlx_array };

/// The kernel reads scales and biases group-major; the transposed copies are
/// derived constants of the weight, made once and kept.
const Derived = struct { scT: mlx.mlx_array, biT: mlx.mlx_array };

/// The route's per-weight constants, owned by the model whose weights they
/// derive from: the owner frees them in its `deinit` (on the inference thread),
/// so a key can never outlive the weight it names.
pub const Cache = struct {
    map: std.AutoHashMapUnmanaged(usize, Derived) = .{},

};

var kernel_cache: ?mlx.mlx_fast_metal_kernel = null;
const CfgKey = struct { n: c_int, mpad: c_int, dt: mlx.mlx_dtype, st: mlx.mlx_dtype, neg: bool };
var cfg_cache: std.AutoHashMapUnmanaged(CfgKey, mlx.mlx_fast_metal_kernel_config) = .{};
var engaged_logged = false;

// ── tests ──

const RmsMax = struct { rms: f32, max: f32 };

// This route is LOSSY, so the bar cannot be byte-equality or even stock's own
// error. It is: the int8 activation must not cost more than the quantization
// step it introduces, i.e. error within a small multiple of stock's, never
// NaN/Inf, and the shape/dtype contract preserved.
