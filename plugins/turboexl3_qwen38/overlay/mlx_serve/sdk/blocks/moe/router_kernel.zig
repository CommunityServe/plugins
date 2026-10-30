//! copied_from: src/sdk/blocks/moe/router_kernel.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! sdk/blocks/moe/router_kernel.
//! Owns RouterMode, ROUTER_MAX_EXPERTS, moeRouterSource, ROUTER_SOURCES, ROUTER_NAMES, router_kernels, and 13 more.

const mlx = @import("../../../mlx/ffi.zig");

// ── Fused MoE router (top-K selection + weight normalization in ONE kernel) ──
//
// The op chains above are ~11 dispatches each — sigmoid/softmax, bias add,
// negate, argpartition (a full 256-wide sort), slice, take_along_axis, sum,
// eps add, divide, scale multiply, cast — over a [rows, E] tensor that is 512
// bytes wide at decode. Every one of them is launch-latency bound, and a MoE
// model pays the whole chain once per LAYER: 40 layers x 11 = 440 dispatches
// per token on Laguna, none of which move meaningful data.
//
// One threadgroup per row does all of it: phase 1 computes the selection key
// for every expert into threadgroup memory, phase 2 runs K rounds of
// simdgroup-wide argmax with a LOWEST-INDEX tie-break, phase 3 normalizes the
// selected weights. Selection is exact integer/compare logic, so the picked
// expert set is the same set the chain picks whenever the keys are distinct.
//
// The ARITHMETIC deliberately replicates the chain's rather than improving on
// it — same `fast::exp`, same reciprocal-multiply softmax normalization, same
// MLX `Sigmoid` two-sided form, same intermediate rounding to the output dtype
// at the same points. A "more accurate" fused router is still a DIFFERENT
// model: the first live A/B of this kernel computed the softmax weights in one
// f32 pass and diverged from the chain on gemma4-26B-A4B at token ~80, which
// is the known INT4 near-tie argmax class amplified by an avoidable numerical
// change. The tests below therefore assert BIT equality with the chain, not
// just no-worse-than-reference; only the reduction ORDER over the expert axis
// differs (a simd tree vs MLX's N_READS blocking), which cannot move a bf16
// result except at an exact rounding boundary.
//
// Engagement is LOGGED, never
// inferred — a declined kernel is output-identical to the chain.

/// Largest expert count the kernel will stage in threadgroup memory. The
/// sigmoid arm keeps two f32 planes (key + raw weight), so 2048 experts is
/// 16 KiB of the 32 KiB budget.
pub const ROUTER_MAX_EXPERTS: c_int = 2048;


/// Everything the cached kernel config bakes in. A mismatch re-arms it.
/// Full-shape component for cached metal-kernel config keys. A cached config
/// bakes the OUTPUT SHAPES (and grid) of the call that built it, so the key
/// must carry the caller's actual dims: keying on a PRODUCT (rows, n) collides
/// across layouts with equal element counts and hands the colliding caller an
/// output labeled with the FIRST caller's shape. Live (Laguna XS):
/// a MoE-layer SwiGLU at a 16-token hot-cache-restore prefill is (1,16,512) =
/// 8192 elements — the same count/rank/dtype as a dense mlp_only layer at
/// decode (1,1,8192) — and the mislabeled activation killed the server in the
/// dense down-proj matmul. Every cached config key whose config bakes an
/// input-derived shape uses this instead of derived products.
pub const ShapeKey = struct {
    dims: [8]c_int = @splat(0),
    len: u8 = 0,
    pub fn from(sh: []const c_int) ShapeKey {
        var key = ShapeKey{};
        for (sh, 0..) |d, i| key.dims[i] = d;
        key.len = @intCast(sh.len);
        return key;
    }
};



/// The shared-expert gate's logit, folded into the softmax router: `x` [rows * gate_k] and the flat
/// weight vector `w` [gate_k], one logit per row.
pub const RouterGate = struct { x: mlx.mlx_array, w: mlx.mlx_array };

// ── tests ──
