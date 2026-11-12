//! `TURBOEXL3_GLM53F_CACHED_SCALARS=1` (opt-in, exact): scalars the forward makes again at every call, made once and
//! kept. The clamped SwiGLU (fused_activations' `fusedClampedSwiGLU`: every shared expert's and every dense layer's MLP at
//! decode and verify rows, the MTP layer's too) makes its limit as `scalarOf(limit, dt)` a call: an f32 host scalar cast
//! to the activation's dtype, one GPU launch a call (45 a verify, 42 shared experts and 3 dense layers). Here the cast
//! runs once a device, dtype and value (the copy's own `scalarOf`, evaluated) and every call reads that array
//! (patches/turboexl3-sdk.patch in fused_activations.zig): the same scalar, so the same
//! bytes. src/smallop_gpu_test.zig holds the clamped SwiGLU with the kept limit to the per-call one bit for bit at a
//! shared expert's and a dense layer's widths, 1 to 4 rows.
const std = @import("std");
const kit = @import("mlx_serve");
const mlx = kit.mlx;
const log = @import("mlx_serve/sdk/util/log.zig");
const array = @import("mlx_serve/mlx/array.zig");
const Array = mlx.mlx_array;

/// `TURBOEXL3_GLM53F_CACHED_SCALARS=1`, set by the plugin at load, before any forward.
pub var on: bool = false;
/// The switch's load line (the A/B step reads "<switch>=1: ").
pub const line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_CACHED_SCALARS=1: the clamped SwiGLU's limit made once a device and dtype (the copy's own scalar cast, evaluated, kept) where every shared expert's and dense layer's call cast it again: the same scalar, one launch fewer a call\n";

/// Since the load: the scalars made and the calls that read a kept one (src/smallop_gpu_test.zig reads them).
pub var made: u64 = 0;
pub var uses: u64 = 0;
var engaged = false;

/// The most scalars kept (GLM-5.3-Flash's: the limit in bf16 on the GPU); past it a call makes its own.
pub const max_kept = 8;
const Kept = struct { gpu: bool, dt: mlx.mlx_dtype, bits: u32, arr: Array };
var kept: [max_kept]Kept = undefined;
var n_kept: usize = 0;

/// `scalarOf(v, dt, s)`'s array (a 0-d `dt` scalar of v, cast on `s`'s device) made once a device, dtype and value and
/// evaluated: a reference this file keeps (the caller does not free it). Null with the switch off or `max_kept` kept
/// (the caller makes its own).
pub fn of(s: mlx.mlx_stream, v: f32, dt: mlx.mlx_dtype) !?Array {
    if (!on) return null;
    const gpu = mlx.streamIsGpu(s);
    const bits: u32 = @bitCast(v);
    for (kept[0..n_kept]) |k| if (k.gpu == gpu and k.dt == dt and k.bits == bits) {
        uses += 1;
        return k.arr;
    };
    if (n_kept == max_kept) return null;
    const a = try array.scalarOf(v, dt, s);
    errdefer _ = mlx.mlx_array_free(a);
    try mlx.check(mlx.mlx_array_eval(a));
    kept[n_kept] = .{ .gpu = gpu, .dt = dt, .bits = bits, .arr = a };
    n_kept += 1;
    made += 1;
    uses += 1;
    if (!engaged) {
        engaged = true;
        log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_CACHED_SCALARS engaged: {d} as {s} kept\n", .{ v, @tagName(dt) });
    }
    return a;
}

/// The kept scalars let go (the model's unload; a test's end).
pub fn reset() void {
    for (kept[0..n_kept]) |k| _ = mlx.mlx_array_free(k.arr);
    n_kept = 0;
}

/// The request's line (the plugin's meters).
pub fn summary(buf: []u8) []const u8 {
    return std.fmt.bufPrint(buf, "[turboexl3_glm53f] TURBOEXL3_GLM53F_CACHED_SCALARS made={d} uses={d}\n", .{ made, uses }) catch buf[0..0];
}

test "turboexl3_glm53f stream cached_scalars: off makes nothing (the caller casts its own), and the request's line counts the scalars made and read" {
    const t = std.testing;
    // off: no array, nothing counted (the stream is never read with the switch off)
    try t.expectEqual(@as(?Array, null), try of(std.mem.zeroes(mlx.mlx_stream), 10.0, .bfloat16));
    try t.expectEqual(@as(u64, 0), made + uses);
    var buf: [128]u8 = undefined;
    made = 1;
    uses = 45;
    defer {
        made = 0;
        uses = 0;
    }
    try t.expectEqualStrings("[turboexl3_glm53f] TURBOEXL3_GLM53F_CACHED_SCALARS made=1 uses=45\n", summary(&buf));
}
