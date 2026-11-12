//! TURBOEXL3_GLM53F_DSA_VATTN's device check (`zig build dsa-gpu-test`, run with TURBOEXL3_GLM53F_DSA_GPU=1 beside
//! src/dsa_gpu_test.zig's): glm5_next `sparseLatentAttention` over a verify window (1 to 4 rows, the cache's last) with
//! the switch off (the gather, then MLX's matmuls) and on (src/dsa_vattn.zig: the matmuls reading the latent at the
//! selected rows), bit for bit. The selection is `selectTokens`' over src/dsa_gpu_test.zig's inputs at a 16K context
//! (the window's pools completing between its rows, a partial tail pool: unused slots), its end (17,411 keys), 2K and
//! 32K contexts, and with every head weight zero and keys repeating every 7 pools; the queries and latent rows are
//! random bf16. The switch runs at 2 rows or more (ran=1) and leaves one row to MLX's own matmuls (ran=0). One line a
//! case, then the summary:
//!     TURBOEXL3_GLM53F_DSA_BYTES vattn case=<name> rows=<S> keys=<T> differing=<n>/<total> nonfinite=<n> ran=<n> PASS|FAIL
//!     TURBOEXL3_GLM53F_DSA_VATTN_PARITY lines=<n> fails=<n> PASS|FAIL
const std = @import("std");
const kit = @import("mlx_serve");
const mlx = kit.mlx;
const glm5 = @import("mlx_serve/plugins/glm5_next/glm5_next.zig");
const dsa_gpu_test = @import("dsa_gpu_test.zig");
const dsa_vattn = @import("dsa_vattn.zig");
const Array = mlx.mlx_array;

/// GLM-5.3-Flash's latent attention: 64 heads over 512-wide latent rows.
const heads: c_int = 64;
const latent_w: c_int = 512;
const attn_scale: f32 = 0.0625;

/// The cases: the cache's keys (the window's rows the last of them), the rows, what the selection's inputs hold.
pub const cases = [_]dsa_gpu_test.SelCase{
    .{ .name = "verify16k", .t = 16390, .rows = 3, .fill = .random },
    .{ .name = "verify16k_rows2", .t = 16389, .rows = 2, .fill = .random },
    .{ .name = "verify16k_rows4", .t = 16391, .rows = 4, .fill = .random },
    .{ .name = "verify16k_rows1", .t = 16388, .rows = 1, .fill = .random },
    .{ .name = "verify16k_end", .t = 17411, .rows = 3, .fill = .random },
    .{ .name = "verify2k", .t = 2103, .rows = 3, .fill = .random },
    .{ .name = "verify32k", .t = 32771, .rows = 4, .fill = .random },
    .{ .name = "verify16k_zero_weights", .t = 16390, .rows = 3, .fill = .zero_weights },
    .{ .name = "verify16k_repeating_keys", .t = 16390, .rows = 3, .fill = .repeating_keys },
};

/// glm5_next's sparse latent attention with DSA_VATTN as `on`, evaluated.
fn attend(s: mlx.mlx_stream, q: Array, lat: Array, sel: Array, on: bool) !Array {
    dsa_vattn.on = on;
    defer dsa_vattn.on = false;
    const out = try glm5.sparseLatentAttention(s, q, lat, sel, attn_scale, 64);
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

/// bf16 arrays `a` and `b` (the same shape) compared bit for bit: the elements that differ, the total, and `b`'s
/// non-finite elements.
const Bits = struct { differing: usize, total: usize, nonfinite: usize };
fn bitsOf(s: mlx.mlx_stream, a: Array, b: Array) !Bits {
    if (!std.mem.eql(c_int, mlx.getShape(a), mlx.getShape(b))) return error.DsaGpuShape;
    if (mlx.mlx_array_dtype(a) != .bfloat16 or mlx.mlx_array_dtype(b) != .bfloat16) return error.DsaGpuDtype;
    var ac = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ac);
    try mlx.check(mlx.mlx_contiguous(&ac, a, false, s));
    var bc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bc);
    try mlx.check(mlx.mlx_contiguous(&bc, b, false, s));
    try mlx.check(mlx.mlx_array_eval(ac));
    try mlx.check(mlx.mlx_array_eval(bc));
    const n = mlx.mlx_array_size(ac);
    const pa = (mlx.mlx_array_data_bfloat16(ac) orelse return error.DsaGpuData)[0..n];
    const pb = (mlx.mlx_array_data_bfloat16(bc) orelse return error.DsaGpuData)[0..n];
    var d: Bits = .{ .differing = 0, .total = n, .nonfinite = 0 };
    for (pa, pb) |x, y| {
        d.differing += @intFromBool(x != y);
        d.nonfinite += @intFromBool((y & 0x7F80) == 0x7F80);
    }
    return d;
}

test "turboexl3_glm53f dsa gpu: the verify window's sparse latent attention without the gather writes the gather's output bit for bit at 1 to 4 rows" {
    if (std.c.getenv("TURBOEXL3_GLM53F_DSA_GPU") == null) return error.SkipZigTest;
    defer dsa_vattn.on = false;
    const s = mlx.gpuStream();
    var seed: u64 = 20261109;
    var fails: usize = 0;
    for (cases) |c| {
        var x = try dsa_gpu_test.selInputs(s, &seed, c);
        defer x.deinit();
        const sel = try glm5.selectTokens(s, x.q, x.w, x.cache, dsa_gpu_test.ix);
        defer _ = mlx.mlx_array_free(sel);
        const q = try dsa_gpu_test.normal(s, &seed, &.{ 1, heads, c.rows, latent_w }, .bfloat16, 1.0);
        defer _ = mlx.mlx_array_free(q);
        const lat = try dsa_gpu_test.normal(s, &seed, &.{ 1, 1, c.t, latent_w }, .bfloat16, 1.0);
        defer _ = mlx.mlx_array_free(lat);
        const off = try attend(s, q, lat, sel, false);
        defer _ = mlx.mlx_array_free(off);
        const before = dsa_vattn.runs;
        const on = try attend(s, q, lat, sel, true);
        defer _ = mlx.mlx_array_free(on);
        const ran = dsa_vattn.runs - before;
        const d = try bitsOf(s, off, on);
        const ok = d.differing == 0 and d.total > 0 and ran == @intFromBool(c.rows >= dsa_vattn.min_rows);
        if (!ok) fails += 1;
        std.debug.print("TURBOEXL3_GLM53F_DSA_BYTES vattn case={s} rows={d} keys={d} differing={d}/{d} nonfinite={d} ran={d} {s}\n", .{ c.name, c.rows, c.t, d.differing, d.total, d.nonfinite, ran, if (ok) "PASS" else "FAIL" });
    }
    std.debug.print("TURBOEXL3_GLM53F_DSA_VATTN_PARITY lines={d} fails={d} {s}\n", .{ cases.len, fails, if (fails == 0) "PASS" else "FAIL" });
    try std.testing.expectEqual(@as(usize, 0), fails);
}
