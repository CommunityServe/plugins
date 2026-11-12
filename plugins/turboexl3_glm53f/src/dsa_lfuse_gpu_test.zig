//! TURBOEXL3_GLM53F_DSA_LFUSE's device check (`zig build dsa-gpu-test`, run with TURBOEXL3_GLM53F_DSA_GPU=1 beside
//! src/dsa_gpu_test.zig's): glm5_next `sparseLatentAttention` over a verify window (1 to 8 rows, the cache's last) with
//! the switch off (the stock chain: the q4 copy, the mask, the gather, then latentAttendFolded's multiply, matmul, where,
//! precise softmax and matmul) and on (src/dsa_lfuse.zig: two launches), bit for bit; and on beside DSA_VATTN (the two
//! launches take the window, VATTN never runs). The selections and inputs are src/dsa_vattn_gpu_test.zig's cases' (16K,
//! its end, 2K and 32K contexts, every head weight zero, keys repeating every 7 pools; the window's pools completing
//! between its rows, a partial tail pool: unused slots) with 6 and 8 rows beside them; the queries and latent rows are
//! random bf16. The switch runs at 2 rows or more (ran=1) and leaves one row to the stock chain (ran=0). One line a case,
//! then the summary:
//!     TURBOEXL3_GLM53F_DSA_BYTES lfuse case=<name> rows=<S> keys=<T> differing=<n>/<total> nonfinite=<n> ran=<n> with_vattn=<n> vattn_ran=<n> PASS|FAIL
//!     TURBOEXL3_GLM53F_DSA_LFUSE_PARITY lines=<n> fails=<n> PASS|FAIL
const std = @import("std");
const kit = @import("mlx_serve");
const mlx = kit.mlx;
const glm5 = @import("mlx_serve/plugins/glm5_next/glm5_next.zig");
const dsa_gpu_test = @import("dsa_gpu_test.zig");
const dsa_vattn_gpu_test = @import("dsa_vattn_gpu_test.zig");
const dsa_lfuse = @import("dsa_lfuse.zig");
const dsa_vattn = @import("dsa_vattn.zig");
const Array = mlx.mlx_array;

const attn_scale: f32 = 0.0625;

/// The cases: DSA_VATTN's, then 6 and 8 rows at 16K.
const cases = dsa_vattn_gpu_test.cases ++ [_]dsa_gpu_test.SelCase{
    .{ .name = "verify16k_rows6", .t = 16393, .rows = 6, .fill = .random },
    .{ .name = "verify16k_rows8", .t = 16395, .rows = 8, .fill = .random },
};

/// glm5_next's sparse latent attention with DSA_LFUSE and DSA_VATTN as given, evaluated.
fn attend(s: mlx.mlx_stream, q: Array, lat: Array, sel: Array, lfuse: bool, vattn: bool) !Array {
    dsa_lfuse.on = lfuse;
    dsa_vattn.on = vattn;
    defer dsa_lfuse.on = false;
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

test "turboexl3_glm53f dsa gpu: the verify window's sparse latent attention in two launches writes the stock chain's output bit for bit at 1 to 8 rows, alone and beside DSA_VATTN" {
    if (std.c.getenv("TURBOEXL3_GLM53F_DSA_GPU") == null) return error.SkipZigTest;
    defer dsa_lfuse.on = false;
    defer dsa_vattn.on = false;
    const s = mlx.gpuStream();
    var seed: u64 = 20261110;
    var fails: usize = 0;
    for (cases) |c| {
        var x = try dsa_gpu_test.selInputs(s, &seed, c);
        defer x.deinit();
        const sel = try glm5.selectTokens(s, x.q, x.w, x.cache, dsa_gpu_test.ix);
        defer _ = mlx.mlx_array_free(sel);
        const q = try dsa_gpu_test.normal(s, &seed, &.{ 1, dsa_lfuse.heads, c.rows, dsa_lfuse.latent_w }, .bfloat16, 1.0);
        defer _ = mlx.mlx_array_free(q);
        const lat = try dsa_gpu_test.normal(s, &seed, &.{ 1, 1, c.t, dsa_lfuse.latent_w }, .bfloat16, 1.0);
        defer _ = mlx.mlx_array_free(lat);
        const off = try attend(s, q, lat, sel, false, false);
        defer _ = mlx.mlx_array_free(off);
        const before = dsa_lfuse.runs;
        const on = try attend(s, q, lat, sel, true, false);
        defer _ = mlx.mlx_array_free(on);
        const ran = dsa_lfuse.runs - before;
        const d = try bitsOf(s, off, on);
        // beside DSA_VATTN: the two launches take the window, VATTN never runs
        const before2 = dsa_lfuse.runs;
        const vbefore = dsa_vattn.runs;
        const both = try attend(s, q, lat, sel, true, true);
        defer _ = mlx.mlx_array_free(both);
        const ran2 = dsa_lfuse.runs - before2;
        const vran = dsa_vattn.runs - vbefore;
        const d2 = try bitsOf(s, off, both);
        const takes = @intFromBool(c.rows >= dsa_lfuse.min_rows);
        const ok = d.differing == 0 and d.total > 0 and ran == takes and d2.differing == 0 and ran2 == takes and vran == 0;
        if (!ok) fails += 1;
        std.debug.print("TURBOEXL3_GLM53F_DSA_BYTES lfuse case={s} rows={d} keys={d} differing={d}/{d} nonfinite={d} ran={d} with_vattn={d} vattn_ran={d} {s}\n", .{ c.name, c.rows, c.t, d.differing, d.total, d.nonfinite, ran, d2.differing, vran, if (ok) "PASS" else "FAIL" });
    }
    std.debug.print("TURBOEXL3_GLM53F_DSA_LFUSE_PARITY lines={d} fails={d} {s}\n", .{ cases.len, fails, if (fails == 0) "PASS" else "FAIL" });
    try std.testing.expectEqual(@as(usize, 0), fails);
}
