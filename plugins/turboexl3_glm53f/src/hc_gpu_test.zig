//! The device parity of TURBOEXL3_GLM53F_HC_FUSE's text (src/hc_fuse.zig): in the `kda-gpu-test` binary (build.zig: its
//! filter takes this test's name too, linked to the server's MLX), under TURBOEXL3_GLM53F_KDA_GPU=1.
//!
//! One sublayer boundary of GLM-5.3-Flash's hyper-connection (4 streams of 4096, the mixes of the next sublayer's
//! seeded f32 tables, the copy's 20 Sinkhorn iterations) at each row count: a seeded bf16 stream, the gates of its own
//! collapse (`glm5.collapse`, as the sublayer before made them) and a seeded sublayer output, then the copy's
//! `glm5.expand` and `glm5.collapse` of the new stream (its expand, mixes and sinkhorn_collapse texts) against the fused
//! text's new stream, input x and post and comb gates, bit for bit. A 16K prompt's split main span (16,356 rows), a
//! 995-row prompt and the copy's first mixes-text row count (512) take the text; 511 rows and a split's 31-row tail
//! run the copy's MLX ops for the mixes, which the text does not reproduce, so it declines them (and the caller runs
//! the copy's three). One line an output, one a declined row count, and the run's last line:
//!
//!     TURBOEXL3_GLM53F_HC_BYTES T=<rows> out=<stream|x|post|comb> dtype=<dtype> differing=<n>/<total>
//!         nonfinite=<n> PASS|FAIL
//!     TURBOEXL3_GLM53F_HC_DECLINED T=<rows> PASS|FAIL
//!     TURBOEXL3_GLM53F_HC_PARITY lines=<n> declined=<n> runs=<n> fails=<n> PASS|FAIL
const std = @import("std");
const kit = @import("mlx_serve");
const mlx = kit.mlx;
const glm5 = @import("mlx_serve/plugins/glm5_next/glm5_next.zig");
const hc_fuse = @import("hc_fuse.zig");
const Array = mlx.mlx_array;

const hidden: c_int = 4096;
const mix: c_int = (2 + glm5.HC) * glm5.HC;
/// GLM-5.3-Flash's hyper-connection settings (config.zig: 20 Sinkhorn iterations, the gates' and the norm's epsilon).
const iters: u32 = 20;
const hc_eps: f32 = 1e-6;
const norm_eps: f32 = 1e-6;
/// The row counts: the text takes the first three, declines the last two.
pub const rows_tested = [_]c_int{ 16356, 995, 512, 511, 31 };
pub const outputs = [_][]const u8{ "stream", "x", "post", "comb" };

fn normal(s: mlx.mlx_stream, seed: *u64, shape: []const c_int, dtype: mlx.mlx_dtype, scale: f32) !Array {
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed.*));
    seed.* += 1;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_random_normal(&out, shape.ptr, shape.len, dtype, 0, scale, key, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

fn uniform(s: mlx.mlx_stream, seed: *u64, shape: []const c_int, lo: f32, hi: f32) !Array {
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed.*));
    seed.* += 1;
    const lo_a = mlx.mlx_array_new_float(lo);
    defer _ = mlx.mlx_array_free(lo_a);
    const hi_a = mlx.mlx_array_new_float(hi);
    defer _ = mlx.mlx_array_free(hi_a);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_random_uniform(&out, lo_a, hi_a, shape.ptr, shape.len, .float32, key, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

fn evalAll(arrs: []const Array) !void {
    const vec = mlx.mlx_vector_array_new_data(arrs.ptr, arrs.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_eval(vec));
}

const Diff = struct { differing: usize, total: usize, nonfinite: usize };

/// `got` against `want` element by element, as bits (both made contiguous: copies of the same bits).
fn diff(s: mlx.mlx_stream, want: Array, got: Array) !Diff {
    const dt = mlx.mlx_array_dtype(want);
    if (dt != mlx.mlx_array_dtype(got)) return error.HcGpuDtype;
    if (!std.mem.eql(c_int, mlx.getShape(want), mlx.getShape(got))) return error.HcGpuShape;
    var wc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wc);
    try mlx.check(mlx.mlx_contiguous(&wc, want, false, s));
    var gc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gc);
    try mlx.check(mlx.mlx_contiguous(&gc, got, false, s));
    try evalAll(&.{ wc, gc });
    const n = mlx.mlx_array_size(wc);
    var d: Diff = .{ .differing = 0, .total = n, .nonfinite = 0 };
    switch (dt) {
        .bfloat16 => {
            const pw = (mlx.mlx_array_data_bfloat16(wc) orelse return error.HcGpuData)[0..n];
            const pg = (mlx.mlx_array_data_bfloat16(gc) orelse return error.HcGpuData)[0..n];
            for (pw, pg) |a, b| {
                if (a != b) d.differing += 1;
                if ((b & 0x7F80) == 0x7F80) d.nonfinite += 1;
            }
        },
        .float32 => {
            const pw = (mlx.mlx_array_data_float32(wc) orelse return error.HcGpuData)[0..n];
            const pg = (mlx.mlx_array_data_float32(gc) orelse return error.HcGpuData)[0..n];
            for (pw, pg) |a, b| {
                const ab: u32 = @bitCast(a);
                const bb: u32 = @bitCast(b);
                if (ab != bb) d.differing += 1;
                if ((bb & 0x7F800000) == 0x7F800000) d.nonfinite += 1;
            }
        },
        else => return error.HcGpuDtype,
    }
    return d;
}

test "turboexl3_glm53f hc gpu: the fused expand and next collapse is the copy's expand, mixes and sinkhorn collapse bit for bit at 16,356, 995 and 512 rows, and declines 511 rows and a split's 31-row tail" {
    if (std.c.getenv("TURBOEXL3_GLM53F_KDA_GPU") == null) return error.SkipZigTest;
    const was_on = hc_fuse.on;
    defer hc_fuse.on = was_on;
    hc_fuse.on = true;
    const s = mlx.gpuStream();
    var seed: u64 = 20261107 + 53;
    const k_cols = glm5.HC * hidden;
    // the sublayers' tables, as the SC packs store them (f32): the one before (whose collapse made the gates the
    // expand reads) and the next one (whose collapse the text makes)
    var owned: [6]Array = @splat(.{ .ctx = null });
    defer for (owned) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(a);
    };
    owned[0] = try normal(s, &seed, &.{ mix, k_cols }, .float32, 1.0 / 128.0);
    owned[1] = try uniform(s, &seed, &.{3}, 0.5, 2.0);
    owned[2] = try normal(s, &seed, &.{mix}, .float32, 1.0);
    owned[3] = try normal(s, &seed, &.{ mix, k_cols }, .float32, 1.0 / 128.0);
    owned[4] = try uniform(s, &seed, &.{3}, 0.5, 2.0);
    owned[5] = try normal(s, &seed, &.{mix}, .float32, 1.0);
    var fails: usize = 0;
    var lines: usize = 0;
    var declined: usize = 0;
    const runs_before = hc_fuse.runs;
    for (rows_tested) |rows| {
        const stream = try normal(s, &seed, &.{ 1, rows, glm5.HC, hidden }, .bfloat16, 1.0);
        defer _ = mlx.mlx_array_free(stream);
        const x = try normal(s, &seed, &.{ 1, rows, hidden }, .bfloat16, 1.0);
        defer _ = mlx.mlx_array_free(x);
        // the gates the sublayer before made (its own collapse of the stream)
        var c0 = try glm5.collapse(s, stream, owned[0], owned[1], owned[2], iters, hc_eps, norm_eps);
        defer c0.deinit();
        try evalAll(&.{ c0.x, c0.post, c0.comb });
        // the copy's: expand, then the next sublayer's collapse of the new stream
        const new = try glm5.expand(s, x, stream, &c0);
        defer _ = mlx.mlx_array_free(new);
        var c1 = try glm5.collapse(s, new, owned[3], owned[4], owned[5], iters, hc_eps, norm_eps);
        defer c1.deinit();
        try evalAll(&.{ new, c1.x, c1.post, c1.comb });
        const fused = try hc_fuse.expandCollapse(s, x, stream, &c0, owned[3], owned[4], owned[5], iters, hc_eps, norm_eps);
        const takes = rows >= hc_fuse.min_rows;
        if (fused) |f| {
            var r = f;
            defer {
                _ = mlx.mlx_array_free(r.stream);
                r.next.deinit();
            }
            if (!takes) fails += 1;
            try evalAll(&.{ r.stream, r.next.x, r.next.post, r.next.comb });
            for (outputs, [_]Array{ new, c1.x, c1.post, c1.comb }, [_]Array{ r.stream, r.next.x, r.next.post, r.next.comb }) |name, w, g| {
                const d = try diff(s, w, g);
                const ok = d.differing == 0 and d.nonfinite == 0 and d.total > 0 and takes;
                if (!ok) fails += 1;
                lines += 1;
                std.debug.print("TURBOEXL3_GLM53F_HC_BYTES T={d} out={s} dtype={s} differing={d}/{d} nonfinite={d} {s}\n", .{ rows, name, @tagName(mlx.mlx_array_dtype(w)), d.differing, d.total, d.nonfinite, if (ok) "PASS" else "FAIL" });
            }
        } else {
            const ok = !takes;
            if (!ok) fails += 1;
            declined += 1;
            std.debug.print("TURBOEXL3_GLM53F_HC_DECLINED T={d} {s}\n", .{ rows, if (ok) "PASS" else "FAIL" });
        }
    }
    // the text ran once a row count it takes
    const runs = hc_fuse.runs - runs_before;
    var takes_n: usize = 0;
    for (rows_tested) |rows| takes_n += @intFromBool(rows >= hc_fuse.min_rows);
    if (runs != takes_n or lines != takes_n * outputs.len) fails += 1;
    std.debug.print("TURBOEXL3_GLM53F_HC_PARITY lines={d} declined={d} runs={d} fails={d} {s}\n", .{ lines, declined, runs, fails, if (fails == 0) "PASS" else "FAIL" });
    try std.testing.expectEqual(@as(usize, 0), fails);
}
