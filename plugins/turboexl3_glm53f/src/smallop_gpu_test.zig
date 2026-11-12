//! The device check of `TURBOEXL3_GLM53F_SMALLOP_FUSE` (src/smallop.zig and the hcPre defer2 text of
//! patches/turboexl3-model.patch and patches/turboexl3-sdk.patch), run by the build step `smallop-gpu-test` (linked to
//! the server's MLX)
//! under TURBOEXL3_GLM53F_SMALLOP_GPU=1; `zig build test` compiles it and its filter leaves it out.
//!
//! Each merged piece against the stock ops it replaces, at GLM-5.3-Flash's widths, on seeded random values, bit for bit:
//! - hcPre over a MoE sublayer's two terms (the routed sum y and the shared expert y2): the defer2 text against
//!   `mlx_add(y, y2)` and the defer text (its normed input, post and comb gates, rebuilt stream), and the deferral's
//!   materialize against the stock one, at decode and verify rows 1, 2, 3, 4, 8 and 16 (4 streams of 4096, 20 Sinkhorn
//!   iterations); the defer2 text must run (not decline to the stock add);
//! - a DSA layer's q_a and kv_a norms (1536 and 512 wide) in one launch against two `fast::rms_norm`, rows 1, 2, 3, 4
//!   and 16, and a 17-row call declined;
//! - the draft step's front (the embedding row, both norms, their concatenation) against take, two `fast::rms_norm`
//!   and the concatenation, an int32 and a uint32 token id;
//! - the draft step's last residual (`h + (a + b)`, then the final norm) against two `mlx_add` and `fast::rms_norm`,
//!   rows 1, 3 and 4; and the kit's addNorm (the first residual and the post-norm) against `mlx_add` and
//!   `fast::rms_norm` at the same rows;
//! - a DSA layer's q_a and kv_a (4096 -> 1536 and 4096 -> 512, each at its own K split, 8 and 32) in one launch against
//!   the two `linear`s at the trunk's K6 and the MTP layer's K4, rows 1, 2, 3, 4, 8 and 16 (24 lines); 17 rows and an
//!   unregistered pair run the two linears;
//! - a dense layer's gate and up (4096 -> 12288 each, K6) in one launch against the two `linear`s at rows 1 to 8 and 16
//!   where turboexl3's `glm53f.shared_gu.plan` takes them (its prepare inside fits at chunks of up to 3 rows at this
//!   width's K split of 1: rows 1, 2, 3 and 6, which must include the decode and depth-2 verify rows 1 to 3); the other
//!   row counts, 17 rows and an unregistered pair run the two linears.
//! One line an output or a declined row count, and the run's last line:
//!
//!     TURBOEXL3_GLM53F_SMALLOP_BYTES piece=<piece> rows=<n> out=<output> dtype=<dtype> differing=<n>/<total>
//!         nonfinite=<n> PASS|FAIL
//!     TURBOEXL3_GLM53F_SMALLOP_DECLINED piece=dense_gu rows=<n> PASS|FAIL
//!     TURBOEXL3_GLM53F_SMALLOP_PARITY lines=<n> fails=<n> defer2=<on|declined|unlaunched> fronts=<n> norm_pairs=<n>
//!         dsa_qkv=<n> tails=<n> dense=<n> declined=<n> PASS|FAIL
//!
//! TURBOEXL3_GLM53F_DRAFT_S0's device check (its own test, the same binary and switch): the MTP draft's step 0 pieces
//! (src/draft_s0.zig) against the stock step's ops at 2, 3 and 4 rows, bit for bit: the front against take, two
//! `fast::rms_norm` and the concatenation (int32 and uint32 ids, the last a table's last row); the first residual and
//! post-norm through the kit's addNorm against `mlx_add` and `fast::rms_norm` (both outputs); the last residual at the
//! last row against the MoE's add, the residual's add over every row, the last row taken and `fast::rms_norm`; and,
//! beside HEAD_LAST_ROW, the last residual over a last-row MoE output against `tailLast` (the last row taken, `mlx_add`,
//! `fast::rms_norm`). One and five rows decline (the front and the last residual):
//!
//!     TURBOEXL3_GLM53F_DRAFT_S0_BYTES piece=<front|post|tail|tail_last_row> rows=<n> out=<output> differing=<n>/<total>
//!         nonfinite=<n> PASS|FAIL
//!     TURBOEXL3_GLM53F_DRAFT_S0_DECLINED piece=<front|tail> rows=<n> PASS|FAIL
//!     TURBOEXL3_GLM53F_DRAFT_S0_PARITY lines=<n> fails=<n> fronts=<n> tails=<n> tails_last_row=<n> declined=<n> PASS|FAIL
const std = @import("std");
const kit = @import("mlx_serve");
const mlx = kit.mlx;
const t3 = @import("turboexl3");
const glm5 = @import("mlx_serve/plugins/glm5_next/glm5_next.zig");
const add_norm = @import("mlx_serve/sdk/blocks/norms/add_norm.zig");
const smallop = @import("smallop.zig");
const draft_s0 = @import("draft_s0.zig");
const Array = mlx.mlx_array;

const hidden: c_int = 4096;
const mix: c_int = (2 + glm5.HC) * glm5.HC;
/// GLM-5.3-Flash's settings (config.zig): 20 Sinkhorn iterations, the gates' and the norms' epsilon
const iters: u32 = 20;
const hc_eps: f32 = 1e-6;
const rms_eps: f32 = 1e-6;
/// a DSA layer's q_a and kv_a widths (q_lora_rank, kv_lora_rank)
const q_width: c_int = 1536;
const kv_width: c_int = 512;
/// a dense layer's intermediate width and rate
const dense_inter: u32 = 12288;
const dense_k: u32 = 6;

const hc_rows = [_]c_int{ 1, 2, 3, 4, 8, 16 };
const hc_outputs = [_][]const u8{ "normed", "post", "comb", "stream", "materialized" };
const pair_rows = [_]c_int{ 1, 2, 3, 4, 16 };
const tail_rows = [_]c_int{ 1, 3, 4 };
const dense_rows = [_]c_int{ 1, 2, 3, 4, 5, 6, 7, 8, 16 };
/// the DSA pair: the trunk's K6 and the MTP layer's K4, decode and verify rows
const qkv_rates = [_]u32{ 6, 4 };
const qkv_rows = [_]c_int{ 1, 2, 3, 4, 8, 16 };
/// The byte lines a run prints but the dense pair's: hcPre's five outputs, the norm pair's two, the front's one a token
/// dtype, the tail's one and the kit addNorm's two, the DSA pair's two (the dense pair's two a row count it takes)
pub const fixed_lines = hc_rows.len * hc_outputs.len + pair_rows.len * 2 + 2 + tail_rows.len * 3 + qkv_rates.len * qkv_rows.len * 2;

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

/// A norm weight [d] bf16 in [0.5, 1.5]
fn weight(s: mlx.mlx_stream, seed: *u64, d: c_int) !Array {
    const f = try uniform(s, seed, &.{d}, 0.5, 1.5);
    defer _ = mlx.mlx_array_free(f);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_astype(&out, f, .bfloat16, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

fn evalAll(arrs: []const Array) !void {
    const vec = mlx.mlx_vector_array_new_data(arrs.ptr, arrs.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_eval(vec));
}

const Diff = struct { differing: usize, total: usize, nonfinite: usize };

/// `got` against `want` element by element, as bits (both made contiguous: copies of the same bits)
fn diff(s: mlx.mlx_stream, want: Array, got: Array) !Diff {
    const dt = mlx.mlx_array_dtype(want);
    if (dt != mlx.mlx_array_dtype(got)) return error.SmallopGpuDtype;
    if (!std.mem.eql(c_int, mlx.getShape(want), mlx.getShape(got))) return error.SmallopGpuShape;
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
            const pw = (mlx.mlx_array_data_bfloat16(wc) orelse return error.SmallopGpuData)[0..n];
            const pg = (mlx.mlx_array_data_bfloat16(gc) orelse return error.SmallopGpuData)[0..n];
            for (pw, pg) |a, b| {
                if (a != b) d.differing += 1;
                if ((b & 0x7F80) == 0x7F80) d.nonfinite += 1;
            }
        },
        .float32 => {
            const pw = (mlx.mlx_array_data_float32(wc) orelse return error.SmallopGpuData)[0..n];
            const pg = (mlx.mlx_array_data_float32(gc) orelse return error.SmallopGpuData)[0..n];
            for (pw, pg) |a, b| {
                const ab: u32 = @bitCast(a);
                const bb: u32 = @bitCast(b);
                if (ab != bb) d.differing += 1;
                if ((bb & 0x7F800000) == 0x7F800000) d.nonfinite += 1;
            }
        },
        else => return error.SmallopGpuDtype,
    }
    return d;
}

const Tally = struct {
    lines: usize = 0,
    fails: usize = 0,
    fn check(t: *Tally, s: mlx.mlx_stream, piece: []const u8, rows: c_int, out: []const u8, want: Array, got: Array) !void {
        const d = try diff(s, want, got);
        const ok = d.total > 0 and d.differing == 0 and d.nonfinite == 0;
        t.lines += 1;
        t.fails += @intFromBool(!ok);
        std.debug.print("TURBOEXL3_GLM53F_SMALLOP_BYTES piece={s} rows={d} out={s} dtype={s} differing={d}/{d} nonfinite={d} {s}\n", .{ piece, rows, out, @tagName(mlx.mlx_array_dtype(want)), d.differing, d.total, d.nonfinite, if (ok) "PASS" else "FAIL" });
    }
};

fn rmsNorm(s: mlx.mlx_stream, x: Array, w: Array) !Array {
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_fast_rms_norm(&out, x, w, rms_eps, s));
    return out;
}

fn add(s: mlx.mlx_stream, a: Array, b: Array) !Array {
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_add(&out, a, b, s));
    return out;
}

fn f16Bits(v: f32) u16 {
    return @bitCast(@as(f16, @floatCast(v)));
}

/// A projection `in -> out` at rate `k` as the pack stores it: random codewords [in/16, out/16, 16k], suh f16 [in] in
/// [0.5, 1.5], svh f16 [out] in [0.5, 1.5] / sqrt(in)
fn proj(a: std.mem.Allocator, rnd: std.Random, in: u32, out: u32, k: u32) !t3.Proj {
    const n = 16 * k;
    const t = try a.alloc(u16, (in / 16) * (out / 16) * n);
    defer a.free(t);
    rnd.bytes(std.mem.sliceAsBytes(t));
    const su = try a.alloc(u16, in);
    defer a.free(su);
    for (su) |*v| v.* = f16Bits(0.5 + rnd.float(f32));
    const sv = try a.alloc(u16, out);
    defer a.free(sv);
    for (sv) |*v| v.* = f16Bits((0.5 + rnd.float(f32)) / @sqrt(@as(f32, @floatFromInt(in))));
    const p: t3.Proj = .{
        .trellis = mlx.mlx_array_new_data(t.ptr, &[_]c_int{ @intCast(in / 16), @intCast(out / 16), @intCast(n) }, 3, .uint16),
        .suh = mlx.mlx_array_new_data(su.ptr, &[_]c_int{@intCast(in)}, 1, .float16),
        .svh = mlx.mlx_array_new_data(sv.ptr, &[_]c_int{@intCast(out)}, 1, .float16),
    };
    try evalAll(&.{ p.trellis, p.suh, p.svh });
    return p;
}

fn freeProj(p: t3.Proj) void {
    for ([_]Array{ p.trellis, p.suh, p.svh }) |x| _ = mlx.mlx_array_free(x);
}

/// hcPre over a MoE sublayer's two terms: the defer2 text against the stock add and defer text, and the deferral's
/// materialize, at `rows` rows. The stream's gates come from a first hcPre over it (the direct text), as the sublayer
/// before leaves them.
fn hcCase(s: mlx.mlx_stream, seed: *u64, t: *Tally, rows: c_int, tables: *const [6]Array, nw: [2]Array) !void {
    const stream = try normal(s, seed, &.{ 1, rows, glm5.HC, hidden }, .bfloat16, 1.0);
    defer _ = mlx.mlx_array_free(stream);
    var pre0 = try glm5.hcPre(s, stream, null, tables[0], tables[1], tables[2], nw[0], iters, hc_eps, rms_eps, rms_eps);
    defer pre0.deinit();
    const y = try normal(s, seed, &.{ 1, rows, hidden }, .bfloat16, 1.0);
    defer _ = mlx.mlx_array_free(y);
    const y2 = try normal(s, seed, &.{ 1, rows, hidden }, .bfloat16, 0.5);
    defer _ = mlx.mlx_array_free(y2);
    try evalAll(&.{ pre0.normed, pre0.post, pre0.comb });
    // stock: the MoE's add, then the defer text
    const sum = try add(s, y, y2);
    defer _ = mlx.mlx_array_free(sum);
    var df_s = pre0.defer_(sum, stream);
    defer df_s.deinit();
    var pre_s = try glm5.hcPre(s, null, &df_s, tables[3], tables[4], tables[5], nw[1], iters, hc_eps, rms_eps, rms_eps);
    defer pre_s.deinit();
    const mat_s = try df_s.materialize(s);
    defer _ = mlx.mlx_array_free(mat_s);
    try evalAll(&.{ pre_s.normed, pre_s.post, pre_s.comb, pre_s.h, mat_s });
    // the two terms unadded, the defer2 text
    var df_2 = pre0.defer2_(y, y2, stream);
    defer df_2.deinit();
    var pre_2 = try glm5.hcPre(s, null, &df_2, tables[3], tables[4], tables[5], nw[1], iters, hc_eps, rms_eps, rms_eps);
    defer pre_2.deinit();
    const mat_2 = try df_2.materialize(s);
    defer _ = mlx.mlx_array_free(mat_2);
    try evalAll(&.{ pre_2.normed, pre_2.post, pre_2.comb, pre_2.h, mat_2 });
    for (hc_outputs, [_]Array{ pre_s.normed, pre_s.post, pre_s.comb, pre_s.h, mat_s }, [_]Array{ pre_2.normed, pre_2.post, pre_2.comb, pre_2.h, mat_2 }) |name, w, g| {
        try t.check(s, "hc_defer2", rows, name, w, g);
    }
}

test "turboexl3_glm53f smallop gpu: TURBOEXL3_GLM53F_SMALLOP_FUSE's pieces give the stock ops' bits: hcPre's defer2 text over a MoE layer's two terms, a DSA layer's two norms and its q_a and kv_a, the draft step's front, last residual and the kit's addNorm, and a dense layer's gate and up, at decode and verify rows" {
    if (std.c.getenv("TURBOEXL3_GLM53F_SMALLOP_GPU") == null) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const a = std.heap.page_allocator;
    var seed: u64 = 20261107 + 61;
    var prng = std.Random.DefaultPrng.init(20261107 + 62);
    const rnd = prng.random();
    smallop.reset();
    defer smallop.reset();
    smallop.on = true;
    defer smallop.on = false;
    var t: Tally = .{};
    var dense_taken: usize = 0;
    const eps = mlx.mlx_array_new_float(rms_eps);
    defer _ = mlx.mlx_array_free(eps);

    // hcPre: the sublayer before's tables (its direct text made the gates) and the next sublayer's, as the SC packs
    // store them (f32), and the two input norms
    {
        const k_cols = glm5.HC * hidden;
        var tables: [6]Array = @splat(.{ .ctx = null });
        defer for (tables) |x| if (x.ctx != null) {
            _ = mlx.mlx_array_free(x);
        };
        tables[0] = try normal(s, &seed, &.{ mix, k_cols }, .float32, 1.0 / 128.0);
        tables[1] = try uniform(s, &seed, &.{3}, 0.5, 2.0);
        tables[2] = try normal(s, &seed, &.{mix}, .float32, 1.0);
        tables[3] = try normal(s, &seed, &.{ mix, k_cols }, .float32, 1.0 / 128.0);
        tables[4] = try uniform(s, &seed, &.{3}, 0.5, 2.0);
        tables[5] = try normal(s, &seed, &.{mix}, .float32, 1.0);
        const nw = [2]Array{ try weight(s, &seed, hidden), try weight(s, &seed, hidden) };
        defer for (nw) |x| {
            _ = mlx.mlx_array_free(x);
        };
        for (hc_rows) |rows| try hcCase(s, &seed, &t, rows, &tables, nw);
    }

    // a DSA layer's q_a and kv_a norms
    {
        const w0 = try weight(s, &seed, q_width);
        defer _ = mlx.mlx_array_free(w0);
        const w1 = try weight(s, &seed, kv_width);
        defer _ = mlx.mlx_array_free(w1);
        for (pair_rows) |rows| {
            const x0 = try normal(s, &seed, &.{ 1, rows, q_width }, .bfloat16, 2.0);
            defer _ = mlx.mlx_array_free(x0);
            const x1 = try normal(s, &seed, &.{ 1, rows, kv_width }, .bfloat16, 2.0);
            defer _ = mlx.mlx_array_free(x1);
            const want0 = try rmsNorm(s, x0, w0);
            defer _ = mlx.mlx_array_free(want0);
            const want1 = try rmsNorm(s, x1, w1);
            defer _ = mlx.mlx_array_free(want1);
            const got = (try smallop.normPair(s, x0, w0, x1, w1, eps)) orelse return error.SmallopGpuPairDeclined;
            defer for (got) |x| {
                _ = mlx.mlx_array_free(x);
            };
            try t.check(s, "norm_pair", rows, "q_resid", want0, got[0]);
            try t.check(s, "norm_pair", rows, "lat", want1, got[1]);
        }
        // a prompt's rows: the two norms (null, not counted)
        const p0 = try normal(s, &seed, &.{ 1, 17, q_width }, .bfloat16, 2.0);
        defer _ = mlx.mlx_array_free(p0);
        const p1 = try normal(s, &seed, &.{ 1, 17, kv_width }, .bfloat16, 2.0);
        defer _ = mlx.mlx_array_free(p1);
        if (try smallop.normPair(s, p0, w0, p1, w1, eps)) |got| {
            for (got) |x| _ = mlx.mlx_array_free(x);
            t.fails += 1;
        }
    }

    // the draft step's front: a table of 64 rows, the token ids 37 (int32) and 63 (uint32, the last row)
    {
        const table = try normal(s, &seed, &.{ 64, hidden }, .bfloat16, 1.0);
        defer _ = mlx.mlx_array_free(table);
        const h = try normal(s, &seed, &.{ 1, 1, hidden }, .bfloat16, 1.0);
        defer _ = mlx.mlx_array_free(h);
        const we = try weight(s, &seed, hidden);
        defer _ = mlx.mlx_array_free(we);
        const wh = try weight(s, &seed, hidden);
        defer _ = mlx.mlx_array_free(wh);
        const id32 = [_]i32{37};
        const idu = [_]u32{63};
        const ids = [2]Array{
            mlx.mlx_array_new_data(&id32, &[_]c_int{ 1, 1 }, 2, .int32),
            mlx.mlx_array_new_data(&idu, &[_]c_int{ 1, 1 }, 2, .uint32),
        };
        defer for (ids) |x| {
            _ = mlx.mlx_array_free(x);
        };
        for (ids, [_][]const u8{ "cat_int32", "cat_uint32" }) |id, name| {
            // stock: the trunk's embedding (take, then [1, 1, hidden]), both norms, the concatenation
            var flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(flat);
            try mlx.check(mlx.mlx_reshape(&flat, id, &[_]c_int{1}, 1, s));
            var row = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(row);
            try mlx.check(mlx.mlx_take_axis(&row, table, flat, 0, s));
            var emb = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(emb);
            try mlx.check(mlx.mlx_reshape(&emb, row, &[_]c_int{ 1, 1, hidden }, 3, s));
            const en = try rmsNorm(s, emb, we);
            defer _ = mlx.mlx_array_free(en);
            const hn = try rmsNorm(s, h, wh);
            defer _ = mlx.mlx_array_free(hn);
            var want = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(want);
            const vec = mlx.mlx_vector_array_new_data(&[_]Array{ en, hn }, 2);
            defer _ = mlx.mlx_vector_array_free(vec);
            try mlx.check(mlx.mlx_concatenate_axis(&want, vec, 2, s));
            const got = (try smallop.front(s, table, false, id, h, we, wh, eps)) orelse return error.SmallopGpuFrontDeclined;
            defer _ = mlx.mlx_array_free(got);
            try t.check(s, "front", 1, name, want, got);
        }
    }

    // the draft step's last residual and the kit's addNorm
    {
        const w = try weight(s, &seed, hidden);
        defer _ = mlx.mlx_array_free(w);
        for (tail_rows) |rows| {
            const h = try normal(s, &seed, &.{ 1, rows, hidden }, .bfloat16, 4.0);
            defer _ = mlx.mlx_array_free(h);
            const ya = try normal(s, &seed, &.{ 1, rows, hidden }, .bfloat16, 1.0);
            defer _ = mlx.mlx_array_free(ya);
            const yb = try normal(s, &seed, &.{ 1, rows, hidden }, .bfloat16, 0.5);
            defer _ = mlx.mlx_array_free(yb);
            // stock: the MoE's add, the residual's, the final norm
            const mlp = try add(s, ya, yb);
            defer _ = mlx.mlx_array_free(mlp);
            const y = try add(s, h, mlp);
            defer _ = mlx.mlx_array_free(y);
            const want = try rmsNorm(s, y, w);
            defer _ = mlx.mlx_array_free(want);
            const got = (try smallop.add2Norm(s, h, ya, yb, w, eps)) orelse return error.SmallopGpuTailDeclined;
            defer _ = mlx.mlx_array_free(got);
            try t.check(s, "add2_norm", rows, "normed", want, got);
            // the kit's addNorm over the same rows: `h + ya` and its norm
            const x1 = try add(s, h, ya);
            defer _ = mlx.mlx_array_free(x1);
            const n1 = try rmsNorm(s, x1, w);
            defer _ = mlx.mlx_array_free(n1);
            const r = (try add_norm.addNorm(h, ya, w, eps, s)) orelse return error.SmallopGpuAddNormDeclined;
            defer {
                _ = mlx.mlx_array_free(r.h);
                _ = mlx.mlx_array_free(r.normed);
            }
            try t.check(s, "kit_add_norm", rows, "h", x1, r.h);
            try t.check(s, "kit_add_norm", rows, "normed", n1, r.normed);
        }
    }

    // a DSA layer's q_a and kv_a at each rate, registered as the bank's opening registers them; every rate's pair held to
    // the end (the registry's keys are the q_a trellises, live as the model's)
    {
        var qp: [qkv_rates.len][2]t3.Proj = undefined;
        var qmade: usize = 0;
        defer for (qp[0..qmade]) |ps| for (ps) |pj| freeProj(pj);
        for (qkv_rates, 0..) |k, ki| {
            qp[ki] = .{ try proj(a, rnd, @intCast(hidden), @intCast(q_width), k), try proj(a, rnd, @intCast(hidden), @intCast(kv_width), k) };
            qmade += 1;
            const q_a = qp[ki][0];
            const kv_a = qp[ki][1];
            try smallop.registerQkv(s, q_a, kv_a);
            for (qkv_rows) |rows| {
                const x = try normal(s, &seed, &.{ 1, rows, hidden }, .bfloat16, 1.0);
                defer _ = mlx.mlx_array_free(x);
                const wq = try t3.linear(s, x, q_a.trellis, q_a.suh, q_a.svh, .mul1);
                defer _ = mlx.mlx_array_free(wq);
                const wkv = try t3.linear(s, x, kv_a.trellis, kv_a.suh, kv_a.svh, .mul1);
                defer _ = mlx.mlx_array_free(wkv);
                const got = (try smallop.dsaQkv(s, q_a, kv_a, x)) orelse return error.SmallopGpuQkvDeclined;
                defer for (got) |y| {
                    _ = mlx.mlx_array_free(y);
                };
                try evalAll(&.{ wq, wkv, got[0], got[1] });
                try t.check(s, if (k == 6) "dsa_qkv_k6" else "dsa_qkv_k4", rows, "q_a", wq, got[0]);
                try t.check(s, if (k == 6) "dsa_qkv_k6" else "dsa_qkv_k4", rows, "kv_a", wkv, got[1]);
            }
        }
        // a prompt's rows and a pair the opening did not register: the two linears (null, not counted)
        const xp = try normal(s, &seed, &.{ 1, 17, hidden }, .bfloat16, 1.0);
        defer _ = mlx.mlx_array_free(xp);
        if (try smallop.dsaQkv(s, qp[0][0], qp[0][1], xp)) |got| {
            for (got) |y| _ = mlx.mlx_array_free(y);
            t.fails += 1;
        }
        const xo = try normal(s, &seed, &.{ 1, 1, hidden }, .bfloat16, 1.0);
        defer _ = mlx.mlx_array_free(xo);
        if (try smallop.dsaQkv(s, qp[0][1], qp[0][0], xo)) |got| {
            for (got) |y| _ = mlx.mlx_array_free(y);
            t.fails += 1;
        }
    }

    // a dense layer's gate and up, registered as the bank's opening registers them
    {
        const gate = try proj(a, rnd, @intCast(hidden), dense_inter, dense_k);
        defer freeProj(gate);
        const up = try proj(a, rnd, @intCast(hidden), dense_inter, dense_k);
        defer freeProj(up);
        try smallop.registerDense(s, gate, up);
        const pair = try t3.glm53f.shared_gu.pairOf(gate, up);
        for (dense_rows) |rows| {
            const takes = t3.glm53f.shared_gu.plan(pair, .mul1, @intCast(rows)) != null;
            dense_taken += @intFromBool(takes);
            // the decode and depth-2 verify rows must be taken
            if (rows <= 3 and !takes) t.fails += 1;
            const x = try normal(s, &seed, &.{ 1, rows, hidden }, .bfloat16, 1.0);
            defer _ = mlx.mlx_array_free(x);
            const wg = try t3.linear(s, x, gate.trellis, gate.suh, gate.svh, .mul1);
            defer _ = mlx.mlx_array_free(wg);
            const wu = try t3.linear(s, x, up.trellis, up.suh, up.svh, .mul1);
            defer _ = mlx.mlx_array_free(wu);
            if (try smallop.denseGateUp(s, gate, up, x)) |got| {
                defer for (got) |y| {
                    _ = mlx.mlx_array_free(y);
                };
                if (!takes) t.fails += 1;
                try evalAll(&.{ wg, wu, got[0], got[1] });
                try t.check(s, "dense_gu", rows, "gate", wg, got[0]);
                try t.check(s, "dense_gu", rows, "up", wu, got[1]);
            } else {
                const ok = !takes;
                t.fails += @intFromBool(!ok);
                std.debug.print("TURBOEXL3_GLM53F_SMALLOP_DECLINED piece=dense_gu rows={d} {s}\n", .{ rows, if (ok) "PASS" else "FAIL" });
            }
        }
        // a prompt's rows and a pair the opening did not register: the two linears (null, not counted)
        const xp = try normal(s, &seed, &.{ 1, 17, hidden }, .bfloat16, 1.0);
        defer _ = mlx.mlx_array_free(xp);
        if (try smallop.denseGateUp(s, gate, up, xp)) |got| {
            for (got) |y| _ = mlx.mlx_array_free(y);
            t.fails += 1;
        }
        const xo = try normal(s, &seed, &.{ 1, 1, hidden }, .bfloat16, 1.0);
        defer _ = mlx.mlx_array_free(xo);
        if (try smallop.denseGateUp(s, up, gate, xo)) |got| {
            for (got) |y| _ = mlx.mlx_array_free(y);
            t.fails += 1;
        }
    }

    const defer2 = if (glm5.hc_pre2_ok) |ok| (if (ok) "on" else "declined") else "unlaunched";
    // a dense row count the plan declines is counted declined (the call reached turboexl3); nothing else declines
    const counts_ok = (glm5.hc_pre2_ok orelse false) and smallop.fronts == 2 and smallop.pairs == pair_rows.len and smallop.tails == tail_rows.len and
        smallop.qkv == qkv_rates.len * qkv_rows.len and smallop.qkvLayers() == qkv_rates.len and
        smallop.dense == dense_taken and smallop.declined == dense_rows.len - dense_taken and smallop.denseLayers() == 1;
    const ok = t.fails == 0 and t.lines == fixed_lines + 2 * dense_taken and counts_ok;
    std.debug.print("TURBOEXL3_GLM53F_SMALLOP_PARITY lines={d} fails={d} defer2={s} fronts={d} norm_pairs={d} dsa_qkv={d} tails={d} dense={d} declined={d} {s}\n", .{ t.lines, t.fails, defer2, smallop.fronts, smallop.pairs, smallop.qkv, smallop.tails, smallop.dense, smallop.declined, if (ok) "PASS" else "FAIL" });
    try std.testing.expect(ok);
}

// ── the launch trims: TURBOEXL3_GLM53F_HOST_ZERO (src/host_zero.zig) and TURBOEXL3_GLM53F_CACHED_SCALARS
// (src/cached_scalars.zig) ──
//
// Each DEC-ONE call that zeroes its counters with a launch, run with turboexl3's `dense.dec_host_zero` off (the launch)
// and on (host-made zeros), at decode and verify rows 1 to 4, its outputs compared bit for bit and its host-made counter
// arrays counted (1 a call that takes them, 0 one that does not): a DSA layer's q_a and kv_a in one launch at K6 and K4
// (`smallop.dsaQkv`), a dense layer's gate and up in one launch where its plan takes the rows (`smallop.denseGateUp`),
// and kv_a and q_a alone (`linear`: the prepare inside at kv_a's 512 outputs, and at q_a's 1536 at one row only). Then the clamped
// SwiGLU (sc_3.10's limit 10) at a shared expert's and a dense layer's widths with the per-call limit and the kept one.
// One line an output, and the run's last line:
//
//     TURBOEXL3_GLM53F_TRIM_BYTES piece=<piece> rows=<n> out=<output> dtype=<dtype> differing=<n>/<total> nonfinite=<n>
//         host_zeroed=<n> PASS|FAIL
//     TURBOEXL3_GLM53F_TRIM_PARITY lines=<n> fails=<n> host_zeroed=<n> launch_zeroed=<n> scalars_made=<n>
//         scalars_used=<n> PASS|FAIL

const cached_scalars = @import("cached_scalars.zig");
const fused_act = @import("mlx_serve/sdk/blocks/mlp/fused_activations.zig");

/// The trims' rows: decode and verify rows
const trim_rows = [_]c_int{ 1, 2, 3, 4 };
/// sc_3.10's swiglu_limit, and the clamped SwiGLU's widths: a shared expert's and a dense layer's
const swiglu_limit: f32 = 10.0;
const swiglu_widths = [_]c_int{ 2048, 12288 };
/// The lines a run prints but the dense pair's (two a row count its plan takes): the DSA pair's two at each rate, kv_a's
/// and q_a's one, a row count; the SwiGLU's one a width and row count
pub const trim_fixed_lines = trim_rows.len * (qkv_rates.len * 2 + 2) + swiglu_widths.len * trim_rows.len;

fn setHostZero(on: bool) void {
    t3.dense.dec_host_zero.get().* = on;
}

fn hostZeroed() u64 {
    return t3.dense.dec_host_zeroed.get().*;
}

const TrimTally = struct {
    lines: usize = 0,
    fails: usize = 0,
    /// the host-made counter arrays the on arms took and the off arms took (none)
    zeroed_on: u64 = 0,
    zeroed_off: u64 = 0,
    fn check(t: *TrimTally, s: mlx.mlx_stream, piece: []const u8, rows: c_int, out: []const u8, want: Array, got: Array, zeroed: u64, want_zeroed: u64) !void {
        const d = try diff(s, want, got);
        const ok = d.total > 0 and d.differing == 0 and d.nonfinite == 0 and zeroed == want_zeroed;
        t.lines += 1;
        t.fails += @intFromBool(!ok);
        std.debug.print("TURBOEXL3_GLM53F_TRIM_BYTES piece={s} rows={d} out={s} dtype={s} differing={d}/{d} nonfinite={d} host_zeroed={d} {s}\n", .{ piece, rows, out, @tagName(mlx.mlx_array_dtype(want)), d.differing, d.total, d.nonfinite, zeroed, if (ok) "PASS" else "FAIL" });
    }
};

/// One linear of x by `p` (`root.linear`), with dec_zero's launch (off) and with host-made counters (on), compared; the
/// on arm takes one counter array where its prepare runs inside (`want_zeroed`)
fn trimLinear(s: mlx.mlx_stream, t: *TrimTally, piece: []const u8, rows: c_int, x: Array, p: t3.Proj, want_zeroed: u64) !void {
    setHostZero(false);
    var z0 = hostZeroed();
    const off = try t3.linear(s, x, p.trellis, p.suh, p.svh, .mul1);
    defer _ = mlx.mlx_array_free(off);
    t.zeroed_off += hostZeroed() - z0;
    setHostZero(true);
    z0 = hostZeroed();
    const on = try t3.linear(s, x, p.trellis, p.suh, p.svh, .mul1);
    defer _ = mlx.mlx_array_free(on);
    const dz = hostZeroed() - z0;
    t.zeroed_on += dz;
    try evalAll(&.{ off, on });
    try t.check(s, piece, rows, "y", off, on, dz, want_zeroed);
}

/// A joined call's two outputs (`which`: 0 the DSA pair, 1 the dense pair), off and on, compared; each on arm takes one
/// counter array. False where the call declines in both arms (nothing compared).
fn trimPair(s: mlx.mlx_stream, t: *TrimTally, comptime which: u8, piece: []const u8, rows: c_int, x: Array, m0: t3.Proj, m1: t3.Proj, names: [2][]const u8) !bool {
    var arms: [2]?[2]Array = .{ null, null };
    defer for (arms) |ar| if (ar) |ys| for (ys) |y| {
        _ = mlx.mlx_array_free(y);
    };
    var dz: [2]u64 = .{ 0, 0 };
    for ([_]bool{ false, true }, 0..) |on, i| {
        setHostZero(on);
        const z0 = hostZeroed();
        arms[i] = switch (which) {
            0 => try smallop.dsaQkv(s, m0, m1, x),
            else => try smallop.denseGateUp(s, m0, m1, x),
        };
        dz[i] = hostZeroed() - z0;
    }
    t.zeroed_off += dz[0];
    t.zeroed_on += dz[1];
    const off = arms[0] orelse {
        if (arms[1] != null or dz[1] != 0) t.fails += 1;
        return false;
    };
    const on = arms[1] orelse {
        t.fails += 1;
        return false;
    };
    try evalAll(&.{ off[0], off[1], on[0], on[1] });
    for (names, off, on) |name, w, g| try t.check(s, piece, rows, name, w, g, dz[1], 1);
    return true;
}

test "turboexl3_glm53f smallop gpu: TURBOEXL3_GLM53F_HOST_ZERO's host-made counters give dec_zero's launch's bits (a DSA layer's q_a and kv_a and a dense layer's gate and up in one launch, and a linear with its prepare inside), and TURBOEXL3_GLM53F_CACHED_SCALARS' kept limit the per-call one's (the clamped SwiGLU at a shared expert's and a dense layer's widths), at decode and verify rows 1 to 4" {
    if (std.c.getenv("TURBOEXL3_GLM53F_SMALLOP_GPU") == null) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const a = std.heap.page_allocator;
    var seed: u64 = 20261109 + 71;
    var prng = std.Random.DefaultPrng.init(20261109 + 72);
    const rnd = prng.random();
    smallop.reset();
    defer smallop.reset();
    smallop.on = true;
    defer smallop.on = false;
    defer setHostZero(false);
    cached_scalars.reset();
    defer cached_scalars.reset();
    defer cached_scalars.on = false;
    var t: TrimTally = .{};
    var dense_taken: usize = 0;

    // the DSA pair at each rate and the dense pair, registered as the bank's opening registers them
    var qp: [qkv_rates.len][2]t3.Proj = undefined;
    var qmade: usize = 0;
    defer for (qp[0..qmade]) |ps| for (ps) |pj| freeProj(pj);
    for (qkv_rates, 0..) |k, ki| {
        qp[ki] = .{ try proj(a, rnd, @intCast(hidden), @intCast(q_width), k), try proj(a, rnd, @intCast(hidden), @intCast(kv_width), k) };
        qmade += 1;
        try smallop.registerQkv(s, qp[ki][0], qp[ki][1]);
    }
    const dgate = try proj(a, rnd, @intCast(hidden), dense_inter, dense_k);
    defer freeProj(dgate);
    const dup = try proj(a, rnd, @intCast(hidden), dense_inter, dense_k);
    defer freeProj(dup);
    try smallop.registerDense(s, dgate, dup);
    const dpair = try t3.glm53f.shared_gu.pairOf(dgate, dup);

    for (trim_rows) |rows| {
        const x = try normal(s, &seed, &.{ 1, rows, hidden }, .bfloat16, 1.0);
        defer _ = mlx.mlx_array_free(x);
        for (qkv_rates, 0..) |k, ki| {
            if (!try trimPair(s, &t, 0, if (k == 6) "host_zero_dsa_qkv_k6" else "host_zero_dsa_qkv_k4", rows, x, qp[ki][0], qp[ki][1], .{ "q_a", "kv_a" })) t.fails += 1;
        }
        const takes = t3.glm53f.shared_gu.plan(dpair, .mul1, @intCast(rows)) != null;
        const ran = try trimPair(s, &t, 1, "host_zero_dense_gu", rows, x, dgate, dup, .{ "gate", "up" });
        if (ran != takes) t.fails += 1;
        dense_taken += @intFromBool(ran);
        // the prepare runs inside at kv_a's 512 outputs at every decode row count, at q_a's 1536 at one row
        try trimLinear(s, &t, "host_zero_linear_kv_a", rows, x, qp[0][1], 1);
        try trimLinear(s, &t, "host_zero_linear_q_a", rows, x, qp[0][0], if (rows <= 1) 1 else 0);
    }

    // the clamped SwiGLU with the limit cast at the call and with the kept one
    const made0 = cached_scalars.made;
    const used0 = cached_scalars.uses;
    for (swiglu_widths) |w| {
        for (trim_rows) |rows| {
            // gate and up at a scale that crosses the limit on both sides
            const gate = try normal(s, &seed, &.{ 1, rows, w }, .bfloat16, 8.0);
            defer _ = mlx.mlx_array_free(gate);
            const up = try normal(s, &seed, &.{ 1, rows, w }, .bfloat16, 8.0);
            defer _ = mlx.mlx_array_free(up);
            cached_scalars.on = false;
            const off = (try fused_act.fusedClampedSwiGLU(s, gate, up, swiglu_limit)) orelse return error.SmallopGpuTrimSwiglu;
            defer _ = mlx.mlx_array_free(off);
            cached_scalars.on = true;
            const on = (try fused_act.fusedClampedSwiGLU(s, gate, up, swiglu_limit)) orelse return error.SmallopGpuTrimSwiglu;
            defer _ = mlx.mlx_array_free(on);
            cached_scalars.on = false;
            try evalAll(&.{ off, on });
            try t.check(s, if (w == 2048) "cached_limit_shared_swiglu" else "cached_limit_dense_swiglu", rows, "act", off, on, 0, 0);
        }
    }
    const made = cached_scalars.made - made0;
    const used = cached_scalars.uses - used0;
    // the on arms' counter arrays: each DSA pair, each dense pair the plan takes, kv_a at every row count and q_a at one
    // row; the off arms' none; one limit made and read by every on call
    const want_zeroed: u64 = trim_rows.len * (qkv_rates.len + 1) + dense_taken + 1;
    const ok = t.fails == 0 and t.lines == trim_fixed_lines + 2 * dense_taken and t.zeroed_on == want_zeroed and t.zeroed_off == 0 and made == 1 and used == swiglu_widths.len * trim_rows.len;
    std.debug.print("TURBOEXL3_GLM53F_TRIM_PARITY lines={d} fails={d} host_zeroed={d} launch_zeroed={d} scalars_made={d} scalars_used={d} {s}\n", .{ t.lines, t.fails, t.zeroed_on, t.zeroed_off, made, used, if (ok) "PASS" else "FAIL" });
    try std.testing.expect(ok);
}

/// DRAFT_S0's test: the step-0 rows, the byte lines (the front's two a row count, the post-norm's two, the tail's and the
/// last-row tail's one) and the decline lines (the front and the tail at one and five rows).
const s0_rows = [_]c_int{ 2, 3, 4 };
pub const s0_byte_lines = s0_rows.len * (2 + 2 + 1 + 1);
pub const s0_decline_lines = 4;

const S0Tally = struct {
    lines: usize = 0,
    fails: usize = 0,
    fn check(t: *S0Tally, s: mlx.mlx_stream, piece: []const u8, rows: c_int, out: []const u8, want: Array, got: Array) !void {
        const d = try diff(s, want, got);
        const ok = d.total > 0 and d.differing == 0 and d.nonfinite == 0;
        t.lines += 1;
        t.fails += @intFromBool(!ok);
        std.debug.print("TURBOEXL3_GLM53F_DRAFT_S0_BYTES piece={s} rows={d} out={s} differing={d}/{d} nonfinite={d} {s}\n", .{ piece, rows, out, d.differing, d.total, d.nonfinite, if (ok) "PASS" else "FAIL" });
    }
    fn declined(t: *S0Tally, piece: []const u8, rows: c_int, got: ?Array) void {
        const ok = got == null;
        if (got) |x| _ = mlx.mlx_array_free(x);
        t.lines += 1;
        t.fails += @intFromBool(!ok);
        std.debug.print("TURBOEXL3_GLM53F_DRAFT_S0_DECLINED piece={s} rows={d} {s}\n", .{ piece, rows, if (ok) "PASS" else "FAIL" });
    }
};

/// The last row of `y` ([1, rows, d]) as a view: [1, 1, d].
fn lastRowOf(s: mlx.mlx_stream, y: Array) !Array {
    const sh = mlx.getShape(y);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_slice(&out, y, &[_]c_int{ 0, sh[1] - 1, 0 }, 3, &[_]c_int{ 1, sh[1], sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    return out;
}

/// The stock step's front over `ids` [1, rows]: the embedding rows taken ([1, rows, hidden]), both norms, the
/// concatenation along the last axis.
fn stockFront(s: mlx.mlx_stream, table: Array, ids: Array, h: Array, we: Array, wh: Array, rows: c_int) !Array {
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, ids, &[_]c_int{rows}, 1, s));
    var taken = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(taken);
    try mlx.check(mlx.mlx_take_axis(&taken, table, flat, 0, s));
    var emb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(emb);
    try mlx.check(mlx.mlx_reshape(&emb, taken, &[_]c_int{ 1, rows, hidden }, 3, s));
    const en = try rmsNorm(s, emb, we);
    defer _ = mlx.mlx_array_free(en);
    const hn = try rmsNorm(s, h, wh);
    defer _ = mlx.mlx_array_free(hn);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    const vec = mlx.mlx_vector_array_new_data(&[_]Array{ en, hn }, 2);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 2, s));
    return out;
}

test "turboexl3_glm53f smallop gpu: TURBOEXL3_GLM53F_DRAFT_S0's step-0 pieces give the stock step's bits at 2, 3 and 4 rows (the front, the first residual and post-norm, the last residual at the last row, with and without HEAD_LAST_ROW's last-row MoE) and decline one and five rows" {
    if (std.c.getenv("TURBOEXL3_GLM53F_SMALLOP_GPU") == null) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var seed: u64 = 20261109 + 91;
    const saved = .{ draft_s0.fronts, draft_s0.tails, draft_s0.tails_last_row, draft_s0.declined };
    draft_s0.fronts = 0;
    draft_s0.tails = 0;
    draft_s0.tails_last_row = 0;
    draft_s0.declined = 0;
    draft_s0.on = true;
    defer {
        draft_s0.on = false;
        draft_s0.fronts = saved[0];
        draft_s0.tails = saved[1];
        draft_s0.tails_last_row = saved[2];
        draft_s0.declined = saved[3];
    }
    var t: S0Tally = .{};
    const eps = mlx.mlx_array_new_float(rms_eps);
    defer _ = mlx.mlx_array_free(eps);
    // a table of 64 rows; the front's ids walk it, the uint32 ids ending at its last row
    const table = try normal(s, &seed, &.{ 64, hidden }, .bfloat16, 1.0);
    defer _ = mlx.mlx_array_free(table);
    const we = try weight(s, &seed, hidden);
    defer _ = mlx.mlx_array_free(we);
    const wh = try weight(s, &seed, hidden);
    defer _ = mlx.mlx_array_free(wh);
    const wp = try weight(s, &seed, hidden);
    defer _ = mlx.mlx_array_free(wp);
    const wn = try weight(s, &seed, hidden);
    defer _ = mlx.mlx_array_free(wn);
    for (s0_rows) |rows| {
        const n: usize = @intCast(rows);
        // the front: int32 ids 5, 19, 33, .. and uint32 ids ending at 63
        {
            const h = try normal(s, &seed, &.{ 1, rows, hidden }, .bfloat16, 1.0);
            defer _ = mlx.mlx_array_free(h);
            var id32: [4]i32 = undefined;
            var idu: [4]u32 = undefined;
            for (0..n) |i| {
                id32[i] = @intCast(5 + 14 * i);
                idu[i] = @intCast(63 - (n - 1 - i) * 9);
            }
            const ids = [2]Array{
                mlx.mlx_array_new_data(&id32, &[_]c_int{ 1, rows }, 2, .int32),
                mlx.mlx_array_new_data(&idu, &[_]c_int{ 1, rows }, 2, .uint32),
            };
            defer for (ids) |x| {
                _ = mlx.mlx_array_free(x);
            };
            for (ids, [_][]const u8{ "cat_int32", "cat_uint32" }) |id, name| {
                const want = try stockFront(s, table, id, h, we, wh, rows);
                defer _ = mlx.mlx_array_free(want);
                const got = (try draft_s0.front(s, table, false, id, h, we, wh, eps)) orelse return error.DraftS0GpuFrontDeclined;
                defer _ = mlx.mlx_array_free(got);
                try t.check(s, "front", rows, name, want, got);
            }
        }
        // the first residual and post-norm: the kit's addNorm (head.zig's narrowAddNorm) against the stock add and norm
        {
            const x = try normal(s, &seed, &.{ 1, rows, hidden }, .bfloat16, 4.0);
            defer _ = mlx.mlx_array_free(x);
            const attn = try normal(s, &seed, &.{ 1, rows, hidden }, .bfloat16, 1.0);
            defer _ = mlx.mlx_array_free(attn);
            const x1 = try add(s, x, attn);
            defer _ = mlx.mlx_array_free(x1);
            const n1 = try rmsNorm(s, x1, wp);
            defer _ = mlx.mlx_array_free(n1);
            const r = (try add_norm.addNorm(x, attn, wp, eps, s)) orelse return error.DraftS0GpuPostDeclined;
            defer {
                _ = mlx.mlx_array_free(r.h);
                _ = mlx.mlx_array_free(r.normed);
            }
            try t.check(s, "post", rows, "h", x1, r.h);
            try t.check(s, "post", rows, "normed", n1, r.normed);
        }
        // the last residual at the last row: the MoE's two terms, then the residual over every row, the last row, the norm
        {
            const h = try normal(s, &seed, &.{ 1, rows, hidden }, .bfloat16, 4.0);
            defer _ = mlx.mlx_array_free(h);
            const ya = try normal(s, &seed, &.{ 1, rows, hidden }, .bfloat16, 1.0);
            defer _ = mlx.mlx_array_free(ya);
            const yb = try normal(s, &seed, &.{ 1, rows, hidden }, .bfloat16, 0.5);
            defer _ = mlx.mlx_array_free(yb);
            const mlp = try add(s, ya, yb);
            defer _ = mlx.mlx_array_free(mlp);
            const y = try add(s, h, mlp);
            defer _ = mlx.mlx_array_free(y);
            const last = try lastRowOf(s, y);
            defer _ = mlx.mlx_array_free(last);
            const want = try rmsNorm(s, last, wn);
            defer _ = mlx.mlx_array_free(want);
            const got = (try draft_s0.add2NormLast(s, h, ya, yb, wn, eps)) orelse return error.DraftS0GpuTailDeclined;
            defer _ = mlx.mlx_array_free(got);
            try t.check(s, "tail", rows, "normed", want, got);
            // beside HEAD_LAST_ROW: the last-row MoE output; `tailLast` takes the residual's last row, adds, norms
            const mlp1 = try normal(s, &seed, &.{ 1, 1, hidden }, .bfloat16, 1.0);
            defer _ = mlx.mlx_array_free(mlp1);
            const hl = try lastRowOf(s, h);
            defer _ = mlx.mlx_array_free(hl);
            const yl = try add(s, hl, mlp1);
            defer _ = mlx.mlx_array_free(yl);
            const want1 = try rmsNorm(s, yl, wn);
            defer _ = mlx.mlx_array_free(want1);
            const got1 = (try draft_s0.addNormLast(s, h, mlp1, wn, eps)) orelse return error.DraftS0GpuTailLastDeclined;
            defer _ = mlx.mlx_array_free(got1);
            try t.check(s, "tail_last_row", rows, "normed", want1, got1);
        }
    }
    // one row (step 1: SMALLOP's own pieces) and five rows decline
    for ([_]c_int{ 1, 5 }) |rows| {
        const n: usize = @intCast(rows);
        const h = try normal(s, &seed, &.{ 1, rows, hidden }, .bfloat16, 1.0);
        defer _ = mlx.mlx_array_free(h);
        var id32: [5]i32 = undefined;
        for (0..n) |i| id32[i] = @intCast(i);
        const ids = mlx.mlx_array_new_data(&id32, &[_]c_int{ 1, rows }, 2, .int32);
        defer _ = mlx.mlx_array_free(ids);
        t.declined("front", rows, try draft_s0.front(s, table, false, ids, h, we, wh, eps));
        t.declined("tail", rows, try draft_s0.add2NormLast(s, h, h, h, wn, eps));
    }
    const counts_ok = draft_s0.fronts == 2 * s0_rows.len and draft_s0.tails == s0_rows.len and draft_s0.tails_last_row == s0_rows.len and draft_s0.declined == s0_decline_lines;
    const ok = t.fails == 0 and t.lines == s0_byte_lines + s0_decline_lines and counts_ok;
    std.debug.print("TURBOEXL3_GLM53F_DRAFT_S0_PARITY lines={d} fails={d} fronts={d} tails={d} tails_last_row={d} declined={d} {s}\n", .{ t.lines, t.fails, draft_s0.fronts, draft_s0.tails, draft_s0.tails_last_row, draft_s0.declined, if (ok) "PASS" else "FAIL" });
    try std.testing.expect(ok);
}
