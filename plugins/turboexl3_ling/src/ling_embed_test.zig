//! EMBED_ROWS' byte test (GPU; `MLX_SERVE_LING_GPU_TESTS=1` runs it, else it skips): the embedding's rows read from the
//! pack (src/ling_embed.zig `rowsOf`, which the copied trunk.zig `embedding` returns) against the resident table's
//! take, computed here as the trunk computed it before EMBED_ROWS (MLX's take of the flattened ids over the evaluated
//! table, the astype to bf16, the reshape).
//! - The tables: a test pack's in a fresh /tmp folder (2,048 x 4,096 finite bf16 random words, after another tensor so
//!   its data does not start at 0; src/ling_embed_rows.zig `TestPack`), and with MLX_SERVE_LING_PACK=<pack dir> the
//!   pack's own `model.word_embeddings.weight` (157,184 x 4,096), the reference read as src/ling_weights.zig reads the
//!   shards.
//! - The ids at the forwards the trunk embeds: decode's 1 row and a verify's 2 and 4, each still being computed on the
//!   GPU when `rowsOf` takes it (so the read waits for it, as a decode-phase forward's does), as int32, uint32 and
//!   int64; an MTP head history of 5; a prompt span of 4,096 host ids with repeats and the table's first and last rows;
//!   and a span's [1, L] view of a longer prompt's ids (the layer-major pass's `hiddenSpans`).
//! - Checks: every word of every case, read against the take; the output's dtype and shape; the run-path mark (absent
//!   before the first read, present after); MLX's memory in use grows by the output array alone (no table made
//!   resident); `keepsLazy` holds the table's handle back from the trunk's eval with or without a table open.
//! - Red checks in the same run: the rows one id over differ; one flipped bit counts as one differing word; an id past
//!   the table and a negative id are refused by name; with no table open (before `open`, after `close`) the rows are
//!   refused as LingEmbedRowsNotOpen, since the trunk has no resident table to take from.
//! Each check prints a `TURBOEXL3_LING_EMBED_ROWS` line ending in PASS or FAIL; the last line counts them.
const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const ling_embed = @import("ling_embed.zig");
const er = @import("ling_embed_rows.zig");
const ling_weights = @import("ling_weights.zig");
const ran = @import("ling_ran.zig");

const Array = mlx.mlx_array;
const Stream = mlx.mlx_stream;
const tag = "TURBOEXL3_LING_EMBED_ROWS";
const gpa = std.heap.c_allocator;
const key = er.TestPack.test_key;
const hidden: u32 = 4096;

fn gpu() !Stream {
    // the GPU tests' own variable (test-only): at 1 they run, at any other value or unset they skip
    const run_gpu = std.c.getenv("MLX_SERVE_LING_GPU_TESTS") orelse return error.SkipZigTest;
    if (!std.mem.eql(u8, std.mem.span(run_gpu), "1")) return error.SkipZigTest;
    mlx.installErrorHandler();
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    return s;
}

fn free(a: Array) void {
    if (a.ctx != null) _ = mlx.mlx_array_free(a);
}

/// The resident table's take, as the copied trunk.zig `embedding` ran it before EMBED_ROWS, over the evaluated table
/// `table` (bf16 [V, hidden]): the take of the flattened `ids`, the astype to bf16, the reshape to [B, L, hidden];
/// evaluated. The reference is computed here, not through the trunk.
fn takeRows(s: Stream, table: Array, ids: Array) !Array {
    const sh = mlx.getShape(ids);
    var flat = mlx.mlx_array_new();
    defer free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, ids, &[_]c_int{sh[0] * sh[1]}, 1, s));
    var tw = mlx.mlx_array_new();
    defer free(tw);
    try mlx.check(mlx.mlx_take_axis(&tw, table, flat, 0, s));
    var emb = mlx.mlx_array_new();
    defer free(emb);
    try mlx.check(mlx.mlx_astype(&emb, tw, .bfloat16, s));
    var out = mlx.mlx_array_new();
    errdefer free(out);
    try mlx.check(mlx.mlx_reshape(&out, emb, &[_]c_int{ sh[0], sh[1], @intCast(hidden) }, 3, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

/// The words of `a` and `b` (bf16, evaluated, row-contiguous) that differ; maxInt when their sizes differ.
fn differing(a: Array, b: Array) usize {
    const n = mlx.mlx_array_size(a);
    if (mlx.mlx_array_size(b) != n) return std.math.maxInt(usize);
    const pa = mlx.mlx_array_data_bfloat16(a) orelse return std.math.maxInt(usize);
    const pb = mlx.mlx_array_data_bfloat16(b) orelse return std.math.maxInt(usize);
    var d: usize = 0;
    for (0..n) |i| d += @intFromBool(pa[i] != pb[i]);
    return d;
}

/// Whether `r`, a `rowsOf` result, is the refusal `name`; rows given are freed.
fn refusedAs(r: anyerror!Array, name: anyerror) bool {
    const got = r catch |e| return e == name;
    free(got);
    return false;
}

/// How a case hands `rowsOf` its ids: the host's (evaluated) int32, int32 or uint32 still being computed on the GPU
/// (an add of 0 not yet run), an int64 cast not yet run, or a [1, L] view of a longer host array.
const Kind = enum { host_i32, lazy_i32, lazy_u32, lazy_i64, view };

/// `vals` as ids [1, vals.len] of `kind` (the caller frees them).
fn idsOf(s: Stream, vals: []const u32, kind: Kind) !Array {
    const n: c_int = @intCast(vals.len);
    switch (kind) {
        .lazy_u32 => {
            const base = mlx.mlx_array_new_data(vals.ptr, &[_]c_int{ 1, n }, 2, .uint32);
            defer free(base);
            const zero_val: u32 = 0;
            const zero = mlx.mlx_array_new_data(&zero_val, &[_]c_int{}, 0, .uint32);
            defer free(zero);
            var out = mlx.mlx_array_new();
            errdefer free(out);
            try mlx.check(mlx.mlx_add(&out, base, zero, s));
            return out;
        },
        .view => {
            const pad: usize = 500;
            const long = try gpa.alloc(i32, vals.len + 2 * pad);
            defer gpa.free(long);
            @memset(long, 7);
            for (vals, 0..) |v, i| long[pad + i] = @intCast(v);
            const base = mlx.mlx_array_new_data(long.ptr, &[_]c_int{ 1, @intCast(long.len) }, 2, .int32);
            defer free(base);
            var out = mlx.mlx_array_new();
            errdefer free(out);
            try mlx.check(mlx.mlx_slice(&out, base, &[_]c_int{ 0, @intCast(pad) }, 2, &[_]c_int{ 1, @intCast(pad + vals.len) }, 2, &[_]c_int{ 1, 1 }, 2, s));
            return out;
        },
        else => {
            const as_i32 = try gpa.alloc(i32, vals.len);
            defer gpa.free(as_i32);
            for (vals, as_i32) |v, *o| o.* = @intCast(v);
            const base = mlx.mlx_array_new_data(as_i32.ptr, &[_]c_int{ 1, n }, 2, .int32);
            if (kind == .host_i32) return base;
            defer free(base);
            var out = mlx.mlx_array_new();
            errdefer free(out);
            if (kind == .lazy_i32) {
                const zero = mlx.mlx_array_new_int(0);
                defer free(zero);
                try mlx.check(mlx.mlx_add(&out, base, zero, s));
            } else try mlx.check(mlx.mlx_astype(&out, base, .int64, s));
            return out;
        },
    }
}

const Counts = struct {
    lines: usize = 0,
    failed: usize = 0,
    reds: usize = 0,

    fn check(c: *Counts, ok: bool, red: bool, comptime fmt: []const u8, args: anytype) void {
        c.lines += 1;
        c.failed += @intFromBool(!ok);
        c.reds += @intFromBool(ok and red);
        std.debug.print(tag ++ " " ++ fmt ++ " {s}\n", args ++ .{if (ok) "PASS" else "FAIL"});
    }
};

/// One case: `rowsOf` of `vals` handed as `kind`, against `takeRows` over `table`, word for word; the dtype and shape
/// checked too. `rowsOf` runs first, so lazy ids are still unevaluated when it takes them.
fn oneCase(c: *Counts, s: Stream, src: []const u8, table: Array, vals: []const u32, kind: Kind) !void {
    const ids = try idsOf(s, vals, kind);
    defer free(ids);
    const got = ling_embed.rowsOf(s, ids, hidden) catch |e| {
        c.check(false, false, "{s}: {d} ids as {t}: the rows refused as {s}", .{ src, vals.len, kind, @errorName(e) });
        return;
    };
    defer free(got);
    try mlx.check(mlx.mlx_array_eval(got));
    const want = try takeRows(s, table, ids);
    defer free(want);
    const sh = mlx.getShape(got);
    const shape_ok = sh.len == 3 and sh[0] == 1 and sh[1] == @as(c_int, @intCast(vals.len)) and sh[2] == @as(c_int, @intCast(hidden));
    const d = differing(got, want);
    c.check(shape_ok and mlx.mlx_array_dtype(got) == .bfloat16 and d == 0, false, "{s}: {d} ids as {t}: bf16 [1, {d}, {d}], {d} of {d} words differ from the resident table's take", .{ src, vals.len, kind, vals.len, hidden, d, vals.len * hidden });
}

/// `n` ids below `rows`: random, every fifth one a repeat of the one before, and (past 2) the table's first and last rows.
fn someIds(rnd: std.Random, n: usize, rows: u32) ![]u32 {
    const out = try gpa.alloc(u32, n);
    for (out, 0..) |*v, i| v.* = if (i % 5 == 4) out[i - 1] else rnd.uintLessThan(u32, rows);
    if (n > 2) {
        out[0] = 0;
        out[n - 1] = rows - 1;
    }
    return out;
}

/// The cases over one table: decode, a verify, the head's history, a prompt span (host ids) and a span's view.
fn cases(c: *Counts, s: Stream, src: []const u8, table: Array, rows: u32, rnd: std.Random) !void {
    const plan = [_]struct { n: usize, kind: Kind }{
        .{ .n = 1, .kind = .lazy_u32 },
        .{ .n = 2, .kind = .lazy_i32 },
        .{ .n = 4, .kind = .lazy_u32 },
        .{ .n = 4, .kind = .lazy_i64 },
        .{ .n = 5, .kind = .lazy_i32 },
        .{ .n = 4096, .kind = .host_i32 },
        .{ .n = 1965, .kind = .view },
    };
    for (plan) |p| {
        const vals = try someIds(rnd, p.n, rows);
        defer gpa.free(vals);
        try oneCase(c, s, src, table, vals, p.kind);
    }
}

test "turboexl3_ling EMBED_ROWS: the embedding's rows read from the pack give the resident table's take word for word at decode, verify, head-history and prompt-span ids (lazy int32, uint32 and int64, host int32, a span's view), make no table resident, mark the run path, refuse an id past the table, and are refused by name with no table open (MLX_SERVE_LING_GPU_TESTS=1)" {
    const s = try gpu();
    ran.clearForTests();
    defer ran.clearForTests();
    defer ling_embed.close();
    var c: Counts = .{};
    var prng = std.Random.DefaultPrng.init(0xE3B0);
    const rnd = prng.random();
    // the test pack: finite bf16 words (a normal f32's top half), so no NaN payload can be at stake in a copy
    const rows: u32 = 2048;
    const words = try gpa.alloc(u16, @as(usize, rows) * hidden);
    defer gpa.free(words);
    for (words) |*w| w.* = @truncate(@as(u32, @bitCast(rnd.floatNorm(f32))) >> 16);
    var pack = try er.TestPack.make(gpa, .{ .rows = rows, .cols = hidden, .words = words, .tag = 0xE3 });
    defer pack.deinit(gpa);
    const io = std.Io.Threaded.global_single_threaded.io();
    // red: with no table open, the rows are refused by name (the trunk has no resident table to take from), and no
    // run-path mark is set
    {
        const ids_none = try idsOf(s, &.{ 1, 2 }, .host_i32);
        defer free(ids_none);
        c.check(refusedAs(ling_embed.rowsOf(s, ids_none, hidden), error.LingEmbedRowsNotOpen) and !ran.has(.EMBED_ROWS), true, "red: nothing open: the rows refused as LingEmbedRowsNotOpen, no run-path mark", .{});
    }
    {
        const w = try ling_weights.Weights.open(gpa, io, pack.dir, "model.layers.", s);
        defer w.deinit();
        const table: Array = .{ .ctx = w.src.tensor(key).ctx };
        // the trunk's eval holds the table's handle back with no table open and with one open, and no other array
        const other = mlx.mlx_array_new_int(0);
        defer free(other);
        const lazy_closed = ling_embed.keepsLazy(table, table);
        try ling_embed.open(pack.dir, key, rows, hidden);
        c.check(lazy_closed and ling_embed.keepsLazy(table, table) and !ling_embed.keepsLazy(other, table), false, "keepsLazy: the table's handle held back with no table open and with one open, another array not", .{});
        // memory: a 4,096-row read adds its output array (32 MiB) to MLX's memory in use, and no table
        {
            const vals = try someIds(rnd, 4096, rows);
            defer gpa.free(vals);
            const ids = try idsOf(s, vals, .host_i32);
            defer free(ids);
            _ = mlx.mlx_clear_cache();
            var before: usize = 0;
            _ = mlx.mlx_get_active_memory(&before);
            const got = try ling_embed.rowsOf(s, ids, hidden);
            defer free(got);
            try mlx.check(mlx.mlx_array_eval(got));
            var after: usize = 0;
            _ = mlx.mlx_get_active_memory(&after);
            const out_bytes: usize = vals.len * hidden * 2;
            const grew = after -| before;
            c.check(grew >= out_bytes and grew <= out_bytes + (64 << 10), false, "memory: MLX in use grew {d} B for a {d} B output (no table: it would add {d} B)", .{ grew, out_bytes, @as(usize, rows) * hidden * 2 });
            c.check(ran.has(.EMBED_ROWS), false, "the run-path mark EMBED_ROWS after the first read", .{});
        }
        // the table, resident for the reference only from here on
        try mlx.check(mlx.mlx_array_eval(table));
        try cases(&c, s, "test pack", table, rows, rnd);
        // red: the rows one id over differ from the take
        {
            const vals = try someIds(rnd, 64, rows - 1);
            defer gpa.free(vals);
            const plus = try gpa.alloc(u32, vals.len);
            defer gpa.free(plus);
            for (vals, plus) |v, *p| p.* = v + 1;
            const ids = try idsOf(s, vals, .host_i32);
            defer free(ids);
            const ids_plus = try idsOf(s, plus, .host_i32);
            defer free(ids_plus);
            const got = try ling_embed.rowsOf(s, ids_plus, hidden);
            defer free(got);
            try mlx.check(mlx.mlx_array_eval(got));
            const want = try takeRows(s, table, ids);
            defer free(want);
            const d = differing(got, want);
            c.check(d > 0 and d != std.math.maxInt(usize), true, "red: the rows one id over: {d} of {d} words differ", .{ d, vals.len * hidden });
            // red: one flipped bit of the rows read is one differing word
            const n = mlx.mlx_array_size(want);
            const copy = try gpa.alloc(u16, n);
            defer gpa.free(copy);
            const po = mlx.mlx_array_data_bfloat16(want) orelse return error.NoData;
            @memcpy(copy, po[0..n]);
            copy[n / 3] ^= 1;
            const flipped = mlx.mlx_array_new_data(copy.ptr, mlx.getShape(want).ptr, 3, .bfloat16);
            defer free(flipped);
            c.check(differing(flipped, want) == 1, true, "red: one flipped bit counts as {d} differing word", .{differing(flipped, want)});
        }
        // red: an id past the table, and a negative one, refused by name
        {
            const ids = try idsOf(s, &.{ 3, rows }, .lazy_u32);
            defer free(ids);
            c.check(refusedAs(ling_embed.rowsOf(s, ids, hidden), error.LingEmbedRowsTokenPastTable), true, "red: id {d} past a {d}-row table refused as LingEmbedRowsTokenPastTable", .{ rows, rows });
            const neg = [_]i32{ 5, -1 };
            const nids = mlx.mlx_array_new_data(&neg, &[_]c_int{ 1, 2 }, 2, .int32);
            defer free(nids);
            c.check(refusedAs(ling_embed.rowsOf(s, nids, hidden), error.LingEmbedRowsTokenPastTable), true, "red: id -1 refused as LingEmbedRowsTokenPastTable", .{});
        }
        ling_embed.close();
        const ids_closed = try idsOf(s, &.{ 1, 2 }, .host_i32);
        defer free(ids_closed);
        c.check(refusedAs(ling_embed.rowsOf(s, ids_closed, hidden), error.LingEmbedRowsNotOpen), true, "red: closed: the rows refused as LingEmbedRowsNotOpen", .{});
    }
    // the pack's own table, where one is named
    if (std.c.getenv("MLX_SERVE_LING_PACK")) |raw| {
        const dir = std.mem.span(raw);
        const vocab: u32 = 157184;
        const w = try ling_weights.Weights.open(gpa, io, dir, "model.layers.", s);
        defer w.deinit();
        const table: Array = .{ .ctx = w.src.tensor(key).ctx };
        if (table.ctx == null) {
            c.check(false, false, "the pack: no {s}", .{key});
        } else {
            try ling_embed.open(dir, key, vocab, hidden);
            defer ling_embed.close();
            const t = ling_embed.servedTable().?;
            std.debug.print(tag ++ " the pack: {s} in {s} at byte {d}, {d} x {d} ({d} MiB)\n", .{ key, ling_embed.servedShard(), t.offset, t.rows, t.cols, t.bytes() >> 20 });
            try mlx.check(mlx.mlx_array_eval(table));
            try cases(&c, s, "the pack", table, vocab, rnd);
        }
    }
    std.debug.print(tag ++ " DONE lines={d} failed={d} reds={d}\n", .{ c.lines, c.failed, c.reds });
    try std.testing.expectEqual(@as(usize, 0), c.failed);
}
