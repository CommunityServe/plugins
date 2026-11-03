//! `MLX_SERVE_DSV41_FLASH_ATTN` (read at load; unset, empty or `0`: off; `1`: on; `check`: off's gather served, the flash
//! kernel run beside every wide span and compared; anything else is refused by name): a wide prompt layer's attention
//! span as one kernel (kernels/attn_flash.metal) in place of the gather's three launches (Q_TAIL's qk_t, sm, pv) and the
//! SCRATCH_ATTN ring's scores and probabilities.
//!
//! One threadgroup is a query row and `hgn` groups of 16 heads, which share the row's keys (one KV head; the keys are the
//! values). The keys go in blocks of 128 (the window's rows, then the picks in their order); per block each simdgroup
//! scores its 32 keys for its 16 heads on the tensor units, the block's max joins the running max, exp(scale * s - m)
//! goes to threadgroup memory as f16, and the simdgroup accumulates its 128 output dims over the block, its f32
//! accumulator scaled by exp(m_old - m_new). The sink starts the state (m = sink, l = 1). Epilogue: O / l, the last 64
//! dims inverse-roped, stored grouped, or (with SCRATCH_HEADS) straight into the run region as pv's heads twin does.
//!
//! NOT exact: the probabilities round to f16 against the running max before the division (the gather's sm divides
//! first), the accumulator is rescaled in f32, and the sum's order differs. `check` reports per wide layer the heads'
//! bf16 words that differ, the largest absolute difference and the largest relative one (floored at the served heads'
//! rms), on sampled spans the picks' overlap between neighbouring rows and the probability blocks no head needs, and
//! `MLX_SERVE_DSV41_FLASH_DUMP=<absolute dir>` writes each request's last prompt row's logits (f32) as
//! `fa1-logits-<pid>-<request>.bin`, for a reader to compare arms.

const std = @import("std");
const mlx = @import("mlx_serve").mlx;

const A = mlx.mlx_array;
const none: A = .{ .ctx = null };

pub const plan = @import("flash_plan.zig");
pub const Mode = plan.Mode;
pub const Error = plan.Error;
pub const parse = plan.parse;
pub const hgn = plan.hgn;
pub const source = plan.source;
pub const regionTwin = plan.regionTwin;

/// Off until the plugin's load configures it (on unless the setting says otherwise): a harness that loads the model
/// without the plugin's configure keeps the gather's words.
pub var mode: Mode = .off;
/// The setting named on itself (`=1`), not by default: a span the kernel cannot take is refused, not the gather's.
pub var explicit: bool = false;

pub fn configure(value: ?[]const u8, dump: ?[]const u8) Error!void {
    mode = try parse(value);
    explicit = plan.explicitOn(value);
    dump_dir = try plan.parseDump(dump);
}

// ── check ──

/// A wide layer's `check` tallies: spans compared, their words, bf16 words that differ, the largest absolute and
/// relative differences (relative: floored at the span's served rms), and the sums of squares (difference, served).
pub const Tally = struct { spans: u64 = 0, words: u64 = 0, differ: u64 = 0, max_abs: f32 = 0, max_rel: f32 = 0, se: f64 = 0, ss: f64 = 0 };
pub var tally: Tally = .{};

/// `check`: the flash heads `got` against the served gather's `want` (one shape, bf16), on the device, into `tally`.
pub fn compare(s: mlx.mlx_stream, got: A, want: A) !void {
    var t: [16]A = undefined;
    for (&t) |*x| x.* = mlx.mlx_array_new();
    defer for (t) |x| {
        _ = mlx.mlx_array_free(x);
    };
    try mlx.check(mlx.mlx_view(&t[0], got, .uint16, s));
    try mlx.check(mlx.mlx_view(&t[1], want, .uint16, s));
    try mlx.check(mlx.mlx_equal(&t[2], t[0], t[1], s));
    try mlx.check(mlx.mlx_astype(&t[3], t[2], .float32, s));
    try mlx.check(mlx.mlx_sum(&t[4], t[3], false, s));
    try mlx.check(mlx.mlx_astype(&t[5], got, .float32, s));
    try mlx.check(mlx.mlx_astype(&t[6], want, .float32, s));
    try mlx.check(mlx.mlx_subtract(&t[7], t[5], t[6], s));
    try mlx.check(mlx.mlx_abs(&t[8], t[7], s));
    try mlx.check(mlx.mlx_max(&t[9], t[8], false, s));
    try mlx.check(mlx.mlx_square(&t[10], t[7], s));
    try mlx.check(mlx.mlx_sum(&t[11], t[10], false, s));
    try mlx.check(mlx.mlx_square(&t[12], t[6], s));
    try mlx.check(mlx.mlx_sum(&t[13], t[12], false, s));
    const parts = [_]A{ t[4], t[9], t[11], t[13] };
    const vp = mlx.mlx_vector_array_new_data(&parts, parts.len);
    defer _ = mlx.mlx_vector_array_free(vp);
    try mlx.check(mlx.mlx_stack_axis(&t[14], vp, 0, s));
    try mlx.check(mlx.mlx_array_eval(t[14]));
    const r = mlx.mlx_array_data_float32(t[14]) orelse return error.NoData;
    const n: u64 = mlx.mlx_array_size(got);
    const same: u64 = @intFromFloat(r[0]);
    const nf: f64 = @floatFromInt(n);
    const rms: f32 = @floatCast(@sqrt(@as(f64, r[3]) / nf));
    // the largest relative difference, floored at this span's served rms
    var u: [5]A = undefined;
    u[0] = mlx.mlx_array_new_float(@max(rms, std.math.floatMin(f32)));
    for (u[1..]) |*x| x.* = mlx.mlx_array_new();
    defer for (u) |x| {
        _ = mlx.mlx_array_free(x);
    };
    try mlx.check(mlx.mlx_abs(&u[1], t[6], s));
    try mlx.check(mlx.mlx_maximum(&u[2], u[1], u[0], s));
    try mlx.check(mlx.mlx_divide(&u[3], t[8], u[2], s));
    try mlx.check(mlx.mlx_max(&u[4], u[3], false, s));
    try mlx.check(mlx.mlx_array_eval(u[4]));
    const rel = (mlx.mlx_array_data_float32(u[4]) orelse return error.NoData)[0];
    tally.spans += 1;
    tally.words += n;
    tally.differ += n - @min(n, same);
    tally.max_abs = @max(tally.max_abs, r[1]);
    tally.max_rel = @max(tally.max_rel, rel);
    tally.se += r[2];
    tally.ss += r[3];
}

/// `check`: the wide layer's line; the tallies start over.
pub fn report(layer: usize) void {
    const t = tally;
    if (t.spans > 0) std.debug.print("[flash-attn] check layer={d} spans={d} words={d} differ={d} differ_pct={d:.3} max_abs={d:.6} max_rel={d:.6} rms_rel={d:.6}\n", .{ layer, t.spans, t.words, t.differ, 100.0 * @as(f64, @floatFromInt(t.differ)) / @as(f64, @floatFromInt(@max(t.words, 1))), t.max_abs, t.max_rel, @sqrt(t.se / @max(t.ss, std.math.floatMin(f64))) });
    tally = .{};
}

// ── check: the sorting questions' data (main 2026-11-03), on sampled spans ──

/// The span `wideLayer` is about to run: its index, the layer's span count, the layer and whether it is an index layer.
var span_k: usize = 0;
var span_n: usize = 0;
var span_layer: usize = 0;
var span_index_layer = false;
pub fn spanBegin(k: usize, n: usize, layer: usize, index_layer: bool) void {
    span_k = k;
    span_n = n;
    span_layer = layer;
    span_index_layer = index_layer;
}
/// `check` samples the middle and the last span of each wide layer.
pub fn sampled() bool {
    return mode == .check and span_n > 1 and (span_k == span_n / 2 or span_k + 1 == span_n);
}

/// `check`, a sampled span: the picks' stats (index layers: the six layers after one share its picks) and the
/// probabilities' (`p` the gather's fresh [rows, heads, nk] f16, `picks` [rows, k] i32 or null at a window-only layer).
pub fn sampleLine(gpa: std.mem.Allocator, s: mlx.mlx_stream, p: A, picks: ?A, rows: usize, heads: usize, nk: usize, k: usize) !void {
    _ = s;
    var ps: plan.PickStats = .{};
    if (picks) |pk| if (span_index_layer and k > 0) {
        try mlx.check(mlx.mlx_array_eval(pk));
        const d = mlx.mlx_array_data_int32(pk) orelse return error.NoData;
        ps = try plan.pickStats(gpa, d[0 .. rows * k], rows, k);
    };
    try mlx.check(mlx.mlx_array_eval(p));
    const ph = mlx.mlx_array_data_float16(p) orelse return error.NoData;
    const pf = try gpa.alloc(f32, rows * heads * nk);
    defer gpa.free(pf);
    for (pf, ph[0..pf.len]) |*o, v| o.* = v;
    const ms = plan.massStats(pf, rows, heads, nk);
    std.debug.print("[flash-attn] sample layer={d} span={d}/{d} nk={d} picks_neigh={d:.4} union4={d:.3} union8={d:.3} union16={d:.3} blocks={d} skip_all_2^-11={d:.4} skip_all_2^-8={d:.4} skip_head_2^-11={d:.4}\n", .{ span_layer, span_k, span_n, nk, ps.neigh, ps.union4, ps.union8, ps.union16, ms.blocks, ms.all11, ms.all8, ms.head11 });
}

// ── MLX_SERVE_DSV41_FLASH_DUMP ──

pub var dump_dir: ?[]const u8 = null;
var dump_req: u64 = 0;

/// A request begins (the prompt's resume): its dumps take the next number.
pub fn requestBegin() void {
    dump_req += 1;
}

/// A prompt forward's `logits` [R, vocab]: its last row as f32 into `<dir>/fa1-logits-<pid>-<request>.bin` (each prompt
/// forward of the request overwrites it, so the last chunk's row stays). Off without the dump dir; a failure is a line.
pub fn dumpLogits(s: mlx.mlx_stream, logits: A) void {
    const dir = dump_dir orelse return;
    dumpOne(s, dir, logits) catch |e| std.debug.print("[flash-attn] dump failed: {s}\n", .{@errorName(e)});
}

fn dumpOne(s: mlx.mlx_stream, dir: []const u8, logits: A) !void {
    const sh = mlx.getShape(logits);
    if (sh.len != 2 or sh[0] < 1) return error.FlashDumpShape;
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, logits, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f));
    const d = mlx.mlx_array_data_float32(f) orelse return error.NoData;
    const v: usize = @intCast(sh[1]);
    const r: usize = @intCast(sh[0]);
    const row = d[(r - 1) * v ..][0..v];
    var path_buf: [1024]u8 = undefined;
    const path = std.fmt.bufPrint(path_buf[0 .. path_buf.len - 1], "{s}/fa1-logits-{d}-{d}.bin", .{ dir, std.c.getpid(), dump_req }) catch return error.FlashDumpPath;
    path_buf[path.len] = 0;
    const fd = std.c.open(path_buf[0..path.len :0].ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o644));
    if (fd < 0) return error.FlashDumpOpen;
    defer _ = std.c.close(fd);
    const bytes = std.mem.sliceAsBytes(row);
    var done: usize = 0;
    while (done < bytes.len) {
        const w = std.c.write(fd, bytes[done..].ptr, bytes.len - done);
        if (w <= 0) return error.FlashDumpWrite;
        done += @intCast(w);
    }
}

