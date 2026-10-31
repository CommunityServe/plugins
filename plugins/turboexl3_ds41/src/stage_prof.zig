//! The stage profiler (`MLX_SERVE_DSV41_STAGE_PROF`, read at load): each forward's time by stage, summed over a request
//! by the forward's kind and rows and logged at its end (the table's `meters`), or every N forwards as well. The copied
//! model marks its stages (patches/turboexl3-stage-prof.patch): a mark evaluates the stage's outputs and reads a
//! monotonic clock, and the time since the previous mark is the stage's.
//!
//! The times are stage-synced: the GPU drains at every mark, so nothing overlaps (the host building the next stage's
//! graph, the GPU running this one) and their sum is above an unprofiled forward's time. They say where a forward's
//! time goes, not what it serves at.
//!
//! Off (the default) every hook returns before it touches an array: the graphs, when they are evaluated and the
//! compiled decode stretches are the unprofiled ones. On, the model compiles no decode stretch (`compileRegions`): each
//! runs its body eager, the same ops on the same kernels, so the marks inside it can sync.

const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const log = @import("mlx_serve/sdk/util/log.zig");

/// What a forward is: a decode step (one row), a DSpark verify (its rows, `t1` and the drafts), a prompt chunk (its
/// rows), the DSpark stages' draft.
pub const Kind = enum { decode, verify, prompt, draft };

/// A forward's stages, in its order. A layer's stages are marked once a layer (once a span of a wide prompt chunk,
/// which counts one call a layer), the rest once a forward.
pub const Stage = enum {
    /// The streamed bank's phase change (prompt or decode slot rows), where the forward changes it.
    phase,
    /// The token rows (and an image's), the stream and mix they start, the chunk's Engram ids and their read started.
    embed,
    /// A layer's Engram write (its rows off the SSD, `wkv`, the gate).
    engram,
    /// A DSpark target layer's main row.
    ds_main,
    /// A wide prompt chunk's routes predicted before its attention, read back, and the streamer's read-ahead seeded.
    read_ahead,
    /// The attention-side hyper-connection: the mix and the normed collapse of the stream.
    attn_hc,
    /// The q/kv projections, the window row, its ring write.
    qkv,
    /// The compressor (a KV source's pooled latents and index keys).
    compressor,
    /// The indexer's score and top-k.
    indexer,
    /// The attention core.
    attn_core,
    /// The output projections (the inverse rope, the grouped `wo_a`, `wo_b`).
    attn_out,
    /// The FFN-side hyper-connection: the attention's output into the stream, the FFN's mix and normed input.
    ffn_hc,
    /// The router.
    router,
    /// The routed experts: the streamer's call (the DSpark stages: their resident bank).
    routed,
    /// The shared expert.
    shared,
    /// The routed and shared sum into the stream.
    combine,
    /// CED's replay-in: the split layer's compressor over the chunk, the replay tail's spans.
    ced_replay,
    /// What a layer's own marks left unevaluated (its end-of-layer eval): near zero when the marks cover the layer.
    rest,
    /// The DSpark main rows into the stages' windows.
    ds_commit,
    /// The final collapse, norm and LM head.
    head,
    /// The draft's Markov chain and confidence, read back.
    markov,
};

pub const n_stages = std.enums.values(Stage).len;
/// Verify rows apart (1..16), prompt chunks by powers of two (1, 2, 3-4, ..., the last also every wider one).
const verify_groups = 16;
const prompt_groups = 22;
pub const n_groups = 1 + verify_groups + prompt_groups + 1;

pub const Cell = struct {
    ns: u64 = 0,
    calls: u64 = 0,
    /// The `Profile.seq` its last call counted at: a stage marked again within a layer (a prompt span's) is one call.
    seq: u64 = 0,
};

pub const Group = struct {
    forwards: u64 = 0,
    layers: u64 = 0,
    wall_ns: u64 = 0,
    rows_lo: usize = std.math.maxInt(usize),
    rows_hi: usize = 0,
    cells: [n_stages]Cell = @splat(.{}),
};

/// A request's stage times by group (kind and rows), MLX-free: the hooks below feed it the clock.
pub const Profile = struct {
    groups: [n_groups]Group = @splat(.{}),
    /// The forward running, by group.
    open: ?usize = null,
    /// One a forward and one a layer: what a stage's calls count.
    seq: u64 = 0,
    start: u64 = 0,
    last: u64 = 0,
    forwards: u64 = 0,

    pub fn groupOf(kind: Kind, rows: usize) usize {
        return switch (kind) {
            .decode => 0,
            .verify => std.math.clamp(rows, 1, verify_groups),
            // Typed: a bare @min narrows to the bound's width, which the sum overflows.
            .prompt => 1 + verify_groups + @as(usize, @min(prompt_groups - 1, if (rows <= 1) 0 else @as(usize, std.math.log2_int_ceil(usize, rows)))),
            .draft => n_groups - 1,
        };
    }

    fn kindOf(g: usize) Kind {
        if (g == 0) return .decode;
        if (g <= verify_groups) return .verify;
        if (g < n_groups - 1) return .prompt;
        return .draft;
    }

    pub fn begin(p: *Profile, kind: Kind, rows: usize, t: u64) void {
        _ = p.end(t);
        const g = groupOf(kind, rows);
        const gr = &p.groups[g];
        gr.forwards += 1;
        gr.rows_lo = @min(gr.rows_lo, rows);
        gr.rows_hi = @max(gr.rows_hi, rows);
        p.forwards += 1;
        p.seq += 1;
        p.open = g;
        p.start = t;
        p.last = t;
    }

    pub fn layer(p: *Profile) void {
        const g = p.open orelse return;
        p.groups[g].layers += 1;
        p.seq += 1;
    }

    /// `stage` ended at `t`: the time since the last mark is its.
    pub fn lap(p: *Profile, stage: Stage, t: u64) void {
        const g = p.open orelse return;
        const c = &p.groups[g].cells[@intFromEnum(stage)];
        c.ns += t -| p.last;
        if (c.seq != p.seq) {
            c.calls += 1;
            c.seq = p.seq;
        }
        p.last = t;
    }

    /// The forward ended at `t`; false when none was running.
    pub fn end(p: *Profile, t: u64) bool {
        const g = p.open orelse return false;
        p.groups[g].wall_ns += t -| p.start;
        p.open = null;
        return true;
    }

    /// A new request: the groups emptied, a forward still running (none should be) counted from here.
    pub fn reset(p: *Profile) void {
        p.groups = @splat(.{});
        p.forwards = 0;
    }

    /// Group `g`'s log line, or null when it ran no forward: `kind rows=lo[-hi] forwards= layers= wall_ms=`, then
    /// `stage=total_ms/calls/ms_per_call` for every stage it marked.
    pub fn line(p: *const Profile, g: usize, buf: []u8) ?[]const u8 {
        const gr = &p.groups[g];
        if (gr.forwards == 0) return null;
        var w = std.Io.Writer.fixed(buf);
        w.print("{s} rows={d}", .{ @tagName(kindOf(g)), gr.rows_lo }) catch return w.buffered();
        if (gr.rows_hi != gr.rows_lo) w.print("-{d}", .{gr.rows_hi}) catch return w.buffered();
        w.print(" forwards={d} layers={d} wall_ms={d:.1}", .{ gr.forwards, gr.layers, ms(gr.wall_ns) }) catch return w.buffered();
        for (gr.cells, 0..) |c, s| {
            if (c.calls == 0) continue;
            const total = ms(c.ns);
            w.print(" {s}={d:.1}/{d}/{d:.3}", .{ @tagName(@as(Stage, @enumFromInt(s))), total, c.calls, total / @as(f64, @floatFromInt(c.calls)) }) catch break;
        }
        return w.buffered();
    }
};

fn ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / std.time.ns_per_ms;
}

/// The profiler's setting from `MLX_SERVE_DSV41_STAGE_PROF`: null (off) when unset or starting with `0`; else the
/// forwards between reports beside each request's end (a whole number of 2 or more), 0 for the request's end only.
pub fn parse(value: ?[]const u8) ?u32 {
    const v = value orelse return null;
    if (v.len > 0 and v[0] == '0') return null;
    const n = std.fmt.parseInt(u32, v, 10) catch return 0;
    return if (n >= 2) n else 0;
}

// ── the hooks the model and plugin.zig call ──

/// On (`configure`, at load, before the model compiles its decode stretches): every hook below times.
pub var on: bool = false;
var every: u32 = 0;
var prof: Profile = .{};

pub fn configure(value: ?[]const u8) void {
    const setting = parse(value);
    on = setting != null;
    every = setting orelse 0;
    prof = .{};
}

/// A forward of `kind` over `rows` rows begins: `pre` (a decode token the engine's sampler made, still lazy) is
/// evaluated and the stream drained before the clock starts, so nothing earlier is the forward's.
pub fn begin(s: mlx.mlx_stream, kind: Kind, rows: usize, pre: []const mlx.mlx_array) !void {
    if (!on) return;
    try evalAll(pre);
    try mlx.check(mlx.mlx_synchronize(s));
    prof.begin(kind, rows, now());
}

/// The forward enters its next layer (a trunk layer, or a DSpark stage of a draft).
pub fn layer() void {
    if (on) prof.layer();
}

/// `stage` ended: its outputs `xs` (null handles skipped) are evaluated, and the time since the last mark is its. No
/// forward running (a load's probe, a path that began none): nothing is evaluated.
pub fn mark(stage: Stage, xs: []const mlx.mlx_array) !void {
    if (!on or prof.open == null) return;
    try evalAll(xs);
    prof.lap(stage, now());
}

/// The forward ended (its error path too); every `every` forwards the request so far is logged.
pub fn end() void {
    if (!on or !prof.end(now())) return;
    if (every > 0 and prof.forwards % every == 0) report();
}

/// The request ended: its stages are logged, then forgotten.
pub fn requestEnd() void {
    if (!on) return;
    report();
    prof.reset();
}

fn report() void {
    if (prof.forwards == 0) return;
    log.info("[stage-prof] {d} forwards, every stage synced (no overlap): stage=total_ms/calls/ms_per_call, a call a layer's (a forward's: phase, embed, ds_commit, head, markov)\n", .{prof.forwards});
    var buf: [2048]u8 = undefined;
    for (0..n_groups) |g| if (prof.line(g, &buf)) |l| log.info("[stage-prof] {s}\n", .{l});
}

fn now() u64 {
    const io = std.Io.Threaded.global_single_threaded.io();
    return @intCast(std.Io.Timestamp.now(io, .awake).toNanoseconds());
}

fn evalAll(xs: []const mlx.mlx_array) !void {
    var buf: [32]mlx.mlx_array = undefined;
    var n: usize = 0;
    for (xs) |x| {
        if (x.ctx == null) continue;
        buf[n] = x;
        n += 1;
        if (n < buf.len) continue;
        try evalSome(buf[0..n]);
        n = 0;
    }
    if (n > 0) try evalSome(buf[0..n]);
}

fn evalSome(xs: []const mlx.mlx_array) !void {
    const vec = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_eval(vec));
}

// ── tests ──

const testing = std.testing;

test "the stage profiler's setting: off unset or at 0, the request's end at 1 or any word, every N forwards beside it at N" {
    for ([_]?[]const u8{ null, "0", "00", "0x" }) |v| try testing.expectEqual(@as(?u32, null), parse(v));
    for ([_][]const u8{ "", "1", "on", "yes" }) |v| try testing.expectEqual(@as(?u32, 0), parse(v));
    try testing.expectEqual(@as(?u32, 64), parse("64"));
}

test "the stage profiler's groups: decode, verify by rows, prompt chunks by powers of two, the draft" {
    try testing.expectEqual(@as(usize, 0), Profile.groupOf(.decode, 1));
    for (1..9) |r| try testing.expectEqual(r, Profile.groupOf(.verify, r));
    try testing.expectEqual(Profile.groupOf(.prompt, 16384), Profile.groupOf(.prompt, 8193));
    try testing.expect(Profile.groupOf(.prompt, 8192) != Profile.groupOf(.prompt, 8193));
    try testing.expectEqual(Profile.groupOf(.prompt, 1 << 30), Profile.groupOf(.prompt, 1 << 21));
    try testing.expectEqual(@as(usize, n_groups - 1), Profile.groupOf(.draft, 5));
    for ([_]Kind{ .decode, .verify, .prompt, .draft }, [_]usize{ 1, 6, 300, 5 }) |k, r| try testing.expectEqual(k, Profile.kindOf(Profile.groupOf(k, r)));
}

test "the stage profiler counts a layer's stage once a layer however many spans mark it, a forward's once a forward" {
    var p: Profile = .{};
    const ns = std.time.ns_per_ms;
    // Two 6-row verifies of two layers: 1 ms of embed, each layer 2 ms of attention core over two marks and 3 of
    // routed experts, 1 ms of head.
    for (0..2) |f| {
        var t: u64 = @as(u64, f) * 100 * ns;
        p.begin(.verify, 6, t);
        t += ns;
        p.lap(.embed, t);
        for (0..2) |_| {
            p.layer();
            t += ns;
            p.lap(.attn_core, t);
            t += ns;
            p.lap(.attn_core, t);
            t += 3 * ns;
            p.lap(.routed, t);
        }
        t += ns;
        p.lap(.head, t);
        try testing.expect(p.end(t + ns));
    }
    try testing.expect(!p.end(0));
    // A mark outside a forward times nothing.
    p.lap(.head, 10_000 * ns);
    var buf: [512]u8 = undefined;
    try testing.expectEqualStrings(
        "verify rows=6 forwards=2 layers=4 wall_ms=26.0 embed=2.0/2/1.000 attn_core=8.0/4/2.000 routed=12.0/4/3.000 head=2.0/2/1.000",
        p.line(Profile.groupOf(.verify, 6), &buf).?,
    );
    try testing.expectEqual(@as(?[]const u8, null), p.line(Profile.groupOf(.decode, 1), &buf));
    p.reset();
    try testing.expectEqual(@as(?[]const u8, null), p.line(Profile.groupOf(.verify, 6), &buf));
}

test "the stage profiler's prompt groups show the rows they held" {
    var p: Profile = .{};
    p.begin(.prompt, 9000, 0);
    _ = p.end(5 * std.time.ns_per_ms);
    p.begin(.prompt, 16384, 0);
    _ = p.end(5 * std.time.ns_per_ms);
    var buf: [256]u8 = undefined;
    try testing.expectEqualStrings("prompt rows=9000-16384 forwards=2 layers=0 wall_ms=10.0", p.line(Profile.groupOf(.prompt, 16384), &buf).?);
}
