//! C4, the bounded layer-major pass (MLX): a run of wide prompt spans through the copied table_forward.zig
//! `hiddenSpans` one segment at a time (src/ling_segments.zig `Plan`), each segment through every layer before the
//! next, and a forward's hidden rows written once into one array (`HiddenRows`). plugin.zig `spanMajorOr` runs it with
//! the bank's steps and the host's checkpoint op (`Hooks`); src/ling_lm_pass_test.zig runs it with its own, at two
//! segment sizes, and holds the two to the same words.
//! - The segments are the whole run's routed calls cut at call ends, at most `ling_segments.segment_rows` rows a
//!   segment as served, a last group under `ling_segments.segment_tail_rows` rows joined to the segment before it, so a
//!   run of 32,768 rows or fewer is one segment, today's single pass, and a segment is at most 40,959 rows.
//! - A checkpoint at a span end inside a segment is filed (`Hooks.file`) once the segment has passed every layer, in the
//!   run's order, and its references are dropped then. The host makes its own copy of each (mlx-serve's
//!   PromptCheckpoints `file`), so the pass holds one segment's checkpoints, not the run's.
//! - A forward's hidden rows (the MTP head's history): each span's final-normed rows written at their rows of one
//!   `[1, rows, H]` array (`HiddenRows.put`: MLX's slice update, which writes into the array's own buffer when nothing
//!   else holds it), in place of every span's rows kept until one join at the end, both alive then.
const std = @import("std");
const kit = @import("mlx_serve");
const abi = kit.abi.plugin;
const mlx = kit.mlx;
const trunk = @import("mlx_serve/plugins/glm5_next/trunk.zig");
const slots = @import("mlx_serve/plugins/glm5_next/slots.zig");
const fwd = @import("mlx_serve/plugins/bailing_hybrid/table_forward.zig");
const segments = @import("ling_segments.zig");
const ran = @import("ling_ran.zig");

const Array = mlx.mlx_array;

/// What `runWide` asks of its caller around each segment.
pub const Hooks = struct {
    ctx: *anyopaque,
    /// Before each segment's pass; `first` at the run's first segment.
    begin: *const fn (ctx: *anyopaque, first: bool) anyerror!void,
    /// Once a segment has passed every layer: its spans' final-normed hidden rows `outs` (each `[1, S, H]`, in order,
    /// borrowed), the first of them span `j` of the run, at row `row` of the run. The caller takes its own references
    /// (the head's logits, the hidden rows) and evaluates what it keeps.
    after: *const fn (ctx: *anyopaque, j: usize, row: u64, outs: []const Array) anyerror!void,
    /// A checkpoint: every KDA layer's state after position `pos` (borrowed), in the run's order.
    file: *const fn (ctx: *anyopaque, pos: u64, states: []const abi.LayerState) anyerror!void,
};

fn freeAll(xs: []Array) void {
    for (xs) |*x| if (x.ctx != null) {
        _ = mlx.mlx_array_free(x.*);
        x.* = .{ .ctx = null };
    };
}

/// A run of wide spans through every layer, one segment at a time: `rows` each span's rows (in order, the first at
/// position `pos0`), `ids` the run's token ids `[1, sum of rows]`, slot `caches[0]` through `ops`, at most `max_rows`
/// rows a segment (`ling_segments.segment_rows` as served) unless one routed call is wider or a last group under
/// `ling_segments.segment_tail_rows` rows joins it. `checkpoints`: positions at
/// span ends of the run, ascending, each filed after its segment. D2's position `d2_from` as `hiddenSpans` takes it.
/// Returns the segments run.
pub fn runWide(m: *trunk.Model, ops: *const abi.CacheOps, caches: []const abi.Handle, ids: Array, rows: []const u32, pos0: u64, checkpoints: []const u64, d2_from: u64, max_rows: u64, hooks: Hooks) !usize {
    const a = std.heap.c_allocator;
    const n_layers = m.layers.len;
    var plan: segments.Plan = undefined;
    try plan.init(rows, max_rows);
    var row: u64 = 0;
    var ci: usize = 0;
    var n_seg: usize = 0;
    while (plan.next()) |sg| : (n_seg += 1) {
        try hooks.begin(hooks.ctx, sg.first == 0);
        // the checkpoints at this segment's span ends, with each span's index in the segment
        var after: [64]usize = undefined;
        var at_pos: [64]u64 = undefined;
        var n_cp: usize = 0;
        var end_pos = pos0 + row;
        for (sg.first..sg.end) |sj| {
            end_pos += rows[sj];
            while (ci < checkpoints.len and checkpoints[ci] == end_pos) : (ci += 1) {
                if (n_cp == after.len) return error.LayerMajorTooManyCheckpoints;
                after[n_cp] = sj - sg.first;
                at_pos[n_cp] = end_pos;
                n_cp += 1;
            }
        }
        var conv: [64][]Array = undefined;
        var ssm: [64][]Array = undefined;
        var made: usize = 0;
        defer for (conv[0..made], ssm[0..made]) |cs, ss| {
            freeAll(cs);
            freeAll(ss);
            a.free(cs);
            a.free(ss);
        };
        while (made < n_cp) : (made += 1) {
            conv[made] = try a.alloc(Array, n_layers);
            ssm[made] = a.alloc(Array, n_layers) catch |e| {
                a.free(conv[made]);
                return e;
            };
            @memset(conv[made], .{ .ctx = null });
            @memset(ssm[made], .{ .ctx = null });
        }
        const keep: fwd.SpanKeep = .{ .after = after[0..n_cp], .conv = conv[0..n_cp], .ssm = ssm[0..n_cp] };
        const outs = try a.alloc(Array, sg.end - sg.first);
        defer a.free(outs);
        @memset(outs, .{ .ctx = null });
        defer freeAll(outs);
        var idv = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(idv);
        try mlx.check(mlx.mlx_slice(&idv, ids, &[_]c_int{ 0, @intCast(row) }, 2, &[_]c_int{ 1, @intCast(row + sg.rows) }, 2, &[_]c_int{ 1, 1 }, 2, m.s));
        const p = [_]u32{@intCast(pos0 + row)};
        const sl: slots.Slots = .{ .ops = ops, .caches = caches, .positions = &p };
        try fwd.hiddenSpans(m, sl, idv, rows[sg.first..sg.end], sg.calls, &keep, d2_from, outs);
        try hooks.after(hooks.ctx, sg.first, row, outs);
        // the segment's checkpoints, in order: every KDA layer's state (the references go with the defer above)
        for (0..n_cp) |i| {
            var st: [256]abi.LayerState = undefined;
            var ns: usize = 0;
            for (m.layers, 0..) |*lw, li| if (lw.mixer == .kda) {
                if (ns == st.len) return error.LayerMajorTooManyLayers;
                st[ns] = .{ .layer = @intCast(li), .conv = .{ .ctx = conv[i][li].ctx }, .ssm = .{ .ctx = ssm[i][li].ctx } };
                ns += 1;
            };
            try hooks.file(hooks.ctx, at_pos[i], st[0..ns]);
        }
        row += sg.rows;
    }
    if (ci != checkpoints.len) return error.LayerMajorCheckpointOffSpan;
    if (n_seg > 1) segmentsEngaged(n_seg, pos0, row);
    return n_seg;
}

/// LM_SEGMENTS' run-path mark, at each run of two or more segments, and its engaged line, once a process.
var segments_said = false;

fn segmentsEngaged(n_seg: usize, pos0: u64, rows: u64) void {
    ran.mark(.LM_SEGMENTS);
    if (segments_said) return;
    segments_said = true;
    std.log.info("[turboexl3_ling] LM_SEGMENTS engaged: a run of {d} prompt rows from position {d} ran in {d} segments of at most {d} rows (a last group under {d} rows joined to the segment before it), each through every layer before the next, cut at the run's routed-call ends; each segment's checkpoints filed once it passed every layer", .{ rows, pos0, n_seg, segments.segment_rows, segments.segment_tail_rows });
}

/// A forward's hidden rows `[1, rows, H]`, written once: each part at its rows (`put`), then taken whole (`take`). The
/// array starts as zeros at the first part (in the parts' dtype); every row must be written before `take`.
pub const HiddenRows = struct {
    s: mlx.mlx_stream,
    rows: u64,
    dest: Array = .{ .ctx = null },
    /// The rows written so far.
    filled: u64 = 0,

    /// `part` `[1, S, H]` (borrowed) written at rows `[at, at + S)`.
    pub fn put(h: *HiddenRows, at: u64, part: Array) !void {
        const ps = mlx.getShape(part);
        if (ps.len != 3 or ps[0] != 1) return error.LingHiddenRowsShape;
        const end = at + @as(u64, @intCast(ps[1]));
        if (end > h.rows) return error.LingHiddenRowsShape;
        if (h.dest.ctx == null) {
            var z = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(z);
            try mlx.check(mlx.mlx_zeros(&z, &[_]c_int{ 1, @intCast(h.rows), ps[2] }, 3, mlx.mlx_array_dtype(part), h.s));
            h.dest = z;
        }
        var u = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(u);
        try mlx.check(mlx.mlx_slice_update(&u, h.dest, part, &[_]c_int{ 0, @intCast(at), 0 }, 3, &[_]c_int{ 1, @intCast(end), ps[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, h.s));
        // the old handle goes, so the update is the buffer's only holder and MLX writes in place
        _ = mlx.mlx_array_free(h.dest);
        h.dest = u;
        h.filled += @intCast(ps[1]);
    }

    /// The array, the caller's, once every row was written; null when nothing was. Unevaluated past the last
    /// evaluation of `dest`.
    pub fn take(h: *HiddenRows) !?Array {
        if (h.dest.ctx == null) return null;
        if (h.filled != h.rows) return error.LingHiddenRowsUnfilled;
        const r = h.dest;
        h.dest = .{ .ctx = null };
        return r;
    }

    pub fn deinit(h: *HiddenRows) void {
        if (h.dest.ctx != null) _ = mlx.mlx_array_free(h.dest);
        h.dest = .{ .ctx = null };
    }
};
