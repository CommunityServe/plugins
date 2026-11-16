//! glm53f's whole-prompt forward (J5: `Caps.whole_prompt_prefill`, GLM53F_WHOLE_PROMPT, a streaming pack). The engine
//! hands the whole prompt to one forward with the chunk loop's span ends (`Forward.prompt_spans`). Each span but the
//! pair runs as the chunk loop's forward of its own (its bank phase, the table's forward, the settle). The pair, the
//! last chunk and the held-back tail (`span_plan.pairAt`), runs layer by layer with one routed call over both spans'
//! rows (`table_forward.hiddenPair`), as #1196's split prompt forward did, where every routed layer takes the joined
//! call (`moe.joinServes`). Each checkpoint the engine wants (`Forward.ssm_checkpoints`) is filed as soon as its span
//! ends (`CacheOps.checkpoint`, which copies it); the pair's chunk span's after the pair. The bill is the engine's
//! (`Bill.prefill_bytes` stays null): the pair holds at most the tail's rows past one chunk and one checkpoint.
//! copied_from: mlx-serve src/plugins/bailing_hybrid/prompt.zig@84271b33 (`forward`, `Pass.take`, `laneSpan`, the
//! checkpoints' filing and the joined outputs), its wide segments left out.

const std = @import("std");
const kit = @import("mlx_serve");
const abi = kit.abi.plugin;
const mlx = kit.mlx;
const kh = kit.bases.kda_hybrid;
const host = kh.host;
const trunk = kh.trunk;
const moe = kit.bases.blocks.moe;
const fwd = @import("table_forward.zig");
const span_plan = @import("span_plan.zig");
const log = kit.log;

const name = "glm53f";
const gpa = std.heap.c_allocator;
const Array = mlx.mlx_array;
const Parts = std.ArrayList(Array);
const max_captures = 64;

/// Whether `f` is a prompt this forward serves: a prefill naming two spans or more, from a host whose `Forward` carries
/// the whole-prompt fields.
pub fn serves(f: *const abi.Forward) bool {
    if (f.mode != .prefill) return false;
    if (f.size < @offsetOf(abi.Forward, "ssm_checkpoints") + @sizeOf(@FieldType(abi.Forward, "ssm_checkpoints"))) return false;
    return f.prompt_spans.len >= 2;
}

/// The prompt `f` (`serves`) into `out`: the logits it asks for, its hidden rows and captures.
pub fn forward(m: *trunk.Model, ops: *const abi.CacheOps, f: *const abi.Forward, out: *abi.ForwardOut) abi.Verdict {
    var why: abi.Verdict = .ok;
    run(m, ops, f, out, &why) catch |e| {
        host.clear(out, f.capture_layers.len);
        return if (why.isOk()) abi.Verdict.named(.failed, @errorName(e)) else why;
    };
    return .ok;
}

const Pass = struct {
    m: *trunk.Model,
    ops: *const abi.CacheOps,
    f: *const abi.Forward,
    /// `ssm_checkpoints`' next position to file.
    next_cp: usize = 0,
    logits: Parts = .empty,
    hidden: Parts = .empty,
    captures: [max_captures]Parts = @splat(Parts.empty),

    fn deinit(p: *Pass) void {
        freeParts(&p.logits);
        freeParts(&p.hidden);
        for (&p.captures) |*c| freeParts(c);
    }

    /// What a span's forward made, moved into the pass's parts (`so` left empty).
    fn take(p: *Pass, so: *abi.ForwardOut) !void {
        if (so.logits.ctx != null) {
            try p.logits.append(gpa, .{ .ctx = so.logits.ctx });
            so.logits = .{};
        }
        if (so.hidden.ctx != null) {
            if (p.f.want_hidden) try p.hidden.append(gpa, .{ .ctx = so.hidden.ctx }) else _ = mlx.mlx_array_free(.{ .ctx = so.hidden.ctx });
            so.hidden = .{};
        }
        if (so.captured) |c| for (c[0..p.f.capture_layers.len], 0..) |*x, i| {
            try p.captures[i].append(gpa, .{ .ctx = x.ctx });
            x.* = .{};
        };
    }

    /// The checkpoint at `end`, when it is the next the engine wants: every KDA layer's state as `state` gives it (the
    /// slot's own after the span, or the pair's chunk span's kept ones). The engine copies what it files.
    fn file(p: *Pass, end: u64, state: anytype) !void {
        const cps = p.f.ssm_checkpoints.slice();
        if (p.next_cp >= cps.len or cps[p.next_cp] != end) return;
        const op = p.ops.checkpointOp() orelse return error.PromptCheckpointsNotServed;
        var states: std.ArrayList(abi.LayerState) = .empty;
        defer states.deinit(gpa);
        for (p.m.layers, 0..) |*lw, li| if (lw.mixer == .kda) {
            const s = state.of(@intCast(li));
            try states.append(gpa, .{ .layer = @intCast(li), .conv = .{ .ctx = s[0].ctx }, .ssm = .{ .ctx = s[1].ctx } });
        };
        const v = op(p.f.caches.slice()[0], end, .of(states.items));
        if (!v.isOk()) {
            log.err("glm53f: the engine refused the checkpoint at {d}: {s}\n", .{ end, v.why.slice() });
            return error.PromptCheckpointRefused;
        }
        p.next_cp += 1;
    }

    /// A span as the chunk loop forwards it: its bank phase, the table's forward and the settle, its outputs into the
    /// pass, then the checkpoint at its end. The sub-forward always makes its hidden rows, so the settle evaluates the
    /// span's whole graph before the bank's rows move on.
    fn alone(p: *Pass, ids: Array, pos: u32, rows: u32, last: bool, why: *abi.Verdict) !void {
        var sub = p.f.*;
        const at = [_]u32{pos};
        sub.mode = if (rows == 1) .decode else .prefill;
        sub.rows = rows;
        sub.tokens = .{};
        sub.ids = .{ .ctx = ids.ctx };
        sub.positions = .of(&at);
        sub.logits = if (p.f.logits == .all_rows) .all_rows else if (last) p.f.logits else .none;
        sub.want_hidden = true;
        sub.progress = null;
        sub.prompt_spans = .{};
        sub.ssm_checkpoints = .{};
        var captured: [max_captures]abi.Array = @splat(.{});
        var so: abi.ForwardOut = .{ .captured = &captured };
        try trunk.bankPhase(p.m, rows);
        var v = host.forwardOne(name, fwd, p.m, p.ops, &sub, &so);
        if (v.isOk()) v = host.settleBank(p.m, &sub, &so);
        if (!v.isOk()) {
            why.* = v;
            return error.PromptSpanRefused;
        }
        errdefer host.clear(&so, p.f.capture_layers.len);
        try p.take(&so);
        const sl: kh.slots.Slots = .{ .ops = p.ops, .caches = p.f.caches.slice(), .positions = &at };
        try p.file(@as(u64, pos) + rows, SlotState{ .sl = sl });
    }

    /// The pair: the chunk span (`rows_a` rows from `pos`) and the tail span, layer by layer with one routed call each
    /// layer (`fwd.hiddenPair`), the head over the tail's last row, evaluated, then settled. The chunk span's
    /// checkpoint is filed from its kept states, then the tail's own if the engine wants one.
    fn pair(p: *Pass, ids: Array, pos: u32, rows_a: u32, rows_b: u32) !void {
        const m = p.m;
        const n_layers = m.layers.len;
        const conv = try gpa.alloc(Array, n_layers);
        defer gpa.free(conv);
        const ssm = try gpa.alloc(Array, n_layers);
        defer gpa.free(ssm);
        @memset(conv, .{ .ctx = null });
        @memset(ssm, .{ .ctx = null });
        defer for (conv, ssm) |c, s| {
            free(c);
            free(s);
        };
        const ids_a = try sliceIds(m.s, ids, 0, rows_a);
        defer _ = mlx.mlx_array_free(ids_a);
        const ids_b = try sliceIds(m.s, ids, rows_a, rows_a + rows_b);
        defer _ = mlx.mlx_array_free(ids_b);
        const at_a = [_]u32{pos};
        const at_b = [_]u32{pos + rows_a};
        const st_a: fwd.Step = .{ .slots = .{ .ops = p.ops, .caches = p.f.caches.slice(), .positions = &at_a }, .eval_every = p.f.eval_every };
        const st_b: fwd.Step = .{ .slots = .{ .ops = p.ops, .caches = p.f.caches.slice(), .positions = &at_b }, .eval_every = p.f.eval_every };
        try trunk.bankPhase(m, rows_a + rows_b);
        const hs = try fwd.hiddenPair(m, &st_a, &st_b, ids_a, ids_b, true, .{ .conv = conv, .ssm = ssm });
        defer for (hs) |h| free(h);
        pairSaid(rows_a, rows_b);
        if (p.f.want_hidden) for (hs) |h| try p.hidden.append(gpa, ref(h));
        // the head as the tail's own forward runs it (`host.finish`)
        if (p.f.logits != .none) {
            var sub = p.f.*;
            sub.want_hidden = false;
            var so: abi.ForwardOut = .{};
            try host.finish(m, &sub, ref(hs[1]), &so);
            errdefer host.clear(&so, 0);
            try p.take(&so);
        }
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        for (hs) |h| _ = mlx.mlx_vector_array_append_value(vec, h);
        for (p.logits.items) |x| _ = mlx.mlx_vector_array_append_value(vec, x);
        try mlx.check(mlx.mlx_eval(vec));
        if (m.routed) |r| try r.settle();
        try p.file(@as(u64, pos) + rows_a, KeptState{ .conv = conv, .ssm = ssm });
        const sl: kh.slots.Slots = .{ .ops = p.ops, .caches = p.f.caches.slice(), .positions = &at_b };
        try p.file(@as(u64, pos) + rows_a + rows_b, SlotState{ .sl = sl });
    }
};

/// A KDA layer's state as the slot holds it now (borrowed).
const SlotState = struct {
    sl: kh.slots.Slots,
    fn of(self: SlotState, li: u32) [2]Array {
        return .{ self.sl.state(0, li, .conv), self.sl.state(0, li, .ssm) };
    }
};

/// A KDA layer's state as the pair kept it after its chunk span (borrowed).
const KeptState = struct {
    conv: []const Array,
    ssm: []const Array,
    fn of(self: KeptState, li: u32) [2]Array {
        return .{ self.conv[li], self.ssm[li] };
    }
};

/// Whether every routed layer takes one routed call over both spans' rows (the bank declares each prompt row summed by
/// itself, both spans wider than its decode call).
fn joins(m: *const trunk.Model, rows_a: u32, rows_b: u32) bool {
    const rows = [_]c_int{ @intCast(rows_a), @intCast(rows_b) };
    var any = false;
    for (m.layers[0..m.c.num_hidden_layers]) |*lw| switch (lw.mlp) {
        .moe => |*mw| {
            if (!moe.joinServes(mw, &rows)) return false;
            any = true;
        },
        .dense => {},
    };
    return any;
}

var pair_said = false;

fn pairSaid(rows_a: u32, rows_b: u32) void {
    if (pair_said) return;
    pair_said = true;
    log.info("[glm53f] whole-prompt pair engaged: a {d}-row chunk and its {d}-row tail in one routed call a layer\n", .{ rows_a, rows_b });
}

fn run(m: *trunk.Model, ops: *const abi.CacheOps, f: *const abi.Forward, out: *abi.ForwardOut, why: *abi.Verdict) !void {
    if (f.caches.len != 1 or f.positions.len != 1) return error.PromptOneSlot;
    if (f.media.ctx != null) return error.PromptMediaUnsupported;
    const n_cap = f.capture_layers.len;
    if (n_cap > max_captures) return error.TooManyCaptureLayers;
    if (n_cap > 0 and out.captured == null) return error.CaptureLayersWithoutOutput;
    const pos0: u64 = f.positions.slice()[0];
    const ends = f.prompt_spans.slice();
    try span_plan.check(pos0, f.rows, ends, f.ssm_checkpoints.slice());
    const rows = try gpa.alloc(u32, ends.len);
    defer gpa.free(rows);
    try span_plan.rowsOf(pos0, ends, rows);

    var p: Pass = .{ .m = m, .ops = ops, .f = f };
    defer p.deinit();
    const ids = try host.tokenIds(f);
    defer _ = mlx.mlx_array_free(ids);

    // A pair-shaped ending runs as the pair, or apart where a fact of the forward or a routed layer keeps it apart; one
    // line a pass says which (the first-use engaged line is `pairSaid`'s).
    var pair: ?usize = null;
    if (span_plan.pairAt(rows)) |j| {
        const facts: span_plan.Facts = .{ .slots = f.caches.len, .captures = n_cap, .media = f.media.ctx != null, .all_rows = f.logits == .all_rows };
        const apart: ?span_plan.Apart = span_plan.apartOf(facts) orelse if (!joins(m, rows[j], rows[j + 1])) .layer_no_join else null;
        var buf: [160]u8 = undefined;
        log.info("{s}\n", .{try span_plan.passLine(&buf, rows[j], rows[j + 1], apart)});
        if (apart == null) pair = j;
    }
    const alone_end = pair orelse rows.len;
    var at: u32 = 0;
    for (rows[0..alone_end], 0..) |r, j| {
        const idv = try sliceIds(m.s, ids, at, at + r);
        defer _ = mlx.mlx_array_free(idv);
        try p.alone(idv, @intCast(pos0 + at), r, j + 1 == rows.len, why);
        at += r;
        if (f.progress) |pr| @atomicStore(u64, pr, at, .monotonic);
    }
    if (pair) |j| {
        try p.pair(ids, @intCast(pos0 + at), rows[j], rows[j + 1]);
        at += rows[j] + rows[j + 1];
        if (f.progress) |pr| @atomicStore(u64, pr, at, .monotonic);
    }
    if (p.next_cp != f.ssm_checkpoints.len) return error.PromptCheckpointsUnfiled;

    if (p.logits.items.len > 0) out.logits = .{ .ctx = (try joined(&p.logits, 0, m.s)).ctx };
    if (f.want_hidden) out.hidden = .{ .ctx = (try joined(&p.hidden, 1, m.s)).ctx };
    for (p.captures[0..n_cap], 0..) |*c, i| out.captured.?[i] = .{ .ctx = (try joined(c, 1, m.s)).ctx };
}

/// Columns `lo .. hi` of the `[1, n]` ids (a new reference).
fn sliceIds(s: mlx.mlx_stream, ids: Array, lo: u32, hi: u32) !Array {
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_slice(&out, ids, &[_]c_int{ 0, @intCast(lo) }, 2, &[_]c_int{ 1, @intCast(hi) }, 2, &[_]c_int{ 1, 1 }, 2, s));
    return out;
}

/// `parts` joined along `axis` (a new reference; one part is moved out as it is).
fn joined(parts: *Parts, axis: c_int, s: mlx.mlx_stream) !Array {
    if (parts.items.len == 1) return parts.pop().?;
    const vec = mlx.mlx_vector_array_new_data(parts.items.ptr, parts.items.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var r = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(r);
    try mlx.check(mlx.mlx_concatenate_axis(&r, vec, axis, s));
    return r;
}

fn ref(x: Array) Array {
    var r = mlx.mlx_array_new();
    _ = mlx.mlx_array_set(&r, x);
    return r;
}

fn free(a: Array) void {
    if (a.ctx != null) _ = mlx.mlx_array_free(a);
}

fn freeParts(p: *Parts) void {
    for (p.items) |x| _ = mlx.mlx_array_free(x);
    p.deinit(gpa);
}
