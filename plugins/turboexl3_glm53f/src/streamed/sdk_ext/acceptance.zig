//! The draft lanes' acceptance: one rule set, one sampled path and one seeded draw for every arch's draft lane
//! (DeepSeek-V4.1's DSpark lane; MiMo-V2.6's MTP lane next). A lane drafts, verifies [t1, drafts] on its target, and
//! hands this module the verify rows; it gets back how many drafts are accepted and the token after them. Arch state
//! (KV rollback, draft windows, trims) stays in the lane.
//!
//! The rules (`Rule`, chosen once at the lane's construction):
//!
//! - `exact`: draft x at row r is accepted iff x == argmax(logits_r). The control: every emitted token is the target's
//!   greedy token.
//! - `typical` (#475's rule, Medusa's typical acceptance): with p = softmax(logits_r) at temperature 1, unfiltered,
//!   and H(p) = -sum_i p_i ln p_i (nats),
//!
//!       accept x  <=>  p(x) > min(eps, delta * exp(-H(p)))        (strict; eps = 1.0 unless set)
//!
//!   delta scales the floor: a smaller delta lowers it, so acceptance never falls as delta falls (delta -> 0 accepts
//!   every draft with mass; delta 0.3 accepts at least what 0.5 does, row for row). exp(-H) is 1 on a one-hot row and
//!   1 / m on a flat row of m tokens, so the floor follows the row's spread: a confident row (low H) demands a high p(x),
//!   a flat row accepts any of its m tokens once delta < 1. The correction at the first rejected row, and the bonus
//!   after an accepted run, stay the row's argmax. Lossy by design (a draft the target ranks second can be kept).
//!
//! Requests (`Params`, the host's `sdk.SamplingParams`): a greedy request (temperature < 0.01 or top_k == 1) takes the
//! lane's `Rule` as above. A sampled request takes SAMPLE-THEN-VERIFY whatever the rule, over the target row tempered
//! (logits / T) and filtered (top_k, then top_p over the top_k-renormalised set, then min_p against the top
//! probability; ties at the kept set's smallest tempered logit are kept): p. Every row's token is p's draw by inverse
//! CDF at `drawV` (the request's one position-keyed stream, `drawAt` index 0), and draft x is accepted iff it IS that
//! token. Every emitted token is p's draw at its position, so a seeded request emits the same ids on every path: the
//! serial step (`sampler.zig`, the plugin's draw handed to the host), a lane's verify (drafts accepted or not), the
//! prompt's first token. It is exact (each token distributed as p) for any drafter; for an argmax drafter (a point mass
//! at x) a draft is accepted with probability p(x), as under min(1, p / q) speculative sampling. The draws are keyed
//! by (seed, absolute position of the drawn token, draw index), never by round: a seeded request is reproducible however
//! its rounds fall.
//!
//! The rule on every request (`acceptGeneral`, `typicalRows`; a lane opts in at construction by calling them: MiMo's
//! MTP lane does, DeepSeek-V4.1's DSpark lane keeps `accept` / `sampledRows`): `exact` is as above (greedy: argmax
//! equality; sampled: sample-then-verify). `typical` on a sampled request is Medusa's typical acceptance (Cai
//! et al. 2024, section 2.3.1) over the request's own target: p = `distribution` (tempered, then top_k, top_p, min_p,
//! the sampled path's transform, unchanged), draft x at row r accepted iff p_r(x) > min(eps, delta * exp(-H(p_r))) with
//! H over p_r's kept set; every row's token is p_r's plain draw at `drawV(seed, pos)` (inverse CDF): the first
//! rejected row emits its draw (the correction, which may equal the rejected draft: no residual, no resampling
//! adjustment), an accepted run the next row's draw (the bonus). `drawU` is not consumed. Lossy by design: the emitted
//! tokens are not distributed as p. A greedy request has no distribution to temper (the host's greedy token is the
//! argmax), so `typical` on it stays the greedy round above (T = 1 unfiltered test, argmax tokens): at a literal
//! T -> 0 every row is one-hot, the floor is delta, and typical would collapse to `exact` (Medusa's "reverts to greedy
//! decoding"). Medusa's first token is the target's greedy token; here it is the previous round's draw (t1), so a
//! sampled request samples every token the target emits.
//!
//! Two forms of the same rules: the device graphs (`typicalFlags`, `sampledRows`) a lane folds into its verify's eval
//! (written once against any op backend with the ops they call: `shapeOf`, `astype`, `slice`, `logsumexp`, `sub`,
//! `exp`, `neg`, `sum`, `mul`, `minimum`, `scalar`, `hostArray`, `takeAlongAxis`, `reshape`, `greater`, `sort`,
//! `softmax`, `arange`, `less`, `where`, `div`, `cumsum`, `logicalAnd`, `max`, `greaterEqual`, `equal`), and the host
//! reference (`accept`, `distribution`, `specStep`, `typicalAccepts`): the parity tests' oracle, rows of at most
//! `ref_max_vocab` tokens. `prefix` is the walk both share: from the rows' tokens (argmax or drawn) and the per-draft
//! decisions to the accepted count and the next token.

const std = @import("std");
const sdk = @import("sdk");

/// Rows one verify can hand the device graphs (drafts + the bonus row).
pub const max_rows = 17;
/// The host reference's row bound (its scratch lives on the stack).
pub const ref_max_vocab = 4096;

/// The typical rule's parameters: the host's (`sdk.acceptance.Mode.typical`: delta, eps = 1.0).
pub const Typical = @FieldType(sdk.acceptance.Mode, "typical");

/// How a lane accepts drafts, chosen once at its construction. Under `accept` it shapes greedy requests only (sampled
/// requests take exact speculative sampling under either); under `acceptGeneral` `typical` shapes every request.
pub const Rule = union(enum) {
    exact,
    typical: Typical,
};

/// A request's sampling as a lane applies it to the target's verify rows: the host's `sdk.SamplingParams`, field for
/// field.
pub const Params = struct {
    temperature: f32 = 1.0,
    top_p: f32 = 1.0,
    top_k: u32 = 0,
    min_p: ?f32 = null,
    seed: u64 = 0,

    pub fn of(s: sdk.SamplingParams) Params {
        return .{ .temperature = s.temperature, .top_p = s.top_p, .top_k = s.top_k, .min_p = s.min_p, .seed = s.seed };
    }

    /// The host's greedy rule (`sdk.SamplingParams.greedy`): the lane's `Rule` path.
    pub fn greedy(s: Params) bool {
        return s.temperature < 0.01 or s.top_k == 1;
    }

    /// The sampled parameters, or null when the request is greedy (the `Rule` path).
    pub fn active(s: ?Params) ?Params {
        const x = s orelse return null;
        return if (x.greedy()) null else x;
    }

    /// Whether the filter sorts each row (top_k or top_p set).
    pub fn sorts(s: Params) bool {
        return s.top_k > 0 or s.top_p < 1.0;
    }
};

/// mlx-stream's one random stream: the uniform in (0, 1) of draw `index` at absolute token position `pos` of a request
/// seeded `seed`. splitmix64 over (seed ^ the index's key) + pos x the golden gamma, its top 24 bits, centred
/// ((bits + 0.5) / 2^24, never 0 or 1). Index 0 (`token_draw`) is the token's draw (every path samples it: serial,
/// verify, first token); index 1 (`test_draw`) the typical tier's spare. The same (seed, pos, index) is the same
/// uniform on every arch and path, however the rounds fall.
pub fn drawAt(seed: u64, pos: u64, index: u32) f32 {
    var z = (seed ^ draw_keys[index]) +% pos *% 0x9E3779B97F4A7C15;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    z ^= z >> 31;
    const bits: u32 = @intCast(z >> 40);
    return (@as(f32, @floatFromInt(bits)) + 0.5) / 16777216.0;
}

pub const token_draw: u32 = 0;
pub const test_draw: u32 = 1;
const draw_keys = [2]u64{ 0xD1B54A32D192ED03, 0 };

/// The token draw at `pos` (`drawAt` index 0).
pub fn drawV(seed: u64, pos: u64) f32 {
    return drawAt(seed, pos, token_draw);
}

/// The spare uniform at `pos` (`drawAt` index 1).
pub fn drawU(seed: u64, pos: u64) f32 {
    return drawAt(seed, pos, test_draw);
}

/// The reference sampler: the token at absolute position `pos` of a request seeded `seed`, drawn from `dist` (the
/// row's `distribution`) at its token draw. Every path's token is this one.
pub fn sample(dist: []const f64, seed: u64, pos: u64) u32 {
    return draw(dist, drawV(seed, pos));
}

/// The walk from a verify's rows to the round's outcome: `targets[r]` is row r's token (the argmax on the `Rule` path,
/// the drawn token on the sampled path: p's draw at the row's position), `decisions`
/// (one per draft; null = the exact rule: target == draft) the per-draft test. `targets` may be a chunk of the verify
/// (fewer rows than drafts + 1): `next` is null when every draft in it is accepted and it holds no further row.
pub const Prefix = struct { accepted: u32, next: ?u32 };

pub fn prefix(drafts: []const u32, targets: []const u32, decisions: ?[]const bool) Prefix {
    const n = @min(drafts.len, targets.len);
    var acc: u32 = 0;
    while (acc < n) : (acc += 1) {
        const ok = if (decisions) |d| d[acc] else targets[acc] == drafts[acc];
        if (!ok) return .{ .accepted = acc, .next = targets[acc] };
    }
    return .{ .accepted = acc, .next = if (targets.len > n) targets[n] else null };
}

/// One round's counters.
pub const RoundStats = struct {
    drafted: u32,
    accepted: u32,
    /// The next token is a correction (a draft was rejected), else the bonus.
    corrected: bool,
};

pub const Round = struct { n_accepted: u32, next_token: u32, stats: RoundStats };

/// The host reference of one round (the device graphs' oracle and the rules' statement): `logits` = the verify's
/// `drafts.len + 1` rows (row r predicts the token at `base` + r + 1), `draft_q` (null: an argmax drafter, point-mass
/// proposals) the drafts' distributions, one row per draft. `params` null or greedy: `rule`; sampled: exact speculative
/// sampling, seeded per position.
pub fn accept(rule: Rule, params: ?Params, logits: []const f32, drafts: []const u32, draft_q: ?[]const f64, base: u64) Round {
    const rows = drafts.len + 1;
    const vocab = logits.len / rows;
    std.debug.assert(vocab * rows == logits.len and vocab <= ref_max_vocab and rows <= max_rows);
    var targets: [max_rows]u32 = undefined;
    var decisions: [max_rows]bool = undefined;
    var d: [ref_max_vocab]f64 = undefined;
    const dist = d[0..vocab];
    var p: Prefix = undefined;
    if (Params.active(params)) |sm| {
        // Row by row to the first rejection: a row's draws are needed only once it is reached.
        var r: usize = 0;
        while (r < rows) : (r += 1) {
            distribution(logits[r * vocab ..][0..vocab], sm, dist);
            const x: ?u32 = if (r < drafts.len) drafts[r] else null;
            const q: ?[]const f64 = if (x != null and draft_q != null) draft_q.?[r * vocab ..][0..vocab] else null;
            const s = specStep(dist, x, q, drawU(sm.seed, base + r + 1), drawV(sm.seed, base + r + 1));
            targets[r] = s.tok;
            if (x != null) decisions[r] = s.accept;
            if (x == null or !s.accept) break;
        }
        p = prefix(drafts[0..@min(r, drafts.len)], targets[0 .. r + 1], decisions[0..@min(r + 1, drafts.len)]);
    } else {
        for (targets[0..rows], 0..) |*t, r| t.* = argmaxRow(logits[r * vocab ..][0..vocab]);
        const typ: ?[]const bool = switch (rule) {
            .exact => null,
            .typical => |t| blk: {
                for (decisions[0..drafts.len], drafts, 0..) |*f, x, r| {
                    distribution(logits[r * vocab ..][0..vocab], .{ .temperature = 1.0 }, dist);
                    f.* = typicalAccepts(dist, x, t);
                }
                break :blk decisions[0..drafts.len];
            },
        };
        p = prefix(drafts, targets[0..rows], typ);
    }
    return roundOf(p, drafts.len);
}

fn roundOf(p: Prefix, drafted: usize) Round {
    const corrected = p.accepted < drafted;
    return .{ .n_accepted = p.accepted, .next_token = p.next.?, .stats = .{ .drafted = p.accepted + @intFromBool(corrected), .accepted = p.accepted, .corrected = corrected } };
}

/// The host reference of one round with the lane's `Rule` on every request (`typicalRows`' oracle; see the module
/// doc): `exact`, or a greedy request, is `accept`; `typical` on a sampled request is Medusa's typical acceptance over
/// the request's filtered target (`typicalStep` per row, seeded per position). `draft_q` is unused by `typical`.
pub fn acceptGeneral(rule: Rule, params: ?Params, logits: []const f32, drafts: []const u32, draft_q: ?[]const f64, base: u64) Round {
    const t = switch (rule) {
        .exact => return accept(rule, params, logits, drafts, draft_q, base),
        .typical => |t| t,
    };
    const sm = Params.active(params) orelse return accept(rule, null, logits, drafts, draft_q, base);
    const rows = drafts.len + 1;
    const vocab = logits.len / rows;
    std.debug.assert(vocab * rows == logits.len and vocab <= ref_max_vocab and rows <= max_rows);
    var targets: [max_rows]u32 = undefined;
    var decisions: [max_rows]bool = undefined;
    var d: [ref_max_vocab]f64 = undefined;
    const dist = d[0..vocab];
    // Row by row to the first rejection, as `accept`'s sampled path.
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        distribution(logits[r * vocab ..][0..vocab], sm, dist);
        const x: ?u32 = if (r < drafts.len) drafts[r] else null;
        const s = typicalStep(dist, x, t, drawV(sm.seed, base + r + 1));
        targets[r] = s.tok;
        if (x != null) decisions[r] = s.accept;
        if (x == null or !s.accept) break;
    }
    return roundOf(prefix(drafts[0..@min(r, drafts.len)], targets[0 .. r + 1], decisions[0..@min(r + 1, drafts.len)]), drafts.len);
}

/// The first index of the row's largest logit (the device argmax's tie rule).
fn argmaxRow(row: []const f32) u32 {
    var best: u32 = 0;
    for (row, 0..) |v, i| if (v > row[best]) {
        best = @intCast(i);
    };
    return best;
}

/// The typical floor min(eps, delta x exp(-H(dist))) (the host's `typicalThreshold`).
pub fn typicalFloor(dist: []const f64, t: Typical) f64 {
    return sdk.acceptance.typicalThreshold(dist, t.delta, t.eps);
}

/// The typical test on a row's distribution: p(draft) > the floor.
pub fn typicalAccepts(dist: []const f64, draft: u32, t: Typical) bool {
    return dist[draft] > typicalFloor(dist, t);
}

/// The host reference of a sampled row: `out` (the row's length, at most `ref_max_vocab`) gets the tempered, filtered
/// distribution (0 off the kept set), in f64 over the f32 tempered logits.
pub fn distribution(logits: []const f32, s: Params, out: []f64) void {
    const n = logits.len;
    const inv_t: f32 = 1.0 / s.temperature;
    var lt_buf: [ref_max_vocab]f32 = undefined;
    const lt = lt_buf[0..n];
    for (lt, logits) |*d, v| d.* = v * inv_t;
    var thr: f32 = -std.math.inf(f32);
    if (s.sorts()) {
        var srt_buf: [ref_max_vocab]f32 = undefined;
        const srt = srt_buf[0..n];
        @memcpy(srt, lt);
        std.mem.sort(f32, srt, {}, std.sort.desc(f32));
        const smax: f64 = srt[0];
        var sz: f64 = 0;
        for (srt) |v| sz += @exp(@as(f64, v) - smax);
        const kk: usize = if (s.top_k > 0) @min(s.top_k, n) else n;
        var zk: f64 = 0;
        for (srt[0..kk]) |v| zk += @exp(@as(f64, v) - smax) / sz;
        var kept: usize = 0;
        var cum: f64 = 0;
        for (srt[0..kk]) |v| {
            const p = @exp(@as(f64, v) - smax) / sz / zk;
            if (s.top_p < 1.0 and cum >= s.top_p) break;
            cum += p;
            kept += 1;
        }
        thr = srt[kept - 1];
    }
    var mx: f64 = -std.math.inf(f64);
    for (lt) |v| if (v >= thr) {
        mx = @max(mx, v);
    };
    var z: f64 = 0;
    for (lt) |v| if (v >= thr) {
        z += @exp(@as(f64, v) - mx);
    };
    for (out, lt) |*o, v| o.* = if (v >= thr) @exp(@as(f64, v) - mx) / z else 0;
    if (s.min_p) |mp| {
        var pmax: f64 = 0;
        for (out) |p| pmax = @max(pmax, p);
        var z2: f64 = 0;
        for (out) |*p| {
            if (p.* < mp * pmax) p.* = 0;
            z2 += p.*;
        }
        for (out) |*p| p.* /= z2;
    }
}

/// The reference draw: the first token whose cumulative probability reaches `u` x the total.
pub fn draw(dist: []const f64, u: f32) u32 {
    var total: f64 = 0;
    for (dist) |p| total += p;
    var cum: f64 = 0;
    for (dist, 0..) |p, i| {
        cum += p;
        if (cum >= @as(f64, u) * total and p > 0) return @intCast(i);
    }
    return @intCast(dist.len - 1);
}

/// The reference step on one row of a sampled request (sample-then-verify): the row's token is p's draw at v; a draft
/// is accepted iff it is that token. `q` and `u` are not read (any drafter: the emitted token is p's draw).
pub fn specStep(dist: []const f64, draft: ?u32, q: ?[]const f64, u: f32, v: f32) struct { accept: bool, tok: u32 } {
    _ = .{ q, u };
    const tok = draw(dist, v);
    const x = draft orelse return .{ .accept = false, .tok = tok };
    return .{ .accept = x == tok, .tok = tok };
}

/// The reference typical step on one row of a sampled request (`acceptGeneral`): the row's token is p's draw at `v`;
/// with a draft x, accept iff p(x) > the floor (`typicalAccepts`).
pub fn typicalStep(dist: []const f64, draft: ?u32, t: Typical, v: f32) struct { accept: bool, tok: u32 } {
    const tok = draw(dist, v);
    const x = draft orelse return .{ .accept = false, .tok = tok };
    return .{ .accept = typicalAccepts(dist, x, t), .tok = tok };
}

/// `typicalFlags`'s graph over the verify's rows `row2` [W, V] (any float dtype): the typical test on the first
/// `drafts.len` rows (each against its draft), in f32 on the device: bool [drafts.len]. drafts.len >= 1.
pub fn typicalFlags(g: anytype, row2: anytype, drafts: []const u32, t: Typical) !@TypeOf(row2) {
    const vocab = g.shapeOf(row2).dim(-1);
    const drafted: c_int = @intCast(drafts.len);
    const lg = try g.astype(try g.slice(row2, &.{ 0, 0 }, &.{ drafted, vocab }, &.{ 1, 1 }), .float32);
    return typicalTest(g, lg, null, drafts, t);
}

/// The typical test's graph on f32 rows `lg` [drafts.len, V], the logits of the distribution it reads (p =
/// softmax(lg)): -inf off `keep` when one is given (a filtered row), where the entropy's p ln p terms are 0 (not 0 x
/// -inf). Without `keep` the ops are `typicalFlags`' of before (the greedy tier's graph, unchanged).
fn typicalTest(g: anytype, lg: anytype, keep: ?@TypeOf(lg), drafts: []const u32, t: Typical) !@TypeOf(lg) {
    const drafted: c_int = @intCast(drafts.len);
    const log_p = try g.sub(lg, try g.logsumexp(lg, -1, true));
    const p = try g.exp(log_p);
    const plogp = try g.mul(p, log_p);
    const terms = if (keep) |k| try g.where(k, plogp, try g.scalar(0, .float32)) else plogp;
    const entropy = try g.neg(try g.sum(terms, -1, false));
    const floor = try g.minimum(try g.scalar(t.eps, .float32), try g.mul(try g.scalar(t.delta, .float32), try g.exp(try g.neg(entropy))));
    var idb: [max_rows]i32 = undefined;
    for (idb[0..drafts.len], drafts) |*d, v| d.* = @intCast(v);
    const ids = try g.hostArray(std.mem.sliceAsBytes(idb[0..drafts.len]), &.{ drafted, 1 }, .int32);
    return g.greater(try g.reshape(try g.takeAlongAxis(p, ids, -1), &.{-1}), floor);
}

/// A sampled verify's arrays: `tok` uint32 [W] (the rows' tokens, as the argmax rows), `accept` bool [drafts] (null
/// without drafts), `p` the rows' filtered distribution [W, V].
pub fn SampledRows(comptime T: type) type {
    return struct { tok: T, accept: ?T, p: T };
}

/// A sampled request's filtered rows: `lf` the tempered logits (x 1 / T, f32), -inf off the keep set; `keep` the keep
/// mask (null: no filter set); `p` = softmax(lf), the target distribution every sampled path reads; `zero` the f32 0
/// scalar the graph built (reused by the caller's masks, so `sampledRows`' op trace is the one it had before the split).
pub fn Filtered(comptime T: type) type {
    return struct { lf: T, keep: ?T, p: T, zero: T };
}

/// The sampled paths' transform (the host reference `distribution`'s graph, one implementation for `sampledRows` and
/// `typicalRows`): the logits [W, V] tempered (x 1 / T, f32); top_k / top_p (when set: one descending sort per row,
/// the kept set's smallest tempered logit as its threshold, ties kept) and min_p (against the row's top probability)
/// as a keep mask; the kept set's probabilities p.
pub fn filteredRows(g: anytype, row2: anytype, sm: Params) !Filtered(@TypeOf(row2)) {
    const sh = g.shapeOf(row2);
    const vocab = sh.dim(1);
    const lt = try g.mul(try g.astype(row2, .float32), try g.scalar(1.0 / sm.temperature, .float32));
    const ninf = try g.scalar(-std.math.inf(f64), .float32);
    const zero = try g.scalar(0, .float32);
    var keep: ?@TypeOf(row2) = null;
    if (sm.sorts()) {
        const srt = try g.neg(try g.sort(try g.neg(lt), -1));
        const ps = try g.softmax(srt, -1);
        const rank = try g.reshape(try g.arange(0, @floatFromInt(vocab), 1, .int32), &.{ 1, vocab });
        var kk: ?@TypeOf(row2) = if (sm.top_k > 0) try g.less(rank, try g.scalar(@floatFromInt(sm.top_k), .int32)) else null;
        if (sm.top_p < 1.0) {
            var pk = ps;
            if (kk) |m| {
                pk = try g.where(m, ps, zero);
                pk = try g.div(pk, try g.sum(pk, -1, true));
            }
            const before = try g.sub(try g.cumsum(pk, -1), pk);
            const kp = try g.less(before, try g.scalar(sm.top_p, .float32));
            kk = if (kk) |m| try g.logicalAnd(m, kp) else kp;
        }
        const count = try g.sum(try g.astype(kk.?, .int32), -1, true);
        const thr = try g.takeAlongAxis(srt, try g.sub(count, try g.scalar(1, .int32)), -1);
        keep = try g.greaterEqual(lt, thr);
    }
    if (sm.min_p) |mp| {
        const base_l = if (keep) |k| try g.where(k, lt, ninf) else lt;
        const pm = try g.softmax(base_l, -1);
        const km = try g.greaterEqual(pm, try g.mul(try g.max(pm, -1, true), try g.scalar(mp, .float32)));
        keep = if (keep) |k| try g.logicalAnd(k, km) else km;
    }
    const lf = if (keep) |k| try g.where(k, lt, ninf) else lt;
    return .{ .lf = lf, .keep = keep, .p = try g.softmax(lf, -1), .zero = zero };
}

/// The draws' uniforms for `w` rows: v = `drawV(seed, base + row + 1)` (the drawn token's absolute position), f32 [w, 1].
fn drawUniforms(g: anytype, w: c_int, sm: Params, base: u64) !@TypeOf(g.*).T {
    const nw: usize = @intCast(w);
    var vb: [max_rows]f32 = undefined;
    for (vb[0..nw], 0..) |*v, i| v.* = drawV(sm.seed, base + i + 1);
    return g.hostArray(std.mem.sliceAsBytes(vb[0..nw]), &.{ w, 1 }, .float32);
}

/// Each row's draw from `src` [W, V] (unnormalised mass) at its uniform `vv` [W, 1] (`drawUniforms`): the first token
/// whose cumulative mass reaches v x the row's total: uint32 [W].
fn drawRows(g: anytype, src: anytype, vv: @TypeOf(src)) !@TypeOf(src) {
    const sh = g.shapeOf(src);
    const w = sh.dim(0);
    const vocab = sh.dim(1);
    const cum = try g.cumsum(src, -1);
    const total = try g.slice(cum, &.{ 0, vocab - 1 }, &.{ w, vocab }, &.{ 1, 1 });
    return g.astype(try g.sum(try g.astype(try g.less(cum, try g.mul(vv, total)), .int32), -1, false), .uint32);
}

/// A sampled request's rows (sample-then-verify; the host's reference `distribution` / `specStep`): p = `filteredRows`;
/// every row's token is p's draw (`drawRows`, v = `drawV(seed, pos)`, pos = `base` + row + 1, the drawn token's
/// absolute position): the token the serial step draws there. `accept` (bool [drafts.len], null without drafts) = the
/// row's draft is its token. One host array of W floats and one of the drafts.
pub fn sampledRows(g: anytype, row2: anytype, sm: Params, base: u64, drafts: []const u32) !SampledRows(@TypeOf(row2)) {
    const w = g.shapeOf(row2).dim(0);
    const f = try filteredRows(g, row2, sm);
    const tok = try drawRows(g, f.p, try drawUniforms(g, w, sm, base));
    var accept_: ?@TypeOf(row2) = null;
    if (drafts.len > 0) {
        const nd: c_int = @intCast(drafts.len);
        const xs = try g.hostArray(std.mem.sliceAsBytes(drafts), &.{nd}, .uint32);
        accept_ = try g.equal(try g.slice(tok, &.{0}, &.{nd}, &.{1}), xs);
    }
    return .{ .tok = tok, .accept = accept_, .p = f.p };
}

/// A sampled request's rows under the typical rule (`acceptGeneral`'s graph; Medusa's typical acceptance): p =
/// `filteredRows` (the same transform as `sampledRows`). `accept` (bool [drafts.len], null without drafts) = p(x) >
/// min(eps, delta x exp(-H(p))) on the first `drafts.len` rows (`typicalTest` on the filtered logits, H over the kept
/// set); every row's token is p's own draw (`drawRows`, v = `drawV(seed, base + row + 1)`): the correction at a
/// rejected row is a plain draw from the target, no residual. `drawU` is not consumed.
pub fn typicalRows(g: anytype, row2: anytype, sm: Params, base: u64, drafts: []const u32, t: Typical) !SampledRows(@TypeOf(row2)) {
    const f = try filteredRows(g, row2, sm);
    const vv = try drawUniforms(g, g.shapeOf(row2).dim(0), sm, base);
    var accept_: ?@TypeOf(row2) = null;
    if (drafts.len > 0) {
        const vocab = g.shapeOf(row2).dim(1);
        const nd: c_int = @intCast(drafts.len);
        const lf = try g.slice(f.lf, &.{ 0, 0 }, &.{ nd, vocab }, &.{ 1, 1 });
        const keep = if (f.keep) |k| try g.slice(k, &.{ 0, 0 }, &.{ nd, vocab }, &.{ 1, 1 }) else null;
        accept_ = try typicalTest(g, lf, keep, drafts, t);
    }
    return .{ .tok = try drawRows(g, f.p, vv), .accept = accept_, .p = f.p };
}

const testing = std.testing;

/// `rows` x `v` seeded logits, each row with a leader; drafts alternate leader / runner / random.
fn testRows(seed: u64, comptime rows: usize, comptime v: usize, lg: *[rows * v]f32, drafts: *[rows - 1]u32, spread: f32) void {
    var rng: std.Random.DefaultPrng = .init(seed);
    for (lg) |*x| x.* = rng.random().floatNorm(f32) * spread;
    for (0..rows) |r| {
        const lead = rng.random().uintLessThan(u32, @as(u32, v));
        lg[r * v + lead] += 1.0 + 7.0 * rng.random().float(f32);
        if (r < rows - 1) drafts[r] = switch (rng.random().uintLessThan(u32, 4)) {
            0, 1 => lead,
            2 => (lead + 1) % @as(u32, v),
            else => rng.random().uintLessThan(u32, @as(u32, v)),
        };
    }
}

/// The lane's loop on the host reference over a seeded toy target: `model(prev, pos)` is row logits for the token at
/// `pos` after `prev`; the drafter chains a perturbed argmax. Rounds at most `cap` drafts; `n` tokens.
fn simLane(model_seed: u64, rule: Rule, sm: Params, cap: u32, n: usize, out: []u32) void {
    const V = 64;
    const Model = struct {
        fn row(ms: u64, prev: u32, pos: u64, salt: u64, lg: *[V]f32) void {
            var rng: std.Random.DefaultPrng = .init(ms ^ (@as(u64, prev) *% 0x9E3779B97F4A7C15) ^ (pos *% 0xD1B54A32D192ED03) ^ salt);
            for (lg) |*x| x.* = rng.random().floatNorm(f32);
            lg[rng.random().uintLessThan(u32, V)] += 3.0;
        }
    };
    var emitted: usize = 0;
    var t1: u32 = 0;
    var pos: u64 = 100;
    while (emitted < n) {
        const k: u32 = @intCast(@min(cap, n - emitted - 1));
        var ids: [max_rows]u32 = undefined;
        ids[0] = t1;
        for (1..k + 1) |i| {
            var dl: [V]f32 = undefined;
            Model.row(model_seed, ids[i - 1], pos + i, 0x5eed, &dl);
            var tl: [V]f32 = undefined;
            Model.row(model_seed, ids[i - 1], pos + i, 0, &tl);
            for (&dl, tl) |*a, b| a.* = b + 0.6 * a.*;
            ids[i] = argmaxRow(&dl);
        }
        var lg: [(max_rows) * V]f32 = undefined;
        for (0..k + 1) |r| Model.row(model_seed, ids[r], pos + r + 1, 0, lg[r * V ..][0..V]);
        const r = acceptGeneral(rule, sm, lg[0 .. (k + 1) * V], ids[1 .. k + 1], null, pos);
        for (ids[0 .. r.n_accepted + 1]) |tok| {
            out[emitted] = tok;
            emitted += 1;
        }
        t1 = r.next_token;
        pos += r.n_accepted + 1;
    }
}
