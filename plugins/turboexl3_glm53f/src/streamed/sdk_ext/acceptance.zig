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

const testing = std.testing;
