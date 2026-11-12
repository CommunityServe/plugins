//! turboexl3_ling's run-path line: the served paths that ran. Each served path that once printed its own
//! `[turboexl3_ling] X engaged: ...` line marks its bit here instead, at the same place, when it runs (`mark`), and
//! plugin.zig `forward` prints one line, `[turboexl3_ling] served paths ran (prompt): ...` after a prompt forward and
//! `... (decode): ...` after a decode round, whenever a path has run that no earlier line named (`report`). So a log has
//! a line after its first prompt and its first decode round, and another when a later request runs a path the earlier
//! ones did not (a warm-up's short prompt takes no span-major path). Each line names every path that has run since load,
//! in `Path`'s order (the stack's). Each request also gets its own two lines, `served paths ran (request, prompt): ...`
//! after its prompt and `(request, decode): ...` after its first decode round, naming only what that request ran
//! (`requestStart`, `reportRequest`), so marks the load's warm-up set cannot stand in for a request's. Nothing here
//! prints at load: a load line names what was built, not what ran, and the chain counts only lines like these.
//! `names` holds every path's name in one string the binary carries whatever ran, for the chain's carry check.
//! The streamer's files reach this module as `@import("mlx_host").ling_ran`.
const std = @import("std");

/// The served paths with a run-path hook, in the stack's order (CANDIDATES.md S1-S48; the MTP head's composed fallback
/// beside its fused launch). MTP_HEAD_LATENT_COMPOSED marks the composed latent chain, which the served shapes never take
/// (the fused launches take every head call). EMBED_ROWS is S51 (src/ling_embed.zig), set by a forward whose embedding
/// rows were read from the pack. KDA_GATE_EPI is S52 (src/ling_kdaproj.zig `projPromptGate`), set by a KDA prompt
/// projection whose launch made the gate (with KDA_PROJ_PROMPT: P4's launch ran). KDA_NORM_PREP is L5, kept
/// (src/ling_kda_norm_prep.zig `oProj`), set by a KDA layer's prompt span whose norm-gate launch wrote o_proj's
/// prepared rows. RQ is kept (src/ling_rq.zig), set at each re-plan of a request (plugin.zig `rqLine`). D2 is one KV
/// append for an MLA layer's prompt spans past 24,576 (the copied mla.zig `mlaGroup`), set by a forward whose MLA layer
/// appended two or more spans in one call. C4's two, last: LM_SEGMENTS, set by a forward whose run of wide spans ran in
/// two or more segments (src/ling_lm_pass.zig `runWide`), and MTP_HISTORY_CHUNKS, set by an MTP head history forward
/// that ran in two or more forwards (src/ling_mtp.zig `history`).
pub const Path = enum(u6) {
    DEVROUTE,
    COMBINE,
    SHARED_F3,
    MLA_SPANS,
    READS_WIDE,
    PAIR,
    KDA_PROJ_FUSED,
    LAYER_MAJOR,
    PREDICTOR,
    PROMPT_AHEAD,
    LOOKAHEAD_ROWS,
    ROUTER_FUSED,
    ROWS_PATH,
    ROUTER_F16,
    MOE_TAIL,
    LAST_ROW_LOGITS,
    PROMPT_KEEP,
    PROMPT_PARK,
    PROMPT_AHEAD_BF16,
    KDA_EPILOGUE,
    KDA_BPROJ,
    ROUTED_JOIN,
    KDA_PROJ_PROMPT,
    MTP_HEAD_Q4,
    SWIGLU_EPI,
    MTP_HEAD_LATENT,
    MTP_HEAD_LATENT_COMPOSED,
    PROMPT_AHEAD_FILL,
    MEM_REFIT,
    MLA_FLASH,
    SHARED_LAYER_WHAT,
    LAYER_WHAT,
    MLA_ROPE_ONCE,
    KDA_REC_SHAPE,
    PF_F16_PAIRS,
    MLA_LATENT_NAX,
    PF_MOE_TAIL,
    EMBED_ROWS,
    KDA_GATE_EPI,
    KDA_NORM_PREP,
    RQ,
    D2,
    LM_SEGMENTS,
    MTP_HISTORY_CHUNKS,
};

const count = @typeInfo(Path).@"enum".field_names.len;
comptime {
    if (count > 64) @compileError("ling_ran: more paths than bits");
}

/// Every path's name, space-separated after a fixed prefix, as one string in the binary (the chain's carry check reads
/// it with `strings`): `turboexl3_ling served path names: DEVROUTE COMBINE ...`.
pub const names: []const u8 = blk: {
    var s: []const u8 = "turboexl3_ling served path names:";
    for (@typeInfo(Path).@"enum".field_names) |name| s = s ++ " " ++ name;
    break :blk s;
};

var ran_bits = std.atomic.Value(u64).init(0);
var said_bits: u64 = 0;
/// The current request's bits (`requestStart` clears them) and which of its two lines printed.
var req_bits = std.atomic.Value(u64).init(0);
var req_said: [2]bool = .{ false, false };

/// `p` ran: its bit set in the since-load set and the request's (the first time and every time after; two atomic ors).
pub fn mark(comptime p: Path) void {
    const bit = @as(u64, 1) << @intFromEnum(p);
    _ = ran_bits.fetchOr(bit, .monotonic);
    _ = req_bits.fetchOr(bit, .monotonic);
}

/// The paths that have run since load, as bits in `Path`'s order.
pub fn ranBits() u64 {
    return ran_bits.load(.monotonic);
}

/// `p` has run since load (since the last `clearForTests`, in a test).
pub fn has(comptime p: Path) bool {
    return ranBits() & (@as(u64, 1) << @intFromEnum(p)) != 0;
}

/// Tests only: every bit cleared, so a test sees only the marks its own calls set. Served code never calls it.
pub fn clearForTests() void {
    ran_bits.store(0, .monotonic);
    said_bits = 0;
    req_bits.store(0, .monotonic);
    req_said = .{ false, false };
}

pub const Phase = enum { prompt, decode };

/// The current request's paths, since its `requestStart` (main, 2026-11-09: marks set in the load's warm-up must not
/// stand in for a request's).
pub fn requestBits() u64 {
    return req_bits.load(.monotonic);
}

/// A request's prompt starts (plugin.zig `forward`, the first prompt forward after a decode round or at load): its set
/// cleared, and its two lines not yet printed.
pub fn requestStart() void {
    req_bits.store(0, .monotonic);
    req_said = .{ false, false };
}

/// The request's line, once after its prompt (`.prompt`, at its first decode-phase forward) and once after its first
/// decode round (`.decode`): every path the request has run. The warm-up's prompts are requests too, before the
/// server's first POST; the chain's rule reads the lines after it.
pub fn reportRequest(phase: Phase) void {
    const i = @intFromEnum(phase);
    if (req_said[i]) return;
    req_said[i] = true;
    var buf: [1200]u8 = undefined;
    std.log.info("{s}", .{requestLine(&buf, phase, requestBits())});
}

/// The request boundary (plugin.zig `forward`), a pure function of the forwards it is shown: which forward starts a
/// request (`requestStart` before it runs) and which is its request's first decode-phase forward (the request's prompt
/// line before it runs, its decode line after it). MLX-free, so the tests drive it with the host's forward sequences.
/// Main's rule (2026-11-09, Q8): a prompt forward starts a request unless it continues the previous prompt forward. It
/// continues it when its first position is that forward's end (first position + rows) and nothing came between: the
/// host's chunk loop (LAYER_MAJOR=0) and its snapshot tail. So a prompt forward from position 0, one after a
/// decode-phase forward or at load, and one from anywhere else (a restored prefix) each start a request. mlx-serve's
/// warm-up (a decode-shape forward, then 8- and 32-row prefill forwards from 0) is three such starts, so the first real
/// request no longer inherits the warm-up's marks (Q6 started a request only while no prompt was running). The one case
/// it reads as a continuation by mistake: a request whose restored prefix ends exactly where a prompt-only request
/// before it ended, with no decode between.
pub const Boundary = struct {
    /// The previous forward's end when it was a prompt forward (its first position + its rows); null after a
    /// decode-phase forward, and at load.
    prev_prefill_end: ?u64 = null,
    /// The current request has run a prompt forward and no decode-phase forward yet.
    in_prompt: bool = false,

    /// A forward as the boundary reads it: a prompt forward (`.prefill`) or a decode-phase one (decode, verify), its
    /// first position and its rows.
    pub const In = struct { prefill: bool, pos0: u64, rows: u64 };
    /// What the forward does at the boundary.
    pub const Step = struct { start: bool = false, first_decode: bool = false };

    pub fn next(b: *Boundary, f: In) Step {
        var st: Step = .{};
        if (f.prefill) {
            const continues = if (b.prev_prefill_end) |end| end == f.pos0 else false;
            st.start = f.pos0 == 0 or !continues;
            b.in_prompt = true;
            b.prev_prefill_end = f.pos0 + f.rows;
        } else {
            st.first_decode = b.in_prompt;
            b.in_prompt = false;
            b.prev_prefill_end = null;
        }
        return st;
    }
};

/// The request's line: `[turboexl3_ling] served paths ran (request, prompt|decode): ...`, every path the request has run,
/// in `Path`'s order.
pub fn requestLine(buf: []u8, phase: Phase, bits: u64) []const u8 {
    var names_buf: [1024]u8 = undefined;
    return std.fmt.bufPrint(buf, "[turboexl3_ling] served paths ran (request, {t}): {s}", .{ phase, list(&names_buf, bits) }) catch "[turboexl3_ling] served paths ran (request): the line does not fit its buffer";
}

/// The line, when a path has run since the last one: every path that has run since load, in `Path`'s order.
pub fn report(phase: Phase) void {
    const now = ranBits();
    if (now == said_bits) return;
    said_bits = now;
    var buf: [1024]u8 = undefined;
    std.log.info("[turboexl3_ling] served paths ran ({t}): {s}", .{ phase, list(&buf, now) });
}

/// `bits`' names, space-separated, in `Path`'s order (the names string's slices, so the line and `names` agree).
pub fn list(buf: []u8, bits: u64) []const u8 {
    std.mem.doNotOptimizeAway(names.ptr);
    var w: usize = 0;
    inline for (@typeInfo(Path).@"enum".field_names) |name| {
        if (bits & (@as(u64, 1) << @intFromEnum(@field(Path, name))) != 0) {
            if (w + name.len + 1 > buf.len) return buf[0..w];
            if (w > 0) {
                buf[w] = ' ';
                w += 1;
            }
            @memcpy(buf[w..][0..name.len], name);
            w += name.len;
        }
    }
    return buf[0..w];
}

test "the names string holds every path once, in Path's order, after its prefix" {
    var it = std.mem.tokenizeScalar(u8, names["turboexl3_ling served path names:".len..], ' ');
    inline for (@typeInfo(Path).@"enum".field_names) |name| try std.testing.expectEqualStrings(name, it.next().?);
    try std.testing.expect(it.next() == null);
}

test "KDA_GATE_EPI (L6, S52) is the run-path name right after EMBED_ROWS: the names string the carry check reads holds the two in that order, and a request line with every path's bit set names every path in order" {
    try std.testing.expect(std.mem.indexOf(u8, names ++ " ", " EMBED_ROWS KDA_GATE_EPI ") != null);
    const head = "turboexl3_ling served path names:";
    const every: u64 = (@as(u64, 1) << count) - 1;
    var buf: [1200]u8 = undefined;
    try std.testing.expectEqualStrings("[turboexl3_ling] served paths ran (request, prompt):" ++ names[head.len..], requestLine(&buf, .prompt, every));
}

test "KDA_NORM_PREP (L5, kept) is the run-path name right after KDA_GATE_EPI: the names string the carry check reads holds the two in that order, and a request line with every path's bit set names every path in order" {
    try std.testing.expect(std.mem.indexOf(u8, names ++ " ", " KDA_GATE_EPI KDA_NORM_PREP ") != null);
    const head = "turboexl3_ling served path names:";
    const every: u64 = (@as(u64, 1) << count) - 1;
    var buf: [1200]u8 = undefined;
    try std.testing.expectEqualStrings("[turboexl3_ling] served paths ran (request, prompt):" ++ names[head.len..], requestLine(&buf, .prompt, every));
}

test "C4's LM_SEGMENTS and MTP_HISTORY_CHUNKS are the last run-path names, after D2, which follows RQ (kept): the names string the carry check reads ends with them, and a request line with every path's bit set names every path in order, C4's last" {
    try std.testing.expect(std.mem.endsWith(u8, names, " KDA_NORM_PREP RQ D2 LM_SEGMENTS MTP_HISTORY_CHUNKS"));
    const head = "turboexl3_ling served path names:";
    const every: u64 = (@as(u64, 1) << count) - 1;
    var buf: [1200]u8 = undefined;
    try std.testing.expectEqualStrings("[turboexl3_ling] served paths ran (request, prompt):" ++ names[head.len..], requestLine(&buf, .prompt, every));
}

test "has and clearForTests: a mark is seen until the bits are cleared" {
    clearForTests();
    defer clearForTests();
    try std.testing.expect(!has(.PAIR));
    mark(.PAIR);
    try std.testing.expect(has(.PAIR) and !has(.SHARED_F3));
    clearForTests();
    try std.testing.expect(!has(.PAIR));
}

test "list names the set bits in Path's order, and nothing when none is set" {
    var buf: [1024]u8 = undefined;
    try std.testing.expectEqualStrings("", list(&buf, 0));
    const b = (@as(u64, 1) << @intFromEnum(Path.MLA_LATENT_NAX)) | (@as(u64, 1) << @intFromEnum(Path.DEVROUTE));
    try std.testing.expectEqualStrings("DEVROUTE MLA_LATENT_NAX", list(&buf, b));
}

test "a request's set: a mark from before its start (the warm-up's) is absent, its own marks are present, and the since-load set keeps both" {
    clearForTests();
    defer clearForTests();
    mark(.KDA_PROJ_FUSED);
    requestStart();
    mark(.MLA_FLASH);
    const kda = @as(u64, 1) << @intFromEnum(Path.KDA_PROJ_FUSED);
    const flash = @as(u64, 1) << @intFromEnum(Path.MLA_FLASH);
    try std.testing.expectEqual(flash, requestBits());
    try std.testing.expectEqual(kda | flash, ranBits());
    // the next request starts empty again
    requestStart();
    try std.testing.expectEqual(@as(u64, 0), requestBits());
}

test "the request line names the request's paths after its prefix and phase" {
    var buf: [1200]u8 = undefined;
    const b = (@as(u64, 1) << @intFromEnum(Path.MLA_FLASH)) | (@as(u64, 1) << @intFromEnum(Path.DEVROUTE));
    try std.testing.expectEqualStrings("[turboexl3_ling] served paths ran (request, decode): DEVROUTE MLA_FLASH", requestLine(&buf, .decode, b));
    try std.testing.expectEqualStrings("[turboexl3_ling] served paths ran (request, prompt): ", requestLine(&buf, .prompt, 0));
}

/// The boundary tests' forwards: each one through a `Boundary`, its request started where the boundary says, then its
/// marks set. `sets` keeps each request's set as it closes (at the next request's start, or `close`); `printed` keeps the
/// set each request's prompt line prints (at its first decode-phase forward, before that forward's marks).
const Sim = struct {
    b: Boundary = .{},
    open: bool = false,
    sets: [8]u64 = @splat(0),
    n_sets: usize = 0,
    printed: [8]u64 = @splat(0),
    n_printed: usize = 0,

    fn run(sim: *Sim, prefill: bool, pos0: u64, rows: u64, comptime marks: []const Path) void {
        const st = sim.b.next(.{ .prefill = prefill, .pos0 = pos0, .rows = rows });
        if (st.start) {
            if (sim.open) sim.keep(requestBits());
            requestStart();
            sim.open = true;
        }
        if (st.first_decode) {
            sim.printed[sim.n_printed] = requestBits();
            sim.n_printed += 1;
        }
        inline for (marks) |p| mark(p);
    }
    fn keep(sim: *Sim, set: u64) void {
        sim.sets[sim.n_sets] = set;
        sim.n_sets += 1;
    }
    fn close(sim: *Sim) void {
        if (sim.open) sim.keep(requestBits());
        sim.open = false;
    }
};

fn pathBit(comptime p: Path) u64 {
    return @as(u64, 1) << @intFromEnum(p);
}

test "the request boundary, the load's warm-up (decode 1 row, then prefill 8 and prefill 32 from 0) and a prompt from 0: the prompt's set holds its own marks, not the 8-row forward's" {
    clearForTests();
    defer clearForTests();
    var sim: Sim = .{};
    // mlx-serve's warm-up (66aad949 engines/mlx/warmup.zig): a [1, 1] decode-shape forward, then [1, 8] and [1, 32]
    // prefill forwards, each after a cache reset (so from position 0); no decode after them
    sim.run(false, 0, 1, &.{ .PAIR, .SHARED_F3, .ROUTER_F16, .READS_WIDE });
    sim.run(true, 0, 8, &.{.KDA_PROJ_FUSED});
    sim.run(true, 0, 32, &.{.SWIGLU_EPI});
    // the first request: its whole prompt from 0, then its first verify
    sim.run(true, 0, 18380, &.{ .LAYER_MAJOR, .MLA_FLASH });
    sim.run(false, 18380, 2, &.{.MTP_HEAD_LATENT});
    sim.close();
    try std.testing.expectEqual(@as(usize, 1), sim.n_printed);
    try std.testing.expectEqual(pathBit(.LAYER_MAJOR) | pathBit(.MLA_FLASH), sim.printed[0]);
    try std.testing.expect((sim.printed[0] & pathBit(.KDA_PROJ_FUSED)) == 0);
    // each request its own set: the two warm-up prompts, then the request with its verify's mark
    try std.testing.expectEqualSlices(u64, &.{ pathBit(.KDA_PROJ_FUSED), pathBit(.SWIGLU_EPI), pathBit(.LAYER_MAJOR) | pathBit(.MLA_FLASH) | pathBit(.MTP_HEAD_LATENT) }, sim.sets[0..sim.n_sets]);
}

test "the request boundary, requests with no decode between (a prompt from 0, one from a restored prefix at 512, one from 0): each gets its own set" {
    clearForTests();
    defer clearForTests();
    var sim: Sim = .{};
    sim.run(true, 0, 1000, &.{.LAYER_MAJOR});
    // a prefix restored to 512 tokens: the next prompt forward starts at 512, not at the previous prompt's end (1,000)
    sim.run(true, 512, 700, &.{.MLA_FLASH});
    sim.run(true, 0, 300, &.{.PROMPT_AHEAD_FILL});
    sim.run(false, 300, 2, &.{.MTP_HEAD_LATENT});
    sim.close();
    try std.testing.expectEqualSlices(u64, &.{ pathBit(.LAYER_MAJOR), pathBit(.MLA_FLASH), pathBit(.PROMPT_AHEAD_FILL) | pathBit(.MTP_HEAD_LATENT) }, sim.sets[0..sim.n_sets]);
    try std.testing.expectEqual(@as(usize, 1), sim.n_printed);
    try std.testing.expectEqual(pathBit(.PROMPT_AHEAD_FILL), sim.printed[0]);
}

test "the request boundary, LAYER_MAJOR=0's chunk loop after the load's warm-up: chunks at 0, c and 2c stay one request holding all three chunks' marks, and so do one-row chunks after a decode" {
    clearForTests();
    defer clearForTests();
    var sim: Sim = .{};
    sim.run(false, 0, 1, &.{.PAIR});
    sim.run(true, 0, 8, &.{.KDA_PROJ_FUSED});
    sim.run(true, 0, 32, &.{.SWIGLU_EPI});
    // the host's chunk loop (the parity instrument, LAYER_MAJOR=0): chunks of c rows at 0 and c, the rest at 2c, each
    // forward's first position the previous one's end, no decode between
    const c: u64 = 4096;
    sim.run(true, 0, c, &.{.ROWS_PATH});
    sim.run(true, c, c, &.{.COMBINE});
    sim.run(true, 2 * c, 1000, &.{.PF_F16_PAIRS});
    sim.run(false, 2 * c + 1000, 2, &.{.MTP_HEAD_LATENT});
    // the next request at --prefill-chunk 1: one-row chunks at 0 to 4, then its first decode
    sim.run(true, 0, 1, &.{.DEVROUTE});
    for (1..4) |i| sim.run(true, i, 1, &.{});
    sim.run(true, 4, 1, &.{.LOOKAHEAD_ROWS});
    sim.run(false, 5, 1, &.{});
    sim.close();
    try std.testing.expectEqual(@as(usize, 2), sim.n_printed);
    try std.testing.expectEqual(pathBit(.ROWS_PATH) | pathBit(.COMBINE) | pathBit(.PF_F16_PAIRS), sim.printed[0]);
    try std.testing.expectEqual(pathBit(.DEVROUTE) | pathBit(.LOOKAHEAD_ROWS), sim.printed[1]);
}
