//! MiMo's six-component MXFP4 bank on the shared SDK stream. The SDK owns
//! residency, asynchronous reads, windows, pinning and release; this bridge
//! maps absolute model layers and exposes GatherQmm's {w, scales} bindings.
const std = @import("std");
const env = @import("mlx_host").env;
const sdk_ext = @import("../sdk_ext.zig");
const mlx = @import("sdk").mlx;
const bank_mod = @import("bank.zig");
const expert_policy = sdk_ext.expert.policy;
const profile = @import("profile.zig");
const route_trace = @import("route_trace.zig");
const prefill_timeline = @import("prefill_timeline.zig");
const gpu_warm = @import("gpu_warm.zig");
const diag = @import("diag.zig");
/// MiMo's bridge over the shared SDK stream (`stream_source.For`): its bank module, GatherQmm's {w, scales} slot bindings
/// and its read pool's staging.
const Mimo = @import("stream_source.zig").For(bank_mod, .{ .Arrays = sdk_ext.quant.GatherQmm.Arrays, .staging_bytes = io_staging_bytes });
const S = Mimo.S;
pub const Stream = Mimo.Stream;
pub const Options = Mimo.Options;
pub const SlotMemory = Mimo.SlotMemory;
pub const LayerSlotBank = Mimo.LayerSlotBank;
pub const Route = Mimo.Route;
pub const Part = Mimo.Part;
pub const BankKind = Mimo.BankKind;
pub const SlotRef = Mimo.SlotRef;
pub const Error = Mimo.Error;
pub const Stats = Mimo.Stats;
pub const Gates = Mimo.Gates;
pub const max_route_ids = Mimo.max_route_ids;
pub const uses_reader = Mimo.uses_reader;
pub const BankArrays = bank_mod.BankArrays;
pub const BankArraysOf = bank_mod.BankArraysOf;

pub const io_workers: u32 = 4;
pub const io_staging_bytes: u64 = 9 << 20;
/// P1's read-ahead admits min(rows, (tickets - 2 * max_items) / 2) records (`Stream.readAheadSeed`): 512 covers
/// every MiMo row a layer can hold (248 >= 124 at the fill; 256 tickets admitted only 120).
pub const io_tickets: u32 = 512;
pub const records_per_part: u32 = 2;
pub const wide_depth: u8 = 2;
pub const transient_rows: u32 = wide_depth * max_route_ids;
/// R3 (mimo/decode-rows-audit): decode's window 0, chosen once at construction (settings `"decode_window"`). `verify`
/// (the default) sizes it to the decode lane's widest route, `decode_lane_route_ids` = 4 rows x top-8 = 32 (a decode
/// plan's transient loads are its distinct non-admitted misses, at most its ids: `LayerPolicy.planWith`); `route` is
/// the control, the SDK's `max_route_ids` (48). Every decode-phase routed call is a decode-lane call (Module routes any
/// wider forward through the prompt phase first; `Source.forward` refuses a decode-phase wide call under `verify`).
pub const DecodeWindow = enum { verify, route, invalid };
/// The decode lane's widest route: `source.decode_lane_rows` (4: the MTP verify) x `source.top_k` (8), pinned there.
pub const decode_lane_route_ids: u32 = 4 * 8;
comptime {
    if (decode_lane_route_ids > max_route_ids) @compileError("mimo_v2: the decode lane's route would not fit one SDK route");
}
pub fn decodeRouteIds(window: DecodeWindow) u32 {
    return switch (window) {
        .verify => decode_lane_route_ids,
        .route, .invalid => max_route_ids,
    };
}
pub const lookahead = Mimo.lookahead;

pub const options = Mimo.options;

/// P7 (`settings.prefill_read_schedule`): the prompt pass's read schedule, fixed at construction. `current` (the control):
/// the stream's stock schedule. `early`: the stream's `overlap` schedule (`S.PromptReads`): every group of a prompt call
/// planned at its first route exactly as the stock schedule plans it, each read posted as soon as its row is free, the
/// read-ahead posted in rank order (head now, tail resolved by the plans: read once, the double reads dropped) and landed
/// only where a group needs it. Same records in the same rows before any kernel reads them: exact by construction.
/// `early_whole`: `early` with the WHOLE read-ahead posted with the attention, in rank order (`S.AheadPost.whole`):
/// p7-20261009 try 2 measured `early` moving ~0.84 s/pass of GPU idle into the router stage, the tail (ranks past
/// depth x width) being read only after the attention, at the call's plans, while 4 pool workers share the drive with
/// the call's misses; `whole` reads it during the attention instead, at the cost of the stock schedule's double reads
/// (records the plans evict before any hit). The same admissions, plans, slots and kernels: exact by construction.
/// `early_whole_trim` (X21, ahead-trim-20261010): `early_whole`, then at the plans the read-ahead jobs the reader has not
/// started yet are withdrawn from its queue (they wrote no row) and their records are resolved by the plans as `early`'s
/// tail is (`S.AheadPost.whole_trim`): read once if a group hits them first, never read if a plan evicts them first. At
/// the next9 clock (~76 ms to the plans on a sliding layer) the reader is still on the head: the tail (ranks 96+) is
/// queued and its evicted-first records, ~254-263 per 16K pass = ~3.5 GB, are not read; at the clamped clock (~120 ms)
/// every job has started and the schedule is `early_whole`'s. The same admissions, plans, slots and kernels: exact by
/// construction.
/// X20 `auto` (mimo/sched-auto): `early` below a prompt-length threshold, `early_whole` at or above it, chosen once per
/// request at its start (`AutoSchedule`, `StreamSource.chooseReadSchedule`); the stream is built once for both.
pub const ReadSchedule = enum { current, early, early_whole, early_whole_trim, auto, invalid };

/// X20: `auto`'s default threshold (settings `prefill_read_schedule_threshold`): shortprompt1-20261010 measured `early`
/// ahead of `early_whole` on every short prompt (codectx-4k 341.2/341.4 vs 316.3/318.0, codectx-1k 90.6/90.5 vs
/// 85.6/86.8, std-1k 89.1/89.4 vs 85.6/85.8 prefill tok/s) and P7 try 3 measured `early_whole` +7 % over `early` at 16K.
pub const default_read_schedule_threshold: u32 = 8192;
/// X20: the largest threshold a setting may name (the checkpoint's `max_position_embeddings`, 1,048,576).
pub const max_read_schedule_threshold: u32 = 1 << 20;

/// X20 `auto`: the request's schedule by its prompt length, a construction-time table of two exact routes that share
/// everything the stream is built with (the overlap schedule, its ticket ring, the stock admissions, the route triple)
/// and differ only in the read-ahead posted with the attention (`S.AheadPost`): `below` (`early`, the head) for a
/// prompt of fewer than `threshold` tokens, `at_or_above` (`early_whole`, the whole) otherwise. Consulted once per
/// request, at its start, never per layer or call.
pub const AutoSchedule = struct {
    threshold: u32 = default_read_schedule_threshold,

    pub const below: ReadSchedule = .early;
    pub const at_or_above: ReadSchedule = .early_whole;

    pub fn of(t: AutoSchedule, prompt_tokens: u64) ReadSchedule {
        return if (prompt_tokens < t.threshold) below else at_or_above;
    }
};

/// The `early` schedule's ticket ring: its posts run up to the read-ahead (<= 248 records) plus a whole call's loads
/// (<= 256) ahead of their consumers, 2 tickets each; the stock ring (`io_tickets`) is sized for one group's.
pub const io_tickets_early: u32 = 2048;

pub const applyReadSchedule = Mimo.applyReadSchedule;

const aheadPost = Mimo.aheadPost;

pub const eventOptions = Mimo.eventOptions;

const wavesOf = Mimo.wavesOf;

pub const traceBank = Mimo.traceBank;

pub const StreamSource = Mimo.StreamSource;

// ── HITPASS CPU tests: a host-memory MiMo stream (layer 0 dense, two routed layers, 64 experts, 4 rows each) over a
// tmp sidecar; pool pread workers and host slot rows only. No MLX array, device or lock. ──

const HitFixture = struct {
    const geom_args = .{ 32, 32, 64, 1, 2 };
    bytes: []u8,
    side: bank_mod.TestSidecar,
    stream: *Stream,

    /// In place: the stream retains `&f.side.bank`, the SDK bank `&f.side.owner`.
    fn init(f: *HitFixture, a: std.mem.Allocator, rows: []const u32) !void {
        return f.initOpts(a, options(rows, .host));
    }

    fn deinit(f: *HitFixture, a: std.mem.Allocator) void {
        f.stream.deinit();
        f.side.deinit();
        a.free(f.bytes);
    }
};

// ── P7 read schedule CPU proofs (host stream over the HitFixture's sidecar: two routed layers, 64 experts; pool pread
// workers, host rows; no MLX array, device or lock). The executor's depth-2 wide sequence is emulated: read-ahead, the
// executor's await, seed, hot order, groups 0-1 routed, then per group its reads landed, every routed expert's row
// checked against the sidecar, released, group + 2 routed. ──

/// The first P7 proof's 40 calls (seeded 0x7e7a11, rows 28): calls alternate layers 1 and 2, expert spans cycle over
/// 24..44 (3-4 groups of 11, under the fixture's 64); the read-ahead's ranking is the call's true hot order with
/// neighbours swapped, then (odd calls) a few cold experts pulled into the admitted ranks, head or tail (mispredictions:
/// drops, read-ahead rows evicted, seed misses).
const P7Calls40 = struct {
    const n_calls = 40;
    ids_store: [n_calls][64 * 8]u16 = undefined,
    pred_store: [n_calls][64]u16 = undefined,
    ids: [n_calls][]const u16 = undefined,
    pred: [n_calls][]const u16 = undefined,

};

// ── X20 (mimo/sched-auto) CPU proofs: `prefill_read_schedule: auto` on the same host stream fixture. ──

/// The scheduler's math on the trace backend: MiMo's projection type, the banked waves' shapes only.
const HitTraceMath = struct {
    pub const has_banked = true;
    pub const decode_forward_rows: u32 = 1;
    pub fn Projection(comptime A: type) type {
        return sdk_ext.quant.GatherQmm.Arrays(A);
    }
    hidden: c_int,
    inter: c_int,
    const Tr = @import("../ops.zig").TraceOps;
    pub fn gateUp(self: *const HitTraceMath, g: *Tr, x: u32, _: u32, _: Projection(u32), _: Projection(u32)) !u32 {
        return g.kernel(&.{ g.shapeOf(x).d[0], self.inter }, .bfloat16);
    }
    pub fn down(self: *const HitTraceMath, g: *Tr, h: u32, _: u32, _: Projection(u32)) !u32 {
        return g.kernel(&.{ g.shapeOf(h).d[0], self.hidden }, .bfloat16);
    }
    pub fn gateUpBanked(self: *const HitTraceMath, g: *Tr, x: u32, ids: u32, _: anytype) !u32 {
        if (g.shapeOf(ids).d[0] != g.shapeOf(x).d[0]) return error.MathShape;
        return g.kernel(&.{ g.shapeOf(x).d[0], self.inter }, .bfloat16);
    }
};

const HitScript = struct {
    calls: []const []const u16,
    next: usize = 0,
    fn values(self: *HitScript) HitTraceMath.Tr.HostValues {
        return .{ .ctx = self, .ids = ids, .argmax = argmax };
    }
    fn ids(ctx: *anyopaque, out: []u16) anyerror!void {
        const self: *HitScript = @ptrCast(@alignCast(ctx));
        if (self.next >= self.calls.len) return error.ScriptExhausted;
        const c = self.calls[self.next];
        self.next += 1;
        if (c.len != out.len) return error.ScriptShape;
        @memcpy(out, c);
    }
    fn argmax(_: *anyopaque) anyerror!u32 {
        return error.NoPicks;
    }
};

// ── end HITPASS CPU tests ──

// Host-memory stream (two routed layers, 64 experts, 4 rows): the prompt pass routes each layer's experts in the
// installed hot order, so decode starts with exactly the installed seed. CPU only: host slot rows and pread workers.
