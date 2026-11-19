//! The shared SDK stream's construction constants (the read pool, the prefill windows, decode's window 0) and the
//! prompt pass's read schedules, which `stream_source.For` and GLM-5.3-Flash's experts build their stream with.
const expert_policy = @import("../sdk_ext.zig").expert.policy;
/// The SDK route's widest id count (48).
pub const max_route_ids = expert_policy.max_route_ids;

pub const io_workers: u32 = 4;
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
