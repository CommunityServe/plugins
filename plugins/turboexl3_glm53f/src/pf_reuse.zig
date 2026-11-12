//! `TURBOEXL3_GLM53F_PF_REUSE=1` (opt-in, exact): a prompt pass reuses its buffers instead of allocating them afresh. Two
//! parts, one switch:
//!
//! 1. A KDA run's buffers held in MLX's cache from its first layer to its last (`Schedule`, through
//!    patches/turboexl3-model.patch in the copy's table_forward.zig `hidden` and `hiddenSplit`).
//!    Every KDA layer of a forward allocates the same sizes, but the prompt's 2 GiB cache limit releases every
//!    buffer freed past it, and MLX reuses a cached buffer only within two pages of a request, so each KDA layer
//!    allocates its working set afresh. At a run's first KDA layer (the next attention layer KDA too), MLX's cache limit
//!    goes up by `cap_row_bytes` a forward row, the bill's envelope for one layer's working set (stream_room
//!    `envelopeRowBytes`), and stays there through the run: what a layer frees stays cached, and the run's next KDA
//!    layer takes it back. The cache holds only what the run's layers freed and did not take back, at most the cap; the
//!    limit is not lowered inside the run, since MLX trims a cache over its limit from its oldest buffers, the held
//!    ones. At the first attention layer after the run (a DSA layer) and at the forward's end, the limit goes back to the
//!    value in force before the run, and MLX trims the cache at its next allocation: the DSA layers see today's cache.
//! 2. The routed prefill's row-slot-sized arrays at size classes (turboexl3's `Config.pf_size_class`, configured at load
//!    by `applySizeClasses`): a call's arrays at its row-slots rounded up to a multiple of 256, so successive calls'
//!    buffers match each other in the cache.
//!
//! g53prof8 (a 16K prompt at step 14, no stage syncs): the pass made about 375 GB of fresh buffers (the router stage
//! 246 GB, a KDA layer 6.2 to 7.6 GB, the MoE stage 110 GB in 1,558 tiles), and the GPU idled 1.4 s, mostly while the
//! host allocated (about 15 ms a GB). Only which buffer an array uses changes: the same kernels over the same rows, so
//! the same words. MLX's peak memory is never reset here (the server reads it).
//!
//! Memory: through a run's MoE stages the cache holds its KDA layer's working set, which the next layer's attention
//! takes back; the DSA layers, the prompt's peak stage, see today's cache. The pass line prints what a run's first layer
//! left and the most the cache held at a run's layer start; the probe pair and the A/B's peak columns decide whether a
//! peak moved.
//!
//! `TURBOEXL3_GLM53F_PF_MOE_HOLD=1` (opt-in, exact, with PF_REUSE): a DSA layer's routed prefill finds its first two
//! chunks' arrays in MLX's cache. g53pflt (the 62-entry control, a 16K prompt): at each DSA layer the MoE stage's first
//! chunk spent 13-18 ms of host time before its first commit (0.11-0.19 ms at a KDA layer) and the GPU idled 144 ms over
//! the 11 DSA layers. The run's limit goes back at the DSA layer's start, and the layer's attention allocates and frees
//! far more than the 2 GiB cache keeps, oldest first, so no chunk-sized buffer is left when its MoE stage starts and the
//! chunks allocate their arrays afresh. With the switch:
//! - at a DSA layer's start, before the run's limit goes back, the hold takes two chunks' worth of what turboexl3's
//!   routed chain allocates for a `routed_t3.prompt_rows` chunk at its size class: the exact chain's `chunkBytes`
//!   (`hold_arrays` arrays), or with `Config.pf_int8` set (TURBOEXL3_GLM53F_PF_INT8) the int8 chain's `chunkBytesI8`
//!   (`i8_hold_arrays`), each at its MLX buffer's bytes (`Chain`, read from turboexl3's configuration at `holdInstall`).
//!   One launch that writes nothing makes them, so they take what the cache holds of those sizes (the previous layer's
//!   last chunks' buffers). The layer's attention cannot trim them, since they are not in the cache.
//! - at the layer's first routed call (routed_t3 `prefillParts`), the hold goes back to MLX's cache under a limit raised
//!   by its bytes (a buffer freed with the cache at its limit is released, not cached), and the call's first two chunks
//!   take its buffers. The next layer's start (or the forward's end) puts the limit back, and lets go of a hold that no
//!   routed call took.
//! Only which buffer an array uses changes, and nothing reads the held arrays, so the same words. Memory: the hold's
//! bytes (482.5 MiB at GLM-5.3-Flash's geometry, 407.6 MiB on the int8 chain) more through each DSA layer's attention,
//! the prompt's peak stage. Not billed: the A/B's peak column decides. The pass line prints the holds taken, used and
//! let go, the bytes the holds took from the cache, and the most a routed call's first two chunks allocated past the
//! hand-back.
//!
//! `TURBOEXL3_GLM53F_PF_MOE_HOLD_TOP=1` (opt-in, exact, with PF_MOE_HOLD): the hand-back raises the limit from the
//! cache's bytes when they are over it. g53pf63t (the 63-entry control's timeline): at every DSA layer the cache held
//! 2858-2921 MiB at the routed call's entry, over the hand-back's 2530 MiB limit (the 2048 MiB limit and the hold): MLX
//! caches a freed buffer whole while the cache is under its limit, and trims only at its next allocation. A buffer
//! freed over the limit is released: the hand-back released all 26 held buffers (3.7-8.8 ms of host time), and chunk 0
//! allocated its arrays afresh while its first allocation trimmed the cache (its commit 2.7-9.7 ms, 0.1 ms at a KDA
//! layer). With the switch the hand-back's limit is max(the limit, the cache's bytes) + the hold, so every held buffer
//! is cached and the call's first two chunks take them, with nothing to trim; the next layer's start puts the limit
//! back as before. A forward's DSA layers before its first routed call hold nothing (the dense layers
//! before them leave no chunk-sized buffer to take). Memory: the cache keeps what it held over the limit through the
//! DSA layer's MoE stage (about 0.5-1 GB at a 16K prompt) instead of trimming it at the stage's start, and the next KDA
//! run's raise carries it until the next DSA layer's drop; the DSA attention sees what it sees with PF_MOE_HOLD alone.
//!
//! `TURBOEXL3_GLM53F_PF_RUN_TRIM=1` (opt-in, exact, with PF_REUSE): a run's cache is trimmed at its last KDA layer's
//! routed call's end, not at the next DSA layer's first allocation. g53hmt (PF_INT8=4's host-marks probe, a 16K prompt):
//! each DSA layer from L7 on spent 1.3-8.8 ms (median 6.1) of host time with the GPU idle between its start and its KV
//! append, where a KDA layer spends 0.04 ms and L3 (no run before it) 0.23 ms: the run's limit goes back at the DSA
//! layer's start with the cache holding up to 8.4 GB, and MLX's first allocation in the attention releases what is past
//! the limit (each buffer out of the residency set and released, under the allocator's lock) before anything is queued.
//! With the switch the run's last KDA layer (the next attention layer not KDA), once its routed call has two chunks in
//! flight (routed_t3 `prefillParts`; at its end with one), puts the limit before the run in force, makes one small
//! allocation (MLX trims the cache to the limit there, oldest first, while the GPU runs the chunks) and puts the run's
//! limit back: the chunks' own buffers are the newest (kept), what the layer frees after is cached as before (its later
//! chunks take their predecessors', the DSA layer's hold its last chunks'), and the DSA layer's drop finds only that.
//! Only which buffers MLX releases when moves, so the same words. Memory: the cache holds less, sooner. The pass line at a
//! forward's end (`TURBOEXL3_GLM53F_PF_RUN_TRIM pass rows= trims= trimmed_mb= drop_excess_mb_max= drop_excess_mb_sum=`)
//! prints the trims and what the DSA layers' drops still found past the limit.
//!
//! With PF_MOE_HOLD (TOP or not), the forward's end prints the take, each count read from MLX's allocator around one
//! step (`TURBOEXL3_GLM53F_PF_MOE_HOLD take`):
//! - each held buffer at the hand-back, by MLX's active and cached bytes around its free: cached (active down and the
//!   cache up by its bytes), released (active down, the cache not up), or still held by its launch's command buffer
//!   (active not down: it is freed at the command buffer's completion, under the limit then in force);
//! - the hand-backs that left the cache over its limit, and the excess (what the next allocation trims at least);
//! - the cache's drop across the call's first allocation (chunk 0's row table): the bytes MLX trims there, since it
//!   trims a cache over its limit after an allocation (the table itself reuses at most a 32 KiB buffer);
//! - the cache's drop and the active growth across chunks 0 and 1's commits, where MLX allocates the chain's arrays as
//!   it encodes them (signed: a buffer taken from the cache moves its bytes from the cache to active, a fresh one adds
//!   to active only; what other work frees meanwhile lowers both);
//! - the chunk outputs whose buffer is a held one the hand-back cached (by its address).
const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const t3 = @import("turboexl3");
const log = @import("host/log.zig");

/// `TURBOEXL3_GLM53F_PF_REUSE=1`, set by the plugin at load, before any forward.
pub var on: bool = false;
/// A run's raise of MLX's cache limit, a forward row (stream_room `envelopeRowBytes`), set at load.
pub var cap_row_bytes: u64 = 0;
/// The switch's load line (an A/B step reads "<switch>=1: "): the cap a forward row (KiB) and turboexl3's size class.
pub const line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_PF_REUSE=1: a prompt pass's buffers reused: a KDA run's held in MLX's cache from its first layer to its last (the limit raised by {d} KiB a forward row, the bill's layer envelope, through the run, then put back), and the routed prefill's row-slot arrays at multiples of {d} row-slots (turboexl3's size classes); the same kernels on the same rows\n";
/// A forward's rows below this hold nothing (a decode or verify window, a short prompt).
pub const min_rows: u64 = 2048;

/// The arrays a routed prompt chunk allocates at its size class, on the exact chain (`chunkBytes`) and on the int8 chain
/// (`chunkBytesI8`), and the chunks the hold covers: the two that routed_t3's `prefillParts` keeps in flight.
pub const chunk_arrays = 13;
pub const i8_chunk_arrays = 16;
pub const hold_chunks = 2;
/// The most arrays a chunk's hold takes, and a hold's (the int8 chain's).
const max_chunk_arrays = @max(chunk_arrays, i8_chunk_arrays);
const max_hold_arrays = max_chunk_arrays * hold_chunks;
/// The routed chain whose arrays the hold takes: turboexl3's exact chain, or its int8 chain (`Config.pf_int8` 1 or 2:
/// the same shapes either scheme).
pub const Chain = enum {
    exact,
    int8,

};
/// `TURBOEXL3_GLM53F_PF_RUN_TRIM=1`, set by the plugin at load (it needs PF_REUSE); the schedule reads it at each
/// forward's start.
pub var run_trim: bool = false;
pub const run_trim_pass_line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_PF_RUN_TRIM pass rows={d}: trims={d} trimmed_mb={d} drop_excess_mb_max={d} drop_excess_mb_sum={d}\n";
/// The take's line at a forward's end (the file's doc), with PF_MOE_HOLD.
pub const take_line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_PF_MOE_HOLD take rows={d}: top={d} handbacks={d} handback_buffers={d} cached={d} ({d} MiB) released={d} still_in_flight={d} over_limit={d} excess_mb={d} first_alloc_trims={d} first_alloc_cache_drop_mb={d} chunk01_commits={d} commit_cache_drop_mb={d} commit_active_mb={d} outputs_from_hold={d}/{d}\n";
/// What a hand-back freed, by MLX's active and cached bytes around each buffer's free: the held buffers; those the
/// cache kept (active down and the cache up by the buffer's bytes) and their bytes; those released (active down, the
/// cache not up); those still held by their launch's command buffer (active not down).
pub const HandBack = struct { bufs: u32 = 0, cached: u32 = 0, released: u32 = 0, deferred: u32 = 0, cached_bytes: u64 = 0 };
/// The take's points in a measured routed call (routed_t3 `prefillParts`): around chunk 0's first allocation, and
/// before chunks 0 and 1's commits (after each, `committed`).
pub const Point = enum { first_alloc_before, first_alloc_after, commit_before };
/// The most the cache drops across the call's first allocation without a trim: the row table's 16 KiB taken from a
/// cached buffer under twice its bytes.
const first_alloc_reuse_max: usize = 32 << 10;

/// A forward's counts (its pass line at its end).
pub const Stats = struct {
    /// The forward's rows (both spans of a split forward) and its cap (bytes).
    rows: u64 = 0,
    cap: u64 = 0,
    /// Runs started (a first KDA layer's limit raised), later KDA layers of a run (they take the held buffers back),
    /// runs ended (the limit put back).
    runs: u32 = 0,
    holds: u32 = 0,
    drops: u32 = 0,
    /// The bytes a run's first layer left in the cache, smallest and largest over the forward's runs, and the most the
    /// cache held at a later layer's start.
    held_min: u64 = std.math.maxInt(u64),
    held_max: u64 = 0,
    cache_max: u64 = 0,
    /// The highest limit set.
    limit_max: u64 = 0,
    /// TURBOEXL3_GLM53F_PF_MOE_HOLD: holds taken (at DSA layers), handed back at a routed call, let go unused, and
    /// refused by MLX; the bytes the holds took from the cache (summed); the routed calls whose first chunks were
    /// measured, and the most and the sum of what those chunks allocated past MLX's active and cached bytes at the
    /// hand-back.
    mh_taken: u32 = 0,
    mh_used: u32 = 0,
    mh_unused: u32 = 0,
    mh_failed: u32 = 0,
    mh_cached: u64 = 0,
    mh_measured: u32 = 0,
    mh_fresh_max: u64 = 0,
    mh_fresh_sum: u64 = 0,
    /// The take (the file's doc): the held buffers freed at the hand-backs, cached, released and still held by their
    /// command buffer, and the cached ones' bytes; the hand-backs that left the cache over its limit and the excess
    /// summed; the calls whose first allocation trimmed the cache, and the cache's drop across it summed; chunks 0 and
    /// 1's commits measured, and the cache's drop and active growth across them summed (signed); the chunk outputs
    /// looked at and those whose buffer is a held one.
    mh_hb_bufs: u32 = 0,
    mh_hb_cached: u32 = 0,
    mh_hb_released: u32 = 0,
    mh_hb_deferred: u32 = 0,
    mh_hb_cached_bytes: u64 = 0,
    mh_over_limit: u32 = 0,
    mh_excess: u64 = 0,
    mh_first_trims: u32 = 0,
    mh_first_drop: u64 = 0,
    mh_commits: u32 = 0,
    mh_commit_drop: i64 = 0,
    mh_commit_alloc: i64 = 0,
    mh_y_seen: u32 = 0,
    mh_y_held: u32 = 0,
    /// TURBOEXL3_GLM53F_PF_RUN_TRIM: the runs' last KDA layers' trims at their routed calls' ends and the bytes they
    /// released; at the DSA layers' drops, what the cache held past the limit put back (the next allocation's trim), the
    /// most and the sum.
    rt_trims: u32 = 0,
    rt_trimmed: u64 = 0,
    rt_excess_max: u64 = 0,
    rt_excess_sum: u64 = 0,
};

fn signedDiff(a: usize, b: usize) i64 {
    return @as(i64, @intCast(a)) - @as(i64, @intCast(b));
}

/// A routed call's take while it is measured (`Schedule.routed` to `committed`): MLX's active and cached bytes summed
/// after the hand-back (PF_MOE_HOLD's first-chunks measure); the cache before the call's first allocation; the cache
/// and active bytes before a commit, while one is open.
const Measure = struct { sum: usize, first_c: usize = 0, commit_c: usize = 0, commit_a: usize = 0, commit_open: bool = false };

/// The schedule over a backend `B`: MLX's for the plugin (`Mlx`), a recording fake for the MLX-free tests. `B` has
/// `cacheBytes() usize`, `setCacheLimit(limit) usize` (the previous limit) and `peak() usize` (MLX's peak since the
/// process began: never reset here); for TURBOEXL3_GLM53F_PF_MOE_HOLD also `activeBytes() usize`, `hold() bool` (the
/// hold's arrays made: false when MLX refused them), `unhold() HandBack` (the hold's arrays freed, counted buffer by
/// buffer) and `fromHold(addr) bool` (a buffer address is one the last hand-back cached); for
/// TURBOEXL3_GLM53F_PF_RUN_TRIM `trimNow()` (one small allocation, after which MLX trims a cache over its limit).
pub fn Schedule(comptime B: type) type {
    return struct {
        const Self = @This();
        b: B,
        /// `begin` calls not yet ended (a forward inside a forward is left to the outer one).
        depth: u32 = 0,
        /// The forward holds (a prompt forward of `min_rows` or more with a cap).
        active: bool = false,
        /// The limit in force before the run's raise, put back at its end; null outside a run.
        base: ?usize = null,
        /// In a run's first layer: the cache's bytes when it started (what it left is measured at the next layer).
        first: bool = false,
        c0: usize = 0,
        stats: Stats = .{},
        engaged_said: bool = false,
        /// TURBOEXL3_GLM53F_PF_MOE_HOLD: the hold's bytes (0: no hold), set when it is installed; the hold's arrays out
        /// (taken at a DSA layer's start, not yet handed back); the limit before a routed call's raise (put back at
        /// the next layer's start or the forward's end); MLX's active and cached bytes at the hand-back, while the
        /// call's first chunks are measured.
        hold: usize = 0,
        holding: bool = false,
        raised: ?usize = null,
        measure: ?Measure = null,
        hold_said: bool = false,
        /// TURBOEXL3_GLM53F_PF_MOE_HOLD_TOP: the hand-back's limit from the cache's bytes when they are over it, and no
        /// hold before the forward's first routed call (`seen_routed`: one has begun).
        top: bool = false,
        seen_routed: bool = false,
        /// TURBOEXL3_GLM53F_PF_RUN_TRIM: a run's last KDA layer trims the cache once its routed call has two chunks in
        /// flight (`trim_armed` from the layer's start to the trim or the next layer's start).
        run_trim: bool = false,
        trim_armed: bool = false,

        /// A forward's start: `prompt` a prompt forward's rows (`rows` of them, both spans of a split forward), `cap_row`
        /// the raise a row.
        pub fn begin(self: *Self, prompt: bool, rows: u64, cap_row: u64) void {
            self.depth += 1;
            if (self.depth != 1) return;
            self.active = prompt and rows >= min_rows and cap_row > 0;
            self.base = null;
            self.first = false;
            self.seen_routed = false;
            self.stats = .{ .rows = rows, .cap = rows *| cap_row };
        }

        /// Before layer `li`'s attention: `kda` this layer's attention KDA, `next_kda` the forward's next layer's.
        pub fn layer(self: *Self, li: usize, kda: bool, next_kda: bool) void {
            if (!self.active or self.depth != 1) return;
            // TURBOEXL3_GLM53F_PF_MOE_HOLD: the last layer's raise put back and a hold no routed call took let go, then
            // a DSA layer's hold taken before the run's limit goes back (the cache still holds what the run freed)
            self.settle();
            // TOP: none before the forward's first routed call (no chunk-sized buffer to take yet)
            if (!kda and self.hold > 0 and (!self.top or self.seen_routed)) self.take();
            if (self.base) |base| {
                // the run is over at its first other attention layer (TURBOEXL3_GLM53F_PF_RUN_TRIM: what the cache holds
                // past the limit put back, which MLX trims at the next allocation, counted)
                if (!kda) {
                    if (self.run_trim) {
                        const x = self.b.cacheBytes() -| base;
                        self.stats.rt_excess_max = @max(self.stats.rt_excess_max, x);
                        self.stats.rt_excess_sum +|= x;
                    }
                    return self.drop();
                }
                // TURBOEXL3_GLM53F_PF_RUN_TRIM: the run's last KDA layer (the next attention layer is not KDA) trims once
                // its routed call has two chunks in flight
                if (self.run_trim and !next_kda) self.trim_armed = true;
                // a later KDA layer of the run: it takes back what the run's layers left under the run's limit
                const c = self.b.cacheBytes();
                self.stats.cache_max = @max(self.stats.cache_max, c);
                self.stats.holds += 1;
                if (!self.first) return;
                self.first = false;
                const held = c -| self.c0;
                self.stats.held_min = @min(self.stats.held_min, held);
                self.stats.held_max = @max(self.stats.held_max, held);
                if (!self.engaged_said) {
                    self.engaged_said = true;
                    log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_PF_REUSE engaged: a KDA run's first layer left {d} MiB in MLX's cache for the run's later layers (at layer {d} the cache {d} MiB, the run's limit {d} MiB, {d} MiB before it)\n", .{ held >> 20, li, c >> 20, (base +| self.stats.cap) >> 20, base >> 20 });
                }
                return;
            }
            if (!(kda and next_kda)) return;
            // a run's first layer: what the run's layers free stays cached, at most the cap (the limit in force read by
            // raising it: a raise never trims)
            const prev = self.b.setCacheLimit(std.math.maxInt(usize));
            const limit = prev +| self.stats.cap;
            _ = self.b.setCacheLimit(limit);
            self.stats.limit_max = @max(self.stats.limit_max, limit);
            self.base = prev;
            self.c0 = self.b.cacheBytes();
            self.first = true;
            self.stats.runs += 1;
        }

        /// The forward's end: a run still open ends (the limit put back), and the forward's pass line.
        pub fn end(self: *Self) void {
            if (self.depth == 0) return;
            self.depth -= 1;
            if (self.depth != 0 or !self.active) return;
            self.settle();
            self.drop();
            self.active = false;
            const st = self.stats;
            log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_PF_REUSE pass rows={d}: runs={d} holds={d} drops={d} held_mb={d}..{d} cache_mb_max={d} cap_mb={d} limit_mb_max={d} mlx_peak_mb={d}\n", .{ st.rows, st.runs, st.holds, st.drops, if (st.runs == 0 or st.held_min == std.math.maxInt(u64)) 0 else st.held_min >> 20, st.held_max >> 20, st.cache_max >> 20, st.cap >> 20, st.limit_max >> 20, self.b.peak() >> 20 });
            if (self.hold > 0) log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_PF_MOE_HOLD pass rows={d}: hold_mb={d} taken={d} used={d} unused={d} failed={d} from_cache_mb={d} first_chunks_measured={d} fresh_kb_max={d} fresh_kb_sum={d}\n", .{ st.rows, self.hold >> 20, st.mh_taken, st.mh_used, st.mh_unused, st.mh_failed, st.mh_cached >> 20, st.mh_measured, st.mh_fresh_max >> 10, st.mh_fresh_sum >> 10 });
            if (self.run_trim) log.info(run_trim_pass_line, .{ st.rows, st.rt_trims, st.rt_trimmed >> 20, st.rt_excess_max >> 20, st.rt_excess_sum >> 20 });
            if (self.hold > 0) log.info(take_line, .{ st.rows, @intFromBool(self.top), st.mh_used, st.mh_hb_bufs, st.mh_hb_cached, st.mh_hb_cached_bytes >> 20, st.mh_hb_released, st.mh_hb_deferred, st.mh_over_limit, st.mh_excess >> 20, st.mh_first_trims, st.mh_first_drop >> 20, st.mh_commits, @divTrunc(st.mh_commit_drop, 1 << 20), @divTrunc(st.mh_commit_alloc, 1 << 20), st.mh_y_held, st.mh_y_seen });
        }

        /// TURBOEXL3_GLM53F_PF_MOE_HOLD: a routed prompt call's start (routed_t3 `prefillParts`). A DSA layer's hold
        /// goes back to MLX's cache under a limit raised by its bytes, for the call's first chunks; the next layer's start
        /// puts the limit back. A forward inside the forward takes nothing.
        pub fn routed(self: *Self) void {
            if (self.depth != 1) return;
            // TURBOEXL3_GLM53F_PF_MOE_HOLD_TOP: the forward's routed calls have begun
            if (self.active) self.seen_routed = true;
            if (!self.holding) return;
            // the limit in force read by raising it (a raise never trims), then raised by the hold's bytes: from the
            // limit, or with TOP from the cache's bytes when they are over it (MLX caches a freed buffer only under it)
            const prev = self.b.setCacheLimit(std.math.maxInt(usize));
            const c_in = self.b.cacheBytes();
            const from = if (self.top) @max(prev, c_in) else prev;
            const limit = from +| self.hold;
            _ = self.b.setCacheLimit(limit);
            self.raised = prev;
            const hb = self.b.unhold();
            self.holding = false;
            self.stats.mh_used += 1;
            self.stats.mh_hb_bufs += hb.bufs;
            self.stats.mh_hb_cached += hb.cached;
            self.stats.mh_hb_released += hb.released;
            self.stats.mh_hb_deferred += hb.deferred;
            self.stats.mh_hb_cached_bytes +|= hb.cached_bytes;
            const c = self.b.cacheBytes();
            // a cache over its limit is trimmed by at least the excess at the next allocation (chunk 0's)
            if (c > limit) {
                self.stats.mh_over_limit += 1;
                self.stats.mh_excess +|= c - limit;
            }
            self.measure = .{ .sum = self.b.activeBytes() +| c };
            if (!self.hold_said) {
                self.hold_said = true;
                log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_PF_MOE_HOLD engaged: a DSA layer's hold of {d} MiB handed back to MLX's cache at its first routed call: {d} of {d} held buffers cached, {d} released, {d} still in flight; the cache {d} MiB, its limit {d} MiB (from {d} MiB, {d} MiB before it; top={d})\n", .{ self.hold >> 20, hb.cached, hb.bufs, hb.released, hb.deferred, c >> 20, limit >> 20, from >> 20, prev >> 20, @intFromBool(self.top) });
            }
        }

        /// TURBOEXL3_GLM53F_PF_MOE_HOLD's take at point `p` of a measured routed call (`Point`).
        pub fn at(self: *Self, p: Point) void {
            if (self.depth != 1) return;
            const m = if (self.measure) |*mp| mp else return;
            switch (p) {
                .first_alloc_before => m.first_c = self.b.cacheBytes(),
                .first_alloc_after => {
                    const d = m.first_c -| self.b.cacheBytes();
                    self.stats.mh_first_drop +|= d;
                    if (d > first_alloc_reuse_max) self.stats.mh_first_trims += 1;
                },
                .commit_before => {
                    m.commit_c = self.b.cacheBytes();
                    m.commit_a = self.b.activeBytes();
                    m.commit_open = true;
                },
            }
        }

        /// TURBOEXL3_GLM53F_PF_MOE_HOLD: after a routed call's chunk `k` (from 1) is committed (`out` its output's
        /// buffer address, 0: unknown), and at the call's end (`k` 0). A commit opened at `.commit_before` is measured
        /// (the take); after the second chunk, or at the end of a call with one, what the call's first chunks allocated
        /// past MLX's active and cached bytes at the hand-back (the pass line's measure), and the call is done.
        pub fn committed(self: *Self, k: usize, out: usize) void {
            if (self.depth != 1) return;
            // TURBOEXL3_GLM53F_PF_RUN_TRIM: the run's last KDA layer's routed call has its first two chunks in flight (or
            // ended with one)
            if (self.trim_armed and (k == 2 or k == 0)) self.trimRun();
            const m = if (self.measure) |*mp| mp else return;
            if (k >= 1) {
                if (m.commit_open) {
                    m.commit_open = false;
                    self.stats.mh_commits += 1;
                    self.stats.mh_commit_drop += signedDiff(m.commit_c, self.b.cacheBytes());
                    self.stats.mh_commit_alloc += signedDiff(self.b.activeBytes(), m.commit_a);
                }
                if (out != 0) {
                    self.stats.mh_y_seen += 1;
                    if (self.b.fromHold(out)) self.stats.mh_y_held += 1;
                }
                if (k == 1) return;
            }
            const sum = m.sum;
            self.measure = null;
            const fresh = (self.b.activeBytes() +| self.b.cacheBytes()) -| sum;
            self.stats.mh_measured += 1;
            self.stats.mh_fresh_max = @max(self.stats.mh_fresh_max, fresh);
            self.stats.mh_fresh_sum +|= fresh;
        }

        /// A DSA layer's hold: its arrays made (the bytes they took from the cache counted); a hold MLX refuses is
        /// counted and the layer runs without it.
        fn take(self: *Self) void {
            const c = self.b.cacheBytes();
            if (!self.b.hold()) {
                self.stats.mh_failed += 1;
                return;
            }
            self.holding = true;
            self.stats.mh_taken += 1;
            self.stats.mh_cached +|= c -| self.b.cacheBytes();
        }

        /// TURBOEXL3_GLM53F_PF_RUN_TRIM: the run's last KDA layer's routed call has its first two chunks in flight (or
        /// has ended with one): the cache trimmed to the limit before the run now (MLX releases a cache's buffers past
        /// its limit at its next allocation, oldest first: one small allocation), while the GPU runs them, not at the
        /// next DSA layer's first allocation with the GPU idle; then the run's limit back, so what the layer frees after
        /// (its later chunks' buffers, which the chunks after them take, and its last chunks', which the DSA layer's hold
        /// takes) is cached as before.
        fn trimRun(self: *Self) void {
            self.trim_armed = false;
            const base = self.base orelse return;
            const c0 = self.b.cacheBytes();
            if (c0 <= base) return;
            const prev = self.b.setCacheLimit(base);
            self.b.trimNow();
            _ = self.b.setCacheLimit(prev);
            self.stats.rt_trims += 1;
            self.stats.rt_trimmed +|= c0 -| self.b.cacheBytes();
        }

    };
}

/// MLX's backend.
pub const Mlx = struct {
    pub fn cacheBytes(_: *const Mlx) usize {
        var c: usize = 0;
        _ = mlx.mlx_get_cache_memory(&c);
        return c;
    }
    pub fn setCacheLimit(_: *const Mlx, limit: usize) usize {
        var prev: usize = 0;
        _ = mlx.mlx_set_cache_limit(&prev, limit);
        return prev;
    }
    pub fn activeBytes(_: *const Mlx) usize {
        var a: usize = 0;
        _ = mlx.mlx_get_active_memory(&a);
        return a;
    }
    pub fn unhold(_: *const Mlx) HandBack {
        return freeHeld();
    }
    /// TURBOEXL3_GLM53F_PF_RUN_TRIM: one small allocation, after which MLX trims a cache over its limit (its malloc's
    /// last step), and its free.
    pub fn trimNow(_: *const Mlx) void {
        const a = mlx.mlx_array_new_float(0);
        _ = mlx.mlx_array_free(a);
    }
    pub fn fromHold(_: *const Mlx, addr: usize) bool {
        if (addr == 0) return false;
        for (hold_addrs[0..hold_addr_n]) |h| if (h == addr) return true;
        return false;
    }
};

var schedule: Schedule(Mlx) = .{ .b = .{} };

// ── TURBOEXL3_GLM53F_PF_MOE_HOLD's arrays ──

var hold_sizes: [max_hold_arrays]usize = @splat(0);
var hold_out: [max_hold_arrays]mlx.mlx_array = undefined;
var hold_n: usize = 0;
/// The hold's buffer addresses, recorded once its launches are encoded; a buffer a hand-back released is forgotten (its
/// address may come back with a fresh buffer). A chunk output's identity (`Mlx.fromHold`).
var hold_addrs: [max_hold_arrays]usize = @splat(0);
var hold_addr_n: usize = 0;

/// The hold's arrays freed one at a time, MLX's active and cached bytes read around each free (`HandBack`); a released
/// buffer's address forgotten.
fn freeHeld() HandBack {
    var hb: HandBack = .{};
    for (hold_out[0..hold_n], 0..) |a, i| {
        var c0: usize = 0;
        var a0: usize = 0;
        _ = mlx.mlx_get_cache_memory(&c0);
        _ = mlx.mlx_get_active_memory(&a0);
        _ = mlx.mlx_array_free(a);
        var c1: usize = 0;
        var a1: usize = 0;
        _ = mlx.mlx_get_cache_memory(&c1);
        _ = mlx.mlx_get_active_memory(&a1);
        hb.bufs += 1;
        const bytes = hold_sizes[i];
        if (a0 -| a1 < bytes) {
            hb.deferred += 1;
        } else if (c1 -| c0 >= bytes) {
            hb.cached += 1;
            hb.cached_bytes += bytes;
        } else {
            hb.released += 1;
            if (i < hold_addr_n) hold_addrs[i] = 0;
        }
    }
    hold_n = 0;
    return hb;
}

/// The copy's `hidden` and `hiddenSplit`, at a forward's start (a defer'd `end` at its end).
pub fn begin(prompt: bool, rows: u64) void {
    if (!on) return;
    schedule.run_trim = run_trim;
    beginOn(Mlx, &schedule, prompt, rows);
}
/// A forward's start on `s`: the run's raise `cap_row_bytes` a row.
fn beginOn(comptime B: type, s: *Schedule(B), prompt: bool, rows: u64) void {
    s.begin(prompt, rows, cap_row_bytes);
}
/// The copy's layer loops, before each layer's attention.
pub fn layer(li: usize, kda: bool, next_kda: bool) void {
    if (!on) return;
    schedule.layer(li, kda, next_kda);
}
pub fn end() void {
    if (!on) return;
    schedule.end();
}
/// routed_t3's `prefillParts`, at a routed prompt call's start: TURBOEXL3_GLM53F_PF_MOE_HOLD's hand-back.
pub fn routed() void {
    if (!on) return;
    schedule.routed();
}
/// routed_t3's `prefillParts`, after its chunk `k` (from 1) is committed (`out` its output's buffer address, 0 when
/// unknown) and at its end (`k` 0).
pub fn committed(k: usize, out: usize) void {
    if (!on) return;
    schedule.committed(k, out);
}
/// Whether a routed call's take is being measured (routed_t3 reads the take's points and an output's address only
/// then).
pub fn measuring() bool {
    return on and schedule.measure != null;
}
/// routed_t3's `prefillParts`, at the take's point `p` of a measured call.
pub fn at(p: Point) void {
    if (!on) return;
    schedule.at(p);
}

// ── tests (MLX-free) ──

const testing = std.testing;
const mib: usize = 1 << 20;

/// A backend that records the limits set; the cache's bytes a value the test moves, trimmed to a lowered limit (MLX
/// trims at its next allocation; `lazy_trim`: not at all, the test moves it). TURBOEXL3_GLM53F_PF_MOE_HOLD: a hold of
/// `hold_bytes` (`hold_bufs` buffers of equal bytes) takes what the cache holds of them and allocates the rest; handed
/// back, each buffer is cached whole while the cache is under its limit (MLX's free), else released, but the first
/// `deferred_bufs` stay active (their command buffer holds them); `refuse` refuses every hold; `held_addr` the one
/// address `fromHold` knows. `events` the order of limit sets (L), holds (H) and hand-backs (U).
const Fake = struct {
    limit: usize = 2048 * mib,
    cache: usize = 0,
    active: usize = 0,
    sets: [128]usize = undefined,
    n: usize = 0,
    hold_bytes: usize = 0,
    hold_bufs: u32 = 1,
    deferred_bufs: u32 = 0,
    refuse: bool = false,
    lazy_trim: bool = false,
    held_addr: usize = 0,
    events: [256]u8 = undefined,
    ne: usize = 0,
};

/// GLM-5.3-Flash's attention pattern: layers 3, 7, ... 43 DSA, the rest KDA (45 layers).
fn glmKda(li: usize) bool {
    return li % 4 != 3;
}

/// A forward of `n` layers over `s` as the copy's layer loop calls it, the cache growing by `freed` over a run's first
/// layer (as far as the limit lets it) and by `more` over its later ones; `limits[li]` the limit in force through layer
/// `li`.
fn forward(s: *Schedule(Fake), n: usize, rows: u64, freed: usize, more: usize, limits: []usize) void {
    s.begin(true, rows, 1 * mib);
    for (0..n) |li| {
        const kda = glmKda(li);
        s.layer(li, kda, li + 1 < n and glmKda(li + 1));
        limits[li] = s.b.limit;
        if (s.base != null) s.b.cache = @min(s.b.cache + (if (s.first) freed else more), s.b.limit);
    }
    s.end();
}
