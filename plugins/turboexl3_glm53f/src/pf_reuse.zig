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
const log = @import("mlx_serve/sdk/util/log.zig");

/// `TURBOEXL3_GLM53F_PF_REUSE=1`, set by the plugin at load, before any forward.
pub var on: bool = false;
/// A run's raise of MLX's cache limit, a forward row (stream_room `envelopeRowBytes`), set at load.
pub var cap_row_bytes: u64 = 0;
/// The switch's load line (an A/B step reads "<switch>=1: "): the cap a forward row (KiB) and turboexl3's size class.
pub const line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_PF_REUSE=1: a prompt pass's buffers reused: a KDA run's held in MLX's cache from its first layer to its last (the limit raised by {d} KiB a forward row, the bill's layer envelope, through the run, then put back), and the routed prefill's row-slot arrays at multiples of {d} row-slots (turboexl3's size classes); the same kernels on the same rows\n";
/// A forward's rows below this hold nothing (a decode or verify window, a short prompt).
pub const min_rows: u64 = 2048;

/// `TURBOEXL3_GLM53F_PF_MOE_HOLD=1`, set by the plugin at load (it needs PF_REUSE), before any forward; the hold itself
/// is installed with the model's stream (`holdInstall`).
pub var hold_on: bool = false;
/// The arrays a routed prompt chunk allocates at its size class, on the exact chain (`chunkBytes`) and on the int8 chain
/// (`chunkBytesI8`), and the chunks the hold covers: the two that routed_t3's `prefillParts` keeps in flight.
pub const chunk_arrays = 13;
pub const i8_chunk_arrays = 16;
pub const hold_chunks = 2;
pub const hold_arrays = chunk_arrays * hold_chunks;
pub const i8_hold_arrays = i8_chunk_arrays * hold_chunks;
/// The most arrays a chunk's hold takes, and a hold's (the int8 chain's).
const max_chunk_arrays = @max(chunk_arrays, i8_chunk_arrays);
const max_hold_arrays = max_chunk_arrays * hold_chunks;
/// The rows turboexl3 cc93b93's int8 chain allocates its int8 copies and their scales past a chunk's size class.
pub const i8_pad_rows = 128;
/// The routed chain whose arrays the hold takes: turboexl3's exact chain, or its int8 chain (`Config.pf_int8` 1 or 2:
/// the same shapes either scheme).
pub const Chain = enum {
    exact,
    int8,

    /// The chain turboexl3 runs a prompt chunk on when `Config.pf_int8` is `pf_int8`. turboexl3 takes int8 beside the
    /// 128-row tiles without `pf_t128_zfit` on the deferred MUL1 map: the hold's size classes need the first two, and
    /// GLM's map is MUL1.
    pub fn of(pf_int8: u8) Chain {
        return if (pf_int8 != 0) .int8 else .exact;
    }

    /// A chunk's held arrays on this chain.
    pub fn arrays(c: Chain) usize {
        return switch (c) {
            .exact => chunk_arrays,
            .int8 => i8_chunk_arrays,
        };
    }
};
/// `TURBOEXL3_GLM53F_PF_MOE_HOLD_TOP=1`, set by the plugin at load (it needs PF_MOE_HOLD); the schedule reads it at
/// `holdInstall`.
pub var hold_top: bool = false;
/// `TURBOEXL3_GLM53F_PF_RUN_TRIM=1`, set by the plugin at load (it needs PF_REUSE); the schedule reads it at each
/// forward's start.
pub var run_trim: bool = false;
/// RUN_TRIM's load line (an A/B step reads "<switch>=1: ") and its pass line at a forward's end.
pub const run_trim_line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_PF_RUN_TRIM=1: a KDA run's last layer trims MLX's cache to the limit before the run once its routed call has two chunks in flight, not at the next DSA layer's first allocation with the GPU idle; the run's limit back after, so the layer's later frees are cached; the same kernels on the same rows\n";
pub const run_trim_pass_line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_PF_RUN_TRIM pass rows={d}: trims={d} trimmed_mb={d} drop_excess_mb_max={d} drop_excess_mb_sum={d}\n";
/// TOP's load line (an A/B step reads "<switch>=1: ").
pub const hold_top_line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_PF_MOE_HOLD_TOP=1: the hold's hand-back raises MLX's cache limit from the cache's bytes when they are over the limit, so every held buffer is cached and the call's first two chunks take them, and a forward's DSA layers before its first routed call hold nothing; the same kernels on the same rows\n";
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
/// The hold's load line (an A/B step reads "<switch>=1: "): its arrays, its MiB, the chunk rows and the chain.
pub const hold_line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_PF_MOE_HOLD=1: a DSA layer's routed prefill finds its first two chunks' arrays in MLX's cache: {d} arrays ({d} MiB, two {d}-row chunks of the {s} chain at their size class) taken at the layer's start before the run's limit goes back, kept through its attention, and handed back to the cache at its first routed call under a limit raised by as much until the next layer; nothing reads them, the same kernels on the same rows\n";

/// A routed prompt chunk's arrays at its size class `cap` (rows), as turboexl3 3439b9d's `moeT128With` allocates them
/// with `Config.pf_size_class` and without `pf_t128_zfit`, in its order: prep's two x copies (f16 [cap, hidden]) and
/// their block sums (f32 [cap, hidden / 128]); gate|up's mixed and 128-row outputs (f16 [cap, inter], two each); mid's
/// output and its sums (f16 [cap, inter], f32 [cap, inter / 128]); down's mixed and 128-row outputs (f16 [cap, hidden]);
/// y (bf16 [cap, hidden]). The arrays sized by the chunk's own rows (prep's inverse order, the route's tables, the
/// weights) are not held. src/pf_hold_gpu_test.zig holds the sizes to the chain's own allocations.
pub fn chunkBytes(cap: usize, hidden: usize, inter: usize) [chunk_arrays]usize {
    const xh = cap * hidden * 2;
    const xi = cap * inter * 2;
    const sh = cap * (hidden / 128) * 4;
    const si = cap * (inter / 128) * 4;
    return .{ xh, xh, sh, sh, xi, xi, xi, xi, xi, si, xh, xh, xh };
}

/// A routed prompt chunk's arrays at its size class `cap` (rows) on turboexl3 cc93b93's int8 chain (`moeI8Cap`, under
/// `Config.pf_int8` 1 or 2: the same shapes), in its order: prep's two int8 x copies (int8 [cap + 128, hidden]), their
/// block sums (f32 [cap, hidden / 128]) and their block scales (f32 [cap + 128, hidden / 128]); gate|up's mixed and
/// 128-row outputs (f16 [cap, inter], two each); mid's int8 output, its sums and its scales (int8 [cap + 128, inter], f32
/// [cap, inter / 128], f32 [cap + 128, inter / 128]); down's mixed and 128-row outputs (f16 [cap, hidden]); y (bf16 [cap,
/// hidden], the exact chain's finish). The 128 rows are `i8_pad_rows`. As `chunkBytes`, the arrays sized by the chunk's
/// own rows are not held; src/pf_hold_gpu_test.zig holds the sizes to the chain's own allocations under both schemes.
pub fn chunkBytesI8(cap: usize, hidden: usize, inter: usize) [i8_chunk_arrays]usize {
    const qh = (cap + i8_pad_rows) * hidden;
    const sh = cap * (hidden / 128) * 4;
    const ch = (cap + i8_pad_rows) * (hidden / 128) * 4;
    const xi = cap * inter * 2;
    const qi = (cap + i8_pad_rows) * inter;
    const si = cap * (inter / 128) * 4;
    const ci = (cap + i8_pad_rows) * (inter / 128) * 4;
    const xh = cap * hidden * 2;
    return .{ qh, qh, sh, sh, ch, ch, xi, xi, xi, xi, qi, si, ci, xh, xh, xh };
}

/// An array's MLX buffer bytes: MLX's Metal allocator rounds a request past one page (`vm_page_size`) up to whole pages,
/// and its cache keeps and matches buffers at those bytes. Every exact-chain array at a size class is whole pages; the
/// int8 chain's mid scales (16.5 pages at the 4096-row class) are not.
pub fn bufferBytes(bytes: usize) usize {
    const page = std.heap.pageSize();
    return if (bytes > page) std.mem.alignForward(usize, bytes, page) else bytes;
}

/// A routed prompt chunk's held arrays on `chain` at its size class `cap` (rows), each at its buffer's bytes
/// (`bufferBytes`): `chunkBytes` or `chunkBytesI8`, the first `chain.arrays()` entries, the rest 0.
pub fn chainBytes(chain: Chain, cap: usize, hidden: usize, inter: usize) [max_chunk_arrays]usize {
    var out: [max_chunk_arrays]usize = @splat(0);
    switch (chain) {
        .exact => for (chunkBytes(cap, hidden, inter), 0..) |b, i| {
            out[i] = bufferBytes(b);
        },
        .int8 => for (chunkBytesI8(cap, hidden, inter), 0..) |b, i| {
            out[i] = bufferBytes(b);
        },
    }
    return out;
}

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

        /// A layer's start and the forward's end: a routed call's raise put back, and a hold no routed call took let go.
        fn settle(self: *Self) void {
            if (self.holding) {
                _ = self.b.unhold();
                self.holding = false;
                self.stats.mh_unused += 1;
            }
            if (self.raised) |r| {
                _ = self.b.setCacheLimit(r);
                self.raised = null;
            }
            self.measure = null;
            self.trim_armed = false;
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

        /// The run's end: the limit before it put back (MLX trims the cache at its next allocation).
        fn drop(self: *Self) void {
            const base = self.base orelse return;
            _ = self.b.setCacheLimit(base);
            self.base = null;
            self.first = false;
            self.stats.drops += 1;
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
    pub fn peak(_: *const Mlx) usize {
        var p: usize = 0;
        _ = mlx.mlx_get_peak_memory(&p);
        return p;
    }
    pub fn activeBytes(_: *const Mlx) usize {
        var a: usize = 0;
        _ = mlx.mlx_get_active_memory(&a);
        return a;
    }
    pub fn hold(_: *const Mlx) bool {
        holdArrays(false) catch return false;
        return true;
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

/// The hold's chain and its arrays a chunk (`holdInstall`'s), its sizes (`chainBytes`, `hold_chunks` times), its arrays
/// while they are out, the model's stream (borrowed: `holdInstall` to `holdReset`) and the launch that makes them (with
/// its outputs a launch).
var hold_chain: Chain = .exact;
var hold_per: usize = chunk_arrays;
var hold_sizes: [max_hold_arrays]usize = @splat(0);
var hold_out: [max_hold_arrays]mlx.mlx_array = undefined;
var hold_n: usize = 0;
/// The hold's buffer addresses, recorded once its launches are encoded; a buffer a hand-back released is forgotten (its
/// address may come back with a fresh buffer). A chunk output's identity (`Mlx.fromHold`).
var hold_addrs: [max_hold_arrays]usize = @splat(0);
var hold_addr_n: usize = 0;
var hold_stream: ?mlx.mlx_stream = null;
var hold_kernel: ?mlx.mlx_fast_metal_kernel = null;
var hold_kernel_outs: usize = 0;
/// The launch's body: nothing (one thread, no reads, no writes). MLX allocates a launch's outputs when it encodes it,
/// from its cache where a buffer of the size is there, and never fills them (no init value).
const hold_source = "";

/// The launch with `hold_per` outputs (a kernel made for another count freed first).
fn holdKernel() !mlx.mlx_fast_metal_kernel {
    if (hold_kernel) |k| {
        if (hold_kernel_outs == hold_per) return k;
        _ = mlx.mlx_fast_metal_kernel_free(k);
        hold_kernel = null;
    }
    const names = [max_chunk_arrays][*:0]const u8{ "h0", "h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9", "h10", "h11", "h12", "h13", "h14", "h15" };
    const iv = mlx.mlx_vector_string_new();
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&names, hold_per);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new("turboexl3_glm53f_pf_moe_hold", iv, ov, hold_source, "", true, false);
    if (k.ctx == null) return error.Glm53fPfMoeHoldKernel;
    hold_kernel = k;
    hold_kernel_outs = hold_per;
    return k;
}

/// The hold's arrays made on the model's stream, one launch a chunk, and evaluated (`sync`: waited for; else
/// committed, the buffers taken as MLX encodes the launches).
fn holdArrays(sync: bool) !void {
    const s = hold_stream orelse return error.Glm53fPfMoeHoldStream;
    if (hold_n != 0) _ = freeHeld();
    hold_addr_n = 0;
    const k = try holdKernel();
    errdefer _ = freeHeld();
    for (0..hold_chunks) |c| {
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        for (hold_sizes[c * hold_per ..][0..hold_per]) |bytes| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{@intCast(bytes)}, 1, .uint8));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 1, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 1, 1, 1));
        const iv = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(iv);
        var ov = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(ov);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, k, iv, cfg, s));
        if (mlx.mlx_vector_array_size(ov) != hold_per) return error.Glm53fPfMoeHoldOutputs;
        for (0..hold_per) |i| {
            var x = mlx.mlx_array_new();
            mlx.check(mlx.mlx_vector_array_get(&x, ov, i)) catch |e| {
                _ = mlx.mlx_array_free(x);
                return e;
            };
            hold_out[hold_n] = x;
            hold_n += 1;
        }
    }
    const vec = mlx.mlx_vector_array_new_data(&hold_out, hold_n);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(if (sync) mlx.mlx_eval(vec) else mlx.mlx_async_eval(vec));
    // encoded: every array has its buffer
    for (hold_out[0..hold_n], 0..) |a, i| hold_addrs[i] = if (mlx.mlx_array_data_uint8(a)) |p| @intFromPtr(p) else 0;
    hold_addr_n = hold_n;
}

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

/// TURBOEXL3_GLM53F_PF_MOE_HOLD for this model: the hold's sizes from its routed geometry (`chainBytes` at the size
/// class of `chunk_rows`, routed_t3's chunk, `hold_chunks` times) on the chain turboexl3's configuration names now
/// (`Chain.of`: the plugin applies TURBOEXL3_GLM53F_PF_INT8 before), its launch built and run once on `s` (the library
/// made here, before any request) and its arrays let go. Returns the hold's bytes (`heldArrays` and `heldChain` its
/// arrays and chain). Refused by name: a stream that is not a GPU's, a geometry not in 128-wide blocks, a launch MLX
/// refuses.
pub fn holdInstall(s: mlx.mlx_stream, hidden: usize, inter: usize, chunk_rows: usize) !usize {
    if (!mlx.streamIsGpu(s)) return error.Glm53fPfMoeHoldStream;
    if (hidden == 0 or inter == 0 or hidden % 128 != 0 or inter % 128 != 0 or chunk_rows == 0) return error.Glm53fPfMoeHoldGeometry;
    if (hold_n != 0) _ = freeHeld();
    const chain = Chain.of(t3.config.get().pf_int8);
    const per = chain.arrays();
    const one = chainBytes(chain, t3.prefill.classRows(@intCast(chunk_rows)), hidden, inter);
    var total: usize = 0;
    for (0..hold_chunks) |c| for (one[0..per], 0..) |bytes, i| {
        hold_sizes[c * per + i] = bytes;
        total += bytes;
    };
    hold_chain = chain;
    hold_per = per;
    hold_stream = s;
    errdefer holdReset();
    try holdArrays(true);
    _ = freeHeld();
    schedule.hold = total;
    schedule.top = hold_top;
    return total;
}

/// The hold goes with the model: its arrays freed, the stream forgotten, the launch freed.
pub fn holdReset() void {
    _ = freeHeld();
    schedule.hold = 0;
    schedule.holding = false;
    schedule.top = false;
    hold_stream = null;
    if (hold_kernel) |k| _ = mlx.mlx_fast_metal_kernel_free(k);
    hold_kernel = null;
    hold_kernel_outs = 0;
}

/// The installed hold's arrays (`hold_chunks` chunks of its chain's) and its chain (`holdInstall`'s).
pub fn heldArrays() usize {
    return hold_per * hold_chunks;
}
pub fn heldChain() Chain {
    return hold_chain;
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

/// turboexl3's configuration with the routed prefill's size classes (`Config.pf_size_class`), before any prefill
/// engine is made.
pub fn applySizeClasses() !void {
    var c = t3.config.get().*;
    c.pf_size_class = true;
    try t3.configure(c);
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
    pub fn cacheBytes(self: *const Fake) usize {
        return self.cache;
    }
    pub fn setCacheLimit(self: *Fake, limit: usize) usize {
        const prev = self.limit;
        self.limit = limit;
        if (!self.lazy_trim) self.cache = @min(self.cache, limit);
        self.sets[self.n] = limit;
        self.n += 1;
        self.ev('L');
        return prev;
    }
    pub fn peak(_: *const Fake) usize {
        return 0;
    }
    pub fn activeBytes(self: *const Fake) usize {
        return self.active;
    }
    pub fn hold(self: *Fake) bool {
        if (self.refuse) return false;
        self.cache -= @min(self.cache, self.hold_bytes);
        self.active += self.hold_bytes;
        self.ev('H');
        return true;
    }
    pub fn unhold(self: *Fake) HandBack {
        var hb: HandBack = .{};
        const each = self.hold_bytes / self.hold_bufs;
        for (0..self.hold_bufs) |i| {
            hb.bufs += 1;
            if (i < self.deferred_bufs) {
                hb.deferred += 1;
                continue;
            }
            self.active -= each;
            if (self.cache < self.limit) {
                self.cache += each;
                hb.cached += 1;
                hb.cached_bytes += each;
            } else hb.released += 1;
        }
        self.ev('U');
        return hb;
    }
    pub fn trimNow(self: *Fake) void {
        self.cache = @min(self.cache, self.limit);
        self.ev('T');
    }
    pub fn fromHold(self: *const Fake, addr: usize) bool {
        return addr != 0 and addr == self.held_addr;
    }
    fn ev(self: *Fake, c: u8) void {
        self.events[self.ne] = c;
        self.ne += 1;
    }
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

test "turboexl3_glm53f stream pf reuse: a KDA run's first layer raises the limit by the cap for the whole run, the next DSA layer and the forward's end put it back; GLM's 45 layers make 11 runs, 22 held layers and 11 drops" {
    var s: Schedule(Fake) = .{ .b = .{ .cache = 1500 * mib } };
    var limits: [45]usize = undefined;
    // a 16K forward: the cap 16 GiB (1 MiB a row); a run's first layer leaves 7 GiB, each later one 64 MiB more
    forward(&s, 45, 16384, 7168 * mib, 64 * mib, &limits);
    try testing.expectEqual(@as(u32, 11), s.stats.runs);
    try testing.expectEqual(@as(u32, 22), s.stats.holds);
    try testing.expectEqual(@as(u32, 11), s.stats.drops);
    try testing.expectEqual(@as(u64, 16384 * mib), s.stats.cap);
    // the first run starts over 1500 MiB, the others over the 2048 MiB the drop trimmed to
    try testing.expectEqual(@as(u64, 7168 * mib), s.stats.held_max);
    try testing.expectEqual(@as(u64, 7168 * mib), s.stats.held_min);
    try testing.expectEqual(@as(u64, (2048 + 7168 + 64) * mib), s.stats.cache_max);
    // layers 0 to 2 at the raised limit, 3 (DSA) put back, 4 raised again; 43 put back, 44 (KDA, last, after a DSA
    // layer) no run
    for ([_]usize{ 0, 1, 2, 4, 42 }) |li| try testing.expectEqual((2048 + 16384) * mib, limits[li]);
    for ([_]usize{ 3, 43, 44 }) |li| try testing.expectEqual(2048 * mib, limits[li]);
    try testing.expectEqual(2048 * mib, s.b.limit);
    try testing.expectEqual(@as(u64, (2048 + 16384) * mib), s.stats.limit_max);
    try testing.expect(!s.active and s.base == null and s.depth == 0);
    // every set in order: per run the read (a raise to the most), the raise, the drop
    try testing.expectEqual(@as(usize, 11 * 3), s.b.n);
    try testing.expectEqualSlices(usize, &.{ std.math.maxInt(usize), (2048 + 16384) * mib, 2048 * mib, std.math.maxInt(usize) }, s.b.sets[0..4]);
}

test "turboexl3_glm53f stream pf reuse: no limit set outside a prompt forward of 2048 rows or more with a cap, inside a nested forward, or over layers that make no run; a forward that ends inside a run puts the limit back; the cap bounds what is held" {
    var limits: [45]usize = undefined;
    // a decode, a short prompt, no cap: nothing set
    var s: Schedule(Fake) = .{ .b = .{} };
    s.begin(false, 16384, 1 * mib);
    s.layer(0, true, true);
    s.end();
    s.begin(true, 2047, 1 * mib);
    s.layer(0, true, true);
    s.end();
    s.begin(true, 16384, 0);
    s.layer(0, true, true);
    s.end();
    try testing.expectEqual(@as(usize, 0), s.b.n);
    // a KDA layer before a DSA layer, a DSA layer, the last layer: no run
    s.begin(true, 4096, 1 * mib);
    s.layer(2, true, false);
    s.layer(3, false, true);
    s.layer(44, true, false);
    s.end();
    try testing.expectEqual(@as(usize, 0), s.b.n);
    // a nested forward changes nothing; the outer one's end inside a run puts the limit back
    s.begin(true, 4096, 1 * mib);
    s.layer(0, true, true);
    s.begin(true, 4096, 1 * mib);
    s.layer(0, true, true);
    s.end();
    try testing.expectEqual(@as(usize, 2), s.b.n);
    s.end();
    try testing.expectEqual(@as(usize, 3), s.b.n);
    try testing.expectEqual(2048 * mib, s.b.limit);
    try testing.expectEqual(@as(u32, 1), s.stats.drops);
    // the cap bounds what the cache holds (4096 rows: 4 GiB over the limit before the run)
    var c: Schedule(Fake) = .{ .b = .{ .cache = 2048 * mib } };
    forward(&c, 45, 4096, 9000 * mib, 512 * mib, &limits);
    try testing.expectEqual(@as(u64, 4096 * mib), c.stats.held_max);
    try testing.expectEqual(@as(u64, (2048 + 4096) * mib), c.stats.cache_max);
    try testing.expectEqual(2048 * mib, c.b.limit);
}

test "turboexl3_glm53f stream pf reuse: a forward's hold over GLM's 45 layers makes 11 runs from the load's cap" {
    const cap0 = cap_row_bytes;
    defer cap_row_bytes = cap0;
    cap_row_bytes = 1 * mib;
    var s: Schedule(Fake) = .{ .b = .{} };
    beginOn(Fake, &s, true, 16384);
    for (0..45) |li| s.layer(li, glmKda(li), li + 1 < 45 and glmKda(li + 1));
    s.end();
    try testing.expectEqual(@as(u32, 11), s.stats.runs);
    try testing.expectEqual(@as(usize, 11 * 3), s.b.n);
    try testing.expectEqual(2048 * mib, s.b.limit);
}

test "turboexl3_glm53f stream pf reuse: TURBOEXL3_GLM53F_PF_RUN_TRIM trims a run's cache to the limit before the run once its last KDA layer's routed call has two chunks in flight (at its end with one) and puts the run's limit back, so the DSA layer's drop finds only what the layer freed after; off, nothing moves" {
    // MLX trims at its next allocation (lazy_trim): without the switch the DSA layer's first allocation meets the run's
    // leftovers
    var s: Schedule(Fake) = .{ .b = .{ .cache = 2048 * mib, .lazy_trim = true }, .run_trim = true };
    s.begin(true, 16384, 1 * mib);
    s.layer(4, true, true);
    s.b.cache = 7168 * mib;
    s.layer(5, true, true);
    s.routed();
    s.committed(1, 0);
    s.committed(0, 0);
    // not the run's last layer: no trim
    try testing.expectEqual(@as(u32, 0), s.stats.rt_trims);
    s.layer(6, true, false);
    try testing.expect(s.trim_armed);
    s.b.cache = 8192 * mib;
    s.routed();
    s.committed(1, 0);
    try testing.expectEqual(@as(u32, 0), s.stats.rt_trims);
    // two chunks in flight: the cache trimmed to the 2048 MiB before the run, the run's limit back
    const n0 = s.b.n;
    s.committed(2, 0);
    try testing.expectEqual(@as(u32, 1), s.stats.rt_trims);
    try testing.expectEqual(@as(u64, 6144 * mib), s.stats.rt_trimmed);
    try testing.expectEqual(2048 * mib, s.b.cache);
    try testing.expectEqual((2048 + 16384) * mib, s.b.limit);
    try testing.expectEqualSlices(usize, &.{ 2048 * mib, (2048 + 16384) * mib }, s.b.sets[n0..s.b.n]);
    try testing.expectEqualSlices(u8, "LTL", s.b.events[s.b.ne - 3 .. s.b.ne]);
    try testing.expect(!s.trim_armed);
    // later chunks and the call's end trim nothing more
    s.committed(3, 0);
    s.committed(0, 0);
    try testing.expectEqual(@as(u32, 1), s.stats.rt_trims);
    // the layer's later frees (its last chunks) cached under the run's limit; the DSA layer's drop finds them only
    s.b.cache += 512 * mib;
    s.layer(7, false, true);
    try testing.expectEqual(2048 * mib, s.b.limit);
    try testing.expectEqual(@as(u64, 512 * mib), s.stats.rt_excess_max);
    // a later call's end trims nothing (disarmed at the DSA layer's start)
    s.committed(0, 0);
    try testing.expectEqual(@as(u32, 1), s.stats.rt_trims);
    s.end();
    // off: no trim at the call's end, the DSA layer's drop leaves the run's 6 GiB to the next allocation
    var o: Schedule(Fake) = .{ .b = .{ .cache = 2048 * mib, .lazy_trim = true } };
    o.begin(true, 16384, 1 * mib);
    o.layer(4, true, true);
    o.layer(5, true, true);
    o.layer(6, true, false);
    try testing.expect(!o.trim_armed);
    o.b.cache = 8192 * mib;
    o.routed();
    o.committed(0, 0);
    try testing.expectEqual(8192 * mib, o.b.cache);
    o.layer(7, false, true);
    o.end();
    try testing.expectEqual(@as(u32, 0), o.stats.rt_trims);
    try testing.expectEqual(8192 * mib, o.b.cache);
    // on, a run's last layer whose call has one chunk trims at the call's end
    var c: Schedule(Fake) = .{ .b = .{ .cache = 2048 * mib, .lazy_trim = true }, .run_trim = true };
    c.begin(true, 16384, 1 * mib);
    c.layer(4, true, true);
    c.layer(5, true, false);
    c.b.cache = 4096 * mib;
    c.routed();
    c.committed(1, 0);
    try testing.expectEqual(@as(u32, 0), c.stats.rt_trims);
    c.committed(0, 0);
    try testing.expectEqual(@as(u32, 1), c.stats.rt_trims);
    try testing.expectEqual(2048 * mib, c.b.cache);
    c.layer(6, false, true);
    c.end();
}

/// TURBOEXL3_GLM53F_PF_MOE_HOLD's bytes at GLM-5.3-Flash's routed geometry: two 4096-row chunks at their size class.
fn glmHold() usize {
    var sum: usize = 0;
    for (chunkBytes(4096, 4096, 2048)) |b| sum += b;
    return hold_chunks * sum;
}

test "turboexl3_glm53f stream pf reuse: TURBOEXL3_GLM53F_PF_MOE_HOLD's chunk at GLM-5.3-Flash's routed geometry is 13 arrays, 241.25 MiB at the 4096-row size class (five [cap, hidden] and five [cap, inter] 2-byte arrays, three f32 block sums), two chunks 482.5 MiB" {
    try testing.expectEqual(@as(u32, 4096), t3.prefill.classRows(4096));
    try testing.expectEqual(@as(u32, 4096), t3.prefill.classRows(3969));
    const one = chunkBytes(4096, 4096, 2048);
    var n: [4]usize = @splat(0);
    for (one) |b| switch (b) {
        32 * mib => n[0] += 1,
        16 * mib => n[1] += 1,
        512 * 1024 => n[2] += 1,
        256 * 1024 => n[3] += 1,
        else => return error.TestUnexpectedResult,
    };
    try testing.expectEqualSlices(usize, &.{ 5, 5, 2, 1 }, &n);
    try testing.expectEqual(@as(usize, 26), hold_arrays);
    try testing.expectEqual(@as(usize, 482 * mib + 512 * 1024), glmHold());
}

test "turboexl3_glm53f stream pf reuse: TURBOEXL3_GLM53F_PF_MOE_HOLD's chunk on PF_INT8's int8 chain at GLM-5.3-Flash's routed geometry is 16 arrays at the 4096-row size class, 213,696,512 bytes at their buffers (the mid's scales 16.5 pages, a 17-page buffer), two chunks 407.6 MiB; the exact chain's arrays are whole pages; PF_INT8 1 and 2 name the int8 chain, 0 the exact one" {
    const page = std.heap.pageSize();
    // in the int8 chain's order: the two int8 x copies, their sums and scales, gate|up's four outputs, mid's int8 output,
    // its sums and scales, down's two outputs and y
    const qh: usize = 4224 * 4096;
    const ch: usize = 4224 * (4096 / 128) * 4;
    const qi: usize = 4224 * 2048;
    const ci: usize = 4224 * (2048 / 128) * 4;
    try testing.expectEqualSlices(usize, &.{ qh, qh, 512 * 1024, 512 * 1024, ch, ch, 16 * mib, 16 * mib, 16 * mib, 16 * mib, qi, 256 * 1024, ci, 32 * mib, 32 * mib, 32 * mib }, &chunkBytesI8(4096, 4096, 2048));
    try testing.expectEqual(@as(usize, 32), i8_hold_arrays);
    // a buffer's bytes: the request rounded up to whole pages past one page, a request within a page as it is
    for (chunkBytes(4096, 4096, 2048)) |b| try testing.expectEqual(b, bufferBytes(b));
    try testing.expectEqual(std.mem.alignForward(usize, ci, page), bufferBytes(ci));
    try testing.expectEqual(page, bufferBytes(page));
    try testing.expectEqual(2 * page, bufferBytes(page + 1));
    try testing.expectEqual(@as(usize, 100), bufferBytes(100));
    if (page == 16 << 10) {
        try testing.expectEqual(@as(usize, 17 * page), bufferBytes(ci));
        var sum: usize = 0;
        for (chainBytes(.int8, 4096, 4096, 2048)) |b| sum += b;
        try testing.expectEqual(@as(usize, 213_696_512), sum);
        try testing.expectEqual(@as(usize, 407 * mib + 608 * 1024), hold_chunks * sum);
    }
    // the chain: PF_INT8 1 and 2 the int8 chain, 0 the exact one, whose sizes are chunkBytes' (the rest 0)
    try testing.expectEqual(Chain.exact, Chain.of(0));
    try testing.expectEqual(Chain.int8, Chain.of(1));
    try testing.expectEqual(Chain.int8, Chain.of(2));
    try testing.expectEqual(@as(usize, chunk_arrays), Chain.exact.arrays());
    try testing.expectEqual(@as(usize, i8_chunk_arrays), Chain.int8.arrays());
    const ex = chainBytes(.exact, 4096, 4096, 2048);
    try testing.expectEqualSlices(usize, &chunkBytes(4096, 4096, 2048), ex[0..chunk_arrays]);
    for (ex[chunk_arrays..]) |b| try testing.expectEqual(@as(usize, 0), b);
}

test "turboexl3_glm53f stream pf reuse: TURBOEXL3_GLM53F_PF_MOE_HOLD takes a DSA layer's hold at its start before the run's limit goes back, hands it back at the layer's first routed call under the limit raised by its bytes, and puts the limit back at the next layer's start; GLM's 45 layers take and use 11" {
    const hold = glmHold();
    var s: Schedule(Fake) = .{ .b = .{ .cache = 1500 * mib, .hold_bytes = hold }, .hold = hold };
    var moe_limits: [45]usize = undefined;
    s.begin(true, 16384, 1 * mib);
    for (0..45) |li| {
        const kda = glmKda(li);
        s.layer(li, kda, li + 1 < 45 and glmKda(li + 1));
        // the attention: a run's first layer leaves 7 GiB in the cache, a later one 64 MiB (as `forward`)
        if (s.base != null) s.b.cache = @min(s.b.cache + (if (s.first) 7168 * mib else 64 * mib), s.b.limit);
        // the MoE stage: its first routed call, two chunks taking the hold's sizes (from the cache as far as it holds
        // them), done and cached again; a later call of the layer takes nothing
        s.routed();
        moe_limits[li] = s.b.limit;
        s.b.cache -= @min(s.b.cache, hold);
        s.b.active += hold;
        s.committed(1, 0);
        s.committed(2, 0);
        s.b.active -= hold;
        s.b.cache = @min(s.b.cache + hold, s.b.limit);
        s.routed();
        s.committed(0, 0);
    }
    s.end();
    const st = s.stats;
    // PF_REUSE's own counts as without the hold
    try testing.expectEqual(@as(u32, 11), st.runs);
    try testing.expectEqual(@as(u32, 22), st.holds);
    try testing.expectEqual(@as(u32, 11), st.drops);
    // every DSA layer's hold taken from the cache, used at its routed call, its two chunks allocating nothing more
    try testing.expectEqual(@as(u32, 11), st.mh_taken);
    try testing.expectEqual(@as(u32, 11), st.mh_used);
    try testing.expectEqual(@as(u32, 0), st.mh_unused);
    try testing.expectEqual(@as(u32, 0), st.mh_failed);
    try testing.expectEqual(@as(u64, 11 * hold), st.mh_cached);
    try testing.expectEqual(@as(u32, 11), st.mh_measured);
    try testing.expectEqual(@as(u64, 0), st.mh_fresh_max);
    // the MoE stages' limits: a run's, a DSA layer's base and the hold, the last layer's (after a DSA layer, no run) base
    for (0..45) |li| {
        const want: usize = if (li == 44) 2048 * mib else if (glmKda(li)) (2048 + 16384) * mib else 2048 * mib + hold;
        try testing.expectEqual(want, moe_limits[li]);
    }
    try testing.expectEqual(2048 * mib, s.b.limit);
    try testing.expect(!s.holding and s.raised == null and s.base == null and s.depth == 0);
    // in order: the run's read and raise; at layer 3 the hold, then the run's drop, the routed call's read and raise and
    // the hand-back; at layer 4 the raise put back, then the next run's read and raise
    try testing.expectEqualStrings("LLHLLLULLL", s.b.events[0..10]);
    try testing.expectEqualSlices(usize, &.{ std.math.maxInt(usize), (2048 + 16384) * mib, 2048 * mib, std.math.maxInt(usize), 2048 * mib + hold, 2048 * mib, std.math.maxInt(usize), (2048 + 16384) * mib }, s.b.sets[0..8]);
    try testing.expectEqual(@as(usize, 11 * 3 + 11 * 3), s.b.n);
}

test "turboexl3_glm53f stream pf reuse: TURBOEXL3_GLM53F_PF_MOE_HOLD holds nothing without its bytes, at a KDA layer, outside a prompt forward or for a nested one; a hold no routed call takes is let go at the next layer's start or the forward's end; what a call's first chunks allocate past the hand-back is counted; a hold MLX refuses is counted and the layer runs without it" {
    const hold = 64 * mib;
    // no hold's bytes: a DSA layer and a routed call set nothing
    var s0: Schedule(Fake) = .{ .b = .{ .hold_bytes = hold } };
    s0.begin(true, 4096, 1 * mib);
    s0.layer(3, false, true);
    s0.routed();
    s0.committed(2, 0);
    s0.end();
    try testing.expectEqual(@as(usize, 0), s0.b.ne);
    // a decode, a short prompt, a KDA layer: nothing
    var s: Schedule(Fake) = .{ .b = .{ .hold_bytes = hold }, .hold = hold };
    s.begin(false, 16384, 1 * mib);
    s.layer(3, false, true);
    s.routed();
    s.end();
    s.begin(true, 2047, 1 * mib);
    s.layer(3, false, true);
    s.routed();
    s.end();
    s.begin(true, 4096, 1 * mib);
    s.layer(2, true, false);
    s.routed();
    s.end();
    try testing.expectEqual(@as(usize, 0), s.b.ne);
    // holds no routed call takes: let go at the next layer's start and at the forward's end, no limit set
    s.begin(true, 4096, 1 * mib);
    s.layer(3, false, false);
    s.layer(4, true, false);
    s.layer(7, false, false);
    s.end();
    try testing.expectEqualStrings("HUHU", s.b.events[0..s.b.ne]);
    try testing.expectEqual(@as(u32, 2), s.stats.mh_taken);
    try testing.expectEqual(@as(u32, 2), s.stats.mh_unused);
    try testing.expectEqual(@as(u32, 0), s.stats.mh_used);
    try testing.expectEqual(2048 * mib, s.b.limit);
    // a nested forward's routed call takes nothing; the outer one's next call takes the hold, its one chunk allocating
    // 3 MiB past the hand-back (measured at the call's end), and the forward's end puts the limit back
    s.b.ne = 0;
    s.begin(true, 4096, 1 * mib);
    s.layer(3, false, false);
    s.begin(true, 4096, 1 * mib);
    s.routed();
    s.end();
    try testing.expect(s.holding);
    s.routed();
    try testing.expectEqual(2048 * mib + hold, s.b.limit);
    s.b.active += 3 * mib;
    s.committed(1, 0);
    s.committed(0, 0);
    s.end();
    try testing.expectEqualStrings("HLLUL", s.b.events[0..s.b.ne]);
    try testing.expectEqual(@as(u32, 1), s.stats.mh_used);
    try testing.expectEqual(@as(u32, 1), s.stats.mh_measured);
    try testing.expectEqual(@as(u64, 3 * mib), s.stats.mh_fresh_max);
    try testing.expectEqual(2048 * mib, s.b.limit);
    // MLX refuses the hold: counted, and the layer's routed call sets nothing
    s.b.refuse = true;
    s.b.ne = 0;
    s.begin(true, 4096, 1 * mib);
    s.layer(3, false, false);
    s.routed();
    s.end();
    try testing.expectEqual(@as(usize, 0), s.b.ne);
    try testing.expectEqual(@as(u32, 1), s.stats.mh_failed);
    try testing.expectEqual(@as(u32, 0), s.stats.mh_taken);
}

test "turboexl3_glm53f stream pf reuse: TURBOEXL3_GLM53F_PF_MOE_HOLD_TOP raises the hand-back's limit from the cache's bytes when they are over it, so the cache keeps every held buffer and is within its limit, where without it the hand-back releases them all and leaves the cache over its limit; with it the DSA layers before the forward's first routed call hold nothing" {
    const each = 16 * mib;
    const hold = 26 * each;
    inline for (.{ false, true }) |top| {
        var s: Schedule(Fake) = .{ .b = .{ .hold_bytes = hold, .hold_bufs = 26, .lazy_trim = true }, .hold = hold, .top = top };
        s.begin(true, 16384, 1 * mib);
        // a KDA layer's routed call (the forward's first), then a DSA layer's hold
        s.layer(2, true, false);
        s.routed();
        s.layer(3, false, false);
        try testing.expect(s.holding);
        // the attention leaves the cache at 2900 MiB, over the 2048 MiB limit (MLX trims only at its next allocation)
        s.b.cache = 2900 * mib;
        s.routed();
        const st = s.stats;
        try testing.expectEqual(@as(u32, 26), st.mh_hb_bufs);
        if (top) {
            try testing.expectEqual(@as(u32, 26), st.mh_hb_cached);
            try testing.expectEqual(@as(u64, hold), st.mh_hb_cached_bytes);
            try testing.expectEqual(@as(u32, 0), st.mh_over_limit);
            try testing.expectEqual(2900 * mib + hold, s.b.limit);
            try testing.expectEqual(2900 * mib + hold, s.b.cache);
        } else {
            try testing.expectEqual(@as(u32, 0), st.mh_hb_cached);
            try testing.expectEqual(@as(u32, 1), st.mh_over_limit);
            try testing.expectEqual(@as(u64, 2900 * mib - (2048 * mib + hold)), st.mh_excess);
            try testing.expectEqual(2048 * mib + hold, s.b.limit);
            try testing.expectEqual(2900 * mib, s.b.cache);
        }
        // the next layer's start, or the forward's end, puts the limit back
        s.end();
        try testing.expectEqual(2048 * mib, s.b.limit);
    }
    // with TOP a DSA layer before the forward's first routed call holds nothing, the next one holds
    var s: Schedule(Fake) = .{ .b = .{ .hold_bytes = 64 * mib }, .hold = 64 * mib, .top = true };
    s.begin(true, 16384, 1 * mib);
    s.layer(3, false, false);
    try testing.expect(!s.holding);
    s.routed();
    s.layer(7, false, false);
    try testing.expect(s.holding);
    s.end();
    try testing.expectEqual(@as(u32, 1), s.stats.mh_taken);
    // PF_MOE_HOLD alone holds at the first DSA layer, as before; a new forward starts with no routed call seen
    var s2: Schedule(Fake) = .{ .b = .{ .hold_bytes = 64 * mib }, .hold = 64 * mib };
    s2.begin(true, 16384, 1 * mib);
    s2.layer(3, false, false);
    try testing.expect(s2.holding);
    s2.end();
    s.begin(true, 16384, 1 * mib);
    s.layer(3, false, false);
    try testing.expect(!s.holding);
    s.end();
}

test "turboexl3_glm53f stream pf reuse: TURBOEXL3_GLM53F_PF_MOE_HOLD's take reads each count around one step: each held buffer's free (cached, released, still in flight), the cache's drop across the call's first allocation (a trim past a 32 KiB reuse), its drop and the active growth across chunks 0 and 1's commits, and the outputs whose buffer is a held one; the trim the netted measure hid shows" {
    const hold = 26 * 16 * mib;
    // with TOP: every held buffer cached (beside 100 MiB of other buffers); no trim at the first allocation (the row
    // table reuses 16 KiB); chunk 0's commit takes half the hold from the cache, chunk 1's the other half and 1 MiB
    // fresh; chunk 0's output is a held buffer; a third chunk and the call's end are not measured
    var s: Schedule(Fake) = .{ .b = .{ .hold_bytes = hold, .hold_bufs = 26, .lazy_trim = true, .held_addr = 0x1000 }, .hold = hold, .top = true };
    s.begin(true, 16384, 1 * mib);
    s.routed();
    s.layer(7, false, false);
    s.b.cache = 100 * mib;
    s.routed();
    s.at(.first_alloc_before);
    s.b.cache -= 16 << 10;
    s.b.active += 16 << 10;
    s.at(.first_alloc_after);
    s.at(.commit_before);
    s.b.cache -= hold / 2;
    s.b.active += hold / 2;
    s.committed(1, 0x1000);
    s.at(.commit_before);
    s.b.cache -= hold / 2;
    s.b.active += hold / 2 + 1 * mib;
    s.committed(2, 0x2000);
    s.at(.commit_before);
    s.committed(3, 0x1000);
    s.committed(0, 0);
    s.end();
    var st = s.stats;
    try testing.expectEqual(@as(u32, 26), st.mh_hb_cached);
    try testing.expectEqual(@as(u32, 0), st.mh_hb_released);
    try testing.expectEqual(@as(u32, 0), st.mh_hb_deferred);
    try testing.expectEqual(@as(u32, 0), st.mh_first_trims);
    try testing.expectEqual(@as(u64, 16 << 10), st.mh_first_drop);
    try testing.expectEqual(@as(u32, 2), st.mh_commits);
    try testing.expectEqual(@as(i64, @intCast(hold)), st.mh_commit_drop);
    try testing.expectEqual(@as(i64, @intCast(hold + 1 * mib)), st.mh_commit_alloc);
    try testing.expectEqual(@as(u32, 2), st.mh_y_seen);
    try testing.expectEqual(@as(u32, 1), st.mh_y_held);
    try testing.expectEqual(@as(u32, 1), st.mh_measured);
    // without TOP, the cache over the raised limit: the hand-back releases every held buffer, the first allocation
    // trims 440 MiB (whole buffers, at least the 436 MiB excess), the commits allocate the hold's bytes afresh. The
    // netted measure reads 0 (the trim and the allocation cancel); the take shows each
    var s2: Schedule(Fake) = .{ .b = .{ .hold_bytes = hold, .hold_bufs = 26, .lazy_trim = true }, .hold = hold };
    s2.begin(true, 16384, 1 * mib);
    s2.layer(3, false, false);
    s2.b.cache = 2900 * mib;
    s2.routed();
    s2.at(.first_alloc_before);
    s2.b.cache -= 440 * mib;
    s2.at(.first_alloc_after);
    s2.at(.commit_before);
    s2.b.active += hold / 2;
    s2.committed(1, 0x3000);
    s2.at(.commit_before);
    s2.b.active += hold / 2;
    s2.committed(2, 0x4000);
    s2.end();
    st = s2.stats;
    try testing.expectEqual(@as(u64, 0), st.mh_fresh_max);
    try testing.expectEqual(@as(u32, 0), st.mh_hb_cached);
    try testing.expectEqual(@as(u32, 26), st.mh_hb_released);
    try testing.expectEqual(@as(u32, 1), st.mh_over_limit);
    try testing.expectEqual(@as(u64, 2900 * mib - (2048 * mib + hold)), st.mh_excess);
    try testing.expectEqual(@as(u32, 1), st.mh_first_trims);
    try testing.expectEqual(@as(u64, 440 * mib), st.mh_first_drop);
    try testing.expectEqual(@as(i64, 0), st.mh_commit_drop);
    try testing.expectEqual(@as(i64, @intCast(hold)), st.mh_commit_alloc);
    try testing.expectEqual(@as(u32, 0), st.mh_y_held);
    // buffers their command buffer still holds at the hand-back: counted, the rest cached
    var s3: Schedule(Fake) = .{ .b = .{ .hold_bytes = hold, .hold_bufs = 26, .deferred_bufs = 24, .lazy_trim = true }, .hold = hold, .top = true };
    s3.begin(true, 16384, 1 * mib);
    s3.routed();
    s3.layer(7, false, false);
    s3.routed();
    s3.end();
    try testing.expectEqual(@as(u32, 24), s3.stats.mh_hb_deferred);
    try testing.expectEqual(@as(u32, 2), s3.stats.mh_hb_cached);
    try testing.expectEqual(@as(u32, 26), s3.stats.mh_hb_bufs);
}
