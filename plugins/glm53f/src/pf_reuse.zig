//! GLM53F_PF_REUSE (on by default, exact): a prompt forward's KDA runs hold their buffers in MLX's cache from a run's
//! first layer to its last, instead of allocating each layer's working set afresh.
//!
//! Every KDA layer of a forward allocates the same sizes, but the prompt's 2 GiB cache limit releases every buffer freed
//! past it, and MLX reuses a cached buffer only within two pages of a request, so each KDA layer allocates its working
//! set afresh. At a run's first KDA layer (the next attention layer KDA too), MLX's cache limit goes up by `cap_row` a
//! forward row (`envelopeRowBytes`: the prompt bill's envelope for one layer's working set) and stays there through the
//! run: what a layer frees stays cached, and the run's next KDA layer takes it back. The cache holds only what the run's
//! layers freed and did not take back, at most the cap; the limit is not lowered inside the run, since MLX trims a cache
//! over its limit from its oldest buffers, the held ones. At the first attention layer after the run (a DSA layer) and
//! at the forward's end, the limit goes back to the value in force before the run, and MLX trims the cache at its next
//! allocation: the DSA layers see the stock cache.
//!
//! Only which buffer an array uses changes: the same kernels over the same rows, so the same words. MLX's peak memory is
//! never reset here (the server reads it). The pass line at a prompt forward's end prints what a run's first layer left
//! and the most the cache held at a run's layer start; the A/B's peak column decides whether a peak moved.
//!
//! The routed prefill's half of the frozen line's TURBOEXL3_GLM53F_PF_REUSE (its row-slot arrays at turboexl3's size
//! classes) is the quant plugin's own switch of the same name (turboexl3_glm53f), which the measured gain had on too.
//! copied_from: CommunityServe plugins/turboexl3_glm53f/src/pf_reuse.zig@ac87fe1 (`Schedule` without
//! TURBOEXL3_GLM53F_PF_MOE_HOLD's hold and TURBOEXL3_GLM53F_PF_RUN_TRIM's trim, which are not ported) and
//! src/stream_room.zig@ac87fe1 (`envelopeRowBytes`, `ffnWidth`), at the hooks patches/turboexl3-model.patch put in
//! glm5_next's table_forward.zig `hidden` and `hiddenSplit` (here `hidden` and `hiddenPair`).

const std = @import("std");
const kit = @import("mlx_serve");
const mlx = kit.mlx;
const log = kit.log;
const options = @import("options.zig");
const Config = kit.bases.kda_hybrid.config.Config;

/// A forward's rows below this hold nothing (a decode or verify window, a short prompt).
pub const min_rows: u64 = 2048;

/// The prompt bill's MLP width (`bills.zig` `prefillFfnWidth`): the routed experts' and the shared expert's, or the
/// dense MLP's where the config declares a wider one.
fn ffnWidth(c: *const Config) u64 {
    const per_tok: u64 = @max(@as(u64, c.num_experts_per_tok), 1);
    const moe_w: u64 = @as(u64, c.moe_intermediate_size) * per_tok + c.shared_expert_intermediate_size;
    const dense_w: u64 = if (c.intermediate_size_declared) c.intermediate_size else 0;
    const w = @max(dense_w, moe_w);
    return if (w > 0) w else c.intermediate_size;
}

/// The prompt bill's MLP envelope (3 x its `mlp` term: a chunk's working set) a forward row: a run's raise of MLX's
/// cache limit a row.
pub fn envelopeRowBytes(c: *const Config) u64 {
    return 3 * 8 * @max(@as(u64, c.hidden_size), ffnWidth(c)) * 2;
}

/// A forward's counts (its pass line at its end).
pub const Stats = struct {
    /// The forward's rows (both spans of a pair forward) and its cap (bytes).
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
};

/// The schedule over a backend `B`: MLX's for the plugin (`Mlx`), a recording fake for the MLX-free tests. `B` has
/// `cacheBytes() usize`, `setCacheLimit(limit) usize` (the previous limit) and `peak() usize` (MLX's peak since the
/// process began: never reset here).
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

        /// A forward's start: `prompt` a prompt forward's rows (`rows` of them, both spans of a pair forward), `cap_row`
        /// the raise a row.
        pub fn begin(self: *Self, prompt: bool, rows: u64, cap_row: u64) void {
            self.depth += 1;
            if (self.depth != 1) return;
            self.active = prompt and rows >= min_rows and cap_row > 0;
            self.base = null;
            self.first = false;
            self.stats = .{ .rows = rows, .cap = rows *| cap_row };
        }

        /// Before layer `li`'s attention: `kda` this layer's attention KDA, `next_kda` the forward's next layer's.
        pub fn layer(self: *Self, li: usize, kda: bool, next_kda: bool) void {
            if (!self.active or self.depth != 1) return;
            if (self.base) |base| {
                // the run is over at its first other attention layer
                if (!kda) return self.drop();
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
                    log.info("[glm53f] GLM53F_PF_REUSE engaged: a KDA run's first layer left {d} MiB in MLX's cache for the run's later layers (at layer {d} the cache {d} MiB, the run's limit {d} MiB, {d} MiB before it)\n", .{ held >> 20, li, c >> 20, (base +| self.stats.cap) >> 20, base >> 20 });
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
            self.drop();
            self.active = false;
            const st = self.stats;
            log.info("[glm53f] GLM53F_PF_REUSE pass rows={d}: runs={d} holds={d} drops={d} held_mb={d}..{d} cache_mb_max={d} cap_mb={d} limit_mb_max={d} mlx_peak_mb={d}\n", .{ st.rows, st.runs, st.holds, st.drops, if (st.runs == 0 or st.held_min == std.math.maxInt(u64)) 0 else st.held_min >> 20, st.held_max >> 20, st.cache_max >> 20, st.cap >> 20, st.limit_max >> 20, self.b.peak() >> 20 });
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
};

var schedule: Schedule(Mlx) = .{ .b = .{} };
var said = false;

/// `hidden` and `hiddenPair`, at a forward's start (a defer'd `end` at its end): `prompt` a prompt forward of `rows`
/// rows over the model of config `c`.
pub fn begin(c: *const Config, prompt: bool, rows: u64) void {
    if (!options.on(.pf_reuse)) return;
    const cap_row = envelopeRowBytes(c);
    if (!said) {
        said = true;
        log.info("[glm53f] GLM53F_PF_REUSE=1: a prompt forward's KDA runs hold their buffers in MLX's cache from a run's first layer to its last (the limit raised by {d} KiB a forward row, the bill's layer envelope, through the run, then put back); the same kernels on the same rows\n", .{cap_row >> 10});
    }
    schedule.begin(prompt, rows, cap_row);
}
/// The layer loops, before each layer's attention.
pub fn layer(li: usize, kda: bool, next_kda: bool) void {
    if (!options.on(.pf_reuse)) return;
    schedule.layer(li, kda, next_kda);
}
pub fn end() void {
    if (!options.on(.pf_reuse)) return;
    schedule.end();
}

// ── tests (MLX-free) ──

const testing = std.testing;
const mib: usize = 1 << 20;

/// A backend that records the limits set; the cache's bytes a value the test moves, trimmed to a lowered limit (MLX trims
/// at its next allocation).
const Fake = struct {
    limit: usize = 2048 * mib,
    cache: usize = 0,
    sets: [128]usize = undefined,
    n: usize = 0,
    pub fn cacheBytes(self: *const Fake) usize {
        return self.cache;
    }
    pub fn setCacheLimit(self: *Fake, limit: usize) usize {
        const prev = self.limit;
        self.limit = limit;
        self.cache = @min(self.cache, limit);
        self.sets[self.n] = limit;
        self.n += 1;
        return prev;
    }
    pub fn peak(_: *const Fake) usize {
        return 0;
    }
};

/// GLM-5.3-Flash's attention pattern: layers 3, 7, ... 43 DSA, the rest KDA (45 layers).
fn glmKda(li: usize) bool {
    return li % 4 != 3;
}

/// A forward of `n` layers over `s` as the layer loop calls it, the cache growing by `freed` over a run's first layer
/// (as far as the limit lets it) and by `more` over its later ones; `limits[li]` the limit in force through layer `li`.
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

test "glm53f pf reuse: a run's raise a forward row is the prompt bill's MLP envelope (GLM-5.3-Flash: 884,736 B, the frozen line's)" {
    const c: Config = .{ .model_type = "glm5_next", .hidden_size = 4096, .intermediate_size = 12288, .intermediate_size_declared = true, .num_experts_per_tok = 8, .moe_intermediate_size = 2048, .shared_expert_intermediate_size = 2048 };
    try testing.expectEqual(@as(u64, 884_736), envelopeRowBytes(&c));
    // the dense width where it is the wider; the hidden width at least
    const d: Config = .{ .hidden_size = 4096, .intermediate_size = 20480, .intermediate_size_declared = true, .num_experts_per_tok = 8, .moe_intermediate_size = 2048, .shared_expert_intermediate_size = 2048 };
    try testing.expectEqual(@as(u64, 3 * 8 * 20480 * 2), envelopeRowBytes(&d));
    const e: Config = .{ .hidden_size = 8192, .intermediate_size = 1024, .num_experts_per_tok = 1, .moe_intermediate_size = 512 };
    try testing.expectEqual(@as(u64, 3 * 8 * 8192 * 2), envelopeRowBytes(&e));
}

test "glm53f pf reuse: a KDA run's first layer raises the limit by the cap for the whole run, the next DSA layer and the forward's end put it back; GLM's 45 layers make 11 runs, 22 held layers and 11 drops" {
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

test "glm53f pf reuse: no limit set outside a prompt forward of 2048 rows or more with a cap, inside a nested forward, or over layers that make no run; a forward that ends inside a run puts the limit back; the cap bounds what is held" {
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

test "glm53f pf reuse: GLM53F_PF_REUSE is on by default and off leaves MLX's limit alone" {
    try testing.expect(options.defaultOn(.pf_reuse));
    options.forced[@backingInt(options.Switch.pf_reuse)] = false;
    defer options.forced[@backingInt(options.Switch.pf_reuse)] = null;
    const c: Config = .{ .hidden_size = 4096, .num_experts_per_tok = 8, .moe_intermediate_size = 2048 };
    // off: the module schedule never starts (a call into MLX here would fail the MLX-free binary)
    begin(&c, true, 16384);
    layer(0, true, true);
    end();
    try testing.expectEqual(@as(u32, 0), schedule.depth);
}
