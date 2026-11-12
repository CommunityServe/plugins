//! `TURBOEXL3_GLM53F_DSA_HOLD` (on by default, exact; a memory-for-compute trade; `0` off): a prompt span's DSA layer keeps its absorbed
//! queries (`q_abs`) and its sparse attention's latent output (`o_lat`), both [1, 64, rows, 512], for the span's next DSA
//! layer, which releases them into MLX's cache just before its own two of that size are allocated, so it takes them
//! instead of two fresh ones (patches/turboexl3-model.patch in the copy's `dsaAttn`).
//!
//! g53q8pt (PF_INT8=4's prompt timeline, a 16K prompt): each DSA layer allocates 7.8-8.4 GB of fresh buffers (a KDA layer
//! after it 4.0-4.3 GB), and the GPU idles about 46 ms a DSA layer, about 9.6 ms of it just before the sparse attention,
//! where the host allocates q_abs and o_lat afresh (2 x 0.94 GB at 14,305 sparse rows). MLX reuses a cached buffer only at
//! its size (up to two pages more), and the prompt's 2 GiB cache is long past them by the next DSA layer, four layers on.
//!
//! The release fits the limits the other prompt holds set at the same DSA layer (src/pf_reuse.zig):
//! - TURBOEXL3_GLM53F_PF_REUSE puts the KDA run's limit back at the DSA layer's start, and MLX trims what the run left in
//!   the cache at its next allocation. The release reads the limit then in force by raising it to the most (a raise never
//!   trims, and MLX caches a buffer freed while the cache is under its limit), frees the held arrays into the cache, and
//!   sets the limit to the limit read plus their bytes: the next allocation still trims what the run left, oldest first,
//!   and the released buffers are the newest. `q_abs` and then `o_lat` are evaluated without the host's wait, each
//!   allocation taking a released buffer of its size, and the limit goes back to exactly the limit read (`restore`,
//!   after both), so the rest of the attention sees PF_REUSE's limit as before.
//! - TURBOEXL3_GLM53F_PF_MOE_HOLD's hold is taken at the layer's start, before the run's limit goes back, and handed back
//!   at the layer's first routed call, after the attention: the release and the restore fall between the two, and
//!   PF_MOE_HOLD_TOP's raise at the hand-back reads the limit the restore put back.
//! Only which buffer an output uses moves, and when its work is committed: the same kernels on the same rows, so the
//! same words.
//!
//! Memory: the held 1.9 GB (at 16K rows) stay live through the three KDA layers and four MoE stages between; not billed.
//! The engaged line and the pass line print the held MiB; the A/B's peak column decides the trade.
//!
//! What holds: a trunk DSA layer (not the MTP head's) of a prompt span with `min_rows` sparse rows or more, on the sparse
//! path, when a later trunk layer of the forward is DSA. A split forward's short span (its 31-row tail) neither keeps,
//! releases nor drops span A's set. A decode or verify forward drops anything left and prints the pass's counts.
const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const log = @import("mlx_serve/sdk/util/log.zig");

/// `TURBOEXL3_GLM53F_DSA_HOLD=1`, set by the plugin at load, before any forward.
pub var on: bool = false;
/// The switch's load line (an A/B step reads "<switch>=1: ").
pub const line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_DSA_HOLD=1: a prompt span's DSA layer keeps its q_abs and o_lat for its next DSA layer, released into MLX's cache just before that layer allocates its own, under the limit in force raised by their bytes (the same kernels on the same rows; a memory trade: held through the KDA and MoE stages between)\n";
/// A span's sparse rows below this hold nothing.
pub const min_rows: c_int = 2048;

/// A prompt pass's counts (printed when a decode or verify forward follows, then zeroed).
pub const Stats = struct {
    /// Releases of a held set under the raised limit; the outputs then counted: their buffer a released one or not.
    releases: u64 = 0,
    hits: u64 = 0,
    misses: u64 = 0,
    /// Sets kept, held sets dropped (another layer's or rows', or left at the pass's end), and the largest held bytes.
    kept: u64 = 0,
    dropped: u64 = 0,
    held_max: u64 = 0,
};

/// The holder over a backend `B`: MLX's for the plugin (`Mlx`), a recording fake for the MLX-free test. `B` has
/// `Array`, `retain(a) Array`, `free(a)`, `nbytes(a) usize`, `dataPtr(a) ?usize`, `cacheBytes() usize`,
/// `setCacheLimit(limit) usize` (the previous limit), `asyncEval(a) !void` and `peak() usize`.
pub fn Holder(comptime B: type) type {
    return struct {
        const Self = @This();
        b: B,
        held: [2]B.Array = undefined,
        n: usize = 0,
        /// The trunk layer the held set is for and its rows.
        for_layer: u32 = 0,
        rows: c_int = 0,
        stats: Stats = .{},
        engaged_said: bool = false,

        /// A release's state for `settle` and `restore`.
        pub const Token = struct {
            active: bool = false,
            prev_limit: usize = 0,
            ptrs: [2]usize = @splat(0),
        };

        /// Before DSA layer `layer`'s q_abs (`rows` sparse rows, `nbytes` each of q_abs and o_lat) is evaluated: the set
        /// held for this layer and rows, of that size, freed into MLX's cache under the limit in force raised by its
        /// bytes (the file's doc); a set held for anything else dropped (no raise).
        pub fn release(self: *Self, layer: u32, rows: c_int, nbytes: usize) Token {
            var t: Token = .{};
            if (self.n == 0) return t;
            const match = self.for_layer == layer and self.rows == rows and self.n == 2 and
                self.b.nbytes(self.held[0]) == nbytes and self.b.nbytes(self.held[1]) == nbytes;
            if (!match) {
                self.drop();
                return t;
            }
            for (self.held[0..2], &t.ptrs) |a, *p| p.* = self.b.dataPtr(a) orelse 0;
            const c = self.b.cacheBytes();
            // the limit in force (PF_REUSE's, put back at this layer's start) read by raising it to the most: a raise
            // never trims, and MLX caches a buffer freed while the cache is under its limit
            const prev = self.b.setCacheLimit(std.math.maxInt(usize));
            for (self.held[0..2]) |a| self.b.free(a);
            self.n = 0;
            // the limit read plus the released bytes: what the cache holds past it is trimmed at the next allocation,
            // oldest first, and the released buffers are the newest
            const limit = prev +| 2 * nbytes;
            _ = self.b.setCacheLimit(limit);
            t.prev_limit = prev;
            t.active = true;
            self.stats.releases += 1;
            if (!self.engaged_said) {
                self.engaged_said = true;
                log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_DSA_HOLD engaged: a DSA layer's q_abs and o_lat ({d} rows, 2 x {d} MiB held) released into MLX's cache for layer {d}: the cache {d} MiB before, its limit {d} MiB (the limit in force {d} MiB and the held bytes) until both are allocated\n", .{ rows, nbytes >> 20, layer, c >> 20, limit >> 20, prev >> 20 });
            }
            return t;
        }

        /// An output (q_abs, then o_lat) evaluated without the host's wait, so its allocation happens now, under the
        /// raised limit; counted a hit when its buffer is a released one.
        pub fn settle(self: *Self, t: *const Token, a: B.Array) !void {
            if (!t.active) return;
            try self.b.asyncEval(a);
            const p = self.b.dataPtr(a) orelse 0;
            const hit = p != 0 and (p == t.ptrs[0] or p == t.ptrs[1]);
            if (hit) self.stats.hits += 1 else self.stats.misses += 1;
        }

        /// The cache limit back to the limit the release read (once a token).
        pub fn restore(self: *Self, t: *Token) void {
            if (!t.active) return;
            _ = self.b.setCacheLimit(t.prev_limit);
            t.active = false;
        }

        /// The layer's q_abs and o_lat kept for trunk layer `next` (the forward's next DSA layer) at these rows, a
        /// reference of the holder's own each; null `next` keeps nothing.
        pub fn keep(self: *Self, next: ?u32, rows: c_int, q_abs: B.Array, o_lat: B.Array) void {
            self.drop();
            const nl = next orelse return;
            self.held = .{ self.b.retain(q_abs), self.b.retain(o_lat) };
            self.n = 2;
            self.for_layer = nl;
            self.rows = rows;
            self.stats.kept += 1;
            self.stats.held_max = @max(self.stats.held_max, self.b.nbytes(q_abs) + self.b.nbytes(o_lat));
        }

        /// A decode or verify forward: what is held dropped and the prompt pass's counts printed.
        pub fn idle(self: *Self) void {
            self.drop();
            const st = self.stats;
            self.stats = .{};
            if (st.releases == 0 and st.kept == 0 and st.dropped == 0) return;
            log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_DSA_HOLD pass: releases={d} hits={d} misses={d} kept={d} dropped={d} held_max_mb={d} mlx_peak_mb={d}\n", .{ st.releases, st.hits, st.misses, st.kept, st.dropped, st.held_max >> 20, self.b.peak() >> 20 });
        }

        fn drop(self: *Self) void {
            if (self.n == 0) return;
            for (self.held[0..self.n]) |a| self.b.free(a);
            self.n = 0;
            self.stats.dropped += 1;
        }
    };
}

/// MLX's backend.
pub const Mlx = struct {
    pub const Array = mlx.mlx_array;
    pub fn retain(_: *const Mlx, a: Array) Array {
        var r = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&r, a);
        return r;
    }
    pub fn free(_: *const Mlx, a: Array) void {
        _ = mlx.mlx_array_free(a);
    }
    pub fn nbytes(_: *const Mlx, a: Array) usize {
        return mlx.mlx_array_size(a) * mlx.mlx_array_itemsize(a);
    }
    /// The buffer's address for an evaluated array (a held one, and an output after `settle`'s evaluation).
    pub fn dataPtr(_: *const Mlx, a: Array) ?usize {
        const p = mlx.mlx_array_data_uint8(a) orelse return null;
        return @intFromPtr(p);
    }
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
    pub fn asyncEval(_: *const Mlx, a: Array) !void {
        const v = mlx.mlx_vector_array_new_data(&[_]Array{a}, 1);
        defer _ = mlx.mlx_vector_array_free(v);
        try mlx.check(mlx.mlx_async_eval(v));
    }
    pub fn peak(_: *const Mlx) usize {
        var p: usize = 0;
        _ = mlx.mlx_get_peak_memory(&p);
        return p;
    }
};

var holder: Holder(Mlx) = .{ .b = .{} };
/// A release's state (the plugin's holder's `Token`).
pub const Hold = Holder(Mlx).Token;

/// Whether DSA layer `layer`'s call holds: a trunk layer's (not the MTP head's: `trunk_layer`), a prompt span of
/// `min_rows` sparse rows or more.
pub fn holds(trunk_layer: bool, rows: c_int) bool {
    return on and trunk_layer and rows >= min_rows;
}
/// `dsaAttn` after building q_abs (`nbytes` its bytes, o_lat's too).
pub fn release(trunk_layer: bool, layer: u32, rows: c_int, nbytes: usize) Hold {
    if (!holds(trunk_layer, rows)) return .{};
    return holder.release(layer, rows, nbytes);
}
pub fn settle(t: *const Hold, a: mlx.mlx_array) !void {
    if (!on) return;
    try holder.settle(t, a);
}
pub fn restore(t: *Hold) void {
    if (!on) return;
    holder.restore(t);
}
/// `dsaAttn` after its sparse path's o_lat: kept for the forward's next trunk DSA layer `next` (null: none).
pub fn keep(trunk_layer: bool, next: ?u32, rows: c_int, q_abs: mlx.mlx_array, o_lat: mlx.mlx_array) void {
    if (!holds(trunk_layer, rows)) return;
    holder.keep(next, rows, q_abs, o_lat);
}
/// `dsaAttn` at a decode or verify forward's rows.
pub fn idle() void {
    if (!on) return;
    holder.idle();
}
/// An array's bytes (its shape and dtype; an unevaluated one too).
pub fn bytesOf(a: mlx.mlx_array) usize {
    return mlx.mlx_array_size(a) * mlx.mlx_array_itemsize(a);
}

// ── tests (MLX-free) ──

const testing = std.testing;

/// A backend that records the holder's calls: arrays are ids with a size and a buffer address each, the cache a byte
/// count and its limit a value.
const Fake = struct {
    pub const Array = u32;
    refs: [16]u32 = @splat(0),
    bytes: [16]usize = @splat(0),
    buf: [16]usize = @splat(0),
    cache: usize = 300 << 20,
    limit: usize = 2 << 30,
    limits: [8]usize = @splat(0),
    n_limits: usize = 0,
    evals: u32 = 0,
    /// The limit in force at each free (the release must free under a limit that caches).
    free_limits: [16]usize = @splat(0),
    n_frees: usize = 0,
    /// The buffer an evaluation hands out next.
    next_buf: usize = 0,
    pub fn retain(self: *Fake, a: Array) Array {
        self.refs[a] += 1;
        return a;
    }
    pub fn free(self: *Fake, a: Array) void {
        self.refs[a] -= 1;
        self.free_limits[self.n_frees] = self.limit;
        self.n_frees += 1;
    }
    pub fn nbytes(self: *const Fake, a: Array) usize {
        return self.bytes[a];
    }
    pub fn dataPtr(self: *const Fake, a: Array) ?usize {
        return if (self.buf[a] == 0) null else self.buf[a];
    }
    pub fn cacheBytes(self: *const Fake) usize {
        return self.cache;
    }
    pub fn setCacheLimit(self: *Fake, limit: usize) usize {
        const prev = self.limit;
        self.limit = limit;
        self.limits[self.n_limits] = limit;
        self.n_limits += 1;
        return prev;
    }
    pub fn asyncEval(self: *Fake, a: Array) !void {
        self.evals += 1;
        if (self.buf[a] == 0) self.buf[a] = self.next_buf;
    }
    pub fn peak(_: *const Fake) usize {
        return 0;
    }
};

test "turboexl3_glm53f stream: TURBOEXL3_GLM53F_DSA_HOLD keeps a DSA layer's q_abs and o_lat for the span's next DSA layer, frees them there under the most limit, then holds the limit in force plus their bytes until both are allocated and puts it back exactly, and drops a set held for another layer, other rows or past the prompt" {
    const mb: usize = 1 << 20;
    var h: Holder(Fake) = .{ .b = .{} };
    // layer 3 keeps its two (ids 1, 2: 937 MiB each, buffers 0xa000 and 0xb000) for layer 7 at 14,305 rows
    h.b.bytes[1] = 937 * mb;
    h.b.bytes[2] = 937 * mb;
    h.b.buf[1] = 0xa000;
    h.b.buf[2] = 0xb000;
    h.b.refs[1] = 1;
    h.b.refs[2] = 1;
    h.keep(7, 14305, 1, 2);
    try testing.expectEqual(@as(u32, 2), h.b.refs[1]);
    try testing.expectEqual(@as(u64, 2 * 937 * mb), h.stats.held_max);
    // layer 7: PF_REUSE put the run's limit back to 2 GiB with the cache over it (5 GiB the run left); q_abs (3) and
    // o_lat (4) take the released buffers
    h.b.cache = 5 << 30;
    h.b.bytes[3] = 937 * mb;
    h.b.bytes[4] = 937 * mb;
    h.b.refs[3] = 1;
    h.b.refs[4] = 1;
    var t = h.release(7, 14305, 937 * mb);
    try testing.expect(t.active);
    try testing.expectEqual(@as(u32, 1), h.b.refs[1]);
    // both freed under the most limit (cached whatever the cache holds), then the limit in force plus their bytes
    try testing.expectEqual(@as(usize, 2), h.b.n_frees);
    for (h.b.free_limits[0..2]) |l| try testing.expectEqual(@as(usize, std.math.maxInt(usize)), l);
    try testing.expectEqual((2048 + 2 * 937) * mb, h.b.limit);
    h.b.next_buf = 0xb000;
    try h.settle(&t, 3);
    h.b.next_buf = 0xa000;
    try h.settle(&t, 4);
    h.restore(&t);
    h.restore(&t);
    try testing.expectEqual(@as(usize, 2) << 30, h.b.limit);
    try testing.expectEqual(@as(usize, 3), h.b.n_limits);
    try testing.expectEqual(@as(u64, 2), h.stats.hits);
    try testing.expectEqual(@as(u64, 0), h.stats.misses);
    // a set kept for layer 11 meets layer 15 (or other rows): dropped, no limit set
    h.keep(11, 14305, 3, 4);
    const t2 = h.release(15, 14305, 937 * mb);
    try testing.expect(!t2.active and h.n == 0 and h.stats.dropped == 1);
    h.keep(11, 14305, 3, 4);
    const t3 = h.release(11, 16000, 937 * mb);
    try testing.expect(!t3.active and h.n == 0);
    // the forward's last DSA layer keeps nothing; a decode forward drops what is left
    h.keep(null, 14305, 3, 4);
    try testing.expectEqual(@as(usize, 0), h.n);
    h.keep(19, 14305, 3, 4);
    h.idle();
    try testing.expectEqual(@as(usize, 0), h.n);
    try testing.expectEqual(@as(u32, 1), h.b.refs[3]);
    try testing.expectEqual(@as(usize, 3), h.b.n_limits);
    // nothing held: a release changes nothing
    const t4 = h.release(23, 14305, 937 * mb);
    try testing.expect(!t4.active);
    try testing.expectEqual(@as(usize, 3), h.b.n_limits);
    // with the switch on: a trunk layer's span of min_rows sparse rows or more holds; the split's tail and the MTP head
    // do not
    on = true;
    defer on = false;
    try testing.expect(holds(true, 14305) and !holds(true, 31) and !holds(false, 14305));
}
