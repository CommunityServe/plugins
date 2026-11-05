//! zzzza5 ("Remove the copies", David 2026-11-04), MLX-free: a wide prompt layer's KV-source compressor run for every span
//! before the span loop (deepseek_v41.zig's `wideLayer`, `Shared.compressed`), against the per-span order it replaces
//! (each span's `attention` running the compressor, then reading the caches). Std only, tested by `zig build test`.
//!
//! The caches (`Rows`: comp, idx_k) are append-only, so the rows a span reads, its causal prefix, are the same whether
//! the later spans' rows were appended before its read or after: a span at positions [p0 + lo, p0 + hi) reads the first
//! `spanClen` rows, which only positions before p0 + hi made. Appending every span's rows first leaves no span's view of
//! an older version pending while the next append runs, so MLX updates the one buffer in place instead of copying the
//! whole cache once a span (the lad281b / mc281 curve's growth).

const std = @import("std");

/// The compressed rows a span ending at position p0 + hi reads (`attention`'s clen): the groups of `ratio` positions
/// closed before it ends.
pub fn spanClen(p0: usize, hi: usize, ratio: usize) usize {
    return (p0 + hi) / ratio;
}

// ── tests ──
const testing = std.testing;

/// A CPU model of one KV-source layer's compressor and cache: positions arrive in calls (`run`); every `ratio` positions
/// close a group, whose row is a fold of the group's positions' values; an open group carries over to the next call
/// (the layer's pend_kv / pend_gate). `copies` counts the whole-cache copies an append makes when an earlier version is
/// still being read (a view taken since the last append).
const Model = struct {
    ratio: usize,
    rows: [512]u64 = undefined,
    n: usize = 0,
    pend: u64 = 0,
    pend_n: usize = 0,
    viewed: bool = false,
    copies: usize = 0,

    fn value(pos: usize) u64 {
        return (@as(u64, pos) *% 0x9E3779B97F4A7C15) ^ 0x5851F42D4C957F2D;
    }
    fn run(self: *Model, p0: usize, n: usize) void {
        if (self.viewed) self.copies += 1; // this call's append copies the version a pending view still reads
        self.viewed = false;
        for (p0..p0 + n) |pos| {
            self.pend = (self.pend *% 31) +% value(pos);
            self.pend_n += 1;
            if (self.pend_n == self.ratio) {
                self.rows[self.n] = self.pend;
                self.n += 1;
                self.pend = 0;
                self.pend_n = 0;
            }
        }
    }
    fn view(self: *Model, clen: usize, out: []u64) void {
        std.debug.assert(clen <= self.n);
        @memcpy(out[0..clen], self.rows[0..clen]);
        self.viewed = true;
    }
};

test "comp_once: a span's causal prefix is the groups closed before it ends" {
    for ([_]usize{ 1, 2 }) |ratio| {
        for ([_]usize{ 0, 1, 40, 41, 256 }) |p0| {
            var m: Model = .{ .ratio = ratio };
            // the positions before the chunk, as earlier chunks left them
            m.run(0, p0);
            var hi: usize = 0;
            while (hi < 100) : (hi += 7) {
                var mm = m;
                mm.run(p0, hi);
                try testing.expectEqual(mm.n, spanClen(p0, hi, ratio));
            }
        }
    }
}

test "comp_once: every span reads the same rows, compressor first or per span; one copy-free cache" {
    const span = 8;
    for ([_]usize{ 1, 2 }) |ratio| {
        for ([_]usize{ 0, 1, 40, 41 }) |p0| {
            for ([_]usize{ 8, 64, 61, 63 }) |L| {
                const n_spans = (L + span - 1) / span;
                // per span (as before): each span's compressor, then its views
                var a: Model = .{ .ratio = ratio };
                a.run(0, p0);
                var va: [16][512]u64 = undefined;
                var ca: [16]usize = undefined;
                for (0..n_spans) |k| {
                    const lo = k * span;
                    const hi = @min(L, lo + span);
                    a.run(p0 + lo, hi - lo);
                    ca[k] = spanClen(p0, hi, ratio);
                    a.view(ca[k], &va[k]);
                }
                // compressor first (zzzza5): every span's rows appended, then each span's views of the one cache
                var b: Model = .{ .ratio = ratio };
                b.run(0, p0);
                for (0..n_spans) |k| {
                    const lo = k * span;
                    const hi = @min(L, lo + span);
                    b.run(p0 + lo, hi - lo);
                }
                for (0..n_spans) |k| {
                    const hi = @min(L, k * span + span);
                    const clen = spanClen(p0, hi, ratio);
                    try testing.expectEqual(ca[k], clen);
                    var vb: [512]u64 = undefined;
                    b.view(clen, &vb);
                    try testing.expectEqualSlices(u64, va[k][0..clen], vb[0..clen]);
                }
                // the same cache and the same open group at the end
                try testing.expectEqual(a.n, b.n);
                try testing.expectEqualSlices(u64, a.rows[0..a.n], b.rows[0..b.n]);
                try testing.expectEqual(a.pend_n, b.pend_n);
                try testing.expectEqual(a.pend, b.pend);
                // per span: a copy for every span after the first; compressor first: none
                try testing.expectEqual(n_spans - 1, a.copies);
                try testing.expectEqual(@as(usize, 0), b.copies);
            }
        }
    }
}
