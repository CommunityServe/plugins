//! The routed experts' decode on turboexl3, for the copied expert streamer (sdk/weights/streamed/exl3_source.zig) when
//! the default (`MLX_SERVE_DSV41_ROUTED=streamer` keeps the streamer's own EXL3 kernels throughout). The
//! streamer's decode waves run its banked route on turboexl3's banked decode (`moeClampedBanked`): a wave's pairs over the
//! layer's three slot banks in one call, each pair one row-slot whose id is packed (bank << 24 | slot row), DEVROUTE's
//! device wave among them (patches/turboexl3-t3-banked.patch binds both). The call runs whole at the wave's down:
//! turboexl3's gate|up finish prepares the down's input, so it reads the down projection's suh, and a gated miss part's
//! down arrays are waited only for its down (the streamer's own gate|up reads gate and up arrays alone). The slot arrays
//! are turboexl3's banks (code as the trellis, rin as suh, rout as svh); each pair's row comes back in f32 at weight 1 and
//! the streamer joins and weighs the rows itself. Everything else is the streamer's own math (`Stock`): a wide prompt
//! call's waves and cold rows on its bounded DIG-X lane, whose working set the prompt bill covers.

const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const t3 = @import("turboexl3");
const t3_group = @import("t3_group.zig");

/// Set at load from `MLX_SERVE_DSV41_ROUTED` (`config.routedT3`), before the bank opens.
pub var enabled: bool = false;

/// At load, after `MLX_SERVE_DSV41_T3_GROUP` (src/t3_group.zig) and before the bank opens (`warm` builds the banked
/// kernels at it): turboexl3's banked GEMVs at the setting's row-slots of one packed id per lead.
pub fn configureGroup() error{InvalidTurboexl3Config}!void {
    var c = t3.config.get().*;
    c.banked_group = t3_group.bankedGroup();
    try t3.configure(c);
}

/// The streamer's math over backend `G` (its `ops.MlxOps`), slot arrays `Proj` ({code, rout, rin}), a bank's three
/// `Bank`, a prompt call's `PrefillRows`, and `Stock`, the streamer's own math (its `QuantMath`), which runs all but the
/// banked decode.
pub fn Math(comptime G: type, comptime Proj: type, comptime Bank: type, comptime PrefillRows: type, comptime Stock: type) type {
    return struct {
        const Self = @This();
        const T = G.T;
        /// The widest decode-lane call, the stock math's (a wide call's cold rows go in chunks of it).
        pub const max_decode_rows: u32 = Stock.max_decode_rows;
        /// The most pairs a banked call takes here: a verify's 8 rows of 6 routes.
        pub const max_banked_pairs: u32 = 48;
        state: *State,
        stock: Stock,

        /// The banked route's counters; `begin` at each layer call (the source's `rows`).
        pub const State = struct {
            limit: c_int,
            /// Cleared at each layer call by the source's `rows` (patches/turboexl3.patch); nothing reads it since the
            /// prompt calls run on the stock math.
            n: usize = 0,
            tally: Tally = .{},
            /// the current layer call's banked calls and its last one's pairs
            call_banked: u32 = 0,
            call_last: u32 = 0,

            /// A layer call begins: the last one's banked calls counted (`Tally`).
            pub fn begin(st: *State) void {
                if (st.call_banked > 1) st.tally.last_part_pairs += st.call_last;
                st.call_banked = 0;
                st.call_last = 0;
            }

            /// `[routed-t3]` since the last request's end: the banked route's layer calls, banked calls and miss parts.
            pub fn meters(st: *State, buf: []u8) []const u8 {
                if (!enabled) return buf[0..0];
                st.begin();
                const t = st.tally;
                st.tally = .{};
                var w = std.Io.Writer.fixed(buf);
                w.print("[routed-t3] banked_calls={d} layer_calls={d} parts={d} part_pairs={d} last_part_pairs={d}\n", .{ t.banked, t.layer_calls, t.parts, t.part_pairs, t.last_part_pairs }) catch {};
                return w.buffered();
            }
        };

        /// The banked route's work since the last request's end. A layer call's first banked call is its resident pairs'
        /// (DEVROUTE's device wave in decode, the hit wave of a prompt call of at most 48 pairs), each later one a miss
        /// part, which runs whole once its down bytes landed (`downBanked`): in the streamer's own route a part's gate|up
        /// runs once its gate and up bytes land, so a call's last part (`last_part_pairs`) is where that order can
        /// lengthen it.
        pub const Tally = struct { banked: u64 = 0, layer_calls: u64 = 0, parts: u64 = 0, part_pairs: u64 = 0, last_part_pairs: u64 = 0 };

        /// The stock math's decode GEMV (a wide prompt call's cold rows; the decode waves are banked).
        pub fn gateUp(self: *const Self, g: *G, x: T, ids: T, gate: Proj, up: Proj) !T {
            return self.stock.gateUp(g, x, ids, gate, up);
        }

        pub fn down(self: *const Self, g: *G, h: T, ids: T, d: Proj) !T {
            return self.stock.down(g, h, ids, d);
        }

        /// The streamer's banked route (its hook's `banked`, and DEVROUTE's device wave over it).
        pub const has_banked = true;

        /// DEVROUTE's LUT entry for a pair whose expert no bank holds: turboexl3's skip id on its grouped decode
        /// (`MLX_SERVE_DSV41_T3_GROUP`: the row-slot runs no expert, its row 0), else 0 (base row 0, computed). The miss
        /// part's row replaces it in the join either way.
        pub fn lutMissing(self: *const Self) u32 {
            _ = self;
            return if (t3_group.on) t3.banked_skip_id else 0;
        }

        /// A wave's gate|up: nothing runs yet; `x` [pairs, hidden] (row-slot r's row) is what `downBanked` receives. The
        /// banked call reads every projection's arrays (its gate|up finish reads the down's suh), and a gated miss part's
        /// banks hold waited down arrays only at its down.
        pub fn gateUpBanked(self: *const Self, g: *G, x: T, ids: T, banks: *const [3]Bank) !T {
            _ = self;
            _ = g;
            _ = ids;
            _ = banks;
            return x;
        }

        /// The wave over the three banks (base, ext, transient), whole: `x` (`gateUpBanked`'s), `ids` u32 [pairs] packed
        /// (bank << 24 | slot row) -> [pairs, hidden] f32, each pair's row at weight 1. `banks` are the down's: in a gated
        /// miss part, every array waited (gate and up at the gate value, down at the part's).
        pub fn downBanked(self: *const Self, g: *G, x: T, ids: T, banks: *const [3]Bank) !T {
            const tb = t3Banks(banks);
            const y = try g.adopt(try t3.moeClampedBanked(g.s, x, &tb, ids, .mul1, self.state.limit));
            const st = self.state;
            const pairs_n: u32 = @intCast(g.shapeOf(ids).dim(0));
            st.tally.banked += 1;
            if (st.call_banked == 0) st.tally.layer_calls += 1 else {
                st.tally.parts += 1;
                st.tally.part_pairs += pairs_n;
            }
            st.call_banked += 1;
            st.call_last = pairs_n;
            return y;
        }

        /// What `warm` compiled: its calls and their wall time.
        pub const Warmed = struct { calls: u32 = 0, ms: u64 = 0 };

        /// At load, before any request: the banked decode's kernels for every pair count, compiled, on the first layer
        /// of each code width (K3, K2). A banked call's pair count is a template constant of its kernels (RKC), so each
        /// count 1..max_banked_pairs is a kernel set of its own. Each call reads the layer's base bank at slot row 0 with
        /// x zero in the trunk's dtype (`act`; bf16 served) and keeps nothing.
        pub fn warm(self: *const Self, g: *G, banks: []const [3]?Bank, hidden: c_int, act: mlx.mlx_dtype) !Warmed {
            var out: Warmed = .{};
            const t0 = nowNs();
            var seen: [2]bool = .{ false, false };
            for (banks) |lb| {
                const b = lb[0] orelse continue;
                const ki: usize = if (g.shapeOf(b.gate.code).dim(3) == 32) 1 else 0;
                if (seen[ki]) continue;
                seen[ki] = true;
                const three = [3]Bank{ b, b, b };
                for (1..max_banked_pairs + 1) |r| {
                    const m = g.mark();
                    defer g.resetTo(m);
                    const x = try g.zeros(&.{ @intCast(r), hidden }, act);
                    const ids = try g.zeros(&.{@intCast(r)}, .uint32);
                    try g.evalAll(&.{try self.downBanked(g, try self.gateUpBanked(g, x, ids, &three), ids, &three)});
                    out.calls += 1;
                }
            }
            // the warm-up's calls are no request's
            self.state.tally = .{};
            self.state.call_banked = 0;
            self.state.call_last = 0;
            out.ms = (nowNs() - t0) / std.time.ns_per_ms;
            return out;
        }

        fn nowNs() u64 {
            var ts: std.c.timespec = undefined;
            _ = std.c.clock_gettime(.MONOTONIC, &ts);
            return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
        }

        /// The banks as turboexl3 takes them: code (I16: the banked decode reads its words as they are) as the trellis,
        /// rin as suh, rout as svh.
        fn t3Banks(banks: *const [3]Bank) [3]t3.Bank {
            var out: [3]t3.Bank = undefined;
            for (banks, &out) |b, *o| {
                inline for (.{ "gate", "up", "down" }) |f| {
                    const p = @field(b, f);
                    @field(o.*, f) = .{ .trellis = p.code, .suh = p.rin, .svh = p.rout };
                }
            }
            return out;
        }

        /// A wide prompt call's waves: the stock math's (its DIG-X lane).
        pub fn prefill(self: *const Self, g: *G, layer: u32, x: T, rows: PrefillRows, bank: Bank) !T {
            return self.stock.prefill(g, layer, x, rows, bank);
        }

        pub fn finishPrefill(self: *const Self, g: *G) !void {
            return self.stock.finishPrefill(g);
        }

        /// The stock math's unjoined prefill, as it has one.
        pub const has_parts = Stock.has_parts;
        pub fn prefillParts(self: *const Self, g: *G, layer: u32, x: T, rows: PrefillRows, bank: Bank, alloc: std.mem.Allocator, outs: *std.ArrayList(T), pos: *std.ArrayList(u32)) !void {
            return self.stock.prefillParts(g, layer, x, rows, bank, alloc, outs, pos);
        }
    };
}
