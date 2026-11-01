//! The routed experts' math on turboexl3, for the copied expert streamer (sdk/weights/streamed/exl3_source.zig) when
//! `MLX_SERVE_DSV41_ROUTED=turboexl3` asks for it; the default is the streamer's own EXL3 kernels. The streamer's decode
//! waves run its banked route on turboexl3's banked decode (`moeClampedBanked`): a wave's pairs over the layer's three slot
//! banks in one call, each pair one row-slot whose id is packed (bank << 24 | slot row), DEVROUTE's device wave among them
//! (patches/turboexl3-t3-banked.patch binds both). The call runs whole at the wave's down: turboexl3's gate|up finish
//! prepares the down's input, so it reads the down projection's suh, and a gated miss part's down arrays are waited only
//! for its down (the streamer's own gate|up reads gate and up arrays alone). Its prompt calls, and a wave of a hook
//! bound without the banked route, hand it each bank's pairs: every pair is one row of a top-1 `moeClamped` call at weight
//! 1 on the bank's slot arrays. Either way the slot arrays are turboexl3's bank (code as the trellis, rin as suh, rout as
//! svh) and the rows come back in f32, each pair's at weight 1; the streamer joins and weighs the pairs' rows itself.

const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const t3 = @import("turboexl3");

/// Set at load from `MLX_SERVE_DSV41_ROUTED` (`config.routedT3`), before the bank opens.
pub var enabled: bool = false;

/// The streamer's math over backend `G` (its `ops.MlxOps`), slot arrays `Proj` ({code, rout, rin}), a bank's three
/// `Bank`, and a prompt call's `PrefillRows`.
pub fn Math(comptime G: type, comptime Proj: type, comptime Bank: type, comptime PrefillRows: type) type {
    return struct {
        const Self = @This();
        const T = G.T;
        /// The widest decode-lane call: a verify's 8 rows of 6 routes.
        pub const max_decode_rows: u32 = 48;
        state: *State,

        /// A decode group's gate and up arrays, from its `gateUp` to its `down`: the streamer runs a call's gate/ups before
        /// their downs, and each group's ids are an array of their own. `begin` at each layer call (the source's `rows`).
        pub const State = struct {
            limit: c_int,
            pending: [256]Pending = undefined,
            n: usize = 0,
            tally: Tally = .{},
            /// the current layer call's banked calls and its last one's pairs
            call_banked: u32 = 0,
            call_last: u32 = 0,

            /// A layer call begins: the last one's banked calls counted (`Tally`), the pending list emptied.
            pub fn begin(st: *State) void {
                if (st.call_banked > 1) st.tally.last_part_pairs += st.call_last;
                st.call_banked = 0;
                st.call_last = 0;
                st.n = 0;
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
        const Pending = struct { ids: ?*anyopaque, gate: Proj, up: Proj };

        /// The banked route's work since the last request's end. A layer call's first banked call is its resident pairs'
        /// (DEVROUTE's device wave in decode, the hit wave before a prompt's misses), each later one a miss part, which
        /// runs whole once its down bytes landed (`downBanked`): in the streamer's own route a part's gate|up runs once its
        /// gate and up bytes land, so a call's last part (`last_part_pairs`) is where that order can lengthen it.
        pub const Tally = struct { banked: u64 = 0, layer_calls: u64 = 0, parts: u64 = 0, part_pairs: u64 = 0, last_part_pairs: u64 = 0 };

        /// Keeps the group's gate and up for its `down`, which runs the pairs: `x` is what `down` receives.
        pub fn gateUp(self: *const Self, g: *G, x: T, ids: T, gate: Proj, up: Proj) !T {
            _ = g;
            const st = self.state;
            if (st.n == st.pending.len) return error.RoutedT3PendingFull;
            st.pending[st.n] = .{ .ids = ids.ctx, .gate = gate, .up = up };
            st.n += 1;
            return x;
        }

        pub fn down(self: *const Self, g: *G, h: T, ids: T, d: Proj) !T {
            const p = try self.pendingOf(ids);
            return pairs(self.state.limit, g, h, ids, .{ .gate = p.gate, .up = p.up, .down = d });
        }

        /// The streamer's banked route (its hook's `banked`, and DEVROUTE's device wave over it).
        pub const has_banked = true;

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

        /// At load, before any request: every kernel the routed experts can launch, compiled, on the first layer of each
        /// code width (K3, K2). A banked call's pair count is a template constant of its kernels (RKC), so each count
        /// 1..max_decode_rows is a kernel set of its own; the prompt lane's top-1 `moeClamped` calls take its decode
        /// launches up to 16 rows (each count its own as well) and its prefill chain above. Each call reads the layer's
        /// base bank at slot row 0 with x zero in the trunk's dtype (`act`; bf16 served) and keeps nothing. Unwarmed
        /// counts compile at their first launch, as before.
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
                for (1..max_decode_rows + 1) |r| {
                    const m = g.mark();
                    defer g.resetTo(m);
                    const x = try g.zeros(&.{ @intCast(r), hidden }, act);
                    const ids = try g.zeros(&.{@intCast(r)}, .uint32);
                    try g.evalAll(&.{try self.downBanked(g, try self.gateUpBanked(g, x, ids, &three), ids, &three)});
                    out.calls += 1;
                }
                for ([_]c_int{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 64 }) |r| {
                    const m = g.mark();
                    defer g.resetTo(m);
                    const x = try g.zeros(&.{ r, hidden }, .float32);
                    const ids = try g.zeros(&.{r}, .uint32);
                    try g.evalAll(&.{try pairs(self.state.limit, g, x, ids, b)});
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

        /// The gate|up state of the group whose ids are `ids`, taken off the list.
        fn pendingOf(self: *const Self, ids: T) !Pending {
            const st = self.state;
            const i = for (st.pending[0..st.n], 0..) |q, j| {
                if (q.ids == ids.ctx) break j;
            } else return error.RoutedT3NoGateUp;
            const p = st.pending[i];
            st.pending[i] = st.pending[st.n - 1];
            st.n -= 1;
            return p;
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

        /// A prompt call's pairs: `rows.slot` each pair's slot row, `rows.act_row` its row of `x` (null: x's rows are
        /// the pairs'). The result is kept: the streamer's wide path holds it past the call's scope and releases it.
        pub fn prefill(self: *const Self, g: *G, layer: u32, x: T, rows: PrefillRows, bank: Bank) !T {
            _ = layer;
            const n: c_int = @intCast(rows.slot.len);
            const ids = try g.hostArray(std.mem.sliceAsBytes(rows.slot), &.{n}, .uint32);
            const xs = if (rows.act_row) |r| try g.take(x, try g.hostArray(std.mem.sliceAsBytes(r), &.{n}, .uint32), 0) else x;
            return g.keep(try pairs(self.state.limit, g, xs, ids, bank));
        }

        pub fn finishPrefill(_: *const Self, _: *G) !void {}

        /// Each row of `x` [n, hidden] through its pair's expert (`ids` u32 [n], slot rows of `b`) at weight 1:
        /// [n, hidden] f32.
        fn pairs(limit: c_int, g: *G, x: T, ids: T, b: Bank) !T {
            const n = g.shapeOf(x).dim(0);
            const hidden = g.shapeOf(x).dim(1);
            var bank: t3.Bank = undefined;
            inline for (.{ "gate", "up", "down" }) |f| {
                const p = @field(b, f);
                var code = mlx.mlx_array_new();
                mlx.check(mlx.mlx_view(&code, p.code, .uint16, g.s)) catch |e| {
                    _ = mlx.mlx_array_free(code);
                    return e;
                };
                @field(bank, f) = .{ .trellis = try g.adopt(code), .suh = p.rin, .svh = p.rout };
            }
            const x3 = try g.reshape(try g.astype(x, .float32), &.{ 1, n, hidden });
            const inds = try g.reshape(ids, &.{ 1, n, 1 });
            const ones = try g.ones(&.{ 1, n, 1 }, .float32);
            const y = try g.adopt(try t3.moeClamped(g.s, x3, bank, inds, ones, .mul1, limit));
            return g.reshape(y, &.{ n, hidden });
        }
    };
}
