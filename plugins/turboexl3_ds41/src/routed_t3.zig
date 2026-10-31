//! The routed experts' math on turboexl3, for the copied expert streamer (sdk/weights/streamed/exl3_source.zig) when
//! `MLX_SERVE_DSV41_ROUTED=turboexl3` asks for it; the default is the streamer's own EXL3 kernels. The
//! streamer's decode waves and prompt calls hand it each bank's routed pairs: every pair is one row of a top-1
//! `moeClamped` call at weight 1 on the bank's slot arrays (code read as u16, rin as suh, rout as svh), in f32, as the
//! streamer's own lane gives them. The streamer joins and weighs the pairs' rows itself.

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

        /// A decode group's gate and up arrays, from its `gateUp` to its `down`: the streamer runs a call's gate/ups
        /// before their downs, and each group's ids are an array of their own.
        pub const State = struct {
            limit: c_int,
            pending: [256]Pending = undefined,
            n: usize = 0,
        };
        const Pending = struct { ids: ?*anyopaque, gate: Proj, up: Proj };

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
            const st = self.state;
            const i = for (st.pending[0..st.n], 0..) |q, j| {
                if (q.ids == ids.ctx) break j;
            } else return error.RoutedT3NoGateUp;
            const p = st.pending[i];
            st.pending[i] = st.pending[st.n - 1];
            st.n -= 1;
            return pairs(st.limit, g, h, ids, .{ .gate = p.gate, .up = p.up, .down = d });
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

