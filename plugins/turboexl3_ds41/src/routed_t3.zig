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
//!
//! `MLX_SERVE_DSV41_FRAC` (src/frac.zig): a layer at a half-integer EXL3 rate (code tiles of 24 or 40 halfwords, which
//! the DIG-X lane has no texts for) runs every call here on turboexl3 instead: a wide prompt call's pairs (`prefill`,
//! `prefillParts`) sorted by slot, as `moeClamped` calls at top-1 and weight 1 over the bank's slot arrays of at most
//! `frac.prefill_rows` pairs each (two in flight), each routed row through its slot's expert (in its assignment row, in
//! the lane's dtype), and a cold-row chunk (`gateUp` then `down`) the same way at its down: every prompt pair on
//! turboexl3's prefill GEMM (`fracGemm`; a call of 16 pairs or fewer padded into it with zero-score rows), so a pair's
//! bytes do not depend on how the bank's residency split the call. The decode waves stay banked. A call's layer is told
//! by its code array's tile width; a bank without such a layer (`State.any_frac`, set by `warm`) never reads it, so
//! every call goes to the stock math exactly as before.

const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const t3 = @import("turboexl3");
const t3_group = @import("t3_group.zig");
const hw4b = @import("hw4b.zig");
const hw2 = @import("hw2.zig");
const frac = @import("frac.zig");
const exact = @import("one_exact.zig");

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
            /// MLX_SERVE_DSV41_FRAC: the bank holds a layer at a half-integer rate (`warm`); false, no call reads a width.
            any_frac: bool = false,
            /// A fractional cold-row chunk's gate and up arrays, from its `gateUp` to its `down`.
            parked: frac.Parked(Proj) = .{},

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

        /// The stock math's decode GEMV (a wide prompt call's cold rows; the decode waves are banked). A fractional layer's
        /// (MLX_SERVE_DSV41_FRAC): the rows, bf16, its gate and up parked for its `down`, which runs the whole expert.
        pub fn gateUp(self: *const Self, g: *G, x: T, ids: T, gate: Proj, up: Proj) !T {
            if (self.state.any_frac and fracCode(g, gate.code)) {
                const xb = if (g.dtypeOf(x) == .bfloat16) x else try g.astype(x, .bfloat16);
                self.state.parked.put(xb.ctx, gate, up);
                return xb;
            }
            return self.stock.gateUp(g, x, ids, gate, up);
        }

        /// A fractional layer's: the parked gate and up with `d` as one bank, the chunk's rows through their slots' experts
        /// on the hot rows' path (`fracGemm`: turboexl3's prefill GEMM, a call of 16 pairs or fewer padded into it), so a
        /// pair's bytes are the same whether the bank held its expert (the hot call) or read it for the chunk (here): f32
        /// [rows, hidden] at weight 1 as the stock down returns them, each value the hot path's (rounded to bf16 where the
        /// lane stores its expert outputs in bf16).
        pub fn down(self: *const Self, g: *G, h: T, ids: T, d: Proj) !T {
            if (self.state.any_frac and fracCode(g, d.code)) {
                const gu = self.state.parked.take(h.ctx) orelse return error.FracDownWithoutGateUp;
                const b: Bank = .{ .gate = gu.gate, .up = gu.up, .down = d };
                const rows: usize = @intCast(g.shapeOf(ids).dim(0));
                const tb = try t3View(g, b);
                const y32 = try self.fracGemm(g, h, ids, rows, tb);
                const y = if (self.stock.q.expert_bf16) try g.astype(try g.astype(y32, .bfloat16), .float32) else y32;
                frac.stats.cold_calls += 1;
                frac.stats.cold_rows += rows;
                if (frac.mode == .check) try self.checkRows(g, 0, h, ids, null, y32, b, .cold);
                return y;
            }
            return self.stock.down(g, h, ids, d);
        }

        /// A bank's three projections as turboexl3's moeClamped takes them: code viewed as u16 (the same bits), rin as
        /// suh, rout as svh.
        fn t3View(g: *G, bank: Bank) !t3.Bank {
            var tb: t3.Bank = undefined;
            inline for (.{ "gate", "up", "down" }) |f| {
                const pj = @field(bank, f);
                @field(tb, f) = .{ .trellis = try viewU16(g, pj.code), .suh = pj.rin, .svh = pj.rout };
            }
            return tb;
        }

        /// MLX_SERVE_DSV41_FRAC's one path for a fractional layer's prompt pairs: `rows` pairs (`xb` bf16 [rows, hidden],
        /// `slot` u32 [rows], each pair's slot in `tb`) through turboexl3's moeClamped at top-1 and weight 1, x read in
        /// f32 -> f32 [rows, hidden]. moeClamped runs its decode kernels at 16 rows or fewer (`kernels.DECODE_ROWS_MAX`)
        /// and its prefill GEMM above, and the two agree within tolerance, not byte for byte; how many pairs a call holds
        /// follows the bank's residency (its groups, its hot and cold rows), so a call of 16 pairs or fewer is padded to
        /// `frac.gemm_min_rows` with copies of its first pair at score 0 (their rows dropped after): every pair takes the
        /// GEMM, whose row bytes do not depend on the call's other rows (turboexl3 prefill.zig), and a pair's bytes do not
        /// depend on how the call was split.
        fn fracGemm(self: *const Self, g: *G, xb: T, slot: T, rows: usize, tb: t3.Bank) !T {
            comptime {
                if (frac.gemm_min_rows != t3.kernels.DECODE_ROWS_MAX + 1) @compileError("frac.gemm_min_rows must be turboexl3's DECODE_ROWS_MAX + 1");
            }
            if (rows == 0) return error.FracEmptyCall;
            const hidden: c_int = @intCast(g.shapeOf(xb).dim(1));
            const small = !t3.kernels.usesPrefillArm(rows);
            const n_call: usize = if (small) frac.gemm_min_rows else rows;
            const nc: c_int = @intCast(n_call);
            var xc = xb;
            var sc_ids = slot;
            var sc: T = undefined;
            if (small) {
                // the call's rows, then copies of its first at score 0
                var idx: [frac.gemm_min_rows]u32 = undefined;
                var w: [frac.gemm_min_rows]f32 = undefined;
                for (&idx, &w, 0..) |*ix, *wt, i| {
                    ix.* = if (i < rows) @intCast(i) else 0;
                    wt.* = if (i < rows) 1.0 else 0.0;
                }
                const ia = try g.hostArray(std.mem.sliceAsBytes(idx[0..]), &.{nc}, .uint32);
                xc = try g.take(xb, ia, 0);
                sc_ids = try g.take(slot, ia, 0);
                sc = try g.hostArray(std.mem.sliceAsBytes(w[0..]), &.{ 1, nc, 1 }, .float32);
                frac.stats.padded_calls += 1;
            } else sc = try g.ones(&.{ 1, nc, 1 }, .float32);
            const x3 = try g.reshape(try g.astype(xc, .float32), &.{ 1, nc, hidden });
            const inds = try g.reshape(sc_ids, &.{ 1, nc, 1 });
            const y3 = try g.adopt(try t3.moeClamped(g.s, x3, tb, inds, sc, .mul1, self.state.limit));
            const y_all = try g.reshape(y3, &.{ nc, hidden });
            if (!small) return y_all;
            return g.slice(y_all, &.{ 0, 0 }, &.{ @intCast(rows), hidden }, &.{ 1, 1 });
        }

        /// Whether `code`'s tiles are a fractional width (off the DIG-X lane's 32 and 48 halfwords).
        fn fracCode(g: *G, code: T) bool {
            return !frac.isDigx(@intCast(g.shapeOf(code).dim(3)));
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
            // MLX_SERVE_DSV41_HW2's check, or MLX_SERVE_DSV41_HW4B's call (its check, or turboexl3's call as configured);
            // one check mode at a time (the load refuses both)
            const yb = if (hw2.mode == .check) try hw2.banked(g.s, x, &tb, ids, .mul1, self.state.limit) else try hw4b.banked(g.s, x, &tb, ids, .mul1, self.state.limit);
            const y = try g.adopt(yb);
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
            // Each code width's first layer, in layer order (a K3 / K2 bank: its first K3 layer, then its first K2, as
            // before); MLX_SERVE_DSV41_FRAC's half-integer widths too, and the bank's widths named in a load line.
            var seen: [4]u32 = @splat(0);
            var n_seen: usize = 0;
            var widths: [64]u32 = @splat(0);
            for (banks, 0..) |lb, li| {
                const b = lb[0] orelse continue;
                const w: u32 = @intCast(g.shapeOf(b.gate.code).dim(3));
                if (li < widths.len) widths[li] = w;
                if (!frac.isDigx(w)) {
                    if (!frac.served(w)) return error.FracWidthNotServed;
                    self.state.any_frac = true;
                }
                if (std.mem.indexOfScalar(u32, seen[0..n_seen], w) != null) continue;
                if (n_seen == seen.len) return error.RoutedT3TooManyWidths;
                seen[n_seen] = w;
                n_seen += 1;
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
            if (self.state.any_frac) {
                var lb: [512]u8 = undefined;
                std.debug.print("{s}", .{frac.loadLine(&lb, widths[0..@min(banks.len, widths.len)])});
            }
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
            if (self.state.any_frac and fracCode(g, bank.gate.code)) return self.fracPrefill(g, layer, x, rows, bank);
            return self.stock.prefill(g, layer, x, rows, bank);
        }

        /// A fractional layer's wide call (MLX_SERVE_DSV41_FRAC), unjoined: the call's pairs (x's rows, or `rows.act_row`'s)
        /// sorted by slot (a counting sort, each slot's pairs in their order), then in chunks of at most
        /// `frac.prefill_rows` each one turboexl3 `moeClamped` call at top-1 and weight 1 over the bank's slot arrays (code
        /// read as u16, rin as suh, rout as svh), x read in f32 as ROUTED_ONE's call reads it. The stock waves' lifecycle:
        /// a chunk's ops between a mark and a `resetTo` right after it is submitted (its intermediates live only while the
        /// GPU still needs them), async-evaluated, at most two chunks in flight. Each chunk's KEPT output (f32 [rows,
        /// hidden], rounded to bf16 where the lane stores its expert outputs in bf16: `routeExpertBf16`) is appended to
        /// `outs` (the caller releases it) and each of its rows' assignment rows to `pos`.
        fn fracParts(self: *const Self, g: *G, layer: u32, x: T, rows: PrefillRows, bank: Bank, alloc: std.mem.Allocator, outs: *std.ArrayList(T), pos: *std.ArrayList(u32)) !void {
            const t0 = nowNs();
            const n = rows.slot.len;
            if (n == 0) return;
            const a = self.stock.q.a;
            var max_slot: u32 = 0;
            for (rows.slot) |sv| max_slot = @max(max_slot, sv);
            const starts = try a.alloc(u32, @as(usize, max_slot) + 2);
            defer a.free(starts);
            @memset(starts, 0);
            for (rows.slot) |sv| starts[sv + 1] += 1;
            for (1..starts.len) |i| starts[i] += starts[i - 1];
            const order = try a.alloc(u32, n);
            defer a.free(order);
            for (rows.slot, 0..) |sv, i| {
                order[starts[sv]] = @intCast(i);
                starts[sv] += 1;
            }
            const sorted_slot = try a.alloc(u32, n);
            defer a.free(sorted_slot);
            const sorted_act = try a.alloc(u32, n);
            defer a.free(sorted_act);
            for (order, sorted_slot, sorted_act) |pi, *so, *ao| {
                so.* = rows.slot[pi];
                ao.* = if (rows.act_row) |act| act[pi] else pi;
            }
            const m0 = g.mark();
            defer g.resetTo(m0);
            const tb = try t3View(g, bank);
            var flight: [2]T = undefined;
            var n_flight: usize = 0;
            defer for (flight[0..n_flight]) |fl| g.release(fl);
            var c0: usize = 0;
            while (c0 < n) : (c0 += frac.prefill_rows) {
                const c1 = @min(c0 + frac.prefill_rows, n);
                const mc: c_int = @intCast(c1 - c0);
                // at most one older chunk in flight while this one is encoded
                if (n_flight == flight.len) {
                    try g.evalAll(flight[0..1]);
                    g.release(flight[0]);
                    flight[0] = flight[1];
                    n_flight = 1;
                }
                try outs.ensureUnusedCapacity(alloc, 1);
                try pos.ensureUnusedCapacity(alloc, c1 - c0);
                const m = g.mark();
                const ai = try g.hostArray(std.mem.sliceAsBytes(sorted_act[c0..c1]), &.{mc}, .uint32);
                const xr = try g.take(x, ai, 0);
                const xb = if (g.dtypeOf(xr) == .bfloat16) xr else try g.astype(xr, .bfloat16);
                const sid = try g.hostArray(std.mem.sliceAsBytes(sorted_slot[c0..c1]), &.{mc}, .uint32);
                const y32 = try self.fracGemm(g, xb, sid, c1 - c0, tb);
                if (frac.mode == .check) try self.checkRows(g, layer, xb, sid, sorted_slot[c0..c1], y32, bank, .prefill);
                const y = if (self.stock.q.expert_bf16) try g.astype(y32, .bfloat16) else y32;
                outs.appendAssumeCapacity(g.keep(y));
                pos.appendSliceAssumeCapacity(order[c0..c1]);
                try g.asyncEval(&.{y});
                flight[n_flight] = g.keep(y);
                n_flight += 1;
                g.resetTo(m);
                frac.stats.prefill_chunks += 1;
            }
            frac.stats.prefill_calls += 1;
            frac.stats.prefill_rows += n;
            frac.stats.prefill_ns += nowNs() - t0;
        }

        /// `fracParts` joined: a KEPT [A, hidden] in assignment-row order (the stock call's `take(concatenate(waves),
        /// argsort(positions))`, the permutation made on the host).
        fn fracPrefill(self: *const Self, g: *G, layer: u32, x: T, rows: PrefillRows, bank: Bank) !T {
            if (rows.slot.len == 0) return error.FracEmptyCall;
            const a = self.stock.q.a;
            var parts: std.ArrayList(T) = .empty;
            defer {
                for (parts.items) |pt| g.release(pt);
                parts.deinit(a);
            }
            var at: std.ArrayList(u32) = .empty;
            defer at.deinit(a);
            try self.fracParts(g, layer, x, rows, bank, a, &parts, &at);
            const m = g.mark();
            defer g.resetTo(m);
            const joined = if (parts.items.len == 1) parts.items[0] else try g.concat(parts.items, 0);
            const inv = try a.alloc(u32, at.items.len);
            defer a.free(inv);
            for (at.items, 0..) |pi, j| inv[pi] = @intCast(j);
            const ord = try g.hostArray(std.mem.sliceAsBytes(inv), &.{@intCast(inv.len)}, .uint32);
            return g.keep(try g.take(joined, ord, 0));
        }

        pub fn finishPrefill(self: *const Self, g: *G) !void {
            return self.stock.finishPrefill(g);
        }

        const CheckKind = enum { prefill, cold };
        const Ex = exact.Exact(t3.format);

        /// MLX_SERVE_DSV41_FRAC=check: a fractional call's rows `y` (f32 [rows, hidden], before any bf16 store) against
        /// each sampled row's expert in f64 (src/one_exact.zig over turboexl3's format, from the bank's own bytes: the
        /// slot's three projections and the clamped SwiGLU), at most `frac.check_calls` calls a layer a request
        /// (`frac.takeCheck`), `frac.check_rows` rows each. `x` the call's input rows (bf16), `ids` their slots (or
        /// `slots`, the host's copy).
        fn checkRows(self: *const Self, g: *G, layer: u32, x: T, ids: T, slots: ?[]const u32, y: T, bank: Bank, kind: CheckKind) !void {
            if (!frac.takeCheck(layer, kind == .cold)) return;
            const rows: usize = @intCast(g.shapeOf(y).dim(0));
            const hidden: usize = @intCast(g.shapeOf(y).dim(1));
            if (rows == 0) return;
            var sample: [frac.check_rows]u32 = undefined;
            var nr: usize = 0;
            const cand = [_]usize{ 0, rows / 3, (2 * rows) / 3, rows - 1 };
            next: for (cand) |r| {
                for (sample[0..nr]) |q| if (q == r) continue :next;
                sample[nr] = @intCast(r);
                nr += 1;
            }
            const a = std.heap.page_allocator;
            const sel = try g.hostArray(std.mem.sliceAsBytes(sample[0..nr]), &.{@intCast(nr)}, .uint32);
            const hy = try a.alloc(f32, 2 * nr * hidden);
            defer a.free(hy);
            const hx = hy[nr * hidden ..];
            _ = try g.hostF32(try g.astype(try g.take(y, sel, 0), .float32), hy[0 .. nr * hidden]);
            _ = try g.hostF32(try g.astype(try g.take(x, sel, 0), .float32), hx);
            var sl: [frac.check_rows]u32 = undefined;
            if (slots) |sh| {
                for (sample[0..nr], 0..) |r, i| sl[i] = sh[r];
            } else _ = try g.hostU32(try g.take(ids, sel, 0), sl[0..nr]);
            // The bank's bytes as the call read them (evaluated: the call above waited for their reads).
            try g.evalAll(&.{ bank.gate.code, bank.gate.rin, bank.gate.rout, bank.up.code, bank.up.rin, bank.up.rout, bank.down.code, bank.down.rin, bank.down.rout });
            const gp = try HostProj.of(bank.gate);
            const upp = try HostProj.of(bank.up);
            const dp = try HostProj.of(bank.down);
            const inter = gp.out;
            if (gp.in != hidden or upp.out != inter or dp.in != inter or dp.out != hidden) return error.FracCheckShape;
            const ex = Ex.init();
            const limit: f64 = @floatFromInt(self.state.limit);
            const wide = @max(hidden, inter);
            const f = try a.alloc(f64, 2 * hidden + wide + 3 * inter);
            defer a.free(f);
            const x64 = f[0..hidden];
            const ts = f[hidden..][0..wide];
            const gv = f[hidden + wide ..][0..inter];
            const uv = f[hidden + wide + inter ..][0..inter];
            const td = f[hidden + wide + 2 * inter ..][0..inter];
            const dv = f[hidden + wide + 3 * inter ..][0..hidden];
            const hf: f64 = @floatFromInt(hidden);
            var worst_rms: f64 = 0;
            var worst_ratio: f64 = 0;
            var fails: u32 = 0;
            for (0..nr) |i| {
                for (hx[i * hidden ..][0..hidden], x64) |v, *o| o.* = v;
                const s0 = sl[i];
                if (s0 >= gp.slots) return error.FracCheckSlot;
                const sg = gp.slot(s0);
                const su = upp.slot(s0);
                const sd = dp.slot(s0);
                const xs = [_][]const f64{x64};
                const tss = [_][]f64{ts};
                ex.project(sg.code, sg.suh, sg.svh, hidden, inter, gp.rate, &xs, &tss, &[_][]f64{gv});
                ex.project(su.code, su.suh, su.svh, hidden, inter, upp.rate, &xs, &tss, &[_][]f64{uv});
                for (gv, uv) |*o, uu| o.* = exact.swigluClamped(o.*, uu, limit);
                ex.project(sd.code, sd.suh, sd.svh, inter, hidden, dp.rate, &[_][]const f64{gv}, &[_][]f64{td}, &[_][]f64{dv});
                var ss_d: f64 = 0;
                var ss_r: f64 = 0;
                var max_d: f64 = 0;
                for (hy[i * hidden ..][0..hidden], dv) |got, want| {
                    const d = @as(f64, got) - want;
                    ss_d += d * d;
                    ss_r += want * want;
                    max_d = @max(max_d, @abs(d));
                }
                const rms_ref = @sqrt(ss_r / hf);
                const fl = if (rms_ref > 0) rms_ref else 1;
                const rr = @sqrt(ss_d / hf) / fl;
                const ra = max_d / fl;
                // NaN compares false: a NaN row fails.
                if (!(rr <= frac.bar_rms and ra <= frac.bar_ratio)) fails += 1;
                if (!(rr <= worst_rms)) worst_rms = rr;
                if (!(ra <= worst_ratio)) worst_ratio = ra;
            }
            frac.noteCheck(layer, nr, fails, worst_rms, worst_ratio);
            std.debug.print("[frac] check layer={d} kind={t} n={d} rows={d} sampled={d} rms_ratio={e:.3} max_ratio={e:.3} bar={e:.1}/{e:.1} {s}\n", .{ layer, kind, gp.rate.halfwords(), rows, nr, worst_rms, worst_ratio, frac.bar_rms, frac.bar_ratio, if (fails == 0) "PASS" else "FAIL" });
        }

        /// One projection of a slot bank on the host (src/routed_one.zig's): code u16 halfwords [slots, in/16, out/16, n],
        /// rin (suh) and rout (svh) f16 bits, read in place from the stream's unified-memory arrays.
        const HostProj = struct {
            code: []const u16,
            rin: []const u16,
            rout: []const u16,
            in: usize,
            out: usize,
            slots: usize,
            rate: t3.format.Rate,
            stride: usize,

            fn of(p: Proj) !HostProj {
                const sh = mlx.getShape(p.code);
                if (sh.len != 4) return error.FracCheckShape;
                const in: usize = @intCast(sh[1] * 16);
                const out: usize = @intCast(sh[2] * 16);
                const rate = t3.format.Rate.fromN(@intCast(sh[3])) orelse return error.FracCheckRate;
                return .{ .code = try hostU16(p.code), .rin = try hostU16(p.rin), .rout = try hostU16(p.rout), .in = in, .out = out, .slots = @intCast(sh[0]), .rate = rate, .stride = (in / 16) * (out / 16) * rate.halfwords() };
            }

            fn slot(p: HostProj, s: usize) struct { code: []const u16, suh: []const u16, svh: []const u16 } {
                return .{ .code = p.code[s * p.stride ..][0..p.stride], .suh = p.rin[s * p.in ..][0..p.in], .svh = p.rout[s * p.out ..][0..p.out] };
            }
        };

        fn hostU16(arr: mlx.mlx_array) ![]const u16 {
            const p = mlx.mlx_array_data_uint8(arr) orelse return error.FracCheckNoData;
            const words: [*]const u16 = @ptrCast(@alignCast(p));
            return words[0..mlx.mlx_array_size(arr)];
        }

        /// The slot code as u16 (the same bits), tracked by the backend (src/routed_one.zig's `viewU16`).
        fn viewU16(g: *G, code: T) !T {
            var v = mlx.mlx_array_new();
            mlx.check(mlx.mlx_view(&v, code, .uint16, g.s)) catch |e| {
                _ = mlx.mlx_array_free(v);
                return e;
            };
            return g.adopt(v);
        }

        /// The stock math's unjoined prefill, as it has one.
        pub const has_parts = Stock.has_parts;
        pub fn prefillParts(self: *const Self, g: *G, layer: u32, x: T, rows: PrefillRows, bank: Bank, alloc: std.mem.Allocator, outs: *std.ArrayList(T), pos: *std.ArrayList(u32)) !void {
            // A fractional layer's call: `fracParts`' chunks as its parts, each row in its assignment row.
            if (self.state.any_frac and fracCode(g, bank.gate.code)) return self.fracParts(g, layer, x, rows, bank, alloc, outs, pos);
            return self.stock.prefillParts(g, layer, x, rows, bank, alloc, outs, pos);
        }
    };
}
