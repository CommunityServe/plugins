//! The routed experts' decode on turboexl3, for the copied expert streamer (sdk/weights/streamed/exl3_source.zig). The
//! streamer's decode waves run its banked route on turboexl3's banked decode (`moeClampedBanked`): a wave's pairs over the
//! layer's three slot banks in one call, each pair one row-slot whose id is packed (bank << 24 | slot row), DEVROUTE's
//! device wave among them (patches/turboexl3-stream.patch binds both). The call runs whole at the wave's down:
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
const frac = @import("frac.zig");

/// The GEMV variants (turboexl3 `group_variants` names) the banked calls run: the gate|up GEMV's and the down GEMV's,
/// picked on the M5 Max.
pub const pick_gu = "t4";
pub const pick_dn = "t4";

/// At load, before the bank opens (`warm` builds the banked kernels at it): turboexl3's banked decode on its grouped
/// texts (up to four row-slots of one expert a weight decode, a pair whose expert no bank holds skipped), with the M = 3
/// member path (`banked_m3`: a lead with three members of an expert runs them, not the M = 4 path padded) and the
/// picked GEMV variants.
pub fn configure() error{ InvalidTurboexl3Config, RoutedT3Variant }!void {
    var c = t3.config.get().*;
    c.banked_group = 4;
    c.banked_m3 = true;
    c.banked_variant_gu = t3.groupVariantOf(pick_gu) orelse return error.RoutedT3Variant;
    c.banked_variant_dn = t3.groupVariantOf(pick_dn) orelse return error.RoutedT3Variant;
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

        /// The banked route's state: the clamp limit.
        pub const State = struct {
            limit: c_int,
            /// Cleared at each layer call by the source's `rows` (patches/turboexl3-stream.patch); nothing reads it
            /// since the prompt calls run on the stock math.
            n: usize = 0,
            /// MLX_SERVE_DSV41_FRAC: the bank holds a layer at a half-integer rate (`warm`); false, no call reads a width.
            any_frac: bool = false,
            /// A fractional cold-row chunk's gate and up arrays, from its `gateUp` to its `down`.
            parked: frac.Parked(Proj) = .{},
        };

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

        /// DEVROUTE's LUT entry for a pair whose expert no bank holds: turboexl3's skip id on its grouped decode (the
        /// row-slot runs no expert, its row 0). The miss part's row replaces it in the join.
        pub fn lutMissing(self: *const Self) u32 {
            _ = self;
            return t3.banked_skip_id;
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
            return g.adopt(try t3.moeClampedBanked(g.s, x, &tb, ids, .mul1, self.state.limit));
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
            if (self.state.any_frac and fracCode(g, bank.gate.code)) return self.fracPrefill(g, x, rows, bank);
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
        fn fracParts(self: *const Self, g: *G, x: T, rows: PrefillRows, bank: Bank, alloc: std.mem.Allocator, outs: *std.ArrayList(T), pos: *std.ArrayList(u32)) !void {
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
                const y = if (self.stock.q.expert_bf16) try g.astype(y32, .bfloat16) else y32;
                outs.appendAssumeCapacity(g.keep(y));
                pos.appendSliceAssumeCapacity(order[c0..c1]);
                try g.asyncEval(&.{y});
                flight[n_flight] = g.keep(y);
                n_flight += 1;
                g.resetTo(m);
            }
        }

        /// `fracParts` joined: a KEPT [A, hidden] in assignment-row order (the stock call's `take(concatenate(waves),
        /// argsort(positions))`, the permutation made on the host).
        fn fracPrefill(self: *const Self, g: *G, x: T, rows: PrefillRows, bank: Bank) !T {
            if (rows.slot.len == 0) return error.FracEmptyCall;
            const a = self.stock.q.a;
            var parts: std.ArrayList(T) = .empty;
            defer {
                for (parts.items) |pt| g.release(pt);
                parts.deinit(a);
            }
            var at: std.ArrayList(u32) = .empty;
            defer at.deinit(a);
            try self.fracParts(g, x, rows, bank, a, &parts, &at);
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

        /// The slot code as u16 (the same bits), tracked by the backend.
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
            if (self.state.any_frac and fracCode(g, bank.gate.code)) return self.fracParts(g, x, rows, bank, alloc, outs, pos);
            return self.stock.prefillParts(g, layer, x, rows, bank, alloc, outs, pos);
        }
    };
}
