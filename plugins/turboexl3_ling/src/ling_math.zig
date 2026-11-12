//! Ling-3.1-flash's routed-expert math on turboexl3, for the copied expert streamer (the plugin's
//! overlay/mlx_serve/sdk/weights/streamed/ling_source.zig binds it as the scheduler's math). The streamer's own EXL3
//! texts are DeepSeek-V4.1's and GLM-5.3's shapes (an expert width of 2304 or 2048, K 2 or 3); Ling's experts are 4096 x
//! 1280 at K 2.5 (2.60 bpw) or K 2 (2.10 bpw), so every routed call runs here:
//! - the decode waves (the scheduler's banked route): `moeLingBanked`, a wave's pairs over the layer's three slot banks
//!   in one call, each pair's row at weight 1;
//! - a wide prompt call's pairs: sorted by slot, in chunks of whole experts of at most `prefill_rows` pairs (two in
//!   flight), each chunk's rows gathered from the call's x in turboexl3's prepare (ROWS_PATH, src/ling_rows_path.zig:
//!   `moeLingRows`, no take and no f32 copy) and its routed pairs stored as f16 (PF_F16_PAIRS, src/ling_f16_pairs.zig),
//!   the f32 words they hold. A chunk of 16 pairs or fewer is padded to 17 with zero-score copies of its first, so every
//!   prompt pair takes turboexl3's prefill GEMM and a pair's bytes do not depend on how the bank's residency split the
//!   call (as turboexl3_ds41's half-integer route);
//! - a cold-row chunk: `moeLing` at top-1 and weight 1 over the bank's slot arrays, padded the same way.
//! - PF_DFIN's whole-call rule: a wide call takes turboexl3's fused finish only at a mean of at least
//!   `switches.dfin_call_min` (80) rows an expert (src/ling_switches.zig `dfinCallTakes`, from the call's rows, x's, and
//!   the model's top-k and experts), every other call the served chain; decided once a call, the same for each of its
//!   chunks (`moeLingWith`, `moeLingRowsWith`), the same bytes either way. A decode-width call (`down()`) takes the same
//!   rule from its own rows. Each turboexl3 call is counted by side and ending (src/ling_switches.zig `DfinCalls`);
//!   plugin.zig prints the counts (`logDfinCalls`: `[turboexl3_ling] PF_DFIN calls: ...`) after each prompt forward and
//!   every 64 decode steps.
//! - turboexl3's configuration as this plugin serves it (`configure`).
//! Each layer's SwiGLU is its own (`expert_swiglu_limit_list`: 0 none, else the clamp after the SiLU): the source sets
//! `State.layer` before each layer call, and every call of that layer reads its limit.

const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const t3 = @import("turboexl3");
const ling_predictor = @import("ling_predictor.zig");
/// ROWS_PATH: a prompt chunk's rows gathered in turboexl3's prepare (src/ling_rows_path.zig).
const ling_rows_path = @import("ling_rows_path.zig");
/// PF_F16_PAIRS: a rows-path chunk's pairs stored as f16, the same words (src/ling_f16_pairs.zig).
const ling_f16_pairs = @import("ling_f16_pairs.zig");
const switches = @import("ling_switches.zig");
const ran = @import("ling_ran.zig");

/// The model's routing, config.json's num_experts_per_tok and num_experts (plugin.zig `openBank` sets it before the
/// bank's first routed call, its warm-up): PF_DFIN's whole-call rule reads it.
pub const Routing = struct { topk: u32 = 0, experts: u32 = 0 };
pub var routing: Routing = .{};

/// PF_DFIN's whole-call choice for a call of `rows` rows: whether it takes the fused finish (src/ling_switches.zig
/// `dfinCallTakes` at `switches.dfin_call_min` rows an expert, over the model's routing). Before `routing` is set (no
/// routed call comes then) the rule's integers are 0 and the call takes the fused finish, the ending turboexl3's
/// `Config.pf_dfin` gives a call that names none.
fn dfinTakes(rows: usize) bool {
    return switches.dfinCallTakes(@intCast(rows), routing.topk, routing.experts, switches.dfin_call_min);
}

/// PF_DFIN's counts since load (src/ling_switches.zig `DfinCalls`): each turboexl3 routed call it chose an ending for
/// (a chunk of a wide call, or a `down()` call), by side and ending. `warm()`'s calls at load are not counted; the
/// engine's warm-up prompt's are, like any request's.
var dfin_counts: switches.DfinCalls = .{};

/// PF_DFIN's counts line (`DfinCalls.line`). plugin.zig prints it after each prompt forward, every 64 decode steps, and
/// at decode's last step when the next prompt begins.
pub fn logDfinCalls(at: switches.DfinAt) void {
    var buf: [512]u8 = undefined;
    std.log.info("{s}", .{dfin_counts.line(&buf, at, switches.dfin_call_min)});
}

/// `t3.moeLing` with PF_DFIN's whole-call choice (`dfin`: the fused finish, or the served chain): `t3.moeLingWith`
fn moeLingCall(s: mlx.mlx_stream, x: mlx.mlx_array, bank: t3.Bank, inds: mlx.mlx_array, scores: mlx.mlx_array, limit: c_int, dfin: bool) !mlx.mlx_array {
    return t3.moeLingWith(s, x, bank, inds, scores, .mul1, limit, .{ .dfin = dfin });
}

/// A wide call's pairs a `moeLing` call: bounds a chunk's working set (its rows in bf16 and f32, turboexl3's
/// intermediates, its f32 output) at about half a GiB, two chunks in flight; not a speed setting.
pub const prefill_rows = 4096;
/// The fewest pairs a prompt call takes to `moeLing`: turboexl3 runs 16 rows or fewer on its decode kernels and more on
/// its prefill GEMM, which agree within tolerance, not byte for byte; a smaller call is padded to this many.
pub const gemm_min_rows = 17;

comptime {
    if (gemm_min_rows != t3.kernels.DECODE_ROWS_MAX + 1) @compileError("gemm_min_rows must be turboexl3's DECODE_ROWS_MAX + 1");
}

/// At load, before any turboexl3 launch is built (plugin.zig `load`, and again at the bank's open, the overlay's
/// ling_source.zig): turboexl3's configuration as this plugin serves it, each field the served stack sets, in this order.
/// Every other field keeps turboexl3's default.
pub fn configure() error{InvalidTurboexl3Config}!void {
    var c = t3.config.get().*;
    // The banked decode on turboexl3's grouped texts, as turboexl3_ds41 serves it: up to four row-slots of one expert a
    // weight decode, with the M = 3 member path.
    c.banked_group = 4;
    c.banked_m3 = true;
    // ONE_ROW_TABLE: a one-row trunk linear on its shape's entry of turboexl3's `dense.one_row_table` (the M5 Max's
    // fastest one-row launch for each Ling-3.1-flash trunk shape: output tiles a threadgroup, and DEC-ONE's prepare inside
    // the launch or in its own); the shapes it does not name keep the rest. Launch shape only: the same bytes.
    c.dec_one_row_table = true;
    // BANKED_T4: the grouped banked GEMVs (gate|up and down) on variant t4 of turboexl3's `banked.group_variants`, another
    // schedule of the same loads, decodes and sums, so the same bytes. It runs on the grouped texts, `banked_group` 4
    // above. turboexl3 prints `[turboexl3] banked variant engaged: ...` at the first routed call of each GEMV on it.
    const t4 = t3.banked.groupVariantOf("t4") orelse return error.InvalidTurboexl3Config;
    c.banked_variant_gu = t4;
    c.banked_variant_dn = t4;
    // MLA_LATENT_STAGE and MLA_LATENT_NAX: launch 2 of the MTP head's latent attention (`ling.mlaLatentDecode`) on the
    // matrix units at every call that launch takes, which moves only the head's drafts (the exact verify decides the
    // text), and at the other calls with each 32-key sub-chunk staged in f32 once a threadgroup, launch 2's words.
    // turboexl3 prints `[turboexl3] mla latent attn engaged: ...` at the first call on each.
    c.mla_latent_stage = true;
    c.mla_latent_nax = true;
    // MLA_ROPE_ONCE: a prompt call over the rope-once MLA cache (src/ling_mla_rope1.zig) runs turboexl3's
    // `ling.mlaFlashPrefillRope1`, the flash launch's words over the expanded key.
    c.mla_rope_once = true;
    // PF_DFIN: the routed prompt chain's finish as the down GEMM's epilogue, one launch writing y, at every call that
    // takes it (`dfinTakes`); every other call keeps the served chain.
    c.pf_dfin = true;
    // ROUTER_ROWS: the router's launch at a verify's rows (2 or more), each router threadgroup computing its 16 logits for
    // up to 4 rows, so the f16 router weight is read once for every 4 rows; the same bytes. turboexl3 prints
    // `[turboexl3] router rows engaged: ...` at the first such call.
    c.router_rows = true;
    try t3.configure(c);
}

/// The streamer's math over backend `G` (its `ops.MlxOps`), slot arrays `Proj` ({code, rout, rin}), a bank's three
/// `Bank` and a prompt call's `PrefillRows`.
pub fn Math(comptime G: type, comptime Proj: type, comptime Bank: type, comptime PrefillRows: type) type {
    return struct {
        const Self = @This();
        const T = G.T;
        /// The widest cold-row chunk (the scheduler's decode-lane width).
        pub const max_decode_rows: u32 = 48;
        /// A decode-width forward's rows: 6 x top-8 = 48 routed ids, one route (`policy.max_route_ids`).
        pub const decode_forward_rows: u32 = 6;
        /// The most pairs a banked call takes: one route.
        pub const max_banked_pairs: u32 = 48;
        pub const has_banked = true;
        pub const has_parts = true;

        state: *State,

        pub const State = struct {
            /// Each bank layer's clamp (0: none), set at the source's open.
            limits: []const c_int,
            /// The bank layer of the call in flight, set by the source before each layer call.
            layer: u32 = 0,
            /// A cold-row chunk's gate and up arrays, from its `gateUp` to its `down` (one chunk at a time).
            parked: ?struct { key: ?*anyopaque, gate: Proj, up: Proj } = null,
        };

        fn limit(self: *const Self) c_int {
            return self.state.limits[self.state.layer];
        }

        /// The read-ahead predictor's scores (experts.zig `scoresOf`): Ling's router key (PREDICTOR,
        /// src/ling_predictor.zig).
        pub fn predictorScores(g: *G, xf: T, w: T, bias: T) !T {
            return ling_predictor.lingScores(G, g, xf, w, bias);
        }

        /// A cold-row chunk's gate|up: nothing runs yet; its rows (bf16) and its gate and up arrays are kept for `down`,
        /// which runs the whole expert.
        pub fn gateUp(self: *const Self, g: *G, x: T, ids: T, gate: Proj, up: Proj) !T {
            _ = ids;
            const xb = if (g.dtypeOf(x) == .bfloat16) x else try g.astype(x, .bfloat16);
            self.state.parked = .{ .key = xb.ctx, .gate = gate, .up = up };
            return xb;
        }

        /// The chunk's rows through their slots' experts on the prompt path (`gemm`): f32 [rows, hidden] at weight 1. Not a
        /// wide call: PF_DFIN's whole-call rule takes its own rows for the call's (a side under any cutoff but 0),
        /// counted as a decode-width call. The copied streamer calls it for a decode wave's group off its banked route
        /// and for a wide call's cold-row chunks; this plugin's source runs the banked route (`downBanked`) and sets no
        /// cold rows, so its count stays 0 while it serves (the `PF_DFIN calls:` line's `decode_width_*`).
        pub fn down(self: *const Self, g: *G, h: T, ids: T, d: Proj) !T {
            const p = self.state.parked orelse return error.LingDownWithoutGateUp;
            if (p.key != h.ctx) return error.LingDownWithoutGateUp;
            self.state.parked = null;
            const rows: usize = @intCast(g.shapeOf(ids).dim(0));
            const dfin = dfinTakes(rows);
            const y = try self.gemm(g, h, ids, rows, try t3View(g, .{ .gate = p.gate, .up = p.up, .down = d }), dfin);
            dfin_counts.count(.decode_width, dfin);
            return y;
        }

        /// A wave's gate|up: nothing runs yet; `x` [pairs, hidden] is what `downBanked` receives.
        pub fn gateUpBanked(self: *const Self, g: *G, x: T, ids: T, banks: *const [3]Bank) !T {
            _ = self;
            _ = g;
            _ = ids;
            _ = banks;
            return x;
        }

        /// The wave over the three banks (base, ext, transient), whole: `ids` u32 [pairs] packed (bank << 24 | slot row)
        /// -> [pairs, hidden] f32, each pair's row at weight 1.
        pub fn downBanked(self: *const Self, g: *G, x: T, ids: T, banks: *const [3]Bank) !T {
            const tb = t3Banks(banks);
            return g.adopt(try t3.moeLingBanked(g.s, x, &tb, ids, .mul1, self.limit()));
        }

        /// A wide prompt call's waves, joined: a KEPT [A, hidden] in assignment-row order.
        pub fn prefill(self: *const Self, g: *G, layer: u32, x: T, rows: PrefillRows, bank: Bank) !T {
            _ = layer;
            if (rows.slot.len == 0) return error.LingEmptyCall;
            const a = std.heap.c_allocator;
            var parts: std.ArrayList(T) = .empty;
            defer {
                for (parts.items) |pt| g.release(pt);
                parts.deinit(a);
            }
            var at: std.ArrayList(u32) = .empty;
            defer at.deinit(a);
            try self.chunks(g, x, rows, bank, a, &parts, &at);
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
            _ = self;
            _ = g;
        }

        /// The unjoined prefill: each chunk's KEPT output appended to `outs` (the caller releases it), each of its rows'
        /// assignment rows to `pos`.
        pub fn prefillParts(self: *const Self, g: *G, layer: u32, x: T, rows: PrefillRows, bank: Bank, alloc: std.mem.Allocator, outs: *std.ArrayList(T), pos: *std.ArrayList(u32)) !void {
            _ = layer;
            return self.chunks(g, x, rows, bank, alloc, outs, pos);
        }

        /// The call's pairs (x's rows, or `rows.act_row`'s) sorted by slot (a counting sort), then in chunks of whole experts
        /// of at most `prefill_rows` pairs (one expert wider: alone), each one `rowsCall` over the bank's slot arrays; a
        /// chunk's ops between a mark and a `resetTo` after it is submitted, async-evaluated, at most two chunks in
        /// flight. PF_DFIN's whole-call choice is the wide call's: x [R, hidden] is the call's input in each of its waves,
        /// so every chunk of every wave takes the side of R.
        fn chunks(self: *const Self, g: *G, x: T, rows: PrefillRows, bank: Bank, alloc: std.mem.Allocator, outs: *std.ArrayList(T), pos: *std.ArrayList(u32)) !void {
            const n = rows.slot.len;
            if (n == 0) return;
            const dfin = dfinTakes(@intCast(g.shapeOf(x).dim(0)));
            const a = std.heap.c_allocator;
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
            // ROWS_PATH: each chunk's rows gathered in turboexl3's prepare from the call's x in bf16 (cast once where it is
            // not: the chunks' own rounding)
            const xall = if (g.dtypeOf(x) == .bfloat16) x else try g.astype(x, .bfloat16);
            var flight: [2]T = undefined;
            var n_flight: usize = 0;
            defer for (flight[0..n_flight]) |fl| g.release(fl);
            var c0: usize = 0;
            while (c0 < n) {
                // A chunk ends only between experts (slots): an expert's rows always share one call, so how turboexl3
                // tiles them (its 64- and 128-row tiles, the register path's remainder) depends on the expert's rows in
                // the layer's call alone, never on where `prefill_rows` would cut them, which follows the residency (the
                // bank's slot order). A first expert wider than `prefill_rows` takes a chunk of its own.
                var c1 = c0;
                while (c1 < n) {
                    var run_end = c1 + 1;
                    while (run_end < n and sorted_slot[run_end] == sorted_slot[c1]) run_end += 1;
                    if (c1 > c0 and run_end - c0 > prefill_rows) break;
                    c1 = run_end;
                }
                defer c0 = c1;
                const mc: c_int = @intCast(c1 - c0);
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
                const y = try self.rowsCall(g, xall, ai, sorted_act[c0..c1], sorted_slot[c0..c1], tb, dfin);
                dfin_counts.count(.wide, dfin);
                outs.appendAssumeCapacity(g.keep(y));
                pos.appendSliceAssumeCapacity(order[c0..c1]);
                try g.asyncEval(&.{y});
                flight[n_flight] = g.keep(y);
                n_flight += 1;
                g.resetTo(m);
            }
        }

        /// `rows` pairs (`xb` bf16 [rows, hidden], `slot` u32 [rows]) through `moeLing` at top-1 and weight 1, x read in
        /// f32 -> f32 [rows, hidden], a call of 16 pairs or fewer padded to `gemm_min_rows` with copies of its first pair
        /// at score 0 (their rows dropped after). `dfin`: PF_DFIN's whole-call choice.
        fn gemm(self: *const Self, g: *G, xb: T, slot: T, rows: usize, tb: t3.Bank, dfin: bool) !T {
            if (rows == 0) return error.LingEmptyCall;
            const hidden: c_int = @intCast(g.shapeOf(xb).dim(1));
            const small = !t3.kernels.usesPrefillArm(rows);
            const n_call: usize = if (small) gemm_min_rows else rows;
            const nc: c_int = @intCast(n_call);
            var xc = xb;
            var sc_ids = slot;
            var sc: T = undefined;
            if (small) {
                var idx: [gemm_min_rows]u32 = undefined;
                var w: [gemm_min_rows]f32 = undefined;
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
            const y3 = try g.adopt(try moeLingCall(g.s, x3, tb, inds, sc, self.limit(), dfin));
            const y_all = try g.reshape(y3, &.{ nc, hidden });
            if (!small) return y_all;
            return g.slice(y_all, &.{ 0, 0 }, &.{ @intCast(rows), hidden }, &.{ 1, 1 });
        }

        /// ROWS_PATH: a chunk's pairs through turboexl3's prompt call over gathered rows (`moeLingRows`) over the call's
        /// bf16 x (`xb`), the row map its act rows (`ai`), weight 1 -> [rows, hidden], with the pairs stored as f16
        /// (PF_F16_PAIRS) where turboexl3 takes the call, else f32; a chunk of 16 pairs or fewer padded to
        /// `gemm_min_rows` with copies of its first pair at score 0 (their rows dropped after), as `gemm` pads. Every
        /// row's bytes are `gemm`'s over the chunk's take (turboexl3's models/ling/rows_test.zig). `dfin`: PF_DFIN's
        /// whole-call choice.
        fn rowsCall(self: *const Self, g: *G, xb: T, ai: T, act: []const u32, slot: []const u32, tb: t3.Bank, dfin: bool) !T {
            const rows = act.len;
            if (rows == 0) return error.LingEmptyCall;
            const hidden: c_int = @intCast(g.shapeOf(xb).dim(1));
            const small = !t3.kernels.usesPrefillArm(rows);
            var rm = ai;
            var sid: T = undefined;
            var sc: T = undefined;
            if (small) {
                var r: [gemm_min_rows]u32 = undefined;
                var sl: [gemm_min_rows]u32 = undefined;
                var w: [gemm_min_rows]f32 = undefined;
                for (&r, &sl, &w, 0..) |*rv, *sv, *wv, i| {
                    const j = if (i < rows) i else 0;
                    rv.* = act[j];
                    sv.* = slot[j];
                    wv.* = if (i < rows) 1.0 else 0.0;
                }
                const nc: c_int = gemm_min_rows;
                rm = try g.hostArray(std.mem.sliceAsBytes(r[0..]), &.{nc}, .uint32);
                sid = try g.hostArray(std.mem.sliceAsBytes(sl[0..]), &.{nc}, .uint32);
                sc = try g.hostArray(std.mem.sliceAsBytes(w[0..]), &.{nc}, .float32);
            } else {
                const mc: c_int = @intCast(rows);
                sid = try g.hostArray(std.mem.sliceAsBytes(slot), &.{mc}, .uint32);
                sc = try g.ones(&.{mc}, .float32);
            }
            ran.mark(.ROWS_PATH);
            // PF_F16_PAIRS (src/ling_f16_pairs.zig): the chunk's pairs stored as f16, the words of the f32 call below
            // (top-1 at weight 1; +0 for a pad row). A call turboexl3 refuses (weights not all 1 or +0) keeps the f32 call.
            const n_call: usize = if (small) gemm_min_rows else rows;
            const wbuf = try std.heap.c_allocator.alloc(f32, n_call);
            defer std.heap.c_allocator.free(wbuf);
            const ws = ling_f16_pairs.weightsInto(wbuf, rows, small, gemm_min_rows);
            if (try ling_f16_pairs.call(g.s, xb, rm, tb, sid, ws, self.limit(), dfin)) |y16| {
                ran.mark(.PF_F16_PAIRS);
                const yh = try g.adopt(y16);
                if (!small) return yh;
                return g.slice(yh, &.{ 0, 0 }, &.{ @intCast(rows), hidden }, &.{ 1, 1 });
            }
            const y = try g.adopt(try ling_rows_path.call(g.s, xb, rm, tb, sid, sc, self.limit(), dfin));
            if (!small) return y;
            return g.slice(y, &.{ 0, 0 }, &.{ @intCast(rows), hidden }, &.{ 1, 1 });
        }

        /// A bank's three projections as turboexl3's stacked bank: code viewed as u16 (the same bits), rin as suh, rout
        /// as svh.
        fn t3View(g: *G, bank: Bank) !t3.Bank {
            var tb: t3.Bank = undefined;
            inline for (.{ "gate", "up", "down" }) |f| {
                const pj = @field(bank, f);
                @field(tb, f) = .{ .trellis = try viewU16(g, pj.code), .suh = pj.rin, .svh = pj.rout };
            }
            return tb;
        }

        /// The banks as turboexl3's banked decode takes them: code (I16, read as it is) as the trellis, rin as suh, rout
        /// as svh.
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

        fn viewU16(g: *G, code: T) !T {
            var v = mlx.mlx_array_new();
            mlx.check(mlx.mlx_view(&v, code, .uint16, g.s)) catch |e| {
                _ = mlx.mlx_array_free(v);
                return e;
            };
            return g.adopt(v);
        }

        /// At load, before any request: the banked decode's kernels for every pair count compiled, and each limit's
        /// prompt kernels, on the first layer of each limit (a pair count is a template constant of its kernels).
        pub fn warm(self: *const Self, g: *G, banks: []const [3]?Bank, hidden: c_int) !struct { calls: u32, ms: u64 } {
            var calls: u32 = 0;
            const t0 = nowNs();
            var seen: [4]c_int = @splat(-1);
            var n_seen: usize = 0;
            const saved = self.state.layer;
            defer self.state.layer = saved;
            // PF_DFIN: both endings of a prompt call, the served chain's and the fused finish's (a wide call takes either)
            const sides = [_]bool{ false, true };
            for (banks, 0..) |lb, li| {
                const b = lb[0] orelse continue;
                const lim = self.state.limits[li];
                if (std.mem.indexOfScalar(c_int, seen[0..n_seen], lim) != null) continue;
                if (n_seen == seen.len) break;
                seen[n_seen] = lim;
                n_seen += 1;
                self.state.layer = @intCast(li);
                const three = [3]Bank{ b, b, b };
                for (1..max_banked_pairs + 1) |r| {
                    const m = g.mark();
                    defer g.resetTo(m);
                    const x = try g.zeros(&.{ @intCast(r), hidden }, .bfloat16);
                    const ids = try g.zeros(&.{@intCast(r)}, .uint32);
                    try g.evalAll(&.{try self.downBanked(g, try self.gateUpBanked(g, x, ids, &three), ids, &three)});
                    calls += 1;
                }
                const m = g.mark();
                defer g.resetTo(m);
                const x = try g.zeros(&.{ gemm_min_rows, hidden }, .bfloat16);
                const ids = try g.zeros(&.{gemm_min_rows}, .uint32);
                for (sides) |side| {
                    try g.evalAll(&.{try self.gemm(g, x, ids, gemm_min_rows, try t3View(g, b), side)});
                    calls += 1;
                }
                // ROWS_PATH: the row-map prepare too (the rest of the chain is the calls' above)
                const ones = try g.ones(&.{gemm_min_rows}, .float32);
                try g.evalAll(&.{try g.adopt(try ling_rows_path.call(g.s, x, ids, try t3View(g, b), ids, ones, self.limit(), sides[0]))});
                calls += 1;
                // PF_F16_PAIRS: the f16 store's endings too (each ending a wide call takes)
                for (sides) |side| {
                    var wv: [gemm_min_rows]f32 = undefined;
                    const ws = ling_f16_pairs.weightsInto(&wv, gemm_min_rows, false, gemm_min_rows);
                    if (try ling_f16_pairs.call(g.s, x, ids, try t3View(g, b), ids, ws, self.limit(), side)) |y16| {
                        try g.evalAll(&.{try g.adopt(y16)});
                        calls += 1;
                    }
                }
            }
            return .{ .calls = calls, .ms = (nowNs() - t0) / std.time.ns_per_ms };
        }

        fn nowNs() u64 {
            var ts: std.c.timespec = undefined;
            _ = std.c.clock_gettime(.MONOTONIC, &ts);
            return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
        }
    };
}
