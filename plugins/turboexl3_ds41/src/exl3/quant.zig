//! The EXL3 routed-expert quant, C2's first client (`quant`): the streamed EXL3 bank's pinned
//! Metal texts. The texts decode the mul1 codebook at K = 3 and are compiled for hidden 5120 /
//! inter 2304.
//!   - Decode (at most 48 routed rows) runs the tier of record's PREP=rin chain: in_rin, the two
//!     GEMVs, gu_epi (the clamped SwiGLU, fused), din_rin, the down GEMV, dpost. Every launch
//!     comes from a per-M table of prepared launches.
//!   - Prefill runs the DIG-X waves (DIG2 onepass, SHAPE's schedule), one wave state per layer.
//! Its kernels (`kernels`, 18 of the set) are self-checked by `accept` on the kernel set the load
//! context owns. The routes below the C2 section are the lanes' own launches, moved here verbatim
//! from exl3_kernel_ops.zig.

const std = @import("std");
const env = @import("sdk").env;
/// the waves' per-prompt regions and their twins' text derivation (the
/// plugin's scratch_waves.zig, wave_twins.zig).
const sw = @import("../bank/scratch_waves.zig");
/// the half-integer-rate layers its t3 math runs on turboexl3 (the plugin's
/// frac.zig); this lane's texts decode K 2 and 3 only.
const frac = @import("../bank/frac.zig");
const kset = @import("../streamed/sdk_ext/kernel_set.zig");
const mtl = @import("../streamed/sdk_ext/metal_launch.zig");
const mlx = @import("sdk").mlx;
const xk = @import("kernels.zig");
const selfcheck = @import("selfcheck.zig");
const kr = sdk_ext.kernels.Routes(xk);
const ks = sdk_ext.kernels.KernelSet(xk);
const quant = @import("../streamed/sdk_ext.zig").quant;
const sdk = @import("sdk");
const sdk_ext = @import("../streamed/sdk_ext.zig");

const Allocator = std.mem.Allocator;
const Kernel = xk.Kernel;
const Entry = xk.Entry;
const Vars = xk.Vars;
const LaunchConfig = xk.LaunchConfig;
const Dtype = mlx.mlx_dtype;
const Diag = xk.Diag;
const Refusal = kr.Refusal;
const refuse = kr.refuse;
const argOf = kr.argOf;
const expectInput = kr.expectInput;
const launchRule = kr.launchRule;
const rowsOf = kr.rowsOf;

/// Which text set a code array takes: 0 for K = 3 (48 int16 per tile), 1 for K = 2 (32). A layer's K reaches every
/// dispatch this way: the bank's per-layer arrays are shaped by the v2 manifest's K (the claim admits only bank_ks, and
/// the TURBOEXL3_DS41_FRAC widths, whose calls the pack's quant plugin runs: `routed.zig`). Any other width is refused
/// by name, never decoded as another K's.
fn kIndexOf(comptime G: type, g: *G, code: G.T) error{Exl3CodeWidthNotImplemented}!usize {
    const w = rowsOf(G, g, code, 3);
    return switch (w) {
        48 => 0,
        32 => 1,
        else => error.Exl3CodeWidthNotImplemented,
    };
}
const Statics = kr.Statics;
const RowPlans = kr.RowPlans;
const rowsVars = kr.rowsVars;
const no_vars = kr.no_vars;

// ── C2: the quant ──

pub const name = "exl3-mul1-k3";

/// G5: the kernel set this quant self-checks at accept is pinned by the registry's manifest.
pub const kernel_pin: sdk_ext.kernels.Pin = .{ .manifest_sha256 = xk.manifest_sha256 };

/// This consumer's subset of the kernel set: the EXL3 families (the decode GEMV, the rin stage,
/// the rebuild, DIG-X and its golden-tile check texts).
pub const kernels = [_]Kernel{
    .dsv41_exl3_mul1h_k3_2304,
    .dsv41_exl3_mul1h_k3_5120,
    .q3_exl3_prep_in_rin,
    .q3_exl3_prep_gu_epi,
    .q3_exl3_prep_din_rin,
    .q3_moeprep_dpost,
    .q3_prefill_fused_exl3x3_mul1lut_k3_bf16,
    .q3_prefill_dig_gemm_5120x2304_gu_xmul1hk3,
    .q3_prefill_dig_gemm_2304x5120_xmul1hk3,
    .dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk3_m128,
    .dsv41_prefill_dig_gemm_2304x5120_xmul1hk3_m128,
    .dsv41_prefill_dig_gemm_2304x5120_xmul1hk3_m128w1,
    .dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk3_m128lut,
    .q3_prefill_dig_rot_take2_5120,
    .dsv41_prefill_dig_take2v_5120,
    .q3_prefill_dig_rot_roundx_2304,
    .q3_prefill_dig2_swiglu_2304_x,
    .q3_prefill_dig_rot_widen2_2304,
    .q3_prefill_dig_rot_widen1_5120,
    .dsv41_prefill_dig_rot_widen1_5120_obf16,
    .q3_exl3_dig_decmat_5120x2304_mul1hk3,
    .q3_exl3_dig_decmat_2304x5120_mul1hk3,
    .q3_exl3_dig_decmat_5120x2304_mul1k3,
    .q3_exl3_dig_decmat_2304x5120_mul1k3,
    .dsv41_exl3_pair_k3_5120,
    .dsv41_exl3_guone_k3_2304,
    .dsv41_exl3_b3_mul1h_k3_2304,
    .dsv41_exl3_b3_mul1h_k3_5120,
    .dsv41_exl3_b3_prep_in_rin,
    .dsv41_exl3_b3_prep_gu_epi,
    .dsv41_exl3_b3_prep_din_rin,
    .dsv41_exl3_b3_moeprep_dpost,
    .dsv41_exl3_b3_pair_k3_5120,
    .dsv41_exl3_b3_guone_k3_2304,
    .dsv41_exl3_mul1h_k2_2304,
    .dsv41_exl3_mul1h_k2_5120,
    .dsv41_exl3_pair_k2_5120,
    .dsv41_exl3_guone_k2_2304,
    .dsv41_exl3_b3_mul1h_k2_2304,
    .dsv41_exl3_b3_mul1h_k2_5120,
    .dsv41_exl3_b3_pair_k2_5120,
    .dsv41_exl3_b3_guone_k2_2304,
    .dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk2_m128,
    .dsv41_prefill_dig_gemm_2304x5120_xmul1hk2_m128,
    .dsv41_prefill_dig_gemm_2304x5120_xmul1hk2_m128w1,
    .q3_exl3_dig_decmat_5120x2304_mul1hk2,
    .q3_exl3_dig_decmat_2304x5120_mul1hk2,
    .dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk3_m128d,
    .dsv41_prefill_dig_gemm_2304x5120_xmul1hk3_m128d,
    .dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk2_m128d,
    .dsv41_prefill_dig_gemm_2304x5120_xmul1hk2_m128d,
};

/// The routed decode forms (`DSV41_CELL_ROUTED_FORMS`): each independently selectable, chosen at construction.
pub const Forms = struct { down_pair: bool = false, gu_one: bool = false };

/// One projection's per-slot arrays: the streamer's `ProjArrays` {code, rout, rin}.
pub fn Arrays(comptime T: type) type {
    return ProjArrays(T);
}

/// The activation the texts fuse: the clamped SwiGLU at 10.0 (the literals of
/// q3_exl3_prep_gu_epi.metal:47-52 / :64-69 and of header_dig2_x.metal:34-37).
pub const fused_act: quant.Activation = .{ .swiglu_clamped = 10.0 };
/// The dims the texts are compiled for.
pub const compiled_hidden: u32 = 5120;
pub const compiled_inter: u32 = 2304;
/// The decode tables' widest call: 8 verify rows x top-k 6.
pub const decode_table_rows: u32 = 48;

/// The bank format the texts decode, field by field, as the bank manifest's `quantization`
/// object writes it (exllamav3 v1.4.2: codebook.cuh, exl3_dq.cuh, pack.cu, hadamard.cu).
pub const format = struct {
    pub const mode = "exl3";
    pub const codebook = xk.bank_codebook;
    pub const multiplier = xk.bank_multiplier;
    pub const tile_size = 16;
    pub const tile_layout = "exl3-tensor-core";
    pub const tile_bitstream = "msb-first-swap16";
    pub const hadamard_size = 128;
    pub const hadamard_order = "sylvester-natural";
    pub const hadamard_scale = "1/sqrt(128)";
    pub const scale_rin = "exl3.suh";
    pub const scale_rout = "exl3.svh";
};

fn strIs(v: std.json.Value, field: []const u8, want: []const u8) bool {
    const s = quant.str(v, field) orelse return false;
    return std.mem.eql(u8, s, want);
}

fn intIs(v: std.json.Value, field: []const u8, want: i64) bool {
    return (quant.int(v, field) orelse return false) == want;
}

/// The projections' tensors at K (bankv2 layer_segments): per projection code I16 [in/16,
/// out/16, 16 K], rout F16 [out], rin F16 [in]; gate / up map hidden -> inter, down inter -> hidden.
fn segmentOk(s: quant.Segment, k: u32) ?[]const u8 {
    return segmentOkN(s, 16 * @as(u64, k));
}
/// `segmentOk` at `n` halfwords a tile.
fn segmentOkN(s: quant.Segment, n: u64) ?[]const u8 {
    const P3 = struct { prefix: []const u8, in: u64, out: u64 };
    const projs = [_]P3{ .{ .prefix = "gate_proj.", .in = compiled_hidden, .out = compiled_inter }, .{ .prefix = "up_proj.", .in = compiled_hidden, .out = compiled_inter }, .{ .prefix = "down_proj.", .in = compiled_inter, .out = compiled_hidden } };
    for (projs) |p| {
        if (!std.mem.startsWith(u8, s.name, p.prefix)) continue;
        const part = s.name[p.prefix.len..];
        const want_dtype: []const u8, const want: []const u64 = if (std.mem.eql(u8, part, "code"))
            .{ "I16", &.{ p.in / 16, p.out / 16, n } }
        else if (std.mem.eql(u8, part, "rout"))
            .{ "F16", &.{p.out} }
        else if (std.mem.eql(u8, part, "rin"))
            .{ "F16", &.{p.in} }
        else
            return "an unknown tensor";
        if (!std.mem.eql(u8, s.dtype, want_dtype)) return "its dtype";
        if (!std.mem.eql(u64, s.shape, want)) return "its shape";
        return null;
    }
    return "an unknown projection";
}

/// At load, once per weight group: `.native` for a bank these texts decode in full (the
/// format's every field, every layer's K, the compiled dims, every layer's nine tensors), else
/// null with the first mismatch in `why`.
pub fn claims(peek: *const quant.BankPeek, why: ?*Diag) ?quant.Priority {
    const q = peek.quantization;
    if (!strIs(q, "mode", format.mode)) return quant.decline(why, "exl3 quant: quantization.mode \"{s}\" (the texts decode {s})", .{ quant.str(q, "mode") orelse "", format.mode });
    if (!strIs(q, "codebook", format.codebook)) return quant.decline(why, "exl3 quant: quantization.codebook \"{s}\" (the texts decode {s})", .{ quant.str(q, "codebook") orelse "", format.codebook });
    if (!intIs(q, "codebook_multiplier", format.multiplier)) return quant.decline(why, "exl3 quant: quantization.codebook_multiplier {?d} (mul1 is {d})", .{ quant.int(q, "codebook_multiplier"), format.multiplier });
    const tile = quant.obj(q, "tile");
    if (!intIs(tile, "size", format.tile_size) or !strIs(tile, "layout", format.tile_layout) or !strIs(tile, "bitstream", format.tile_bitstream))
        return quant.decline(why, "exl3 quant: quantization.tile {?d} / {s} / {s} (the texts read {d} / {s} / {s})", .{ quant.int(tile, "size"), quant.str(tile, "layout") orelse "", quant.str(tile, "bitstream") orelse "", format.tile_size, format.tile_layout, format.tile_bitstream });
    const had = quant.obj(q, "hadamard");
    if (!intIs(had, "size", format.hadamard_size) or !strIs(had, "order", format.hadamard_order) or !strIs(had, "scale", format.hadamard_scale))
        return quant.decline(why, "exl3 quant: quantization.hadamard {?d} / {s} / {s} (the texts rotate {d} / {s} / {s})", .{ quant.int(had, "size"), quant.str(had, "order") orelse "", quant.str(had, "scale") orelse "", format.hadamard_size, format.hadamard_order, format.hadamard_scale });
    const sc = quant.obj(q, "scales");
    if (!strIs(sc, "rin", format.scale_rin) or !strIs(sc, "rout", format.scale_rout))
        return quant.decline(why, "exl3 quant: quantization.scales rin {s} / rout {s} (the texts read {s} / {s})", .{ quant.str(sc, "rin") orelse "", quant.str(sc, "rout") orelse "", format.scale_rin, format.scale_rout });
    if (peek.hidden != compiled_hidden or peek.inter != compiled_inter)
        return quant.decline(why, "exl3 quant: dims hidden {d} / inter {d} (the texts are compiled for {d} / {d})", .{ peek.hidden, peek.inter, compiled_hidden, compiled_inter });
    if (peek.n_layers == 0 or peek.layers.len != peek.n_layers) return quant.decline(why, "exl3 quant: {d} layer entries for dims.n_layers {d}", .{ peek.layers.len, peek.n_layers });
    for (peek.layers, 0..) |l, li| {
        // a half-integer K peeks as bits 0; its width is the gate code's tiles'.
        if (l.bits == 0) {
            if (l.segments.len != 9) return quant.decline(why, "exl3 quant: layer {d} has {d} tensors (9)", .{ li, l.segments.len });
            const n = fracWidthOf(l.segments) orelse return quant.decline(why, "exl3 quant: layer {d} has neither a K nor a gate code of a turboexl3 width", .{li});
            if (!frac.served(n)) return quant.decline(why, "exl3 quant: layer {d} at {d} halfwords a tile: TURBOEXL3_DS41_FRAC={t} serves {any} on the t3 route (the texts decode K {any})", .{ li, n, frac.ps().mode, frac.admitted(), xk.bank_ks });
            for (l.segments) |s| if (segmentOkN(s, n)) |bad| return quant.decline(why, "exl3 quant: layer {d} segment {s}: {s} ({s} {any})", .{ li, s.name, bad, s.dtype, s.shape });
            continue;
        }
        if (std.mem.indexOfScalar(u32, &xk.bank_ks, l.bits) == null) return quant.decline(why, "exl3 quant: layer {d} K {d} (the texts decode K {any})", .{ li, l.bits, xk.bank_ks });
        if (l.segments.len != 9) return quant.decline(why, "exl3 quant: layer {d} has {d} tensors (9)", .{ li, l.segments.len });
        for (l.segments) |s| if (segmentOk(s, l.bits)) |bad| return quant.decline(why, "exl3 quant: layer {d} segment {s}: {s} ({s} {any})", .{ li, s.name, bad, s.dtype, s.shape });
    }
    return .native;
}

/// A fractional layer's tile width: its gate code's last dim (null without a 3-dim gate code).
fn fracWidthOf(segs: []const quant.Segment) ?u64 {
    for (segs) |sg| if (std.mem.eql(u8, sg.name, "gate_proj.code") and sg.shape.len == 3) return sg.shape[2];
    return null;
}
/// The Spec the texts implement, checked at construction (never a per-call branch): the fused
/// activation, the compiled dims, a bf16 MoE input (in_rin and take2 read bf16 rows) and decode
/// rows (8 tokens x top_k) within the 48-row tables.
pub fn checkSpec(spec: quant.Spec, diag: *Diag) quant.Refusal!void {
    const lim = fused_act.swiglu_clamped;
    switch (spec.act) {
        .swiglu_clamped => |l| if (l != lim) return quant.refuse(diag, error.ActivationNotFused, "exl3 quant: SwiGLU clamped at {d} (the texts fuse {d})", .{ l, lim }),
        else => return quant.refuse(diag, error.ActivationNotFused, "exl3 quant: activation {t} (the texts fuse the SwiGLU clamped at {d})", .{ spec.act, lim }),
    }
    if (spec.hidden != compiled_hidden or spec.inter != compiled_inter) return quant.refuse(diag, error.DimsNotImplemented, "exl3 quant: dims {d} / {d} (the texts are compiled for {d} / {d})", .{ spec.hidden, spec.inter, compiled_hidden, compiled_inter });
    if (spec.input != .bfloat16) return quant.refuse(diag, error.InputDtype, "exl3 quant: MoE input {t} (in_rin and take2 read bf16)", .{spec.input});
    if (spec.top_k == 0 or spec.top_k * 8 > decode_table_rows) return quant.refuse(diag, error.TopKTooWide, "exl3 quant: top_k {d}: 8 decode rows x top_k exceed the {d}-row tables", .{ spec.top_k, decode_table_rows });
    if (spec.n_layers == 0) return quant.refuse(diag, error.DimsNotImplemented, "exl3 quant: no layers", .{});
}

/// The quant accepted on backend G: the decode chain's routes (built once: statics, the per-M
/// launch tables, prepared configs), the row maps and one prefill wave state per layer.
pub fn Accepted(comptime G: type) type {
    return struct {
        const Self = @This();
        const A = ProjArrays(G.T);
        pub const max_decode_rows: u32 = decode_table_rows;
        a: Allocator,
        reg: *const xk.Registry,
        /// this subset's self-check results (its receipt: `report.writeJsonLines(a)`)
        report: selfcheck.Report = .{},
        gemv: Gemv(G),
        prep: RinPrep(G),
        /// tok[m - 1] = int32 [m] 0..m-1, kept: in_rin's row map for m routed rows already taken
        tok: [48]G.T,
        n_tok: usize = 0,
        /// one DIG-X wave state per layer (the lane has one dispatcher per layer)
        waves: []DigXPrefill(G) = &.{},
        /// the waves' down stage is the fused down GEMM (`routeFusedDown`)
        fused_down: bool = false,
        /// M1 (`routeM1`): the waves' 128-row GEMMs double-buffered (false: the stock texts)
        m1_on: bool = false,
        /// kv16-opt: the waves' expert outputs in bf16 (`routeExpertBf16`)
        expert_bf16: bool = false,
        /// the decode GEMVs' routed forms (`routeForms`); all false: the stock mul1h texts
        forms: Forms = .{},
        /// the banked route (`routeBanked`): every bank's rows of a wave in one launch per stage; null: not installed
        banked: ?Banked(G) = null,
        /// The scratch waves' twins (`routeScratchWaves`); null: the waves' stock launches only
        into: ?*IntoTwins = null,

        /// Once per bank bind and per grow: the three projections' arrays are the kernels' (cap
        /// within the kernels' bound, shapes, dtypes).
        pub fn checkBank(self: *const Self, g: *G, bank: quant.BankArrays(A), diag: *Diag) !void {
            try checkProjArrays(G, g, self.reg, .gate, bank.gate, diag);
            try checkProjArrays(G, g, self.reg, .up, bank.up, diag);
            try checkProjArrays(G, g, self.reg, .down, bank.down, diag);
        }

        /// x [rows, 5120] bf16 (the routed rows, taken), slot_ids u32 [rows], rows 1..48 ->
        /// the clamped SwiGLU [rows, 2304] f32: in_rin -> the gate and up GEMVs -> gu_epi.
        pub fn gateUp(self: *const Self, g: *G, x: G.T, slot_ids: G.T, gate: A, up: A) !G.T {
            const rows = rowsOf(G, g, x, 0);
            std.debug.assert(!(rows < 1 or rows > max_decode_rows));
            const xs = try self.prep.inRin(g, x, self.tok[rows - 1], gate.rin, up.rin, slot_ids);
            const z = try self.gemv.projectGu(g, xs[0], xs[1], slot_ids, gate.code, up.code);
            return self.prep.guEpi(g, z[0], z[1], gate.rout, up.rout, slot_ids);
        }

        /// h [rows, 2304] f32 (gateUp's), slot_ids u32 [rows] -> [rows, 5120] f32: din_rin ->
        /// the down GEMV -> dpost.
        pub fn down(self: *const Self, g: *G, h: G.T, slot_ids: G.T, d: A) !G.T {
            const hd = try self.prep.dinRin(g, h, d.rin, slot_ids);
            const zd = try self.gemv.project(g, .down, hd, slot_ids, d.code);
            return self.prep.dpost(g, zd, d.rout, slot_ids);
        }

        /// The banked route (`routeBanked` installed it): x [rows, 5120] bf16, ids u32 [rows] packed (bank << 24 | slot
        /// row), `banks` the three banks' arrays (base, ext, transient; a bank no row names may repeat another's) -> the
        /// clamped SwiGLU [rows, 2304] f32, every word gateUp's per bank.
        pub fn gateUpBanked(self: *const Self, g: *G, x: G.T, ids: G.T, banks: *const [3]quant.BankArrays(A)) !G.T {
            const rows = rowsOf(G, g, x, 0);
            std.debug.assert(!(rows < 1 or rows > max_decode_rows));
            return self.banked.?.gateUp(g, x, self.tok[rows - 1], ids, banks);
        }

        /// The banked route's down: h [rows, 2304] f32 (gateUpBanked's), the same packed ids -> [rows, 5120] f32.
        pub fn downBanked(self: *const Self, g: *G, h: G.T, ids: G.T, banks: *const [3]quant.BankArrays(A)) !G.T {
            return self.banked.?.down(g, h, ids, banks);
        }

        /// Layer `layer`'s DIG-X waves over the call's routed rows (`DigXPrefill.call`): a KEPT
        /// f32 [rows, 5120] in routed-row order (release it).
        pub fn prefill(self: *Self, g: *G, layer: u32, x: G.T, rows: quant.PrefillRows, bank: quant.BankArrays(A)) !G.T {
            return self.waves[layer].call(g, x, rows, bank);
        }

        /// `prefill` unjoined: the waves' KEPT outputs (wave order) and each wave-ordered row's
        /// assignment row (`DigXPrefill.callParts`).
        pub fn prefillParts(self: *Self, g: *G, layer: u32, x: G.T, rows: quant.PrefillRows, bank: quant.BankArrays(A), alloc: Allocator, outs: *std.ArrayList(G.T), pos: *std.ArrayList(u32)) !void {
            return self.waves[layer].callParts(g, x, rows, bank, alloc, outs, pos);
        }

        /// The prefill boundary: every layer's waves still in flight evaluated, oldest first.
        pub fn finishPrefill(self: *Self, g: *G) !void {
            for (self.waves) |*w| try w.finish(g);
        }

        /// The fused down GEMM arm, at construction (before any prefill): its self-checks (compile, composition,
        /// fused) on `set`, the accepting set, join the report, then every layer's waves launch it in place of the
        /// 128-row down text and rot_widen1. A failed check refuses it by name (SelfCheckFailed, `diag`) and the
        /// waves stay stock.
        pub fn routeFusedDown(self: *Self, set: *const ks.Set, diag: *Diag) !void {
            try set.selfCheck(self.a, &w1_texts, .compile, &self.report, diag);
            for (self.waves) |*w| w.installDown(.fused);
            self.fused_down = true;
        }
        /// M1, at construction (before any prefill and before `routeScratchWaves`, whose twins derive from the GEMM texts
        /// it selects): the double-buffered 128-row GEMMs' self-checks (compile, composition, twin: every z word the
        /// 128-row text's) on `set`, the accepting set, join the report, then every layer's waves run their GEMMs on them
        /// (`DigX.m1_on`). A failed check refuses the load by name (SelfCheckFailed, `diag`).
        pub fn routeM1(self: *Self, set: *const ks.Set, diag: *Diag) !void {
            if (self.into != null) return error.M1AfterScratchWaves;
            try set.selfCheck(self.a, &m1_texts, .full, &self.report, diag);
            for (self.waves) |*w| w.dig.m1_on = true;
            self.m1_on = true;
        }

        /// The decode route texts (the forms', the banked route's) compiled at construction with the rest of the set
        /// (`.compile`: compile, bind, one launch), so no route text compiles at its first decode launch.
        pub fn compileTexts(self: *Self, set: *const ks.Set, texts: []const Kernel, diag: *Diag) !void {
            try set.selfCheck(self.a, texts, .compile, &self.report, diag);
        }

        /// kv16-opt, at construction (before any prefill): every layer's DIG-X waves store their expert outputs as bf16
        /// (rot_widen1's bf16-out instantiation: the same f32 word, rounded to bf16), the reference's `Expert` output
        /// type (`type_as(x)`); the combines read them in f32 as before. The fused down GEMM's epilogue writes f32, so it
        /// refuses with the fused arm installed.
        pub fn routeExpertBf16(self: *Self) !void {
            if (self.fused_down) return error.ExpertBf16NeedsChainDown;
            for (self.waves) |*w| w.dig.widen1_e = self.reg.get(.dsv41_prefill_dig_rot_widen1_5120_obf16);
            self.expert_bf16 = true;
        }
        /// The scratch waves (scratch_waves.zig), at construction (after `routeExpertBf16`, before any prefill): every layer's
        /// uncarried waves launch the write-into twins (`IntoTwins`: derived from the registry's pinned stock texts, each
        /// anchor checked, each compiled now by one launch; a twin that does not refuses the load by name). The twins are
        /// the chain down's with the bf16 expert outputs: refused by name otherwise.
        pub fn routeScratchWaves(self: *Self, g: *G, diag: *Diag) !void {
            if (comptime G.T != mlx.mlx_array) return error.ScratchWavesNeedsMlx;
            if (self.fused_down or !self.expert_bf16)
                return intoRefuse(diag, error.ScratchWavesNeedsBf16Chain, "exl3 scratch waves: the twins are the chain down's with bf16 expert outputs", .{});
            if (self.into != null or self.waves.len == 0) return intoRefuse(diag, error.ScratchWavesTwin, "exl3 scratch waves: routed twice or no waves", .{});
            const tw = try self.a.create(IntoTwins);
            errdefer self.a.destroy(tw);
            tw.* = try IntoTwins.init(self.a, self.reg, self.m1_on, diag);
            errdefer tw.deinit();
            try tw.compileAll(g.s, self.waves[0].tiles_gu, self.waves[0].tiles_dn, diag);
            self.into = tw;
            for (self.waves) |*w| w.into = tw;
        }

        /// The routed decode forms, at construction (before any decode): the GEMVs rebuilt on the forms' texts. Exact by
        /// the registry's twin checks (every word == mul1h's) and kbench v9; no device check here.
        pub fn routeForms(self: *Self, g: *G, forms: Forms) !void {
            // The banked route aliases the installed GEMVs' statics: the forms come first.
            if (self.banked != null) return error.FormsAfterBanked;
            var next = try Gemv(G).initForms(g, self.reg, forms);
            errdefer next.deinit(g);
            self.gemv.deinit(g);
            self.gemv = next;
            self.forms = forms;
        }

        /// The banked route, at construction (before any decode; after `routeForms`, whose forms it takes): the decode
        /// stages on the banked texts, one launch per stage over a wave's rows of all three banks. Exact by the registry's
        /// twin checks (every word == the stock text's, per bank) and kbench v6d / v9b; no device check here. It allocates
        /// no device array: its statics are the installed GEMVs' (aliased by input name), its plans prepared configs.
        pub fn routeBanked(self: *Self, g: *G) !void {
            if (self.banked != null) return error.BankedRoutedTwice;
            self.banked = try Banked(G).init(g, self.reg, self.forms, &self.gemv);
        }

        /// Releases the routes (statics, prepared configs, the row maps, the wave states) and
        /// the plan's results. The kernel set stays the load context's.
        pub fn deinit(self: *Self, g: *G) void {
            if (self.into) |tw| {
                tw.deinit();
                self.a.destroy(tw);
            }
            if (self.banked) |*b| b.deinit(g);
            for (self.waves) |*w| w.deinit(g);
            self.a.free(self.waves);
            for (self.tok[0..self.n_tok]) |x| g.release(x);
            self.prep.deinit(g);
            self.gemv.deinit(g);
            self.report.deinit(self.a);
            self.a.destroy(self);
        }
    };
}

/// Once per backend, before the weights bind: the Spec contract, this subset's self-check plan
/// on the load context's kernel set (judged), then the routes. Refused by name: no kernel set
/// (NoKernelSet), the Spec (ActivationNotFused / DimsNotImplemented / InputDtype / TopKTooWide),
/// a self-check failure (SelfCheckFailed, `diag` naming kernel / check / site). The set's
/// launcher must be installed on `g` first (`Set.install`): a backend that prepares launches
/// prepares them through it.
/// The 128-row DIG-X GEMM texts (their 0b smoke checks them by name).
const m128_texts = [_]Kernel{ .dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk3_m128, .dsv41_prefill_dig_gemm_2304x5120_xmul1hk3_m128, .dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk2_m128, .dsv41_prefill_dig_gemm_2304x5120_xmul1hk2_m128 };
/// The fused down GEMM: a construction-time arm (`Accepted.routeFusedDown` self-checks it, then the waves launch it);
/// the stock accept neither routes nor checks it. Its 0b smoke checks it by name.
pub const w1_texts = [_]Kernel{ .dsv41_prefill_dig_gemm_2304x5120_xmul1hk3_m128w1, .dsv41_prefill_dig_gemm_2304x5120_xmul1hk2_m128w1 };
/// The table-codebook gate|up GEMM: registered, not routed (its 0b smoke checks it by name).
const lut_texts = [_]Kernel{.dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk3_m128lut};
/// M1: the 128-row GEMMs with the B stage double-buffered, a construction-time arm
/// (`Accepted.routeM1` self-checks them, then the waves launch them and the scratch waves' twins derive from them); the
/// stock accept neither routes nor checks them.
pub const m1_texts = [_]Kernel{ .dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk3_m128d, .dsv41_prefill_dig_gemm_2304x5120_xmul1hk3_m128d, .dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk2_m128d, .dsv41_prefill_dig_gemm_2304x5120_xmul1hk2_m128d };
/// The routed decode forms: installed at construction by `Accepted.routeForms` (their twins are registry checks, run in
/// a check window and by kbench v9); the stock accept neither routes nor checks them.
pub const form_texts = [_]Kernel{ .dsv41_exl3_pair_k3_5120, .dsv41_exl3_guone_k3_2304, .dsv41_exl3_pair_k2_5120, .dsv41_exl3_guone_k2_2304 };
/// The banked route's texts: installed at construction by `Accepted.routeBanked` (their twins are registry checks, run
/// in a check window and by kbench v6d / v9b); the stock accept neither routes nor checks them.
pub const banked_texts = [_]Kernel{
    .dsv41_exl3_b3_mul1h_k3_2304,     .dsv41_exl3_b3_mul1h_k3_5120,  .dsv41_exl3_b3_prep_in_rin,  .dsv41_exl3_b3_prep_gu_epi,
    .dsv41_exl3_b3_prep_din_rin,      .dsv41_exl3_b3_moeprep_dpost,  .dsv41_exl3_b3_pair_k3_5120, .dsv41_exl3_b3_guone_k3_2304,
    .dsv41_exl3_b3_mul1h_k2_2304,     .dsv41_exl3_b3_mul1h_k2_5120,  .dsv41_exl3_b3_pair_k2_5120, .dsv41_exl3_b3_guone_k2_2304,
};
pub const checked_at_accept = blk: {
    @setEvalBranchQuota(20_000);
    var out: [kernels.len - w1_texts.len - lut_texts.len - form_texts.len - banked_texts.len - m1_texts.len]Kernel = undefined;
    var n: usize = 0;
    for (kernels) |k| if (std.mem.indexOfScalar(Kernel, &(w1_texts ++ lut_texts ++ form_texts ++ banked_texts ++ m1_texts), k) == null) {
        out[n] = k;
        n += 1;
    };
    break :blk out;
};

pub fn accept(comptime G: type, a: Allocator, g: *G, ctx: quant.Context, spec: quant.Spec, diag: *Diag) !*Accepted(G) {
    const ref = ctx.kernels orelse return quant.refuse(diag, error.NoKernelSet, "exl3 quant: accepted without the load context's kernel set", .{});
    const set = ks.Set.of(ref) orelse return quant.refuse(diag, error.NoKernelSet, "exl3 quant: accepted with a kernel set over another registry (manifest {s})", .{ref.manifest_sha256});
    try checkSpec(spec, diag);
    const acc = try a.create(Accepted(G));
    acc.* = .{ .a = a, .reg = &set.reg, .gemv = undefined, .prep = undefined, .tok = undefined };
    errdefer {
        acc.report.deinit(a);
        a.destroy(acc);
    }
    try set.selfCheck(a, &checked_at_accept, .compile, &acc.report, diag);
    acc.gemv = try Gemv(G).init(g, &set.reg);
    errdefer acc.gemv.deinit(g);
    acc.prep = try RinPrep(G).init(g, &set.reg);
    errdefer acc.prep.deinit(g);
    errdefer for (acc.tok[0..acc.n_tok]) |x| g.release(x);
    var idx: [48]i32 = undefined;
    for (&idx, 0..) |*v, i| v.* = @intCast(i);
    for (1..decode_table_rows + 1) |m| {
        acc.tok[m - 1] = g.keep(try g.hostArray(std.mem.sliceAsBytes(idx[0..m]), &.{@intCast(m)}, .int32));
        acc.n_tok = m;
    }
    const waves = try a.alloc(DigXPrefill(G), spec.n_layers);
    var built: usize = 0;
    errdefer {
        for (waves[0..built]) |*w| w.deinit(g);
        a.free(waves);
    }
    for (waves) |*w| {
        w.* = try DigXPrefill(G).init(a, &set.reg, .tier, null);
        built += 1;
    }
    acc.waves = waves;
    return acc;
}

// ── The routes (moved from exl3_kernel_ops.zig) ──

// ── The expert path over the streamer's slot banks (DSV41_EXL3_BANK = on, PREP = rin) ──

pub const Proj = enum { gate, up, down };

/// One projection's slot-bank arrays, as the streamer's `ProjArrays`: code i16 [rows, in/16,
/// out/16, 48], rout f16 [rows, out], rin f16 [rows, in] (row = slot).
pub fn ProjArrays(comptime T: type) type {
    return struct { code: T, rout: T, rin: T };
}

/// A layer bank's projection arrays are what the kernels read (bind time, once per bank).
pub fn checkBank(comptime G: type, g: *G, reg: *const xk.Registry, proj: Proj, a: ProjArrays(G.T), diag: ?*xk.Diag) Refusal!void {
    var vars: Vars = .initFill(0);
    const cap = rowsOf(G, g, a.code, 0);
    const bound = reg.get(.dsv41_exl3_mul1h_k3_2304).bounds.get(.cap).?;
    if (cap < bound[0] or cap > bound[1]) return refuse(diag, error.RouteInput, "exl3 kernel ops: a bank of {d} slots (the kernels take {d}..{d})", .{ cap, bound[0], bound[1] });
    vars.set(.cap, cap);
    switch (proj) {
        .gate, .up => {
            const ki = kIndexOf(G, g, a.code) catch return refuse(diag, error.RouteInput, "exl3 kernel ops: a gate / up code array of {d} int16 a tile (the texts decode 48 and 32)", .{rowsOf(G, g, a.code, 3)});
            try expectInput(G, g, reg.get(if (ki == 1) .dsv41_exl3_mul1h_k2_2304 else .dsv41_exl3_mul1h_k3_2304), "code", a.code, &vars, diag);
            try expectInput(G, g, reg.get(.q3_exl3_prep_gu_epi), "rg", a.rout, &vars, diag);
            try expectInput(G, g, reg.get(.q3_exl3_prep_in_rin), "rg", a.rin, &vars, diag);
        },
        .down => {
            const ki = kIndexOf(G, g, a.code) catch return refuse(diag, error.RouteInput, "exl3 kernel ops: a down code array of {d} int16 a tile (the texts decode 48 and 32)", .{rowsOf(G, g, a.code, 3)});
            try expectInput(G, g, reg.get(if (ki == 1) .dsv41_exl3_mul1h_k2_5120 else .dsv41_exl3_mul1h_k3_5120), "code", a.code, &vars, diag);
            try expectInput(G, g, reg.get(.q3_moeprep_dpost), "rd", a.rout, &vars, diag);
            try expectInput(G, g, reg.get(.q3_exl3_prep_din_rin), "rn", a.rin, &vars, diag);
        },
    }
}

/// The decode GEMVs (form mul1h, K 3): `Provider.project`.
pub fn Gemv(comptime G: type) type {
    return struct {
        const Self = @This();
        gu: *const Entry,
        dn: *const Entry,
        gu_statics: Statics(G),
        dn_statics: Statics(G),
        gu_p: RowPlans(G, 48),
        dn_p: RowPlans(G, 48),
        /// gu_one: gate and up in one launch (`dsv41_exl3_guone_k3_2304`), bound by `initForms`.
        gu1_p: ?RowPlans(G, 48) = null,
        /// The K = 2 texts (the same forms), for the layers whose code tiles are 32 int16 wide.
        gu2: *const Entry,
        dn2: *const Entry,
        gu2_statics: Statics(G),
        dn2_statics: Statics(G),
        gu2_p: RowPlans(G, 48),
        dn2_p: RowPlans(G, 48),
        gu12_p: ?RowPlans(G, 48) = null,
        /// The gate / up call bound at construction: `guOne` (gu_one's one launch) or `guPair` (the two stock launches).
        gu_call: *const GuCall = guPair,

        const GuCall = fn (self: *const Self, g: *G, xg: G.T, xu: G.T, ids: G.T, code_g: G.T, code_u: G.T) anyerror![2]G.T;

        pub fn init(g: *G, reg: *const xk.Registry) !Self {
            return initForms(g, reg, .{});
        }

        /// The GEMVs under the routed forms (stock when none): down_pair's text for the down projection, gu_one's one
        /// launch for gate and up. Every form computes the stock words (registry twin checks; kbench v9).
        pub fn initForms(g: *G, reg: *const xk.Registry, forms: Forms) !Self {
            const gu = reg.get(.dsv41_exl3_mul1h_k3_2304);
            const dn = reg.get(if (forms.down_pair) .dsv41_exl3_pair_k3_5120 else .dsv41_exl3_mul1h_k3_5120);
            const gu2 = reg.get(.dsv41_exl3_mul1h_k2_2304);
            const dn2 = reg.get(if (forms.down_pair) .dsv41_exl3_pair_k2_5120 else .dsv41_exl3_mul1h_k2_5120);
            var gu_p: RowPlans(G, 48) = try .init(g, gu, null, null);
            errdefer gu_p.deinit(g);
            var dn_p: RowPlans(G, 48) = try .init(g, dn, null, null);
            errdefer dn_p.deinit(g);
            var gu2_p: RowPlans(G, 48) = try .init(g, gu2, null, null);
            errdefer gu2_p.deinit(g);
            var dn2_p: RowPlans(G, 48) = try .init(g, dn2, null, null);
            errdefer dn2_p.deinit(g);
            var gu1_p: ?RowPlans(G, 48) = if (forms.gu_one) try RowPlans(G, 48).init(g, reg.get(.dsv41_exl3_guone_k3_2304), null, null) else null;
            errdefer if (gu1_p) |*x| x.deinit(g);
            var gu12_p: ?RowPlans(G, 48) = if (forms.gu_one) try RowPlans(G, 48).init(g, reg.get(.dsv41_exl3_guone_k2_2304), null, null) else null;
            errdefer if (gu12_p) |*x| x.deinit(g);
            var gs = try Statics(G).init(g, gu);
            errdefer gs.deinit(g);
            var ds = try Statics(G).init(g, dn);
            errdefer ds.deinit(g);
            var gs2 = try Statics(G).init(g, gu2);
            errdefer gs2.deinit(g);
            return .{ .gu = gu, .dn = dn, .gu_statics = gs, .dn_statics = ds, .gu_p = gu_p, .dn_p = dn_p, .gu1_p = gu1_p, .gu2 = gu2, .dn2 = dn2, .gu2_statics = gs2, .dn2_statics = try Statics(G).init(g, dn2), .gu2_p = gu2_p, .dn2_p = dn2_p, .gu12_p = gu12_p, .gu_call = if (forms.gu_one) guOne else guPair };
        }

        pub fn deinit(self: *Self, g: *G) void {
            self.gu_p.deinit(g);
            self.dn_p.deinit(g);
            if (self.gu1_p) |*x| x.deinit(g);
            self.gu_statics.deinit(g);
            self.dn_statics.deinit(g);
            self.gu2_p.deinit(g);
            self.dn2_p.deinit(g);
            if (self.gu12_p) |*x| x.deinit(g);
            self.gu2_statics.deinit(g);
            self.dn2_statics.deinit(g);
        }

        /// Gate and up through the call `initForms` bound (no per-call choice).
        pub fn projectGu(self: *const Self, g: *G, xg: G.T, xu: G.T, ids: G.T, code_g: G.T, code_u: G.T) ![2]G.T {
            return self.gu_call(self, g, xg, xu, ids, code_g, code_u);
        }

        fn guPair(self: *const Self, g: *G, xg: G.T, xu: G.T, ids: G.T, code_g: G.T, code_u: G.T) anyerror![2]G.T {
            return .{ try self.project(g, .gate, xg, ids, code_g), try self.project(g, .up, xu, ids, code_u) };
        }

        fn guOne(self: *const Self, g: *G, xg: G.T, xu: G.T, ids: G.T, code_g: G.T, code_u: G.T) anyerror![2]G.T {
            var out: [2]G.T = undefined;
            const p = if ((try kIndexOf(G, g, code_g)) == 1) &self.gu12_p.? else &self.gu1_p.?;
            try p.launch(g, rowsOf(G, g, xg, 0), &.{ xg, xu, ids, code_g, code_u }, &out);
            return out;
        }

        /// xh [rows, in] f32 (rotated), ids [rows] u32 (each row's slot), code = the projection's
        /// bank code -> z [rows, out] f32 (gate / up: out 2304, down: out 5120).
        pub fn project(self: *const Self, g: *G, proj: Proj, xh: G.T, ids: G.T, code: G.T) !G.T {
            const k2 = (try kIndexOf(G, g, code)) == 1;
            const e, const st, const p = if (proj == .down)
                (if (k2) .{ self.dn2, &self.dn2_statics, &self.dn2_p } else .{ self.dn, &self.dn_statics, &self.dn_p })
            else
                (if (k2) .{ self.gu2, &self.gu2_statics, &self.gu2_p } else .{ self.gu, &self.gu_statics, &self.gu_p });
            var ins: [9]G.T = undefined;
            ins[0] = xh;
            ins[1] = ids;
            ins[2] = code;
            for (3..e.inputs.len) |i| ins[i] = st.arrays[i];
            var out: [1]G.T = undefined;
            try p.launch(g, rowsOf(G, g, xh, 0), ins[0..e.inputs.len], &out);
            return out[0];
        }
    };
}

/// The banked route: the decode stages on the banked texts (`banked_texts`), each launch over a wave's rows of all three
/// banks (a row's bank is its packed id's top byte: bank << 24 | slot row; banks base, ext, transient). The forms are
/// the installed GEMVs' (`Gemv.initForms`), bound here at construction: gu_one's banked one launch or the banked
/// mul1h pair, down_pair's banked text or the banked mul1h. The GEMV statics are the installed GEMVs' arrays (aliased by
/// input name, checked once here): the route keeps no device array of its own.
pub fn Banked(comptime G: type) type {
    return struct {
        const Self = @This();
        const A = ProjArrays(G.T);
        const BA = quant.BankArrays(A);
        in_rin_p: RowPlans(G, 48),
        gu_epi_p: RowPlans(G, 48),
        din_rin_p: RowPlans(G, 48),
        dpost_p: RowPlans(G, 48),
        /// gate / up: the banked mul1h (twice) or gu_one's banked one launch
        gu_p: RowPlans(G, 48),
        dn_p: RowPlans(G, 48),
        gu_e: *const Entry,
        dn_e: *const Entry,
        /// the GEMVs' statics, aliased (the installed `Gemv`'s, released by it)
        gu_st: [xk.max_inputs]G.T = undefined,
        dn_st: [xk.max_inputs]G.T = undefined,
        /// the K = 2 texts (the same forms) and their aliased statics
        gu2_p: RowPlans(G, 48),
        dn2_p: RowPlans(G, 48),
        gu2_e: *const Entry,
        dn2_e: *const Entry,
        gu2_st: [xk.max_inputs]G.T = undefined,
        dn2_st: [xk.max_inputs]G.T = undefined,
        gu_call: *const GuCall,

        const GuCall = fn (self: *const Self, g: *G, xg: G.T, xu: G.T, ids: G.T, banks: *const [3]BA) anyerror![2]G.T;

        pub fn init(g: *G, reg: *const xk.Registry, forms: Forms, gemv: *const Gemv(G)) !Self {
            const gu_e = reg.get(if (forms.gu_one) .dsv41_exl3_b3_guone_k3_2304 else .dsv41_exl3_b3_mul1h_k3_2304);
            const dn_e = reg.get(if (forms.down_pair) .dsv41_exl3_b3_pair_k3_5120 else .dsv41_exl3_b3_mul1h_k3_5120);
            const gu2_e = reg.get(if (forms.gu_one) .dsv41_exl3_b3_guone_k2_2304 else .dsv41_exl3_b3_mul1h_k2_2304);
            const dn2_e = reg.get(if (forms.down_pair) .dsv41_exl3_b3_pair_k2_5120 else .dsv41_exl3_b3_mul1h_k2_5120);
            var self: Self = .{ .in_rin_p = undefined, .gu_epi_p = undefined, .din_rin_p = undefined, .dpost_p = undefined, .gu_p = undefined, .dn_p = undefined, .gu_e = gu_e, .dn_e = dn_e, .gu2_p = undefined, .dn2_p = undefined, .gu2_e = gu2_e, .dn2_e = dn2_e, .gu_call = if (forms.gu_one) guOne else guPair };
            // The statics: the stock GEMV's by name (gu_one has none; its gate / up statics are compiled in).
            if (!forms.gu_one) try alias(gu_e, gemv.gu, &gemv.gu_statics, &self.gu_st);
            try alias(dn_e, gemv.dn, &gemv.dn_statics, &self.dn_st);
            if (!forms.gu_one) try alias(gu2_e, gemv.gu2, &gemv.gu2_statics, &self.gu2_st);
            try alias(dn2_e, gemv.dn2, &gemv.dn2_statics, &self.dn2_st);
            self.in_rin_p = try .init(g, reg.get(.dsv41_exl3_b3_prep_in_rin), null, null);
            errdefer self.in_rin_p.deinit(g);
            self.gu_epi_p = try .init(g, reg.get(.dsv41_exl3_b3_prep_gu_epi), null, null);
            errdefer self.gu_epi_p.deinit(g);
            self.din_rin_p = try .init(g, reg.get(.dsv41_exl3_b3_prep_din_rin), null, null);
            errdefer self.din_rin_p.deinit(g);
            self.dpost_p = try .init(g, reg.get(.dsv41_exl3_b3_moeprep_dpost), null, null);
            errdefer self.dpost_p.deinit(g);
            self.gu_p = try .init(g, gu_e, null, null);
            errdefer self.gu_p.deinit(g);
            self.dn_p = try .init(g, dn_e, null, null);
            errdefer self.dn_p.deinit(g);
            self.gu2_p = try .init(g, gu2_e, null, null);
            errdefer self.gu2_p.deinit(g);
            self.dn2_p = try .init(g, dn2_e, null, null);
            return self;
        }

        /// Every static input of the banked text `be` is the stock text `se`'s input of the same name, dtype and values
        /// (the manifest's banked entries are deep copies): alias the stock GEMV's array.
        fn alias(be: *const Entry, se: *const Entry, st: *const Statics(G), out: *[xk.max_inputs]G.T) !void {
            for (be.inputs, 0..) |*arg, i| {
                if (arg.role != .static) continue;
                const j = for (se.inputs, 0..) |*sa, jj| {
                    if (std.mem.eql(u8, sa.name, arg.name)) break jj;
                } else return error.BankedStaticUnmatched;
                const sa = &se.inputs[j];
                if (sa.role != .static or sa.dtype != arg.dtype or st.mask & (@as(u32, 1) << @intCast(j)) == 0) return error.BankedStaticUnmatched;
                var b0: [1024]u8 = undefined;
                var b1: [1024]u8 = undefined;
                const sh0, const v0 = kr.staticBytes(arg, &b0);
                const sh1, const v1 = kr.staticBytes(sa, &b1);
                if (!std.mem.eql(c_int, sh0.slice(), sh1.slice()) or !std.mem.eql(u8, v0, v1)) return error.BankedStaticUnmatched;
                out[i] = st.arrays[j];
            }
        }

        pub fn deinit(self: *Self, g: *G) void {
            self.in_rin_p.deinit(g);
            self.gu_epi_p.deinit(g);
            self.din_rin_p.deinit(g);
            self.dpost_p.deinit(g);
            self.gu_p.deinit(g);
            self.dn_p.deinit(g);
            self.gu2_p.deinit(g);
            self.dn2_p.deinit(g);
        }

        /// x [tokens, 5120] bf16 (taken rows), tok [rows] i32, ids packed -> the clamped SwiGLU [rows, 2304] f32.
        pub fn gateUp(self: *const Self, g: *G, x: G.T, tok: G.T, ids: G.T, banks: *const [3]BA) !G.T {
            const rows = rowsOf(G, g, tok, 0);
            var xs: [2]G.T = undefined;
            try self.in_rin_p.launch(g, rows, &.{ x, tok, banks[0].gate.rin, banks[1].gate.rin, banks[2].gate.rin, banks[0].up.rin, banks[1].up.rin, banks[2].up.rin, ids }, &xs);
            const z = try self.gu_call(self, g, xs[0], xs[1], ids, banks);
            var out: [1]G.T = undefined;
            try self.gu_epi_p.launch(g, rows, &.{ z[0], z[1], banks[0].gate.rout, banks[1].gate.rout, banks[2].gate.rout, banks[0].up.rout, banks[1].up.rout, banks[2].up.rout, ids }, &out);
            return out[0];
        }

        /// h [rows, 2304] f32, ids packed -> [rows, 5120] f32: din_rin -> the down GEMV -> dpost, banked.
        pub fn down(self: *const Self, g: *G, h: G.T, ids: G.T, banks: *const [3]BA) !G.T {
            const rows = rowsOf(G, g, h, 0);
            var hd: [1]G.T = undefined;
            try self.din_rin_p.launch(g, rows, &.{ h, banks[0].down.rin, banks[1].down.rin, banks[2].down.rin, ids }, &hd);
            const k2 = (try kIndexOf(G, g, banks[0].down.code)) == 1;
            var ins: [xk.max_inputs]G.T = if (k2) self.dn2_st else self.dn_st;
            ins[0] = hd[0];
            ins[1] = ids;
            ins[2] = banks[0].down.code;
            ins[3] = banks[1].down.code;
            ins[4] = banks[2].down.code;
            var zd: [1]G.T = undefined;
            const dn_e, const dn_p = if (k2) .{ self.dn2_e, &self.dn2_p } else .{ self.dn_e, &self.dn_p };
            try dn_p.launch(g, rows, ins[0..dn_e.inputs.len], &zd);
            var out: [1]G.T = undefined;
            try self.dpost_p.launch(g, rows, &.{ zd[0], banks[0].down.rout, banks[1].down.rout, banks[2].down.rout, ids }, &out);
            return out[0];
        }

        fn guPair(self: *const Self, g: *G, xg: G.T, xu: G.T, ids: G.T, banks: *const [3]BA) anyerror![2]G.T {
            const rows = rowsOf(G, g, xg, 0);
            var out: [2]G.T = undefined;
            const k2 = (try kIndexOf(G, g, banks[0].gate.code)) == 1;
            const gu_e, const gu_p = if (k2) .{ self.gu2_e, &self.gu2_p } else .{ self.gu_e, &self.gu_p };
            var ins: [xk.max_inputs]G.T = if (k2) self.gu2_st else self.gu_st;
            ins[1] = ids;
            inline for (.{ .gate, .up }, 0..) |proj, o| {
                ins[0] = if (o == 0) xg else xu;
                ins[2] = @field(banks[0], @tagName(proj)).code;
                ins[3] = @field(banks[1], @tagName(proj)).code;
                ins[4] = @field(banks[2], @tagName(proj)).code;
                var z: [1]G.T = undefined;
                try gu_p.launch(g, rows, ins[0..gu_e.inputs.len], &z);
                out[o] = z[0];
            }
            return out;
        }

        fn guOne(self: *const Self, g: *G, xg: G.T, xu: G.T, ids: G.T, banks: *const [3]BA) anyerror![2]G.T {
            var out: [2]G.T = undefined;
            const p = if ((try kIndexOf(G, g, banks[0].gate.code)) == 1) &self.gu2_p else &self.gu_p;
            try p.launch(g, rowsOf(G, g, xg, 0), &.{ xg, xu, ids, banks[0].gate.code, banks[1].gate.code, banks[2].gate.code, banks[0].up.code, banks[1].up.code, banks[2].up.code }, &out);
            return out;
        }
    };
}

/// PREP = rin: the rin / rout stages around the GEMVs (`build_kernels` of the rinprep lane).
pub fn RinPrep(comptime G: type) type {
    return struct {
        const Self = @This();
        in_rin_e: *const Entry,
        gu_epi_e: *const Entry,
        din_rin_e: *const Entry,
        dpost_e: *const Entry,
        in_rin_p: RowPlans(G, 48),
        gu_epi_p: RowPlans(G, 48),
        din_rin_p: RowPlans(G, 48),
        dpost_p: RowPlans(G, 48),

        pub fn init(g: *G, reg: *const xk.Registry) !Self {
            const a, const b, const c, const d = .{ reg.get(.q3_exl3_prep_in_rin), reg.get(.q3_exl3_prep_gu_epi), reg.get(.q3_exl3_prep_din_rin), reg.get(.q3_moeprep_dpost) };
            var pa: RowPlans(G, 48) = try .init(g, a, null, null);
            errdefer pa.deinit(g);
            var pb: RowPlans(G, 48) = try .init(g, b, null, null);
            errdefer pb.deinit(g);
            var pc: RowPlans(G, 48) = try .init(g, c, null, null);
            errdefer pc.deinit(g);
            return .{ .in_rin_e = a, .gu_epi_e = b, .din_rin_e = c, .dpost_e = d, .in_rin_p = pa, .gu_epi_p = pb, .din_rin_p = pc, .dpost_p = try .init(g, d, null, null) };
        }

        pub fn deinit(self: *Self, g: *G) void {
            self.in_rin_p.deinit(g);
            self.gu_epi_p.deinit(g);
            self.din_rin_p.deinit(g);
            self.dpost_p.deinit(g);
        }

        /// x [tokens, 5120] bf16, tok [rows] i32, rg / ru = gate / up rin, ids [rows] u32 ->
        /// (xg, xu) [rows, 5120] f32 = t128(x[tok] * rin[ids]).
        pub fn inRin(self: *const Self, g: *G, x: G.T, tok: G.T, rg: G.T, ru: G.T, ids: G.T) ![2]G.T {
            var out: [2]G.T = undefined;
            try self.in_rin_p.launch(g, rowsOf(G, g, tok, 0), &.{ x, tok, rg, ru, ids }, &out);
            return out;
        }

        /// zg / zu [rows, 2304] f32, rg / ru = gate / up rout -> clamped SwiGLU [rows, 2304] f32.
        pub fn guEpi(self: *const Self, g: *G, zg: G.T, zu: G.T, rg: G.T, ru: G.T, ids: G.T) !G.T {
            var out: [1]G.T = undefined;
            try self.gu_epi_p.launch(g, rowsOf(G, g, zg, 0), &.{ zg, zu, rg, ru, ids }, &out);
            return out[0];
        }

        /// hid [rows, 2304] f32, rn = down rin -> t128(hid * rin[ids]) [rows, 2304] f32.
        pub fn dinRin(self: *const Self, g: *G, hid: G.T, rn: G.T, ids: G.T) !G.T {
            var out: [1]G.T = undefined;
            try self.din_rin_p.launch(g, rowsOf(G, g, hid, 0), &.{ hid, rn, ids }, &out);
            return out[0];
        }

        /// zd [rows, 5120] f32, rd = down rout -> t128(zd) * rout[ids] [rows, 5120] f32.
        pub fn dpost(self: *const Self, g: *G, zd: G.T, rd: G.T, ids: G.T) !G.T {
            var out: [1]G.T = undefined;
            try self.dpost_p.launch(g, rowsOf(G, g, zd, 0), &.{ zd, rd, ids }, &out);
            return out[0];
        }
    };
}

// ── Prefill: REBUILD (DSV41_PREFILL_FUSED_BANK = exl3, mul1lut) and DIG-X (DIG = 1, DIG2 = onepass) ──

pub const wave_max = 16;

/// The rebuild's slot table: the wave's slots, padded with the first (the lane's `slots8`).
pub fn rebuildSlots(slots: []const u32) [wave_max]i32 {
    var t: [wave_max]i32 = undefined;
    for (&t, 0..) |*v, i| v.* = @intCast(slots[if (i < slots.len) i else 0]);
    return t;
}

/// The EXL3 rebuild of a wave's experts into bf16 weights: `Exl3Rebuild3Kernel.launch`.
pub fn Rebuild(comptime G: type) type {
    return struct {
        const Self = @This();
        e: *const Entry,

        pub fn init(reg: *const xk.Registry) Self {
            return .{ .e = reg.get(.q3_prefill_fused_exl3x3_mul1lut_k3_bf16) };
        }

        /// slots i32 [16] (`rebuildSlots`), `experts` of them used -> (og, ou [n, 5120, 2304], od [n, 2304, 5120]) bf16.
        pub fn call(self: *const Self, g: *G, gate: ProjArrays(G.T), up: ProjArrays(G.T), down: ProjArrays(G.T), slots: G.T, experts: u32) ![3]G.T {
            var vars: Vars = .initFill(0);
            vars.set(.experts, experts);
            var out: [3]G.T = undefined;
            try launchRule(G, g, self.e, &vars, &.{ gate.code, gate.rout, gate.rin, up.code, up.rout, up.rin, down.code, down.rout, down.rin, slots }, &out);
            return out;
        }
    };
}

pub const WaveExpert = struct { slot: u32, rows: u32 };
pub const DigTable = struct { table: [80]i32, tgs: u32 };

/// q3_prefill_dig_candidate.wave_table: per expert (<= 16, rows grouped by expert in order)
/// slot, first row, rows, first threadgroup (unused: INT32_MAX); [64] experts, [65] threadgroups.
/// `tiles` = the GEMM's threadgroups per `bm`-row M tile (`digTiles`; `bm` 64, or 128 for the 128-row GEMM texts):
/// the first-threadgroup column counts ceil(rows / bm) tiles per expert.
pub fn digTableBm(experts: []const WaveExpert, tiles: u32, bm: u32) DigTable {
    var t: [80]i32 = @splat(0);
    var row0: i64 = 0;
    var tg: i64 = 0;
    for (0..wave_max) |j| {
        if (j >= experts.len) {
            t[48 + j] = std.math.maxInt(i32);
            continue;
        }
        t[j] = @intCast(experts[j].slot);
        t[16 + j] = @intCast(row0);
        t[32 + j] = @intCast(experts[j].rows);
        t[48 + j] = @intCast(tg);
        row0 += experts[j].rows;
        tg += @as(i64, @intCast((experts[j].rows + bm - 1) / bm)) * tiles;
    }
    t[64] = @intCast(experts.len);
    t[65] = @intCast(tg);
    return .{ .table = t, .tgs = @intCast(tg) };
}

/// DIG-X on DIG2 onepass: the NAX GEMMs whose B loader decodes the trellis, the rotation
/// stages and the onepass SwiGLU (`DigGemmX`, `RotKernelsX`, `Dig2OnePassX`).
pub fn DigX(comptime G: type) type {
    return struct {
        const Self = @This();
        /// the 64-row GEMM texts: the 128-row texts' twin reference (their device self-check, the 0b smoke), not
        /// launched by the route
        gemm_gu: *const Entry,
        gemm_dn: *const Entry,
        /// the GEMMs the route launches: 128-row M tiles (one decoded B stage feeds 128 rows); the down text and
        /// `widen1_e` are also the fused down GEMM's reference (its device self-check `fused`, the 0b smoke)
        gemm_gu128: *const Entry,
        gemm_dn128: *const Entry,
        /// the lane's take2 text: the retune's bitwise reference (its 0b smoke), not launched by the route
        take2_e: *const Entry,
        /// the take2 retune the route launches (both outputs per simdgroup from one act load)
        take2v_e: *const Entry,
        roundx_e: *const Entry,
        onepass_e: *const Entry,
        widen2_e: *const Entry,
        widen1_e: *const Entry,
        /// the fused arm's down stage (`Accepted.routeFusedDown`): the 128-row down text at BN 128 with rot_widen1 as
        /// its epilogue
        gemm_dn_w1: *const Entry,
        /// the 128-row gate|up text with the mul1h codebook through a threadgroup table: registered, not launched by
        /// the route
        gemm_gu128lut: *const Entry,
        /// the K = 2 twins of the launched GEMMs (layers whose code tiles are 32 int16 wide)
        gemm_gu128_k2: *const Entry,
        gemm_dn128_k2: *const Entry,
        gemm_dn_w1_k2: *const Entry,
        /// M1 (`Accepted.routeM1`): the 128-row GEMMs with the B stage double-buffered (K3, K2; the same words), and
        /// whether the waves' GEMMs run on them in the 128-row texts' place
        gemm_gu128d: *const Entry,
        gemm_dn128d: *const Entry,
        gemm_gu128d_k2: *const Entry,
        gemm_dn128d_k2: *const Entry,
        m1_on: bool = false,

        pub fn init(reg: *const xk.Registry) Self {
            return .{
                .gemm_gu = reg.get(.q3_prefill_dig_gemm_5120x2304_gu_xmul1hk3),
                .gemm_dn = reg.get(.q3_prefill_dig_gemm_2304x5120_xmul1hk3),
                .gemm_gu128 = reg.get(.dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk3_m128),
                .gemm_dn128 = reg.get(.dsv41_prefill_dig_gemm_2304x5120_xmul1hk3_m128),
                .take2_e = reg.get(.q3_prefill_dig_rot_take2_5120),
                .take2v_e = reg.get(.dsv41_prefill_dig_take2v_5120),
                .roundx_e = reg.get(.q3_prefill_dig_rot_roundx_2304),
                .onepass_e = reg.get(.q3_prefill_dig2_swiglu_2304_x),
                .widen2_e = reg.get(.q3_prefill_dig_rot_widen2_2304),
                .widen1_e = reg.get(.q3_prefill_dig_rot_widen1_5120),
                .gemm_dn_w1 = reg.get(.dsv41_prefill_dig_gemm_2304x5120_xmul1hk3_m128w1),
                .gemm_gu128lut = reg.get(.dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk3_m128lut),
                .gemm_gu128_k2 = reg.get(.dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk2_m128),
                .gemm_dn128_k2 = reg.get(.dsv41_prefill_dig_gemm_2304x5120_xmul1hk2_m128),
                .gemm_dn_w1_k2 = reg.get(.dsv41_prefill_dig_gemm_2304x5120_xmul1hk2_m128w1),
                .gemm_gu128d = reg.get(.dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk3_m128d),
                .gemm_dn128d = reg.get(.dsv41_prefill_dig_gemm_2304x5120_xmul1hk3_m128d),
                .gemm_gu128d_k2 = reg.get(.dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk2_m128d),
                .gemm_dn128d_k2 = reg.get(.dsv41_prefill_dig_gemm_2304x5120_xmul1hk2_m128d),
            };
        }

        /// The routed GEMMs' M tile rows: their wave tables count ceil(rows / m_tile) tiles per expert (`digTableBm`).
        pub const m_tile: u32 = 128;

        /// The GEMMs' threadgroups per M tile: gate|up (both operands), down (the same at 64 and 128 rows).
        pub fn digTiles(self: *const Self, proj: enum { gate_up, down }) u32 {
            const e = if (proj == .down) self.gemm_dn else self.gemm_gu;
            return argOf(e, "tbl").domain.tiles;
        }

        /// x0 / x1 [rows, 1, 5120] f16 (take2), gate / up code, the gate|up table at `m_tile` rows -> (zg, zu)
        /// [rows, 2304] f32: the 128-row text, the 64-row text's words.
        pub fn gemmGateUp(self: *const Self, g: *G, x0: G.T, x1: G.T, code_g: G.T, code_u: G.T, tbl: G.T, tgs: u32) ![2]G.T {
            var vars = rowsVars(rowsOf(G, g, x0, 0));
            vars.set(.tgs, tgs);
            const k2 = (try kIndexOf(G, g, code_g)) == 1;
            // M1: the double-buffered text in the 128-row text's place
            const dbuf = if (k2) self.gemm_gu128d_k2 else self.gemm_gu128d;
            var out: [2]G.T = undefined;
            try launchRule(G, g, if (self.m1_on) dbuf else if (k2) self.gemm_gu128_k2 else self.gemm_gu128, &vars, &.{ x0, x1, code_g, code_u, tbl }, &out);
            return out;
        }

        /// `gemmGateUp` through the table-codebook text: the same inputs and words.
        pub fn gemmGateUpLut(self: *const Self, g: *G, x0: G.T, x1: G.T, code_g: G.T, code_u: G.T, tbl: G.T, tgs: u32) ![2]G.T {
            var vars = rowsVars(rowsOf(G, g, x0, 0));
            vars.set(.tgs, tgs);
            var out: [2]G.T = undefined;
            try launchRule(G, g, self.gemm_gu128lut, &vars, &.{ x0, x1, code_g, code_u, tbl }, &out);
            return out;
        }

        /// x [rows, 1, 2304] f16 (onepass), down code, down rout, its table (`widenTiles` N tiles at `m_tile` rows) ->
        /// widen1(z) [rows, 5120] f32 in one launch: the words of `gemmDown`, then `widen1`.
        pub fn gemmDownWiden(self: *const Self, g: *G, x: G.T, code_d: G.T, rout_d: G.T, tbl: G.T, tgs: u32) !G.T {
            var vars = rowsVars(rowsOf(G, g, x, 0));
            vars.set(.tgs, tgs);
            var out: [1]G.T = undefined;
            try launchRule(G, g, if ((try kIndexOf(G, g, code_d)) == 1) self.gemm_dn_w1_k2 else self.gemm_dn_w1, &vars, &.{ x, code_d, rout_d, tbl }, &out);
            return out[0];
        }

        /// The fused down GEMM's threadgroups per M tile (its BN 128 column tiles).
        pub fn widenTiles(self: *const Self) u32 {
            return argOf(self.gemm_dn_w1, "tbl").domain.tiles;
        }

        /// x [rows, 1, 2304] f16 (onepass), down code, the down table at `m_tile` rows -> z [rows, 5120] f32: the
        /// 128-row text, the 64-row text's words.
        pub fn gemmDown(self: *const Self, g: *G, x: G.T, code_d: G.T, tbl: G.T, tgs: u32) !G.T {
            var vars = rowsVars(rowsOf(G, g, x, 0));
            vars.set(.tgs, tgs);
            const k2 = (try kIndexOf(G, g, code_d)) == 1;
            // M1: the double-buffered text in the 128-row text's place
            const dbuf = if (k2) self.gemm_dn128d_k2 else self.gemm_dn128d;
            var out: [1]G.T = undefined;
            try launchRule(G, g, if (self.m1_on) dbuf else if (k2) self.gemm_dn128_k2 else self.gemm_dn128, &vars, &.{ x, code_d, tbl }, &out);
            return out[0];
        }

        /// act [A, 5120] bf16 rows ridx [rows] i32, rhs [rows] u32 (expert per row), slots = a
        /// table -> (f16(t128(act * rin_g[slot])), f16(t128(act * rin_u[slot]))) [rows, 1, 5120]. The retune
        /// (`dsv41_prefill_dig_take2v_5120`): one simdgroup per (row, 128-block) writes both outputs from one act
        /// load, the lane text's butterflies in its bit order, so its words are the lane's (the device self-check
        /// `mlx_chain`, bitwise; the 0b smoke against the lane text on real records).
        pub fn take2(self: *const Self, g: *G, act: G.T, ridx: G.T, rhs: G.T, slots: G.T, rin_g: G.T, rin_u: G.T) ![2]G.T {
            const vars = rowsVars(rowsOf(G, g, ridx, 0));
            var out: [2]G.T = undefined;
            try launchRule(G, g, self.take2v_e, &vars, &.{ act, ridx, rhs, slots, rin_g, rin_u }, &out);
            return out;
        }

        /// act [rows, 1, 2304] f32 -> f16(t128(act * rin_d[slot])) [rows, 1, 2304].
        pub fn roundx(self: *const Self, g: *G, act: G.T, rhs: G.T, slots: G.T, rin_d: G.T) !G.T {
            const vars = rowsVars(rowsOf(G, g, act, 0));
            var out: [1]G.T = undefined;
            try launchRule(G, g, self.roundx_e, &vars, &.{ act, rhs, slots, rin_d }, &out);
            return out[0];
        }

        /// zg / zu [rows, 2304] f32 -> hd = f16(t128(clamped SwiGLU(t128(zg) rout_g, t128(zu) rout_u) rin_d)) [rows, 1, 2304].
        pub fn onePass(self: *const Self, g: *G, zg: G.T, zu: G.T, rhs: G.T, tbl: G.T, rout_g: G.T, rout_u: G.T, rin_d: G.T) !G.T {
            const vars = rowsVars(rowsOf(G, g, zg, 0));
            var out: [1]G.T = undefined;
            try launchRule(G, g, self.onepass_e, &vars, &.{ zg, zu, rhs, tbl, rout_g, rout_u, rin_d }, &out);
            return out[0];
        }

        /// act_g / act_u [rows, 2304] f32 -> t128(act) * rout[slot] [rows, 1, 2304] f32 each.
        pub fn widen2(self: *const Self, g: *G, act_g: G.T, act_u: G.T, rhs: G.T, slots: G.T, rout_g: G.T, rout_u: G.T) ![2]G.T {
            const vars = rowsVars(rowsOf(G, g, act_g, 0));
            var out: [2]G.T = undefined;
            try launchRule(G, g, self.widen2_e, &vars, &.{ act_g, act_u, rhs, slots, rout_g, rout_u }, &out);
            return out;
        }

        /// act [rows, 5120] f32 -> t128(act) * rout_d[slot] [rows, 1, 5120] f32.
        pub fn widen1(self: *const Self, g: *G, act: G.T, rhs: G.T, slots: G.T, rout_d: G.T) !G.T {
            const vars = rowsVars(rowsOf(G, g, act, 0));
            var out: [1]G.T = undefined;
            try launchRule(G, g, self.widen1_e, &vars, &.{ act, rhs, slots, rout_d }, &out);
            return out[0];
        }

        /// `Dig2GemmOnePassX`: take2 -> the gate|up GEMM -> onepass: (hd, zg, zu).
        pub fn gateUpOnePass(self: *const Self, g: *G, act: G.T, ridx: G.T, rhs: G.T, gu: DigTableArray(G), gate: ProjArrays(G.T), up: ProjArrays(G.T), down_rin: G.T) ![3]G.T {
            const x = try self.take2(g, act, ridx, rhs, gu.tbl, gate.rin, up.rin);
            const z = try self.gemmGateUp(g, x[0], x[1], gate.code, up.code, gu.tbl, gu.tgs);
            const hd = try self.onePass(g, z[0], z[1], rhs, gu.tbl, gate.rout, up.rout, down_rin);
            return .{ hd, z[0], z[1] };
        }
    };
}

/// A wave table on the device and its threadgroup count (`digTableBm`).
pub fn DigTableArray(comptime G: type) type {
    return struct { tbl: G.T, tgs: u32 };
}

// ── Prefill: the routed wave dispatch (DIG-X, DIG2 onepass, SHAPE balance / rebuildahead / carry) ──

/// The installed prefill wave shape: the fused point's wave / inflight / row budget
/// (Q3_PREFILL_FUSED_INSTALL config) and SHAPE's carry rows (Q3_PREFILL_SHAPE_INSTALL).
pub const PrefillShape = struct {
    /// experts per wave (1..16, the wave table's expert rows)
    wave: u32,
    /// waves in flight (>= 2, SHAPE's overlap route): a wave first waits for all but `inflight - 1` older ones
    inflight: u32,
    /// assignment rows per wave; an expert above it forms a wave alone (solo: drained before, evaluated after)
    row_budget: u32,
    /// a call of at most `carry_rows` rows leaves its last waves in flight (to the next call or `finish`)
    carry_rows: u32,

    /// Record 3 (run 3r-record3-fast-typical-exl3-30-guard-20260928.log): the lane's shape, the one its wave
    /// samples and the move's pinned launches were taken at.
    pub const record3: PrefillShape = .{ .wave = 4, .inflight = 2, .row_budget = 7168, .carry_rows = 8192 };
    /// The served tier: Record 3 with L1, 8 experts per wave. Under K16 a group call's experts carry about 256
    /// rows each (98,304 routed rows over 384 experts), so Record 3's four-expert cap bound, not the row budget:
    /// eight halve the waves, and with them the per-wave host encode (five launches, four host arrays, one async
    /// eval). Exact: a wave's rows are independent of its composition (each 64-row tile reads one expert's rows
    /// and weights; the join restores the assignment order).
    pub const tier: PrefillShape = .{ .wave = 8, .inflight = record3.inflight, .row_budget = record3.row_budget, .carry_rows = record3.carry_rows };
};

/// A layer bank's three projections (the streamer's `BankArrays`).
pub fn BankArrays(comptime T: type) type {
    return quant.BankArrays(ProjArrays(T));
}

/// One call's routed rows: `slot[i]` = assignment row i's bank slot (its binding's bank_index).
/// Row i reads act row i (the switch's `selected`), or act row `act_row[i]` when given (the
/// chunk's tokens and position / top_k: the same words without the switch's row take).
pub const PrefillRows = quant.PrefillRows;
// ── the scratch waves: the DIG-X wave launches' write-into twins ──
/// The wave launches an arena wave runs on its set's regions (the K3 and K2 GEMMs apart).
pub const IntoTwin = enum { take2v, gemm_gu_k3, gemm_gu_k2, onepass, gemm_dn_k3, gemm_dn_k2, widen1 };
/// Each twin's stock text (the chain down with the bf16 expert outputs, `routeExpertBf16`'s rot_widen1), where that
/// text keeps its `fp contract` pragma (wave_twins.zig anchor), and whether the twin is a dep twin
/// (the stock text with `dep` bound, its outputs MLX's own: rot_widen1's y, the combine's part).
const IntoSpec = struct { k: Kernel, pragma: sw.twins.Pragma, dep_only: bool = false };
const into_specs = std.enums.EnumArray(IntoTwin, IntoSpec).init(.{
    .take2v = .{ .k = .dsv41_prefill_dig_take2v_5120, .pragma = .first },
    .gemm_gu_k3 = .{ .k = .dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk3_m128, .pragma = .none },
    .gemm_gu_k2 = .{ .k = .dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk2_m128, .pragma = .none },
    .onepass = .{ .k = .q3_prefill_dig2_swiglu_2304_x, .pragma = .none },
    .gemm_dn_k3 = .{ .k = .dsv41_prefill_dig_gemm_2304x5120_xmul1hk3_m128, .pragma = .none },
    .gemm_dn_k2 = .{ .k = .dsv41_prefill_dig_gemm_2304x5120_xmul1hk2_m128, .pragma = .none },
    .widen1 = .{ .k = .dsv41_prefill_dig_rot_widen1_5120_obf16, .pragma = .first, .dep_only = true },
});
pub const IntoError = error{ ScratchWavesTwin, ScratchWavesNeedsBf16Chain, ScratchWavesNeedsMlx };
/// Each twin's stock text: into_specs', or with M1 routed (`Accepted.routeM1`) the double-buffered 128-row GEMMs in the
/// GEMM twins' place (the same words; the derivation's anchors hold for both texts: no pragma, no name a twin adds).
fn intoKernels(m1_on: bool) std.enums.EnumArray(IntoTwin, Kernel) {
    var out: std.enums.EnumArray(IntoTwin, Kernel) = undefined;
    for (std.enums.values(IntoTwin)) |t| out.set(t, into_specs.get(t).k);
    if (m1_on) {
        out.set(.gemm_gu_k3, .dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk3_m128d);
        out.set(.gemm_gu_k2, .dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk2_m128d);
        out.set(.gemm_dn_k3, .dsv41_prefill_dig_gemm_2304x5120_xmul1hk3_m128d);
        out.set(.gemm_dn_k2, .dsv41_prefill_dig_gemm_2304x5120_xmul1hk2_m128d);
    }
    return out;
}
fn intoRefuse(diag: ?*Diag, err: IntoError, comptime fmt: []const u8, args: anytype) IntoError {
    if (diag) |d| d.set(fmt, args);
    return err;
}
/// The twins' MLX kernels: made at construction from the registry's pinned stock texts and headers (`init`: each
/// derived by wave_twins.zig, its anchors checked, a refusal naming the kernel) and compiled then
/// (`compileAll`: one launch of each at one row, evaluated). A launch takes its stock entry's own rule (`xk.launchFor`):
/// a write-into twin's inputs are the stock's, each output's region (whole, an evaluated array), then `dep`, and its one
/// output the [1] uint32 carrier; a dep twin's inputs are the stock's then `dep`, its outputs the stock's.
pub const IntoTwins = struct {
    reg: *const xk.Registry,
    kernels: std.enums.EnumArray(IntoTwin, mlx.mlx_fast_metal_kernel) = .initFill(.{ .ctx = null }),
    /// each twin's stock text (`intoKernels`: into_specs', or M1's double-buffered GEMMs)
    ks: std.enums.EnumArray(IntoTwin, Kernel),

    /// `m1_on` (M1 routed): the GEMM twins derive from the double-buffered 128-row texts (the same words).
    pub fn init(a: Allocator, reg: *const xk.Registry, m1_on: bool, diag: ?*Diag) (IntoError || error{OutOfMemory})!IntoTwins {
        var self: IntoTwins = .{ .reg = reg, .ks = intoKernels(m1_on) };
        errdefer self.deinit();
        for (std.enums.values(IntoTwin)) |t| {
            const spec = self.specOf(t);
            const e = reg.get(spec.k);
            const hdr = reg.header(e);
            if (e.outputs.len == 0 or (spec.dep_only and e.outputs.len != 1) or e.inputs.len + e.outputs.len + 1 > xk.max_inputs)
                return intoRefuse(diag, error.ScratchWavesTwin, "exl3 scratch waves: {t}: {d} inputs and {d} outputs, no twin", .{ spec.k, e.inputs.len, e.outputs.len });
            var in_names: [xk.max_inputs][*:0]const u8 = undefined;
            for (e.inputs, 0..) |arg, i| in_names[i] = arg.name.ptr;
            if (spec.dep_only) {
                sw.twins.checkDep(e.source, hdr) catch |err|
                    return intoRefuse(diag, error.ScratchWavesTwin, "exl3 scratch waves: {t}: the stock text is not the dep twin's ({s})", .{ spec.k, @errorName(err) });
                in_names[e.inputs.len] = "dep";
                var out_names: [xk.max_outputs][*:0]const u8 = undefined;
                for (e.outputs, 0..) |o, i| out_names[i] = o.name.ptr;
                const twin_name = try std.fmt.allocPrintSentinel(a, "{s}_dep", .{e.mlx_name}, 0);
                defer a.free(twin_name);
                const k = kset.newKernel(twin_name.ptr, in_names[0 .. e.inputs.len + 1], out_names[0..e.outputs.len], e.source.ptr, hdr.ptr, e.ensure_row_contiguous) orelse
                    return intoRefuse(diag, error.ScratchWavesTwin, "exl3 scratch waves: {t}: its dep twin's mlx_fast_metal_kernel_new failed", .{spec.k});
                self.kernels.set(t, k);
                continue;
            }
            var outs: [xk.max_outputs]sw.twins.Out = undefined;
            for (e.outputs, 0..) |o, i| outs[i] = .{ .name = o.name, .mtype = sw.twins.metalType(@tagName(o.dtype)) orelse
                return intoRefuse(diag, error.ScratchWavesTwin, "exl3 scratch waves: {t}: output {s} is {t}, no twin type", .{ spec.k, o.name, o.dtype }) };
            const text = sw.twins.derive(a, e.source, hdr, outs[0..e.outputs.len], spec.pragma) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                else => return intoRefuse(diag, error.ScratchWavesTwin, "exl3 scratch waves: {t}: the stock text is not the twin's ({s})", .{ spec.k, @errorName(err) }),
            };
            defer a.free(text);
            var dst: [xk.max_outputs][:0]u8 = undefined;
            var n_dst: usize = 0;
            defer for (dst[0..n_dst]) |d| a.free(d);
            for (e.outputs, 0..) |o, i| {
                dst[i] = try std.fmt.allocPrintSentinel(a, "{s}_dst", .{o.name}, 0);
                n_dst += 1;
                in_names[e.inputs.len + i] = dst[i].ptr;
            }
            const n_in = e.inputs.len + e.outputs.len + 1;
            in_names[n_in - 1] = "dep";
            const twin_name = try std.fmt.allocPrintSentinel(a, "{s}_into", .{e.mlx_name}, 0);
            defer a.free(twin_name);
            const out_names = [_][*:0]const u8{"carrier"};
            const k = kset.newKernel(twin_name.ptr, in_names[0..n_in], &out_names, text.ptr, hdr.ptr, e.ensure_row_contiguous) orelse
                return intoRefuse(diag, error.ScratchWavesTwin, "exl3 scratch waves: {t}: its twin's mlx_fast_metal_kernel_new failed", .{spec.k});
            self.kernels.set(t, k);
        }
        return self;
    }
    pub fn deinit(self: *IntoTwins) void {
        for (std.enums.values(IntoTwin)) |t| {
            const k = self.kernels.get(t);
            if (k.ctx != null) mtl.mlx_fast_metal_kernel_free(k);
            self.kernels.set(t, .{ .ctx = null });
        }
    }
    /// Twin `t`'s spec with the stock text this set derives it from (`ks`).
    fn specOf(self: *const IntoTwins, t: IntoTwin) IntoSpec {
        var sp = into_specs.get(t);
        sp.k = self.ks.get(t);
        return sp;
    }
    /// One launch of twin `t` at its stock entry's rule for `vars` on `s`: returns its one output (a write-into twin's
    /// carrier, a dep twin's stock output; a new array the caller owns).
    pub fn launch(self: *const IntoTwins, s: mlx.mlx_stream, t: IntoTwin, vars: *const Vars, inputs: []const mlx.mlx_array) !mlx.mlx_array {
        const spec = self.specOf(t);
        const e = self.reg.get(spec.k);
        const cfg = xk.launchFor(e, vars, null) catch unreachable;
        const one = [_]c_int{1};
        var shape: []const c_int = &one;
        var dtype: mlx.mlx_dtype = .uint32;
        if (spec.dep_only) {
            std.debug.assert(inputs.len == e.inputs.len + 1 and cfg.n_out == 1);
            shape = cfg.out_shapes[0][0..cfg.out_ranks[0]];
            dtype = cfg.out_dtypes[0];
        } else std.debug.assert(inputs.len == e.inputs.len + e.outputs.len + 1);
        const c = try kset.buildConfig(.{ .grid = cfg.grid, .threadgroup = cfg.threadgroup, .out_shapes = &.{shape}, .out_dtypes = &.{dtype}, .template = cfg.template });
        defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        var out: [1]mlx.mlx_array = undefined;
        try kset.applyConfig(self.kernels.get(t), c, inputs, &out, s);
        return out[0];
    }
    /// At construction: each twin launched once at one row on `s` and evaluated (zeros; its wave table that of one
    /// expert at slot 0, at `tiles_gu` / `tiles_dn` threadgroups an M tile), so MLX compiles every twin now and a twin
    /// that does not compile or complete refuses the load by name.
    pub fn compileAll(self: *const IntoTwins, s: mlx.mlx_stream, tiles_gu: u32, tiles_dn: u32, diag: ?*Diag) !void {
        const one_expert = [_]WaveExpert{.{ .slot = 0, .rows = 1 }};
        const gu = digTableBm(&one_expert, tiles_gu, 128);
        const dn = digTableBm(&one_expert, tiles_dn, 128);
        for (std.enums.values(IntoTwin)) |t| {
            const spec = self.specOf(t);
            const e = self.reg.get(spec.k);
            const tbl = switch (t) {
                .gemm_dn_k3, .gemm_dn_k2, .widen1 => &dn,
                else => &gu,
            };
            var vars: Vars = .initFill(1);
            vars.set(.tgs, tbl.tgs);
            var ins: [xk.max_inputs]mlx.mlx_array = undefined;
            var n: usize = 0;
            defer for (ins[0..n]) |x| {
                _ = mlx.mlx_array_free(x);
            };
            for (e.inputs) |*arg| {
                if (arg.role == .static) return intoRefuse(diag, error.ScratchWavesTwin, "exl3 scratch waves: {t}: static input {s}, no compile launch", .{ spec.k, arg.name });
                ins[n] = if (arg.role == .table) mlx.mlx_array_new_data(&tbl.table, &[_]c_int{80}, 1, .int32) else try zerosAt(arg.shape, arg.dtype, &vars, s);
                n += 1;
            }
            if (!spec.dep_only) for (e.outputs) |*o| {
                ins[n] = try zerosAt(o.shape, o.dtype, &vars, s);
                n += 1;
            };
            ins[n] = mlx.mlx_array_new_data(&[_]u32{0}, &[_]c_int{1}, 1, .uint32);
            n += 1;
            const out = self.launch(s, t, &vars, ins[0..n]) catch
                return intoRefuse(diag, error.ScratchWavesTwin, "exl3 scratch waves: {t}: its twin's launch failed at construction", .{spec.k});
            defer _ = mlx.mlx_array_free(out);
            mlx.check(mlx.mlx_array_eval(out)) catch
                return intoRefuse(diag, error.ScratchWavesTwin, "exl3 scratch waves: {t}: its twin did not compile or complete a launch", .{spec.k});
        }
    }
    fn zerosAt(shape: []const xk.Dim, dt: Dtype, vars: *const Vars, s: mlx.mlx_stream) !mlx.mlx_array {
        var sh: [xk.max_rank]c_int = undefined;
        for (shape, 0..) |d, i| sh[i] = @intCast(d.eval(vars));
        var z = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(z);
        try mlx.check(mlx.mlx_zeros(&z, &sh, shape.len, dt, s));
        return z;
    }
};

/// The lane of record's `Tcq3FusedPrefillDispatch...__dig2_onepass_exl3.__call__` over one call's
/// routed rows (one route per layer, as the lane has one dispatcher per layer): experts grouped
/// by slot in first-appearance order, snake-ordered (largest, smallest, ...; stable), packed
/// greedily into waves of <= `wave` experts and <= `row_budget` rows; per wave rot_take2 -> the
/// gate|up GEMM (72-tile table) -> dig2 onepass -> the down GEMM (80-tile table) -> rot_widen1
/// (the route: the take2 retune and the 128-row GEMMs; the fused arm launches the fused down GEMM, the
/// 128-row down text with rot_widen1 as its epilogue (40-tile table), in place of the last two: the lane's
/// words either way);
/// then `take(concatenate(waves), argsort(positions))`, the permutation made on the host. The
/// eval schedule is SHAPE's: a wave waits for all but `inflight - 1` older waves (a solo wave for
/// all, and is evaluated at once); a call above `carry_rows` rows drains every wave and evaluates
/// the join and the result, a carried call leaves its last waves in flight and async-evaluates
/// the result; `finish` (the prefill boundary) drains them. The lane's rebuild-ahead is the next
/// wave's two host tables (no GPU work) and has no counterpart here.
///
/// Wave lifecycle (the backend's `mark` / `resetTo`: `resetTo` frees every array tracked since
/// the mark, kept handles survive): each wave's ops run between a mark and a `resetTo` right
/// after the wave is submitted, so its intermediates live only while the GPU still needs them
/// (a pending graph holds its inputs); the wave's output survives as two kept handles (the
/// join's, the in-flight queue's). The join runs between its own mark and `resetTo`; `call`
/// returns a KEPT result (the caller releases it). Nothing else the call builds outlives it.
pub fn DigXPrefill(comptime G: type) type {
    if (!@hasDecl(G, "mark") or !@hasDecl(G, "resetTo"))
        @compileError("exl3 kernel ops: DigXPrefill needs a backend with mark / resetTo (the wave lifecycle)");
    return struct {
        const Self = @This();
        /// G7: the backend's prompt-pass split (the arch injects it; off on every other backend and build).
        const prof = sdk_ext.profile.of(G).prefill;
        const hidden = 5120;
        /// The wave's down stage, installed at construction (`installDown`): the 128-row down text then rot_widen1
        /// (stock), or the fused down GEMM (the same words in one launch). Each builds its own table.
        pub const Down = enum { chain, fused };
        const DownStage = *const fn (self: *const Self, g: *G, hd: G.T, bank: BankArrays(G.T), ex: []const WaveExpert, rhs: G.T) anyerror!G.T;
        dig: DigX(G),
        shape: PrefillShape,
        tiles_gu: u32,
        tiles_dn: u32,
        /// the fused down GEMM's threadgroups per M tile (`DigX.widenTiles`)
        tiles_w: u32,
        rows_hi: u64,
        down: DownStage = downChain,
        /// The scratch waves (`Accepted.routeScratchWaves`): an uncarried call's waves launch these twins into
        /// the per-prompt regions; `arena_waves` counts them (a wave's set is its count's parity: two in flight never
        /// share one).
        into: ?*const IntoTwins = null,
        arena_waves: u64 = 0,
        /// one wave's launches: the take2 retune, the gate|up GEMM, onepass, then the down stage's (2 stock, 1 fused)
        wave_launches: u32 = 5,
        a: Allocator,
        diag: ?*xk.Diag,
        /// waves in flight, oldest first (kept handles; persist across carried calls)
        flight: std.ArrayList(G.T) = .empty,
        parts: std.ArrayList(G.T) = .empty,
        // host scratch, reused across calls
        group_of: std.ArrayList(i32) = .empty,
        gslot: std.ArrayList(u32) = .empty,
        gcount: std.ArrayList(u32) = .empty,
        gnext: std.ArrayList(u32) = .empty,
        snake: std.ArrayList(u32) = .empty,
        sorted: std.ArrayList(u32) = .empty,
        grows: std.ArrayList(u32) = .empty,
        pos: std.ArrayList(u32) = .empty,
        ridx: std.ArrayList(i32) = .empty,
        rhs: std.ArrayList(u32) = .empty,
        inv: std.ArrayList(u32) = .empty,

        /// `diag` (optional) receives the refusal messages of `init` and of every call.
        pub fn init(a: Allocator, reg: *const xk.Registry, shape: PrefillShape, diag: ?*xk.Diag) Refusal!Self {
            if (shape.wave < 1 or shape.wave > wave_max or shape.inflight < 2 or shape.row_budget < 1)
                return refuse(diag, error.RouteInput, "exl3 kernel ops: prefill shape wave {d} (1..{d}), inflight {d} (>= 2: SHAPE's overlap route), row budget {d} (>= 1)", .{ shape.wave, wave_max, shape.inflight, shape.row_budget });
            const dig = DigX(G).init(reg);
            return .{ .dig = dig, .shape = shape, .tiles_gu = dig.digTiles(.gate_up), .tiles_dn = dig.digTiles(.down), .tiles_w = dig.widenTiles(), .rows_hi = dig.gemm_gu.bounds.get(.rows).?[1], .a = a, .diag = diag };
        }

        /// Releases the waves still in flight (without evaluating them) and the scratch.
        pub fn deinit(self: *Self, g: *G) void {
            for (self.flight.items) |x| g.release(x);
            for (self.parts.items) |x| g.release(x);
            self.flight.deinit(self.a);
            self.parts.deinit(self.a);
            inline for (.{ &self.group_of, &self.gslot, &self.gcount, &self.gnext, &self.snake, &self.sorted, &self.grows, &self.pos, &self.ridx, &self.rhs, &self.inv }) |l| l.deinit(self.a);
        }

        /// The prefill boundary: every wave still in flight evaluated, oldest first.
        pub fn finish(self: *Self, g: *G) !void {
            const t = prof.now();
            while (self.flight.items.len > 0) try self.drainOne(g);
            prof.charge(.drain, t);
        }

        fn drainOne(self: *Self, g: *G) !void {
            const x = self.flight.orderedRemove(0);
            defer g.release(x);
            try g.evalAll(&.{x});
        }

        /// act bf16 [a_rows, 5120], `rows` (A assignment rows), the call's bank -> a KEPT f32
        /// [A, 5120] in assignment-row order (the lane's `result`; release it). The caller's route guarantees A within
        /// 1..the kernels' row bound, every slot inside the bank and the act rows (asserted, not re-checked).
        /// The call's waves (every wave's output KEPT in `parts`, wave order; `pos` maps each wave-ordered
        /// row to its assignment row): true when the call is carried (its last waves left in flight).
        fn runWaves(self: *Self, g: *G, act: G.T, rows: PrefillRows, bank: BankArrays(G.T)) !bool {
            const n_rows = rows.slot.len;
            std.debug.assert(!(n_rows == 0 or n_rows > self.rows_hi));
            const a_rows = rowsOf(G, g, act, 0);
            std.debug.assert(!(rows.act_row == null and a_rows != n_rows));
            if (rows.act_row) |ar| std.debug.assert(ar.len == n_rows);
            var tp = prof.now();
            prof.countCall();
            try self.group(rows, rowsOf(G, g, bank.gate.code, 0), a_rows);
            const carried = n_rows <= self.shape.carry_rows;
            const a = self.a;
            const cnt = self.gcount.items;
            const order = self.snake.items;
            for (self.parts.items) |x| g.release(x);
            self.parts.clearRetainingCapacity();
            var i: usize = 0;
            var off: usize = 0;
            while (i < order.len) {
                const first = i;
                var wave_rows: usize = cnt[order[i]];
                i += 1;
                while (i < order.len and i - first < self.shape.wave and wave_rows + cnt[order[i]] <= self.shape.row_budget) : (i += 1) wave_rows += cnt[order[i]];
                const solo = wave_rows > self.shape.row_budget;
                const keep: usize = if (solo) 0 else self.shape.inflight - 1;
                prof.charge(.encode, tp);
                tp = prof.now();
                while (self.flight.items.len > keep) try self.drainOne(g);
                prof.charge(.drain, tp);
                tp = prof.now();
                prof.count(1, self.wave_launches);
                var ex: [wave_max]WaveExpert = undefined;
                for (order[first..i], 0..) |gi, j| {
                    ex[j] = .{ .slot = self.gslot.items[gi], .rows = cnt[gi] };
                    @memset(self.rhs.items[off..][0..cnt[gi]], @intCast(j));
                    off += cnt[gi];
                }
                const r0 = off - wave_rows;
                try self.parts.ensureUnusedCapacity(a, 1);
                try self.flight.ensureUnusedCapacity(a, 1);
                const m = g.mark();
                // The scratch waves: an uncarried call's wave (not solo) on the regions, the stock words.
                if (comptime G.T == mlx.mlx_array) if (self.into) |tw| if (!carried and !solo) {
                    if (try self.submitArena(g, tw, act, bank, ex[0 .. i - first], self.ridx.items[r0..off], self.rhs.items[r0..off])) |ya| {
                        self.parts.appendAssumeCapacity(g.keep(ya));
                        try g.asyncEval(&.{ya});
                        self.flight.appendAssumeCapacity(g.keep(ya));
                        g.resetTo(m);
                        continue;
                    }
                };
                const y = try self.submit(g, act, bank, ex[0 .. i - first], self.ridx.items[r0..off], self.rhs.items[r0..off]);
                self.parts.appendAssumeCapacity(g.keep(y));
                if (solo) {
                    prof.charge(.encode, tp);
                    tp = prof.now();
                    try g.evalAll(&.{y});
                    prof.charge(.drain, tp);
                    tp = prof.now();
                } else {
                    try g.asyncEval(&.{y});
                    self.flight.appendAssumeCapacity(g.keep(y));
                }
                g.resetTo(m);
            }
            prof.charge(.encode, tp);
            return carried;
        }

        pub fn call(self: *Self, g: *G, act: G.T, rows: PrefillRows, bank: BankArrays(G.T)) !G.T {
            const n_rows = rows.slot.len;
            const carried = try self.runWaves(g, act, rows, bank);
            var tp = prof.now();
            if (!carried) while (self.flight.items.len > 0) try self.drainOne(g);
            prof.charge(.drain, tp);
            tp = prof.now();
            const m = g.mark();
            const joined = try g.concat(self.parts.items, 0);
            for (self.parts.items) |x| g.release(x);
            self.parts.clearRetainingCapacity();
            prof.charge(.join, tp);
            tp = prof.now();
            if (!carried) try g.evalAll(&.{joined});
            prof.charge(.drain, tp);
            tp = prof.now();
            for (self.pos.items, 0..) |p, j| self.inv.items[p] = @intCast(j);
            const ord = try g.hostArray(std.mem.sliceAsBytes(self.inv.items), &.{@intCast(n_rows)}, .uint32);
            const result = try g.take(joined, ord, 0);
            const kept = g.keep(result);
            errdefer g.release(kept);
            prof.charge(.join, tp);
            tp = prof.now();
            if (carried) try g.asyncEval(&.{result}) else try g.evalAll(&.{result});
            prof.charge(.drain, tp);
            g.resetTo(m);
            return kept;
        }

        /// `call` without its join (the CALLFEED scatter join's counterpart for a caller that reads
        /// unjoined outputs, JOINLESS): the waves' KEPT outputs are appended to `outs` in wave order
        /// (the caller releases them) and, per wave-ordered row, its assignment row to `pos`. The
        /// values are `call`'s: the same wave arrays, no concatenate or take.
        pub fn callParts(self: *Self, g: *G, act: G.T, rows: PrefillRows, bank: BankArrays(G.T), alloc: Allocator, outs: *std.ArrayList(G.T), pos: *std.ArrayList(u32)) !void {
            const carried = try self.runWaves(g, act, rows, bank);
            const tp = prof.now();
            if (!carried) while (self.flight.items.len > 0) try self.drainOne(g);
            prof.charge(.drain, tp);
            try outs.ensureUnusedCapacity(alloc, self.parts.items.len);
            try pos.appendSlice(alloc, self.pos.items);
            outs.appendSliceAssumeCapacity(self.parts.items);
            self.parts.clearRetainingCapacity();
        }

        /// The down stage (`Down`), before any call: the stage's function and the wave's launch count.
        pub fn installDown(self: *Self, d: Down) void {
            switch (d) {
                .chain => {
                    self.down = downChain;
                    self.wave_launches = 5;
                },
                .fused => {
                    self.down = downFused;
                    self.wave_launches = 4;
                },
            }
        }

        /// One wave's launches (`wave_launches`) -> its rows' output f32 [R, 5120] (the lane's `y.reshape(-1, H)`).
        fn submit(self: *Self, g: *G, act: G.T, bank: BankArrays(G.T), ex: []const WaveExpert, ridx: []const i32, rhs: []const u32) !G.T {
            const n: c_int = @intCast(ridx.len);
            // at the routed GEMMs' M tile; take2 and onepass read the gate|up table's slot column only
            const tg = digTableBm(ex, self.tiles_gu, DigX(G).m_tile);
            const tgu = try g.hostArray(std.mem.sliceAsBytes(&tg.table), &.{80}, .int32);
            const ridx_a = try g.hostArray(std.mem.sliceAsBytes(ridx), &.{n}, .int32);
            const rhs_a = try g.hostArray(std.mem.sliceAsBytes(rhs), &.{n}, .uint32);
            const hz = try self.dig.gateUpOnePass(g, act, ridx_a, rhs_a, .{ .tbl = tgu, .tgs = tg.tgs }, bank.gate, bank.up, bank.down.rin);
            const y = try self.down(self, g, hz[0], bank, ex, rhs_a);
            return g.reshape(y, &.{ n, hidden });
        }
        /// The scratch waves: `submit`'s launches as their twins (`tw`) on the wave's set (its count's parity), each
        /// binding the launch before's carrier as `dep`; null where the regions do not take the wave (the caller submits
        /// it fresh). Returns the wave's y as `submit` returns it ([R, 5120], rot_widen1's dep twin's output).
        fn submitArena(self: *Self, g: *G, tw: *const IntoTwins, act: G.T, bank: BankArrays(G.T), ex: []const WaveExpert, ridx: []const i32, rhs: []const u32) !?G.T {
            const n: c_int = @intCast(ridx.len);
            const v = (try sw.views(@intCast(self.arena_waves & 1), @intCast(ridx.len))) orelse return null;
            // the regions' handles join the wave's scope (freed at its reset); one not taken yet is freed here on an error
            var hs = [_]G.T{ v.x0, v.x1, v.zg, v.zu, v.hd, v.z };
            for (&hs, 0..) |*h, i| h.* = g.adopt(h.*) catch |e| {
                for (hs[i..]) |r| _ = mlx.mlx_array_free(r);
                return e;
            };
            const x0, const x1, const zg, const zu, const hd, const z = hs;
            self.arena_waves += 1;
            const tg = digTableBm(ex, self.tiles_gu, DigX(G).m_tile);
            const tgu = try g.hostArray(std.mem.sliceAsBytes(&tg.table), &.{80}, .int32);
            const ridx_a = try g.hostArray(std.mem.sliceAsBytes(ridx), &.{n}, .int32);
            const rhs_a = try g.hostArray(std.mem.sliceAsBytes(rhs), &.{n}, .uint32);
            const rows = rowsVars(ridx.len);
            var gu = rows;
            gu.set(.tgs, tg.tgs);
            const gu_k2 = (try kIndexOf(G, g, bank.gate.code)) == 1;
            const dn_k2 = (try kIndexOf(G, g, bank.down.code)) == 1;
            const c1 = try intoLaunch(g, tw, .take2v, &rows, &.{ act, ridx_a, rhs_a, tgu, bank.gate.rin, bank.up.rin, x0, x1, v.dep });
            const c2 = try intoLaunch(g, tw, if (gu_k2) .gemm_gu_k2 else .gemm_gu_k3, &gu, &.{ x0, x1, bank.gate.code, bank.up.code, tgu, zg, zu, c1 });
            const c3 = try intoLaunch(g, tw, .onepass, &rows, &.{ zg, zu, rhs_a, tgu, bank.gate.rout, bank.up.rout, bank.down.rin, hd, c2 });
            const td = digTableBm(ex, self.tiles_dn, DigX(G).m_tile);
            const tdn = try g.hostArray(std.mem.sliceAsBytes(&td.table), &.{80}, .int32);
            var dn = rows;
            dn.set(.tgs, td.tgs);
            const c4 = try intoLaunch(g, tw, if (dn_k2) .gemm_dn_k2 else .gemm_dn_k3, &dn, &.{ hd, bank.down.code, tdn, z, c3 });
            const y = try intoLaunch(g, tw, .widen1, &rows, &.{ z, rhs_a, tdn, bank.down.rout, c4 });
            return try g.reshape(y, &.{ n, hidden });
        }
        /// One twin launch whose output joins the wave's scope.
        fn intoLaunch(g: *G, tw: *const IntoTwins, t: IntoTwin, vars: *const Vars, inputs: []const G.T) !G.T {
            const c = try tw.launch(g.s, t, vars, inputs);
            return g.adopt(c) catch |e| {
                _ = mlx.mlx_array_free(c);
                return e;
            };
        }

        /// Stock: the 128-row down text over its 80-tile table, then rot_widen1 (the table's slot column).
        fn downChain(self: *const Self, g: *G, hd: G.T, bank: BankArrays(G.T), ex: []const WaveExpert, rhs: G.T) anyerror!G.T {
            const td = digTableBm(ex, self.tiles_dn, DigX(G).m_tile);
            const tdn = try g.hostArray(std.mem.sliceAsBytes(&td.table), &.{80}, .int32);
            const zd = try self.dig.gemmDown(g, hd, bank.down.code, tdn, td.tgs);
            return self.dig.widen1(g, zd, rhs, tdn, bank.down.rout);
        }

        /// Fused: the fused down GEMM over its 40-tile table (rot_widen1 as its epilogue reads the table's slot column).
        fn downFused(self: *const Self, g: *G, hd: G.T, bank: BankArrays(G.T), ex: []const WaveExpert, rhs: G.T) anyerror!G.T {
            _ = rhs;
            const tw = digTableBm(ex, self.tiles_w, DigX(G).m_tile);
            const tdw = try g.hostArray(std.mem.sliceAsBytes(&tw.table), &.{80}, .int32);
            return self.dig.gemmDownWiden(g, hd, bank.down.code, bank.down.rout, tdw, tw.tgs);
        }

        /// The host plan: groups (slot -> rows, first-appearance order, rows ascending), the snake
        /// order, and per wave-ordered row its assignment row (`pos`) and act row (`ridx`).
        fn group(self: *Self, rows: PrefillRows, cap: u64, a_rows: u64) !void {
            const a = self.a;
            const n = rows.slot.len;
            try self.group_of.resize(a, @intCast(cap));
            @memset(self.group_of.items, -1);
            self.gslot.clearRetainingCapacity();
            self.gcount.clearRetainingCapacity();
            for (rows.slot, 0..) |s, i| {
                std.debug.assert(!(s >= cap));
                if (rows.act_row) |ar| std.debug.assert(ar[i] < a_rows);
                var gi = self.group_of.items[s];
                if (gi < 0) {
                    gi = @intCast(self.gslot.items.len);
                    self.group_of.items[s] = gi;
                    try self.gslot.append(a, s);
                    try self.gcount.append(a, 0);
                }
                self.gcount.items[@intCast(gi)] += 1;
            }
            const ng = self.gslot.items.len;
            // each group's rows, ascending (the lane appends rows in order)
            try self.gnext.resize(a, ng);
            var start: u32 = 0;
            for (self.gcount.items, self.gnext.items) |c, *nx| {
                nx.* = start;
                start += c;
            }
            try self.grows.resize(a, n);
            for (rows.slot, 0..) |s, i| {
                const gi: usize = @intCast(self.group_of.items[s]);
                self.grows.items[self.gnext.items[gi]] = @intCast(i);
                self.gnext.items[gi] += 1;
            }
            // snake over the rows-descending order (stable: ties keep first appearance)
            try self.sorted.resize(a, ng);
            for (self.sorted.items, 0..) |*v, i| v.* = @intCast(i);
            std.mem.sort(u32, self.sorted.items, @as([]const u32, self.gcount.items), struct {
                fn more(c: []const u32, x: u32, y: u32) bool {
                    return c[x] > c[y];
                }
            }.more);
            try self.snake.resize(a, ng);
            var lo: usize = 0;
            var hi: usize = ng;
            var k: usize = 0;
            while (lo < hi) {
                self.snake.items[k] = self.sorted.items[lo];
                k += 1;
                lo += 1;
                if (lo < hi) {
                    hi -= 1;
                    self.snake.items[k] = self.sorted.items[hi];
                    k += 1;
                }
            }
            // rows in wave order: the snake order's groups, each group's rows ascending
            try self.pos.resize(a, n);
            try self.ridx.resize(a, n);
            try self.rhs.resize(a, n);
            try self.inv.resize(a, n);
            var off: usize = 0;
            for (self.snake.items) |gi| {
                const c = self.gcount.items[gi];
                const first = self.gnext.items[gi] - c;
                for (self.grows.items[first..][0..c]) |row| {
                    self.pos.items[off] = row;
                    self.ridx.items[off] = @intCast(if (rows.act_row) |ar| ar[row] else row);
                    off += 1;
                }
            }
        }
    };
}

/// The per-projection bank check (the route above), under a name `Accepted.checkBank` can call.
const checkProjArrays = checkBank;
