//! glm5_next's routed-expert math over the EXL3 K=3 bank (the `exl3` family): the shared DeepSeek scheduler's math seam
//! (`deepseek_v41_experts.ExpertsWith`'s `M`) on the GLM-shape EXL3 texts (exl3_kernels.zig). The scheduler, the stream,
//! the hit wave / miss parts, the gates, the lookahead and the combine are the NVFP4 family's; only the projection math
//! differs:
//!   - decode: DeepSeek-V4.1's banked PREP=rin chain (in_rin, the gate and up mul1h GEMVs, gu_epi, din_rin, the down GEMV,
//!     dpost), one launch of each per wave over the three banks (`gateUpBanked` / `downBanked`, ids `bank << 24 | row`);
//!     the unbanked route (`decode_moe: stock`) runs the same texts with every bank the call's one bank;
//!   - prompt: DeepSeek-V4.1's DIG-X waves (take2, the gate|up GEMM decoding the trellis in its B loader, the onepass
//!     SwiGLU epilogue, the down GEMM, rot_widen1 into bf16) over the call's rows sorted by slot, a wave at most
//!     `wave_rows` rows (the construction's `exl3_wave_rows`, prompt_levers.zig; default `prompt_levers.default_wave_rows`, the control's `control_wave_rows`) of at most
//!     `xq.wave_max` experts, pipelined as MiMo's drain (one wave in flight behind the next). The rows' slot order is a
//!     stable block sort (the control) or, under `prompt_route_cut: on`, a stable counting sort (the same order).
//! The decode lookahead's predictor is MiMo's (`mimo_math.Predictor`), bound at construction.
//! The decode chain's launches are bound once at construction (`Decode`): lane B's seam texts (exl3_kernels.zig,
//! `exl3_gateup_form` / `exl3_down_form` = seam by name: g22-exl3's chain, unchanged) or the installed forms
//! (exl3_forms.zig `Installed`, the settings' `Exl3Forms`, guone + pair by default); the calls run the bound pair, no
//! per-call choice.

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const ops = @import("../ops.zig");
const quant = @import("../sdk_ext.zig").quant;
const xq = @import("../../exl3/quant.zig");
const mimo_math = @import("../mimo/predictor.zig");
const profile = @import("../mimo/profile.zig");
const kernels = @import("exl3_kernels.zig");
const xf = @import("exl3_forms.zig");
const bank = @import("exl3_bank.zig");
const prompt_levers = @import("prompt_levers.zig");

const T = mlx.mlx_array;
const G = ops.MlxOps;
const Proj = bank.Arrays(T);
const Arrays = bank.BankArraysOf(T);

/// A prompt wave's rows at most on the control (g22-exl3's, `exl3_wave_rows` 1024 by name): MiMo's projection tile (the bill's `expert_workspace` prices a tile of these
/// rows at NVFP4's per-row bytes, above a DIG-X wave's: take2 16 KiB + z 16 KiB + hd 4 KiB + zd 16 KiB + y 8 KiB a row).
/// The construction's `exl3_wave_rows` sets a load's (`Math.wave_rows`); the bill prices two waves of it in flight.
pub const control_wave_rows: u32 = mimo_math.projection_rows;
/// A prompt call's routed rows at most (the module's calls: `graph.prompt_call_rows` tokens x top-8; MiMo's math's bound).
/// A prompt call's tokens at most (the module's call rows).
const max_call_tokens: u32 = 16_384;
pub const max_call_rows: u32 = max_call_tokens * 8;
pub const wave_bytes_per_row: u64 = 2 * kernels.hidden * 2 + 2 * kernels.inter * 4 + kernels.inter * 2 + kernels.hidden * 4 + kernels.hidden * 2 + 8;

comptime {
    // One wave table serves both GEMMs (their threadgroups per M tile agree at GLM's dims).
    if (kernels.gu_tiles != kernels.dn_tiles) @compileError("glm5_next exl3: the gate|up and down GEMMs need their own wave tables");
    if (wave_bytes_per_row > 20 * kernels.hidden + 24 * kernels.inter + 32) @compileError("glm5_next exl3: a wave row outgrows the billed tile row");
    // The setting's default is today's wave; every admitted size is whole M tiles of the GEMMs.
    if (prompt_levers.control_wave_rows != control_wave_rows) @compileError("glm5_next exl3: the wave setting's control is not g22-exl3's wave");
    if (prompt_levers.wave_rows_step % kernels.m_tile != 0) @compileError("glm5_next exl3: a wave size off the GEMMs' M tile");
}

/// The decode chain, bound once at construction: lane B's seam kernels, or an installed form.
pub const Decode = struct {
    ctx: *const anyopaque,
    gate_up: *const fn (ctx: *const anyopaque, g: *G, x: T, ids: T, banks: *const [kernels.n_banks]Arrays) anyerror!T,
    down: *const fn (ctx: *const anyopaque, g: *G, h: T, ids: T, banks: *const [kernels.n_banks]Arrays) anyerror!T,

    pub fn seam(k: *const kernels.Kernels) Decode {
        return .{ .ctx = k, .gate_up = seamGateUp, .down = seamDown };
    }

    fn seamGateUp(ctx: *const anyopaque, g: *G, x: T, ids: T, banks: *const [kernels.n_banks]Arrays) anyerror!T {
        return kernels.gateUp(@ptrCast(@alignCast(ctx)), g, x, ids, banks);
    }
    fn seamDown(ctx: *const anyopaque, g: *G, h: T, ids: T, banks: *const [kernels.n_banks]Arrays) anyerror!T {
        return kernels.down(@ptrCast(@alignCast(ctx)), g, h, ids, banks);
    }
};
/// The order of a prompt call's rows by slot (`Math.pack`, the construction's route): the stable block sort (the
/// control) or the stable counting sort (`prompt_route_cut: on`); both give the same permutation.
pub const Pack = enum { sort, count };

pub const Math = struct {
    const Self = @This();

    /// MiMo's decode-lane bounds and phase routes (mimo_math `MathOf`): the decode lane takes at most 4 rows, the wide
    /// lane from 5; the wide waves are event-gated; the predictor runs in decode only.
    pub const decode_forward_rows: u32 = mimo_math.Math.decode_forward_rows;
    pub const prefill_min_tokens: u32 = decode_forward_rows + 1;
    pub const wide_event_gates = true;
    pub const predictor_decode_only = true;
    pub const Profile = mimo_math.Profile;
    pub const has_banked = true;
    pub const has_parts = true;

    pub fn predictorScores(g: anytype, x: @TypeOf(g.*).T, router: @TypeOf(g.*).T, correction: @TypeOf(g.*).T) !@TypeOf(g.*).T {
        return mimo_math.Math.predictorScores(g, x, router, correction);
    }

    allocator: std.mem.Allocator,
    k: *kernels.Kernels,
    /// The installed forms (null: the seam chain, `Exl3Forms.seam_chain` by name).
    forms: ?*xf.Installed,
    /// The decode chain the calls run (`Decode.seam(k)` or `Decode.forms(forms)`), bound at `init`.
    decode: Decode,
    predictor: mimo_math.Predictor = .{},
    order: []u32,
    /// A wave's host tables: act rows, entries, the wave table.
    ridx: []i32,
    rhs: []u32,
    /// A prompt wave's rows at most (`exl3_wave_rows`, construction).
    wave_rows: u32,
    /// The call's rows by slot (`Pack`, bound at construction) and the counting sort's per-slot cursors.
    pack: *const fn (self: *Self, rows: quant.PrefillRows) error{OutOfMemory}![]u32,
    slot_cursor: std.ArrayList(u32) = .empty,

    /// Construction (the module's GPU stream): the texts installed (`exl3_kernels.install`: pins, kernels, the decode
    /// launches, the compile check), refused by name; the decode forms (`forms`, the settings' `Exl3Forms.forms()`): the
    /// seam chain binds lane B's kernels and installs nothing more; any other form is installed here (exl3_forms.zig
    /// `install`: the pins, its texts and every wave width's launches), refused by name; the prompt
    /// waves at `wave_rows` rows (validated with the settings: whole M tiles, at most `prompt_levers.max_wave_rows`), the
    /// rows' order by `pack`.
    pub fn init(a: std.mem.Allocator, s: mlx.mlx_stream, call_rows: u32, forms: xf.Forms, wave_rows: u32, pack: Pack, diag: *sdk.Diag) !Self {
        std.debug.assert(wave_rows >= prompt_levers.min_wave_rows and wave_rows <= prompt_levers.max_wave_rows);
        const k = try kernels.install(a, s, diag);
        errdefer {
            k.deinit();
            a.destroy(k);
        }
        const inst: ?*xf.Installed = if (std.meta.eql(forms, xf.Forms{})) null else try xf.install(a, k, forms, .bfloat16, diag);
        errdefer if (inst) |f| {
            f.deinit();
            a.destroy(f);
        };
        const order = try a.alloc(u32, call_rows);
        errdefer a.free(order);
        const ridx = try a.alloc(i32, wave_rows);
        errdefer a.free(ridx);
        return .{ .allocator = a, .k = k, .forms = inst, .decode = if (inst) |f| Decode.forms(f) else Decode.seam(k), .order = order, .ridx = ridx, .rhs = try a.alloc(u32, wave_rows), .wave_rows = wave_rows, .pack = switch (pack) {
            .sort => &packRowsSort,
            .count => &packRowsCount,
        } };
    }

    pub fn deinit(self: *Self, _: *G) void {
        self.predictor.deinit();
        if (self.forms) |f| {
            f.deinit();
            self.allocator.destroy(f);
        }
        self.k.deinit();
        self.allocator.destroy(self.k);
        self.allocator.free(self.order);
        self.allocator.free(self.ridx);
        self.allocator.free(self.rhs);
        self.slot_cursor.deinit(self.allocator);
    }

    pub fn predictorScoresBound(self: *const Self, g: *G, x: T, router: T, correction: T) !T {
        return self.predictor.scores(&self.predictor, g, x, router, correction);
    }

    // ── Decode ──

    /// The unbanked route: one bank, its slot rows (bank 0 in the packing), every bank its arrays.
    pub fn gateUp(self: *const Self, g: *G, x: T, ids: T, gate: Proj, up: Proj) !T {
        const one: Arrays = .{ .gate = gate, .up = up, .down = gate };
        const banks: [kernels.n_banks]Arrays = @splat(one);
        return self.decode.gate_up(self.decode.ctx, g, x, ids, &banks);
    }

    pub fn down(self: *const Self, g: *G, h: T, ids: T, d: Proj) !T {
        const one: Arrays = .{ .gate = d, .up = d, .down = d };
        const banks: [kernels.n_banks]Arrays = @splat(one);
        return self.decode.down(self.decode.ctx, g, h, ids, &banks);
    }

    /// x bf16 [P, hidden] (each pair's token row), ids packed `bank << 24 | row` -> the clamped SwiGLU f32 [P, inter].
    pub fn gateUpBanked(self: *const Self, g: *G, x: T, ids: T, banks: *const [kernels.n_banks]Arrays) !T {
        return self.decode.gate_up(self.decode.ctx, g, x, ids, banks);
    }

    // ── Prompt: DIG-X waves ──

    /// The call's rows sorted by slot (stable): each wave's experts are contiguous rows of it. The control: a block sort.
    fn packRowsSort(self: *Self, rows: quant.PrefillRows) error{OutOfMemory}![]u32 {
        const order = self.order[0..rows.slot.len];
        for (order, 0..) |*o, i| o.* = @intCast(i);
        std.sort.block(u32, order, rows.slot, struct {
        }.lt);
        return order;
    }

    /// `packRowsSort`'s order by a stable counting sort (`prompt_route_cut: on`): per slot its rows' count, each slot's
    /// first place, then every row at its slot's cursor in row order: slots ascending, a slot's rows ascending (the
    /// block sort's permutation, which is stable: equal slots keep their rows' order). O(rows + slots), no comparison.
    fn packRowsCount(self: *Self, rows: quant.PrefillRows) error{OutOfMemory}![]u32 {
        const order = self.order[0..rows.slot.len];
        try countingOrder(rows.slot, order, &self.slot_cursor, self.allocator);
        return order;
    }

    /// The stable counting sort of `slots`' indices into `order` (`slots.len` of them), its cursors in `cursor`.
    pub fn countingOrder(slots: []const u32, order: []u32, cursor: *std.ArrayList(u32), a: std.mem.Allocator) error{OutOfMemory}!void {
        var top: u32 = 0;
        for (slots) |sl| top = @max(top, sl);
        try cursor.resize(a, @as(usize, top) + 2);
        const c = cursor.items;
        @memset(c, 0);
        for (slots) |sl| c[sl + 1] += 1;
        for (1..c.len) |i| c[i] += c[i - 1];
        for (slots, 0..) |sl, i| {
            order[c[sl]] = @intCast(i);
            c[sl] += 1;
        }
    }

    /// The call's routed rows' outputs bf16 [rows, hidden] in `rows` order, evaluated and KEPT.
    pub fn prefill(self: *Self, g: *G, _: u32, x: T, rows: quant.PrefillRows, b: Arrays) !T {
        var kept: std.ArrayList(T) = .empty;
        defer {
            for (kept.items) |y| g.release(y);
            kept.deinit(self.allocator);
        }
        const order = try self.waves(g, x, rows, b, &kept, self.allocator);
        const inverse = try self.allocator.alloc(u32, order.len);
        defer self.allocator.free(inverse);
        for (order, 0..) |o, i| inverse[o] = @intCast(i);
        const mark = g.mark();
        const joined = try g.concat(kept.items, 0);
        const result = try g.take(joined, try g.hostArray(std.mem.sliceAsBytes(inverse), &.{@intCast(order.len)}, .uint32), 0);
        try g.evalAll(&.{result});
        const held = g.keep(result);
        g.resetTo(mark);
        return held;
    }

    /// `prefill`'s waves as they are: each wave's KEPT output appended to `outs`, its rows' indices into `rows` to `pos`.
    pub fn prefillParts(self: *Self, g: *G, _: u32, x: T, rows: quant.PrefillRows, b: Arrays, alloc: std.mem.Allocator, outs: *std.ArrayList(T), pos: *std.ArrayList(u32)) !void {
        const order = try self.waves(g, x, rows, b, outs, alloc);
        try pos.appendSlice(alloc, order);
    }

    /// Every wave was evaluated and synchronized before its call returned.
    pub fn finishPrefill(_: *Self, _: *G) !void {}

};

const testing = std.testing;
