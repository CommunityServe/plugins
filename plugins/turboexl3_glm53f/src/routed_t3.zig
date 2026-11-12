//! The routed experts' math on turboexl3, for the copied GLM-5.3-Flash expert streamer (sdk/weights/streamed/glm5n/): the
//! scheduler's math seam (`ExpertsWith`'s `M`) with every call of every routed layer on turboexl3, at the layer's own rate.
//! The streamer's own GLM texts (glm5n/exl3_kernels.zig) decode K = 3 only; turboderp's self-calibrated packs route a
//! layer at K 2, 2.5, 3, 3.5, 4 or 5 (src/streamed/glm5n/experts.zig binds this math).
//!
//! - Decode waves (`gateUpBanked`, `downBanked`): turboexl3's banked decode (`moeClampedBanked`), a wave's pairs over the
//!   layer's three slot banks in one call, each pair one row-slot whose id is packed (bank << 24 | slot row), each pair's
//!   row `moeClamped`'s bytes on its bank at top-1. The call runs whole at the wave's down (turboexl3's gate|up finish
//!   prepares the down's input, so it reads the down projection's suh). The slot arrays are turboexl3's banks: code as
//!   the trellis, rin as suh, rout as svh.
//! - A prompt call's rows (`prefill`, `prefillParts`) and a cold-row chunk (`gateUp` then `down`): the rows sorted by slot,
//!   in chunks of at most `prompt_rows`, each chunk one `moeClamped` call at top-1 and weight 1 over the bank's slot
//!   arrays (code viewed as u16, the same bits), on turboexl3's prefill GEMM: a chunk of 16 rows or fewer is padded to 17
//!   with copies of its first row at score 0 (dropped after), so a row's bytes do not depend on how the residency split
//!   the call.
//! - Every row comes back bf16 [rows, hidden] at weight 1, as the streamer's own texts return it (dpost / widen1 store
//!   bf16); the scheduler joins and weighs the rows itself.
//! - The clamp is GLM-5.3-Flash's `swiglu_limit` (10, the streamer's `exl3_kernels.swiglu_limit`).
//! - A slot bank's scale arrays (`rin`, `rout`) can hold more rows than its code array: the stream allocates every slot
//!   array one VM page at least (the EXL3 bank's `slot_array_min_bytes`, `StreamOf.allocSlots`), so a bank of 1 to 3 rows
//!   (a grown layer's ext rows) has 4 KiB scale rows floored at 4. turboexl3 checks every bank's suh and svh at
//!   [rows, in] and [rows, out] (`bankGeometryAs`) and refuses a longer one (BadExl3Shape). Every call hands turboexl3
//!   the scale arrays sliced to the code array's rows (`fitRows`; the rows past them are never addressed).
//! - A turboexl3 refusal is logged by name here: the streamer's routed seam carries an error across by name and the host
//!   knows none of these (it arrives as EngineFailed).
//! The predictor (the decode lookahead's next-layer scores) is the stock math's.
//!
//! Opt-in fused paths (`Switches`, read once at construction; each off by default, each turboexl3's entry that gives the
//! default path's bytes, held so by turboexl3's src/models/glm53f/fused_test.zig):
//! - `TURBOEXL3_GLM53F_PF_ROWS=1`: a prompt call and a cold-row chunk on `t3.moeClampedRows`: the chain's prepare reads
//!   each row-slot's x row through the call's row map and its finish stores bf16, in place of the default's gather
//!   (`take`), its cast to f32 and the cast of moeClamped's f32 y to bf16 (two full copies of the call's rows and two of
//!   its outputs fewer, four launches fewer a call).
//! - `TURBOEXL3_GLM53F_DEC_BF16=1`: a decode wave on `t3.moeClampedBankedAs` at bf16: the banked down finish stores bf16, in
//!   place of moeClampedBanked's f32 y and its cast (one launch a routed layer fewer a forward).
//!
//! Opt-in schedules (the same kernels' arithmetic, so the same bytes; each off by default):
//! - `TURBOEXL3_GLM53F_PF_FEED=1`: a prompt call returns with its last chunks in flight. The scheduler queues a group's
//!   bank calls back to back and evaluates the group's outputs once before its slots go back (`Wide.seed`'s feed), so
//!   that one wait covers every chunk; the default also waits at each call's end.
//! - `TURBOEXL3_GLM53F_PF_ALIGN=1`: a prompt call's chunks (`prefillParts`) end where no expert's 128-row run is split
//!   (`alignedEnd`: at an expert boundary, or a multiple of 128 rows into the expert crossing the chunk's rows at most),
//!   so each expert takes the 128-row tiles and remainder it takes in one call: a fixed cut through an expert's run adds
//!   a part-full tile or a mixed-launch remainder on each side. Each chunk still holds at most the chunk rows (more than
//!   127 fewer only at the call's end), so no array grows. turboexl3's chain_time_test.zig holds the aligned chunks to
//!   fixed 4096-row ones byte for byte.
//! - `TURBOEXL3_GLM53F_BANKED=g4m3.t4.t4`: the banked decode's schedule, set with `t3.configure` before any banked kernel
//!   is built: turboexl3's grouped texts (`banked_group` 4) with `banked_m3`, the grouped gate|up and down GEMVs at
//!   turboexl3's variant t4 (`t3.groupVariantOf`). It reads, decodes and sums each weight as the default does
//!   (fused_test.zig holds it to the default call at every SC rate).

const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const t3 = @import("turboexl3");
const log = @import("mlx_serve/sdk/util/log.zig");
/// TURBOEXL3_GLM53F_PF_MOE_HOLD: a prompt call's start hands a DSA layer's hold back to MLX's cache (src/pf_reuse.zig).
const pf_reuse = @import("pf_reuse.zig");
const options = @import("options.zig");

/// The rows a slot bank's scale array is sliced to for turboexl3: its code array's rows when the scale array holds more
/// (the stream's page floor), else null (the array as it is).
pub fn fitRows(code_rows: c_int, scale_rows: c_int) ?c_int {
    return if (scale_rows > code_rows) code_rows else null;
}

/// `a` (a scale array [slots, width]) at `rows` rows: `a` itself, or a view of its first `rows` rows (appended to `views`,
/// which the caller frees; a kernel holds its inputs until it runs).
fn fitted(a: mlx.mlx_array, rows: c_int, s: mlx.mlx_stream, views: []mlx.mlx_array, nv: *usize) !mlx.mlx_array {
    const sh = mlx.getShape(a);
    if (sh.len != 2) return a;
    const r = fitRows(rows, sh[0]) orelse return a;
    var v = mlx.mlx_array_new();
    mlx.check(mlx.mlx_slice(&v, a, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ r, sh[1] }, 2, &[_]c_int{ 1, 1 }, 2, s)) catch |e| {
        _ = mlx.mlx_array_free(v);
        return e;
    };
    views[nv.*] = v;
    nv.* += 1;
    return v;
}

/// A prompt chunk's rows at most: one `moeClamped` call (turboexl3's prefill GEMM over the call's slot arrays).
pub const prompt_rows: usize = 4096;
/// The fewest rows a chunk runs at: turboexl3's prefill GEMM's (its decode kernels take 16 rows or fewer).
pub const gemm_min_rows: usize = 17;

comptime {
    std.debug.assert(gemm_min_rows == t3.kernels.DECODE_ROWS_MAX + 1);
}

/// The fused paths' and the prompt schedule's switches (the file's doc): the on/off ones unset or 0 off, 1 on.
pub const Switches = struct {
    pf_rows: bool = false,
    dec_bf16: bool = false,
    pf_feed: bool = false,
    /// `TURBOEXL3_GLM53F_PF_ALIGN`: a prompt call's chunks end where no expert's 128-row run is split (`alignedEnd`)
    pf_align: bool = false,

    /// Each switch from its value (`env`: a variable's value by name, null when unset); a value it does not take refuses
    /// by the switch's name, the load with it.
    pub fn read(env: *const fn ([:0]const u8) ?[]const u8) !Switches {
        return .{
            .pf_rows = parseSwitch(env("TURBOEXL3_GLM53F_PF_ROWS")) catch return error.Glm53fPfRowsSwitch,
            .dec_bf16 = parseSwitch(env("TURBOEXL3_GLM53F_DEC_BF16")) catch return error.Glm53fDecBf16Switch,
            .pf_feed = parseSwitch(env("TURBOEXL3_GLM53F_PF_FEED")) catch return error.Glm53fPfFeedSwitch,
            .pf_align = parseSwitch(env("TURBOEXL3_GLM53F_PF_ALIGN")) catch return error.Glm53fPfAlignSwitch,
        };
    }
};

/// The routed prefill's tile rows (turboexl3's 128-row tiles, `prefill.Options.t128`): `TURBOEXL3_GLM53F_PF_ALIGN` cuts an
/// expert's rows only at a multiple of this many rows into its run.
pub const align_tile_rows: usize = 128;

/// `TURBOEXL3_GLM53F_PF_ALIGN`: the end of the prompt chunk from `c0` over a call's slot-sorted rows `sorted`, at most
/// `max_rows` of them: `c0 + max_rows` (or the end) where that is an expert boundary, else the latest point before it
/// that is a multiple of `align_tile_rows` rows into the expert crossing it (that expert's first row when it starts after
/// c0). Every chunk ends so, so a chunk that starts inside an expert starts a multiple of 128 rows into it, and a chunk
/// before the call's last holds more than `max_rows - 128` rows (at least 128 always moves the cut).
pub fn alignedEnd(sorted: []const u32, c0: usize, max_rows: usize) usize {
    const lim = @min(c0 + max_rows, sorted.len);
    if (lim == sorted.len or sorted[lim - 1] != sorted[lim]) return lim;
    var start = lim - 1;
    while (start > 0 and sorted[start - 1] == sorted[lim - 1]) start -= 1;
    const cut = start + (lim - start) / align_tile_rows * align_tile_rows;
    return if (cut > c0) cut else lim;
}

/// `TURBOEXL3_GLM53F_PF_ALIGN`'s engaged line, at its first aligned call (`{d}` the call's rows, its chunks, the fixed
/// cut's chunks)
pub const pf_align_engaged = "[turboexl3_glm53f] TURBOEXL3_GLM53F_PF_ALIGN engaged: a prompt call of {d} rows in {d} chunks ({d} at fixed cuts)\n";
var pf_align_said: bool = false;

/// `TURBOEXL3_GLM53F_BANKED=g4m3.t4.t4`: the banked decode's schedule (turboexl3's `Config.banked_group` 4 and
/// `banked_m3`, the grouped gate|up and down GEMVs at variant t4: `banked_variant_gu`, `banked_variant_dn`).
pub const BankedSchedule = struct {
    gu: u8,
    dn: u8,

    /// The schedule a value names: `g4m3.t4.t4`, the variants by turboexl3's name (`t3.groupVariantOf`); anything else
    /// error.BadSwitch.
    pub fn parse(s: []const u8) !BankedSchedule {
        if (!std.mem.eql(u8, s, "g4m3.t4.t4")) return error.BadSwitch;
        const t4 = t3.groupVariantOf("t4") orelse return error.BadSwitch;
        return .{ .gu = t4, .dn = t4 };
    }

    /// The texts the schedule runs: turboexl3's grouped texts with `banked_m3`
    pub fn textSet(_: BankedSchedule) []const u8 {
        return "g4m3";
    }

    /// The variable's schedule (`env` as `Switches.read`'s): null when unset; a value `parse` refuses refuses the load by
    /// name.
    pub fn read(env: *const fn ([:0]const u8) ?[]const u8) !?BankedSchedule {
        const v = env("TURBOEXL3_GLM53F_BANKED") orelse return null;
        return parse(v) catch return error.Glm53fBankedSwitch;
    }

    /// turboexl3's configuration with this schedule (every other field as configured), set for the process: before the
    /// first banked call of any geometry (a geometry's banked state reads `banked_group` and `banked_m3` when it is
    /// made).
    pub fn apply(b: BankedSchedule) !void {
        var c = t3.config.get().*;
        c.banked_group = 4;
        c.banked_m3 = true;
        c.banked_variant_gu = b.gu;
        c.banked_variant_dn = b.dn;
        try t3.configure(c);
    }
};

/// `TURBOEXL3_GLM53F_DEC=g1`: the dense decode GEMV's schedule, turboexl3's DEC-ONE variant g1 (`Config.dense_variant`,
/// `t3.dense.dec_variants` by name: GLM-5.3-Flash's own GEMV at its trunk shapes, each lane's tile offsets taken once
/// and the pointers stepped); unset or `0` today's. The same reads, decodes and sums as today's (turboexl3's
/// dense_dec_test.zig holds it to the served configuration's bytes), so the same words.
pub const DecSchedule = struct {
    variant: u8,

    /// The schedule a value names: `g1`; anything else error.BadSwitch.
    pub fn parse(s: []const u8) !DecSchedule {
        if (!std.mem.eql(u8, s, "g1")) return error.BadSwitch;
        return .{ .variant = t3.dense.decVariantOf("g1") orelse return error.BadSwitch };
    }

    /// The variable's schedule (`env` as `Switches.read`'s): null when unset or `0`; a value `parse` refuses refuses the
    /// load by name.
    pub fn read(env: *const fn ([:0]const u8) ?[]const u8) !?DecSchedule {
        const v = env("TURBOEXL3_GLM53F_DEC") orelse return null;
        if (std.mem.eql(u8, v, "0")) return null;
        return parse(v) catch return error.Glm53fDecSwitch;
    }

    /// turboexl3's configuration with this schedule (every other field as configured), set for the process: before the
    /// first dense call (a dense shape's engine reads it when made).
    pub fn apply(d: DecSchedule) !void {
        var c = t3.config.get().*;
        c.dense_variant = d.variant;
        try t3.configure(c);
    }
};

/// `TURBOEXL3_GLM53F_PF_OPTS=bk128`: the routed prefill's option turboexl3 serves at the same bytes, the down GEMM's
/// 128-deep K stages (`Config.pf_bk128`); unset or `0` the served options.
pub const PfOpts = struct {
    bk128: bool = false,

    /// The options a value names (`bk128`); anything else error.BadSwitch.
    pub fn parse(s: []const u8) !PfOpts {
        if (std.mem.eql(u8, s, "bk128")) return .{ .bk128 = true };
        return error.BadSwitch;
    }

    /// The variable's options (`env` as `Switches.read`'s): null when unset or `0`; a value `parse` refuses refuses the
    /// load by name.
    pub fn read(env: *const fn ([:0]const u8) ?[]const u8) !?PfOpts {
        const v = env("TURBOEXL3_GLM53F_PF_OPTS") orelse return null;
        if (std.mem.eql(u8, v, "0")) return null;
        return parse(v) catch return error.Glm53fPfOptsSwitch;
    }

    /// turboexl3's configuration with these options (every other field as configured), set for the process: before the
    /// first prompt call (a prefill engine reads them when it is made).
    pub fn apply(o: PfOpts) !void {
        var c = t3.config.get().*;
        c.pf_bk128 = o.bk128;
        try t3.configure(c);
    }
};

/// `TURBOEXL3_GLM53F_PF_INT8=1`, `2` or `3`: the routed prefill's gate|up and down GEMMs on int8 (turboexl3's
/// `Config.pf_int8`; not exact): each activation row as int8 in 128-blocks and each weight's trellis decoded to int8 in
/// the GEMM's stage, int8 MMAs into int32 sums; 1 a scale each activation 128-block (each block's sums folded into f32 at
/// its scale), 2 one scale a row (one fold). turboexl3 takes it beside the 128-row tiles on the deferred MUL1 map and
/// declines it elsewhere (the exact chain); its src/models/glm53f/fused_test.zig holds both schemes to the exact chain at
/// every SC rate. 3 is 1 with the trunk's KDA qkv and o_proj at prefill rows on int8 too (turboexl3's
/// `Config.pf_trunk_int8` at `trunk_int8_shapes`: the int8 prepare at one scale a row, the trellis decoded to int8 once a
/// call, the int8 GEMM with the finish; fused_test.zig holds every trunk linear to the chain); 4 is 1 with every trunk
/// linear of `t3.glm53f.trunk` on int8 (`trunk_int8_all`: the KDA, DSA and indexer projections, the shared experts' and
/// the dense layers' MLPs). Unset or `0` the exact chain.
pub const PfInt8 = struct {
    scheme: u8,

    /// The scheme a value names: `1` to `4`; anything else error.BadSwitch.
    pub fn parse(s: []const u8) !PfInt8 {
        if (std.mem.eql(u8, s, "1")) return .{ .scheme = 1 };
        if (std.mem.eql(u8, s, "2")) return .{ .scheme = 2 };
        if (std.mem.eql(u8, s, "3")) return .{ .scheme = 3 };
        if (std.mem.eql(u8, s, "4")) return .{ .scheme = 4 };
        return error.BadSwitch;
    }

    /// The trunk linears a scheme puts on int8 (turboexl3's `trunk_q8.shapes`): 3 the KDA's two, 4 every one; null below 3
    pub fn trunkShapes(p: PfInt8) ?[]const t3.trunk_q8.Shape {
        return switch (p.scheme) {
            3 => &trunk_int8_shapes,
            4 => &trunk_int8_all,
            else => null,
        };
    }

    /// The variable's scheme (`env` as `Switches.read`'s): null when unset or `0`; a value `parse` refuses refuses the
    /// load by name.
    pub fn read(env: *const fn ([:0]const u8) ?[]const u8) !?PfInt8 {
        const v = env("TURBOEXL3_GLM53F_PF_INT8") orelse return null;
        if (std.mem.eql(u8, v, "0")) return null;
        return parse(v) catch return error.Glm53fPfInt8Switch;
    }

    /// turboexl3's configuration with this scheme (every other field as configured), set for the process: before the
    /// first prompt call (a prefill engine reads it when it is made).
    /// Schemes 3 and 4 set scheme 1 and the trunk's int8 (`Config.pf_trunk_int8`, turboexl3's `trunk_q8.shapes` at
    /// `trunkShapes`).
    pub fn apply(p: PfInt8) !void {
        var c = t3.config.get().*;
        c.pf_int8 = if (p.scheme >= 3) 1 else p.scheme;
        c.pf_trunk_int8 = p.scheme >= 3;
        try t3.configure(c);
        t3.trunk_q8.shapes.get().* = p.trunkShapes();
    }

    /// The scheme in the load line's words.
    pub fn words(p: PfInt8) []const u8 {
        return switch (p.scheme) {
            2 => "one scale a row",
            3 => "a scale each activation 128-block, and the trunk's KDA qkv and o_proj on int8",
            4 => "a scale each activation 128-block, and every trunk linear on int8",
            else => "a scale each activation 128-block",
        };
    }
};

/// `TURBOEXL3_GLM53F_PF_INT8=3`'s trunk linears on int8 (turboexl3's `trunk_q8.shapes`): the KDA layers' qkv and o_proj,
/// `t3.glm53f.trunk`'s first two
pub const trunk_int8_shapes = [_]t3.trunk_q8.Shape{
    .{ .k = t3.glm53f.trunk[0].k, .n = t3.glm53f.trunk[0].n },
    .{ .k = t3.glm53f.trunk[1].k, .n = t3.glm53f.trunk[1].n },
};

/// `TURBOEXL3_GLM53F_PF_INT8=4`'s trunk linears on int8: every one of `t3.glm53f.trunk` (11 shapes)
pub const trunk_int8_all = blk: {
    var a: [t3.glm53f.trunk.len]t3.trunk_q8.Shape = undefined;
    for (t3.glm53f.trunk, &a) |l, *s| s.* = .{ .k = l.k, .n = l.n };
    break :blk a;
};

comptime {
    std.debug.assert(std.mem.eql(u8, t3.glm53f.trunk[0].name, "kda.qkv_proj") and std.mem.eql(u8, t3.glm53f.trunk[1].name, "kda.o_proj"));
}

/// `TURBOEXL3_GLM53F_PG2`: the trunk's linears at prefill rows on turboexl3's pg2 (MLX's fused NAX GEMM loop at a tile
/// of this machine's with the finish as its epilogue: the chain's words where MLX runs its regular kernel), one tile
/// code a linear of `t3.glm53f.trunk` in order (`t3.glm53f.pg2Table`: `0` the chain, `1`-`9` then `a`- turboexl3's
/// `pgemm.candidates`), from the build window's timing (`TURBOEXL3_GLM_PG2_CODES`); unset or `0` pg2 off.
pub const Pg2 = struct {
    entries: [t3.glm53f.trunk.len]t3.pgemm.Entry = undefined,
    n: usize = 0,

    /// The table a value names; anything `pg2Table` refuses error.BadSwitch.
    pub fn parse(s: []const u8) !Pg2 {
        var p: Pg2 = .{};
        p.n = (t3.glm53f.pg2Table(s, &p.entries) catch return error.BadSwitch).len;
        return p;
    }

    /// The variable's table (`env` as `Switches.read`'s): null when unset or `0`; a value `parse` refuses refuses the
    /// load by name.
    pub fn read(env: *const fn ([:0]const u8) ?[]const u8) !?Pg2 {
        const v = env("TURBOEXL3_GLM53F_PG2") orelse return null;
        if (std.mem.eql(u8, v, "0")) return null;
        return parse(v) catch return error.Glm53fPg2Switch;
    }

    /// The process's pg2 table (it outlives every call) and switch: set before the first prompt.
    var installed: [t3.glm53f.trunk.len]t3.pgemm.Entry = undefined;
    pub fn apply(p: Pg2) void {
        installed = p.entries;
        t3.pgemm.host_table.get().* = installed[0..p.n];
        t3.pgemm.mode.get().* = .on;
    }

    /// Every tabled tile compiled at the output dtypes the trunk's linears run at, so a prompt's first call pays no JIT.
    pub fn warm(s: mlx.mlx_stream) !void {
        for ([_]mlx.mlx_dtype{ .bfloat16, .float16, .float32 }) |dt| try t3.pgemm.warm(s, dt, null);
    }
};

/// A switch's value: unset or "0" off, "1" on, anything else error.BadSwitch.
pub fn parseSwitch(v: ?[]const u8) !bool {
    const s = v orelse return false;
    if (std.mem.eql(u8, s, "0")) return false;
    if (std.mem.eql(u8, s, "1")) return true;
    return error.BadSwitch;
}

/// The process environment's value of `name`, or when unset the plugin's own (src/options.zig `defaultOf`: the production
/// control's value of a declared switch; null for a name it does not declare).
pub fn processEnv(name: [:0]const u8) ?[]const u8 {
    return if (std.c.getenv(name)) |v| std.mem.span(v) else options.defaultOf(name);
}

/// The engagement lines of the switches that are on (the A/Bs read them by these words).
pub const pf_rows_line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_PF_ROWS=1: prompt calls on turboexl3's moeClampedRows (the rows gathered in its prepare, y stored bf16)\n";
pub const dec_bf16_line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_DEC_BF16=1: decode waves on turboexl3's moeClampedBankedAs (y stored bf16 by the banked down finish)\n";
pub const pf_feed_line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_PF_FEED=1: a prompt call returns with its last chunks in flight (the group's one wait before its slots go back covers them)\n";
/// `{d}` the chunk rows at most
pub const pf_align_line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_PF_ALIGN=1: a prompt call's chunks end at an expert boundary or a multiple of 128 rows into the expert crossing the cap (at most {d} rows a chunk), so no expert's 128-row run is split across chunks (a row's bytes are its call's in any chunk)\n";
/// plugin.zig's line for TURBOEXL3_GLM53F_PF_STAGE (the copied scheduler's `Wide.stage`, read at load)
pub const pf_stage_line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_PF_STAGE=1: a prompt call's misses routed into the transient windows before its routing barrier, the residents held, their reads beside the layer's attention\n";
/// plugin.zig's line for TURBOEXL3_GLM53F_PF_DRAIN_LAG (the copied scheduler's `Wide.drain_lag`, read at load)
pub const pf_drain_lag_line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_PF_DRAIN_LAG=1: a prompt MoE group's one drain, its slots' release and the next route run after the next group's first chunk is committed, so the GPU runs the group's last chunks while the host builds the next group's first (the same kernels over the same rows in the same order)\n";
/// The lag's first use: `{d}` the call's rows
const pf_drain_lag_engaged = "[turboexl3_glm53f] TURBOEXL3_GLM53F_PF_DRAIN_LAG engaged: a prompt call of {d} rows committed its first chunk, then the previous group's drain, release and next route ran\n";

// ── TURBOEXL3_GLM53F_PF_DRAIN_LAG ──

/// What a prompt call runs once its first chunk is committed (`prefillParts`): the copied scheduler's pending group
/// (`LagPendingOf.begin` arms it before a group's banks). The prompt pass runs on one thread.
pub const PromptLagHook = struct { ctx: *anyopaque, run: *const fn (*anyopaque) anyerror!void };
var lag_hook: ?PromptLagHook = null;
var lag_said = false;
/// `TURBOEXL3_GLM53F_PF_DRAIN_LAG=1`, set by the plugin at load (the request line's counts).
pub var drain_lag_on: bool = false;
/// Groups drained at the next group's first commit (`lagged`) and after the next group's banks because it committed
/// nothing (`flushed`), since the load; a call's last group is drained at the call's end, as without the lag.
pub var lag_lagged: u64 = 0;
pub var lag_flushed: u64 = 0;

/// The armed hook, disarmed (null when none).
pub fn takeLagHook() ?PromptLagHook {
    const h = lag_hook;
    lag_hook = null;
    return h;
}

/// The copied scheduler's pending group under `Wide.drain_lag`: a group's one drain, then its call's release (with the
/// route `depth` groups on), wait for the next group's first committed chunk (`begin` arms the hook that runs them) or,
/// when that group commits none, its banks' end (`end`). Each group is drained once, in group order, and released after
/// its own drain; the call's last group at `flush`. `Ctx` gives `drain(k0, k1)` (the group's kept outputs evaluated)
/// and `release(gi)`.
pub fn LagPendingOf(comptime Ctx: type) type {
    return struct {
        const Self = @This();
        ctx: Ctx,
        /// The pending group and its kept outputs' range.
        gi: ?usize = null,
        k0: usize = 0,
        k1: usize = 0,

        /// Before a group's banks: the hook armed while a group is pending.
        pub fn begin(p: *Self) void {
            if (p.gi != null) lag_hook = .{ .ctx = p, .run = &run };
        }

        /// After group `gi`'s banks (its kept outputs `k0..k1`): the hook disarmed, the pending group drained and
        /// released if no commit ran it, then `gi` pending.
        pub fn end(p: *Self, gi: usize, k0: usize, k1: usize) !void {
            lag_hook = null;
            if (p.gi != null) {
                lag_flushed += 1;
                try p.flush();
            }
            p.gi = gi;
            p.k0 = k0;
            p.k1 = k1;
        }

        /// The pending group's drain and release (none pending: nothing).
        pub fn flush(p: *Self) !void {
            const gi = p.gi orelse return;
            p.gi = null;
            try p.ctx.drain(p.k0, p.k1);
            try p.ctx.release(gi);
        }

        /// The hook disarmed (the scheduler's every way out).
        pub fn disarm(_: *Self) void {
            lag_hook = null;
        }

        fn run(o: *anyopaque) anyerror!void {
            const p: *Self = @ptrCast(@alignCast(o));
            lag_lagged += 1;
            try p.flush();
        }
    };
}

/// The request line's counts (`{d}` lagged, flushed).
pub const lag_summary = "[turboexl3_glm53f] TURBOEXL3_GLM53F_PF_DRAIN_LAG lagged={d} flushed={d}\n";
/// `{s}` the variable's value, then the texts (`BankedSchedule.textSet`) and the variants' indices
pub const banked_line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_BANKED={s}: the banked decode on turboexl3's {s} texts (banked_group 4), gate|up variant {d}, down variant {d} (the same reads, decodes and sums)\n";
/// `{s}` the variable's value, then the variant's name
pub const dec_line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_DEC={s}: the dense decode GEMV (DEC-ONE) on turboexl3's '{s}' text (the same reads, decodes and sums)\n";
/// `{s}` the variable's value, then bk128
pub const pf_opts_line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_PF_OPTS={s}: the routed prefill with bk128 {} (turboexl3's Config.pf_bk128: the same MMAs in the same order)\n";
/// `{s}` the variable's value, then the scheme's words (`PfInt8.words`)
pub const pf_int8_line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_PF_INT8={s}: the routed prefill's gate|up and down GEMMs on int8, {s} (turboexl3's Config.pf_int8; not exact)\n";
/// TURBOEXL3_GLM53F_PF_INT8=4's second line: `{d}` the trunk's linears on int8
pub const pf_trunk_int8_all_line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_PF_INT8=4: every trunk linear ({d}: the KDA, DSA and indexer projections, the shared experts' and the dense layers' MLPs) at prefill rows on int8, one scale a row, the trellis decoded to int8 once a call, the int8 GEMM with the finish (turboexl3's Config.pf_trunk_int8; not exact)\n";
/// TURBOEXL3_GLM53F_PF_INT8=3's second line: the trunk's linears on int8 (`{d}` each one's in and out)
pub const pf_trunk_int8_line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_PF_INT8=3: the trunk's KDA qkv ({d} x {d}) and o_proj ({d} x {d}) at prefill rows on int8, one scale a row, the trellis decoded to int8 once a call, the int8 GEMM with the finish (turboexl3's Config.pf_trunk_int8; not exact)\n";
/// `{s}` the variable's value, then the linears on pg2 and the trunk's count
pub const pg2_line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_PG2={s}: {d} of the trunk's {d} linears at prefill rows on turboexl3's pg2 (MLX's fused NAX GEMM loop at the coded tile, the finish its epilogue: the chain's words)\n";

test "turboexl3_glm53f stream math: the fused paths' and the prompt schedule's switches: unset and 0 are off, 1 on, any other value refuses by the switch's name" {
    const t = std.testing;
    try t.expectEqual(false, try parseSwitch(null));
    try t.expectEqual(false, try parseSwitch("0"));
    try t.expectEqual(true, try parseSwitch("1"));
    for ([_][]const u8{ "", "2", "on", "true", "01", " 1" }) |v| try t.expectError(error.BadSwitch, parseSwitch(v));
    const Env = struct {
        var pf: ?[]const u8 = null;
        var dec: ?[]const u8 = null;
        var feed: ?[]const u8 = null;
        var al: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            if (std.mem.eql(u8, name, "TURBOEXL3_GLM53F_PF_ALIGN")) return al;
            if (std.mem.eql(u8, name, "TURBOEXL3_GLM53F_PF_ROWS")) return pf;
            if (std.mem.eql(u8, name, "TURBOEXL3_GLM53F_DEC_BF16")) return dec;
            if (std.mem.eql(u8, name, "TURBOEXL3_GLM53F_PF_FEED")) return feed;
            return null;
        }
    };
    try t.expectEqual(Switches{}, try Switches.read(&Env.get));
    Env.pf = "1";
    try t.expectEqual(Switches{ .pf_rows = true }, try Switches.read(&Env.get));
    Env.dec = "1";
    try t.expectEqual(Switches{ .pf_rows = true, .dec_bf16 = true }, try Switches.read(&Env.get));
    Env.feed = "1";
    try t.expectEqual(Switches{ .pf_rows = true, .dec_bf16 = true, .pf_feed = true }, try Switches.read(&Env.get));
    Env.pf = "yes";
    try t.expectError(error.Glm53fPfRowsSwitch, Switches.read(&Env.get));
    Env.pf = null;
    Env.dec = "2";
    try t.expectError(error.Glm53fDecBf16Switch, Switches.read(&Env.get));
    Env.dec = null;
    Env.feed = "on";
    try t.expectError(error.Glm53fPfFeedSwitch, Switches.read(&Env.get));
    Env.feed = null;
    Env.pf = "1";
    // PF_ALIGN: on at 1, off at 0, any other value refused by its name
    Env.al = "1";
    try t.expectEqual(Switches{ .pf_rows = true, .pf_align = true }, try Switches.read(&Env.get));
    Env.al = "0";
    try t.expectEqual(Switches{ .pf_rows = true }, try Switches.read(&Env.get));
    Env.al = "on";
    try t.expectError(error.Glm53fPfAlignSwitch, Switches.read(&Env.get));
}

/// An expert run's 128-row tiles and remainder in each chunk of `ends` over slot-sorted `sorted` (turboexl3's 128-row
/// tile route: full tiles, then one remainder, in each chunk it appears in): the sums over the chunks
fn tileSums(sorted: []const u32, ends: []const usize) [2]usize {
    var full: usize = 0;
    var parts: usize = 0;
    var c0: usize = 0;
    for (ends) |c1| {
        var i = c0;
        while (i < c1) {
            var j = i + 1;
            while (j < c1 and sorted[j] == sorted[i]) j += 1;
            full += (j - i) / align_tile_rows;
            parts += @intFromBool((j - i) % align_tile_rows != 0);
            i = j;
        }
        c0 = c1;
    }
    return .{ full, parts };
}

test "turboexl3_glm53f stream math: TURBOEXL3_GLM53F_PF_ALIGN's chunks cover a call's rows in order, each at most the chunk rows and, before the last, more than 128 fewer; each cut an expert boundary or a multiple of 128 rows into its expert, so every expert keeps its one-call tiles" {
    const t = std.testing;
    var prng = std.Random.DefaultPrng.init(0xA1167);
    const rnd = prng.random();
    var sorted: [22000]u32 = undefined;
    var ends: [64]usize = undefined;
    for (0..20) |round| {
        // a group's rows: 48 experts at lognormal popularity (round 0: one expert past several chunks), slot-sorted
        var w: [48]f64 = undefined;
        var sum: f64 = 0;
        for (&w, 0..) |*v, e| {
            v.* = if (round == 0 and e == 7) 40.0 else @exp(rnd.floatNorm(f64));
            sum += v.*;
        }
        var p: usize = 0;
        for (w, 0..) |v, e| {
            const c: usize = if (e == 47) sorted.len - p else @min(sorted.len - p, @as(usize, @intFromFloat(@floor(@as(f64, @floatFromInt(sorted.len)) * v / sum))));
            for (sorted[p .. p + c]) |*sl| sl.* = @intCast(e * 6);
            p += c;
        }
        for ([_]usize{ 2048, 4096, 8192 }) |max_rows| {
            var nc: usize = 0;
            var c0: usize = 0;
            while (c0 < sorted.len) {
                const c1 = alignedEnd(&sorted, c0, max_rows);
                try t.expect(c1 > c0 and c1 - c0 <= max_rows);
                if (c1 < sorted.len) {
                    try t.expect(c1 - c0 > max_rows - align_tile_rows);
                    if (sorted[c1 - 1] == sorted[c1]) {
                        var start = c1 - 1;
                        while (start > 0 and sorted[start - 1] == sorted[c1]) start -= 1;
                        try t.expectEqual(@as(usize, 0), (c1 - start) % align_tile_rows);
                    }
                }
                ends[nc] = c1;
                nc += 1;
                c0 = c1;
            }
            try t.expectEqual(sorted.len, c0);
            // every expert's tiles as in one call
            try t.expectEqual(tileSums(&sorted, &.{sorted.len}), tileSums(&sorted, ends[0..nc]));
        }
    }
    // edges: an expert crossing the cap cut at the latest multiple of 128 rows into it; a boundary at the cap kept
    var big: [9000]u32 = undefined;
    for (&big, 0..) |*v, i| v.* = if (i < 100) 0 else if (i < 8800) 1 else 2;
    try t.expectEqual(@as(usize, 100 + 3968), alignedEnd(&big, 0, 4096));
    try t.expectEqual(@as(usize, 100 + 3968 + 4096), alignedEnd(&big, 100 + 3968, 4096));
    try t.expectEqual(@as(usize, 9000), alignedEnd(&big, 100 + 3968 + 4096, 4096));
    var exact: [8192]u32 = undefined;
    for (&exact, 0..) |*v, i| v.* = if (i < 4096) 3 else 4;
    try t.expectEqual(@as(usize, 4096), alignedEnd(&exact, 0, 4096));
    // an expert of fewer than 128 rows crossing the cap: cut at its first row
    var short: [4200]u32 = undefined;
    for (&short, 0..) |*v, i| v.* = if (i < 4000) 5 else 6;
    try t.expectEqual(@as(usize, 4000), alignedEnd(&short, 0, 4096));
}

test "turboexl3_glm53f stream math: TURBOEXL3_GLM53F_BANKED takes g4m3.t4.t4, turboexl3's grouped texts with banked_m3 at GEMV variant t4, and sets them in one configure; anything else refuses by name" {
    const t = std.testing;
    const t4 = t3.groupVariantOf("t4").?;
    try t.expectEqual(BankedSchedule{ .gu = t4, .dn = t4 }, try BankedSchedule.parse("g4m3.t4.t4"));
    for ([_][]const u8{ "", "g4", "g4m3", "g4p", "g4m3p", "g4.t4.t4", "g4m3.t4", "g4m3.0.0", "g4m3.t4.0", "g4m3p.t4.t4", "g4m3.t4.t4.0", "G4M3.T4.T4", "g4m3.t4.t4 " }) |v| try t.expectError(error.BadSwitch, BankedSchedule.parse(v));
    const Env = struct {
        var v: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TURBOEXL3_GLM53F_BANKED")) v else null;
        }
    };
    try t.expectEqual(@as(?BankedSchedule, null), try BankedSchedule.read(&Env.get));
    Env.v = "g4m3.t4.t4";
    try t.expectEqual(@as(?BankedSchedule, .{ .gu = t4, .dn = t4 }), try BankedSchedule.read(&Env.get));
    Env.v = "g4.zz";
    try t.expectError(error.Glm53fBankedSwitch, BankedSchedule.read(&Env.get));
    // applied: the schedule's fields in one configure, every other one kept
    const saved = t3.config.get().*;
    defer t3.configure(saved) catch unreachable;
    try (BankedSchedule{ .gu = t4, .dn = t4 }).apply();
    try t.expectEqual(@as(u32, 4), t3.config.get().banked_group);
    try t.expect(t3.config.get().banked_m3);
    try t.expectEqual(t4, t3.config.get().banked_variant_gu);
    try t.expectEqual(t4, t3.config.get().banked_variant_dn);
    try t.expectEqual(saved.banked_ptg, t3.config.get().banked_ptg);
    try t.expectEqual(saved.banked_wave3, t3.config.get().banked_wave3);
}

test "turboexl3_glm53f stream math: TURBOEXL3_GLM53F_DEC takes g1, turboexl3's DEC-ONE variant, and sets it; anything else refuses by name" {
    const t = std.testing;
    const g1 = t3.dense.decVariantOf("g1").?;
    try t.expectEqual(DecSchedule{ .variant = g1 }, try DecSchedule.parse("g1"));
    for ([_][]const u8{ "", "t4", "t1", "ku", "g2", "bt2", "g1.bt2", "g1.", "G1", "1" }) |v| try t.expectError(error.BadSwitch, DecSchedule.parse(v));
    const Env = struct {
        var v: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TURBOEXL3_GLM53F_DEC")) v else null;
        }
    };
    try t.expectEqual(@as(?DecSchedule, null), try DecSchedule.read(&Env.get));
    Env.v = "0";
    try t.expectEqual(@as(?DecSchedule, null), try DecSchedule.read(&Env.get));
    Env.v = "g1";
    try t.expectEqual(@as(?DecSchedule, .{ .variant = g1 }), try DecSchedule.read(&Env.get));
    Env.v = "t5";
    try t.expectError(error.Glm53fDecSwitch, DecSchedule.read(&Env.get));
    // applied: the variant, the output tiles kept
    const saved = t3.config.get().*;
    defer t3.configure(saved) catch unreachable;
    try (DecSchedule{ .variant = g1 }).apply();
    try t.expectEqual(g1, t3.config.get().dense_variant);
    try t.expectEqualSlices(u8, &saved.dec_bt, &t3.config.get().dec_bt);
}

/// The math over backend `G` (the streamer's `ops.MlxOps`), slot arrays `Proj` ({code, rout, rin}), a bank's three
/// `Bank` ({gate, up, down}), a prompt call's `PrefillRows` and `Stock`, the streamer's own math, whose constants, phase
/// routes and predictor this math keeps.
pub fn Math(comptime G: type, comptime Proj: type, comptime Bank: type, comptime PrefillRows: type, comptime Stock: type) type {
    return struct {
        const Self = @This();
        const T = G.T;
        const Predictor = @FieldType(Stock, "predictor");

        pub const decode_forward_rows: u32 = Stock.decode_forward_rows;
        pub const prefill_min_tokens: u32 = Stock.prefill_min_tokens;
        pub const wide_event_gates = Stock.wide_event_gates;
        pub const predictor_decode_only = Stock.predictor_decode_only;
        pub const Profile = Stock.Profile;
        pub const has_banked = true;
        pub const has_parts = true;
        /// The scheduler's pending group under `Wide.drain_lag` (TURBOEXL3_GLM53F_PF_DRAIN_LAG): `prefillParts` runs its
        /// hook once its first chunk is committed.
        pub fn LagPending(comptime Ctx: type) type {
            return LagPendingOf(Ctx);
        }

        pub fn predictorScores(g: anytype, x: @TypeOf(g.*).T, router: @TypeOf(g.*).T, correction: @TypeOf(g.*).T) !@TypeOf(g.*).T {
            return Stock.predictorScores(g, x, router, correction);
        }

        /// A cold-row chunk's gate and up arrays, from its `gateUp` to its `down` (keyed by the rows `gateUp` returned).
        const Parked = struct { key: ?*anyopaque = null, gate: Proj = undefined, up: Proj = undefined };

        allocator: std.mem.Allocator,
        predictor: Predictor = .{},
        limit: c_int,
        parked: *Parked,
        /// the fused paths' switches (the file's doc), read at construction
        sw: Switches = .{},

        /// Construction: the stock math's signature (its texts are not installed: nothing here runs them).
        const init_params = @typeInfo(@TypeOf(Stock.init)).@"fn".param_types;
        pub fn init(a: std.mem.Allocator, s: mlx.mlx_stream, call_rows: u32, forms: init_params[3].?, wave_rows: u32, pack: init_params[5].?, diag: init_params[6].?) !Self {
            _ = .{ s, call_rows, forms, wave_rows, pack, diag };
            const sw = Switches.read(&processEnv) catch |e| {
                log.err("[turboexl3_glm53f] {s}: TURBOEXL3_GLM53F_PF_ROWS, TURBOEXL3_GLM53F_DEC_BF16, TURBOEXL3_GLM53F_PF_FEED and TURBOEXL3_GLM53F_PF_ALIGN take 0 or 1 (unset: 1): the load is refused\n", .{@errorName(e)});
                return e;
            };
            if (sw.pf_rows) log.info(pf_rows_line, .{});
            if (sw.dec_bf16) log.info(dec_bf16_line, .{});
            if (sw.pf_feed) log.info(pf_feed_line, .{});
            if (sw.pf_align) log.info(pf_align_line, .{prompt_rows});
            const p = try a.create(Parked);
            p.* = .{};
            return .{ .allocator = a, .limit = @intFromFloat(limit_f), .parked = p, .sw = sw };
        }

        /// GLM-5.3-Flash's `swiglu_limit`, as the streamer's texts fuse it.
        const limit_f: f64 = 10;

        pub fn deinit(self: *Self, _: *G) void {
            self.predictor.deinit();
            self.allocator.destroy(self.parked);
        }

        pub fn predictorScoresBound(self: *const Self, g: *G, x: T, router: T, correction: T) !T {
            return self.predictor.scores(&self.predictor, g, x, router, correction);
        }

        // ── Decode ──

        /// A cold-row chunk's gate|up: nothing runs yet; its rows (bf16) and its gate and up arrays wait for its `down`.
        pub fn gateUp(self: *const Self, g: *G, x: T, ids: T, gate: Proj, up: Proj) !T {
            _ = ids;
            const xb = if (g.dtypeOf(x) == .bfloat16) x else try g.astype(x, .bfloat16);
            self.parked.* = .{ .key = xb.ctx, .gate = gate, .up = up };
            return xb;
        }

        /// The chunk's rows through their slots' experts (`ids`: each row's slot in the arrays), the parked gate and up
        /// with `d` as one bank, on the prompt rows' path (`gemm`): bf16 [rows, hidden] at weight 1.
        pub fn down(self: *const Self, g: *G, h: T, ids: T, d: Proj) !T {
            if (self.parked.key != h.ctx) return error.RoutedT3DownWithoutGateUp;
            const b: Bank = .{ .gate = self.parked.gate, .up = self.parked.up, .down = d };
            self.parked.key = null;
            const rows: usize = @intCast(g.shapeOf(ids).dim(0));
            const sid = if (g.dtypeOf(ids) == .uint32) ids else try g.astype(ids, .uint32);
            if (self.sw.pf_rows) {
                // the chunk's rows are h's own, in order: the row map is the identity
                const ident = try self.allocator.alloc(u32, rows);
                defer self.allocator.free(ident);
                for (ident, 0..) |*v, i| v.* = @intCast(i);
                const rm = try g.hostArray(std.mem.sliceAsBytes(ident), &.{@intCast(rows)}, .uint32);
                return self.gemmRows(g, h, rm, sid, rows, try t3View(g, b));
            }
            return self.gemm(g, h, sid, rows, try t3View(g, b));
        }

        /// A wave's gate|up: nothing runs yet; `x` [pairs, hidden] (row-slot r's row) is what `downBanked` receives.
        pub fn gateUpBanked(self: *const Self, g: *G, x: T, ids: T, banks: *const [3]Bank) !T {
            _ = .{ self, g, ids, banks };
            return x;
        }

        /// The wave over the three banks (base, ext, transient), whole: `ids` u32 [pairs] packed (bank << 24 | slot row)
        /// -> bf16 [pairs, hidden], each pair's row at weight 1.
        pub fn downBanked(self: *const Self, g: *G, x: T, ids: T, banks: *const [3]Bank) !T {
            var tb: [3]t3.Bank = undefined;
            var rows: [3]c_int = undefined;
            var views: [3 * 6]mlx.mlx_array = undefined;
            var nv: usize = 0;
            defer for (views[0..nv]) |v| {
                _ = mlx.mlx_array_free(v);
            };
            for (banks, &tb, &rows) |b, *o, *r| {
                r.* = mlx.getShape(b.gate.code)[0];
                inline for (.{ "gate", "up", "down" }) |f| {
                    const p = @field(b, f);
                    const n = mlx.getShape(p.code)[0];
                    @field(o.*, f) = .{ .trellis = p.code, .suh = try fitted(p.rin, n, g.s, &views, &nv), .svh = try fitted(p.rout, n, g.s, &views, &nv) };
                }
            }
            // TURBOEXL3_GLM53F_DEC_BF16: the banked down finish stores bf16 (moeClampedBanked's f32 y cast to bf16, bit for bit)
            const y0 = (if (self.sw.dec_bf16)
                t3.moeClampedBankedAs(g.s, x, &tb, ids, .mul1, self.limit, .bfloat16)
            else
                t3.moeClampedBanked(g.s, x, &tb, ids, .mul1, self.limit)) catch |e| {
                log.err("[turboexl3_glm53f] turboexl3 refused a decode wave: {s} ({d} pairs; banks of {d} / {d} / {d} rows)\n", .{ @errorName(e), g.shapeOf(ids).dim(0), rows[0], rows[1], rows[2] });
                return e;
            };
            const y = try g.adopt(y0);
            if (self.sw.dec_bf16) return y;
            return g.astype(y, .bfloat16);
        }

        // ── Prompt ──

        /// `rows` rows (`xb` bf16 [rows, hidden], `slot` u32 [rows], each row's slot in `tb`) through turboexl3's
        /// moeClamped at top-1 and weight 1 (x read in f32): bf16 [rows, hidden]. A call of 16 rows or fewer is padded
        /// to `gemm_min_rows` with copies of its first row at score 0, so every row takes the prefill GEMM, whose row
        /// bytes do not depend on the call's other rows.
        fn gemm(self: *const Self, g: *G, xb: T, slot: T, rows: usize, tb: t3.Bank) !T {
            if (rows == 0) return error.RoutedT3EmptyCall;
            const hidden: c_int = g.shapeOf(xb).dim(1);
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
            const y30 = t3.moeClamped(g.s, x3, tb, inds, sc, .mul1, self.limit) catch |e| {
                log.err("[turboexl3_glm53f] turboexl3 refused a prompt call: {s} ({d} rows; a bank of {d} rows)\n", .{ @errorName(e), rows, mlx.getShape(tb.gate.trellis)[0] });
                return e;
            };
            const y3 = try g.adopt(y30);
            const y_all = try g.astype(try g.reshape(y3, &.{ nc, hidden }), .bfloat16);
            if (!small) return y_all;
            return g.slice(y_all, &.{ 0, 0 }, &.{ @intCast(rows), hidden }, &.{ 1, 1 });
        }

        /// TURBOEXL3_GLM53F_PF_ROWS: `gemm`'s call on turboexl3's moeClampedRows: row-slot i reads x's row `rmap[i]` (x
        /// bf16 [n, hidden], `rmap` u32 [rows]) in the chain's prepare, through slot `slot[i]` at weight 1, y stored bf16
        /// by the chain's finish: `gemm` over `take(x, rmap)`, bit for bit, without the gathered rows, their f32 copy or
        /// the cast of y. A call of 16 rows or fewer is padded as `gemm` pads it, with row-slot 0's row and slot at score 0.
        fn gemmRows(self: *const Self, g: *G, x: T, rmap: T, slot: T, rows: usize, tb: t3.Bank) !T {
            if (rows == 0) return error.RoutedT3EmptyCall;
            const hidden: c_int = g.shapeOf(x).dim(1);
            const small = !t3.kernels.usesPrefillArm(rows);
            const n_call: usize = if (small) gemm_min_rows else rows;
            const nc: c_int = @intCast(n_call);
            var rm = rmap;
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
                rm = try g.take(rmap, ia, 0);
                sc_ids = try g.take(slot, ia, 0);
                sc = try g.hostArray(std.mem.sliceAsBytes(w[0..]), &.{nc}, .float32);
            } else sc = try g.ones(&.{nc}, .float32);
            const xb = if (g.dtypeOf(x) == .bfloat16) x else try g.astype(x, .bfloat16);
            const y0 = t3.moeClampedRows(g.s, xb, rm, tb, sc_ids, sc, .mul1, self.limit, .bfloat16) catch |e| {
                log.err("[turboexl3_glm53f] turboexl3 refused a prompt call: {s} ({d} rows of {d}; a bank of {d} rows)\n", .{ @errorName(e), rows, g.shapeOf(x).dim(0), mlx.getShape(tb.gate.trellis)[0] });
                return e;
            };
            const y = try g.adopt(y0);
            if (!small) return y;
            return g.slice(y, &.{ 0, 0 }, &.{ @intCast(rows), hidden }, &.{ 1, 1 });
        }

        /// A bank's three projections as turboexl3's moeClamped takes them: code viewed as u16 (the same bits), rin as
        /// suh, rout as svh, the scales at the code's rows (`fitRows`; views owned by `g`).
        fn t3View(g: *G, bank: Bank) !t3.Bank {
            var tb: t3.Bank = undefined;
            inline for (.{ "gate", "up", "down" }) |f| {
                const pj = @field(bank, f);
                const n = mlx.getShape(pj.code)[0];
                @field(tb, f) = .{ .trellis = try viewU16(g, pj.code), .suh = try fitG(g, pj.rin, n), .svh = try fitG(g, pj.rout, n) };
            }
            return tb;
        }

        /// `fitted`'s view owned by `g` (the prompt path's arrays live to the call's end).
        fn fitG(g: *G, a: T, rows: c_int) !T {
            const sh = mlx.getShape(a);
            if (sh.len != 2) return a;
            const r = fitRows(rows, sh[0]) orelse return a;
            return g.slice(a, &.{ 0, 0 }, &.{ r, sh[1] }, &.{ 1, 1 });
        }

        fn viewU16(g: *G, code: T) !T {
            var v = mlx.mlx_array_new();
            mlx.check(mlx.mlx_view(&v, code, .uint16, g.s)) catch |e| {
                _ = mlx.mlx_array_free(v);
                return e;
            };
            return g.adopt(v);
        }

        /// TURBOEXL3_GLM53F_PF_MOE_HOLD's take: a committed chunk output's data address (its buffer's, at row 0), 0
        /// when it has none.
        fn bufferAddr(y: T) usize {
            return if (mlx.mlx_array_data_bfloat16(y)) |p| @intFromPtr(p) else 0;
        }

        /// The call's rows (x's rows, or `rows.act_row`'s) sorted by slot (a stable counting sort), then in chunks of at
        /// most `prompt_rows` each one `gemm` call; each chunk's KEPT output (bf16 [rows,
        /// hidden]) is appended to `outs` and its rows' assignment rows to `pos`. At most two chunks in flight; every chunk
        /// evaluated and the stream synchronized before the call returns (the stock waves' contract), or with
        /// `TURBOEXL3_GLM53F_PF_FEED` the last two left in flight for the caller's evaluation of `outs`.
        pub fn prefillParts(self: *const Self, g: *G, layer: u32, x: T, rows: PrefillRows, bank: Bank, alloc: std.mem.Allocator, outs: *std.ArrayList(T), pos: *std.ArrayList(u32)) !void {
            _ = layer;
            const n = rows.slot.len;
            if (n == 0) return;
            // TURBOEXL3_GLM53F_PF_MOE_HOLD: a DSA layer's hold back in MLX's cache, for this call's first chunks
            pf_reuse.routed();
            const a = self.allocator;
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
            const chunk: usize = prompt_rows;
            var c0: usize = 0;
            var c1: usize = 0;
            var chunks: usize = 0;
            while (c0 < n) : (c0 = c1) {
                // TURBOEXL3_GLM53F_PF_ALIGN: the chunk ends where no expert's 128-row run is split
                c1 = if (self.sw.pf_align) alignedEnd(sorted_slot, c0, chunk) else @min(c0 + chunk, n);
                chunks += 1;
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
                // TURBOEXL3_GLM53F_PF_MOE_HOLD's take: the cache across the call's first allocation (where MLX trims a
                // cache over its limit)
                const first = chunks == 1 and pf_reuse.measuring();
                if (first) pf_reuse.at(.first_alloc_before);
                const ai = try g.hostArray(std.mem.sliceAsBytes(sorted_act[c0..c1]), &.{mc}, .uint32);
                if (first) pf_reuse.at(.first_alloc_after);
                const sid = try g.hostArray(std.mem.sliceAsBytes(sorted_slot[c0..c1]), &.{mc}, .uint32);
                // TURBOEXL3_GLM53F_PF_ROWS: x's rows read through `ai` in the chain's prepare (no gathered copy)
                const y = if (self.sw.pf_rows) try self.gemmRows(g, x, ai, sid, c1 - c0, tb) else blk: {
                    const xr = try g.take(x, ai, 0);
                    const xb = if (g.dtypeOf(xr) == .bfloat16) xr else try g.astype(xr, .bfloat16);
                    break :blk try self.gemm(g, xb, sid, c1 - c0, tb);
                };
                outs.appendAssumeCapacity(g.keep(y));
                pos.appendSliceAssumeCapacity(order[c0..c1]);
                // and across the first two chunks' commits (MLX allocates the chain's arrays as it encodes them), with
                // the output's buffer
                const meas = pf_reuse.measuring();
                if (meas) pf_reuse.at(.commit_before);
                try g.asyncEval(&.{y});
                // TURBOEXL3_GLM53F_PF_MOE_HOLD: after the second chunk, what the call's first chunks allocated past the
                // hand-back
                pf_reuse.committed(chunks, if (meas) bufferAddr(y) else 0);
                flight[n_flight] = g.keep(y);
                n_flight += 1;
                g.resetTo(m);
                // TURBOEXL3_GLM53F_PF_DRAIN_LAG: the scheduler's pending group (its drain, release and next route) once
                // this call's first chunk is committed; armed once a group, so a later chunk finds none
                if (takeLagHook()) |h| {
                    try h.run(h.ctx);
                    if (!lag_said) {
                        lag_said = true;
                        log.info(pf_drain_lag_engaged, .{n});
                    }
                }
            }
            // TURBOEXL3_GLM53F_PF_MOE_HOLD: a call of one chunk measured at its end
            pf_reuse.committed(0, 0);
            if (self.sw.pf_align and !pf_align_said) {
                pf_align_said = true;
                log.info(pf_align_engaged, .{ n, chunks, (n + chunk - 1) / chunk });
            }
            // TURBOEXL3_GLM53F_PF_FEED: the last chunks stay in flight; every caller evaluates `outs` before the slots
            // they read go back (the scheduler's group drain, `prefill`'s join)
            if (self.sw.pf_feed) return;
            try g.evalAll(flight[0..n_flight]);
            try mlx.check(mlx.mlx_synchronize(g.s));
        }

        /// `prefillParts` joined: a KEPT, evaluated bf16 [rows, hidden] in the call's row order.
        pub fn prefill(self: *const Self, g: *G, layer: u32, x: T, rows: PrefillRows, bank: Bank) !T {
            if (rows.slot.len == 0) return error.RoutedT3EmptyCall;
            const a = self.allocator;
            var parts: std.ArrayList(T) = .empty;
            defer {
                for (parts.items) |pt| g.release(pt);
                parts.deinit(a);
            }
            var at: std.ArrayList(u32) = .empty;
            defer at.deinit(a);
            try self.prefillParts(g, layer, x, rows, bank, a, &parts, &at);
            const m = g.mark();
            defer g.resetTo(m);
            const joined = if (parts.items.len == 1) parts.items[0] else try g.concat(parts.items, 0);
            const inv = try a.alloc(u32, at.items.len);
            defer a.free(inv);
            for (at.items, 0..) |pi, j| inv[pi] = @intCast(j);
            const ord = try g.hostArray(std.mem.sliceAsBytes(inv), &.{@intCast(inv.len)}, .uint32);
            const result = try g.take(joined, ord, 0);
            try g.evalAll(&.{result});
            return g.keep(result);
        }

        /// Every chunk was evaluated and synchronized before its call returned, or (`TURBOEXL3_GLM53F_PF_FEED`) is in
        /// flight until the caller's evaluation of the call's outputs, which follows this.
        pub fn finishPrefill(_: *const Self, _: *G) !void {}

        /// The kernels a layer's calls build, compiled at load over zero rows and slot row 0 of `b` (one routed layer's
        /// base bank, read only): the banked decode at every pair count of the decode lane (1 to `max_pairs`) and the
        /// prompt path at one padded call. Returns the calls made.
        pub fn prebuild(self: *const Self, g: *G, b: Bank) !u32 {
            var n: u32 = 0;
            const hidden: c_int = g.shapeOf(b.gate.rin).dim(1);
            {
                const m = g.mark();
                defer g.resetTo(m);
                const x = try g.zeros(&.{ 1, hidden }, .bfloat16);
                const sid = try g.zeros(&.{1}, .uint32);
                // the prompt path the switch serves (its kernels built here, as the default's are)
                const y = if (self.sw.pf_rows) try self.gemmRows(g, x, sid, sid, 1, try t3View(g, b)) else try self.gemm(g, x, sid, 1, try t3View(g, b));
                try g.evalAll(&.{y});
                n += 1;
            }
            const banks = [3]Bank{ b, b, b };
            var p: u32 = 1;
            while (p <= max_pairs) : (p += 1) {
                const m = g.mark();
                defer g.resetTo(m);
                const pi: c_int = @intCast(p);
                const ids = try g.zeros(&.{pi}, .uint32);
                const x = try g.zeros(&.{ pi, hidden }, .bfloat16);
                try g.evalAll(&.{try self.downBanked(g, try self.gateUpBanked(g, x, ids, &banks), ids, &banks)});
                n += 1;
            }
            return n;
        }

        /// The decode lane's widest wave: its rows at top-8 (the MTP verify's 4 rows).
        const max_pairs: u32 = decode_forward_rows * 8;
    };
}

test "turboexl3_glm53f stream math: TURBOEXL3_GLM53F_PF_OPTS takes bk128 and sets turboexl3's routed prefill option; unset and 0 keep the served ones, anything else refuses by name" {
    const t = std.testing;
    try t.expectEqual(PfOpts{ .bk128 = true }, try PfOpts.parse("bk128"));
    for ([_][]const u8{ "", "1", "hr", "bk128.hr", "tk", "bk128.tk", "bk128.", "BK128", "bk64", "half_rows", "t128_bk128" }) |v| try t.expectError(error.BadSwitch, PfOpts.parse(v));
    const Env = struct {
        var v: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TURBOEXL3_GLM53F_PF_OPTS")) v else null;
        }
    };
    try t.expectEqual(@as(?PfOpts, null), try PfOpts.read(&Env.get));
    Env.v = "0";
    try t.expectEqual(@as(?PfOpts, null), try PfOpts.read(&Env.get));
    Env.v = "bk128";
    try t.expectEqual(@as(?PfOpts, .{ .bk128 = true }), try PfOpts.read(&Env.get));
    Env.v = "x";
    try t.expectError(error.Glm53fPfOptsSwitch, PfOpts.read(&Env.get));
    // apply sets the field and keeps every other one
    const saved = t3.config.get().*;
    defer t3.configure(saved) catch unreachable;
    try (PfOpts{ .bk128 = true }).apply();
    try t.expect(t3.config.get().pf_bk128);
    try t.expectEqual(saved.pf_half_rows, t3.config.get().pf_half_rows);
    try t.expectEqual(saved.pf_t128_bk128, t3.config.get().pf_t128_bk128);
    try t.expectEqual(saved.pf_t128, t3.config.get().pf_t128);
    try t.expectEqual(saved.banked_group, t3.config.get().banked_group);
}

test "turboexl3_glm53f stream math: TURBOEXL3_GLM53F_PF_INT8 takes 1 to 4 and sets turboexl3's routed prefill int8 scheme (3: scheme 1 and the trunk's KDA qkv and o_proj on int8; 4: scheme 1 and every trunk linear); unset and 0 keep the exact chain, anything else refuses by name" {
    const t = std.testing;
    try t.expectEqual(PfInt8{ .scheme = 1 }, try PfInt8.parse("1"));
    try t.expectEqual(PfInt8{ .scheme = 2 }, try PfInt8.parse("2"));
    try t.expectEqual(PfInt8{ .scheme = 3 }, try PfInt8.parse("3"));
    try t.expectEqual(PfInt8{ .scheme = 4 }, try PfInt8.parse("4"));
    for ([_][]const u8{ "", "0", "5", "12", "01", "1.", " 1", "-1", "on", "true", "int8", "i8t" }) |v| try t.expectError(error.BadSwitch, PfInt8.parse(v));
    const Env = struct {
        var v: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TURBOEXL3_GLM53F_PF_INT8")) v else null;
        }
    };
    try t.expectEqual(@as(?PfInt8, null), try PfInt8.read(&Env.get));
    Env.v = "0";
    try t.expectEqual(@as(?PfInt8, null), try PfInt8.read(&Env.get));
    Env.v = "1";
    try t.expectEqual(@as(?PfInt8, .{ .scheme = 1 }), try PfInt8.read(&Env.get));
    Env.v = "2";
    try t.expectEqual(@as(?PfInt8, .{ .scheme = 2 }), try PfInt8.read(&Env.get));
    Env.v = "3";
    try t.expectEqual(@as(?PfInt8, .{ .scheme = 3 }), try PfInt8.read(&Env.get));
    Env.v = "4";
    try t.expectEqual(@as(?PfInt8, .{ .scheme = 4 }), try PfInt8.read(&Env.get));
    for ([_][]const u8{ "5", "x", "" }) |bad| {
        Env.v = bad;
        try t.expectError(error.Glm53fPfInt8Switch, PfInt8.read(&Env.get));
    }
    // apply sets the scheme (3: scheme 1 and the trunk's int8 at the KDA shapes; 4: at every trunk shape) and keeps
    // every other field
    const saved = t3.config.get().*;
    const saved_shapes = t3.trunk_q8.shapes.get().*;
    defer {
        t3.configure(saved) catch unreachable;
        t3.trunk_q8.shapes.get().* = saved_shapes;
    }
    for ([_]u8{ 1, 2, 3, 4 }) |scheme| {
        try (PfInt8{ .scheme = scheme }).apply();
        try t.expectEqual(if (scheme >= 3) @as(u8, 1) else scheme, t3.config.get().pf_int8);
        try t.expectEqual(scheme >= 3, t3.config.get().pf_trunk_int8);
        if (scheme == 3) {
            const sh = t3.trunk_q8.shapes.get().*.?;
            try t.expectEqual(@as(usize, 2), sh.len);
            try t.expect(sh[0].k == 4096 and sh[0].n == 24576 and sh[1].k == 8192 and sh[1].n == 4096);
        } else if (scheme == 4) {
            const sh = t3.trunk_q8.shapes.get().*.?;
            try t.expectEqual(t3.glm53f.trunk.len, sh.len);
            for (sh, t3.glm53f.trunk) |a, l| try t.expect(a.k == l.k and a.n == l.n);
        } else try t.expect(t3.trunk_q8.shapes.get().* == null);
        try t.expectEqual(saved.pf_t128, t3.config.get().pf_t128);
        try t.expectEqual(saved.pf_size_class, t3.config.get().pf_size_class);
        try t.expectEqual(saved.pf_bk128, t3.config.get().pf_bk128);
        try t.expectEqual(saved.pf_alay, t3.config.get().pf_alay);
        try t.expectEqual(saved.banked_group, t3.config.get().banked_group);
    }
}

test "turboexl3_glm53f stream math: TURBOEXL3_GLM53F_PG2 names one of turboexl3's pg2 tiles a trunk linear (0 the chain) and installs that table; unset and 0 leave pg2 off, any other value refuses by name" {
    const t = std.testing;
    const trunk = t3.glm53f.trunk;
    const none = try Pg2.parse("00000000000");
    try t.expectEqual(@as(usize, 0), none.n);
    const p = try Pg2.parse("20000a00003");
    try t.expectEqual(@as(usize, 3), p.n);
    try t.expect(p.entries[0].k == trunk[0].k and p.entries[0].n == trunk[0].n and std.meta.eql(p.entries[0].tile, t3.pgemm.candidates[1]));
    try t.expect(p.entries[1].k == trunk[5].k and p.entries[1].n == trunk[5].n and std.meta.eql(p.entries[1].tile, t3.pgemm.candidates[9]));
    try t.expect(p.entries[2].k == trunk[10].k and p.entries[2].n == trunk[10].n and std.meta.eql(p.entries[2].tile, t3.pgemm.candidates[2]));
    for ([_][]const u8{ "", "1", "0000000000", "000000000000", "0000000000-", "0000000000Z", "0000000000z" }) |v| try t.expectError(error.BadSwitch, Pg2.parse(v));
    const Env = struct {
        var v: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TURBOEXL3_GLM53F_PG2")) v else null;
        }
    };
    try t.expect((try Pg2.read(&Env.get)) == null);
    Env.v = "0";
    try t.expect((try Pg2.read(&Env.get)) == null);
    Env.v = "x";
    try t.expectError(error.Glm53fPg2Switch, Pg2.read(&Env.get));
    // apply installs the table (turboexl3 then looks a trunk shape up in it) and turns pg2 on
    const saved_table = t3.pgemm.host_table.get().*;
    const saved_mode = t3.pgemm.mode.get().*;
    defer {
        t3.pgemm.host_table.get().* = saved_table;
        t3.pgemm.mode.get().* = saved_mode;
    }
    Env.v = "20000a00003";
    (try Pg2.read(&Env.get)).?.apply();
    try t.expectEqual(t3.pgemm.Mode.on, t3.pgemm.mode.get().*);
    try t.expectEqual(@as(usize, 3), t3.pgemm.host_table.get().*.?.len);
    try t.expect(std.meta.eql(t3.pgemm.tileOf(trunk[5].k, trunk[5].n).?, t3.pgemm.candidates[9]));
    try t.expect(t3.pgemm.tileOf(trunk[1].k, trunk[1].n) == null);
}

test "turboexl3_glm53f stream math: TURBOEXL3_GLM53F_PF_DRAIN_LAG drains each prompt group once and in order, after the next group's first committed chunk (or that group's banks, when it commits none), and releases it after its own drain; the call's last group at its end" {
    const t = std.testing;
    // An event as a number: commit 10000 + group * 100 + chunk, drain 20000 + k0 * 100 + k1, release 30000 + group.
    const Log = struct {
        evs: [32]u32 = undefined,
        n: usize = 0,
        fn add(l: *@This(), e: usize) void {
            l.evs[l.n] = @intCast(e);
            l.n += 1;
        }
    };
    var events: Log = .{};
    const Ctx = struct {
        l: *Log,
        pub fn drain(c: @This(), k0: usize, k1: usize) !void {
            c.l.add(20000 + k0 * 100 + k1);
        }
        pub fn release(c: @This(), gi: usize) !void {
            c.l.add(30000 + gi);
        }
    };
    var p: LagPendingOf(Ctx) = .{ .ctx = .{ .l = &events } };
    defer p.disarm();
    const lagged0 = lag_lagged;
    const flushed0 = lag_flushed;
    // The scheduler's loop over four groups committing 2, 3, 0 and 1 chunks (one kept output a chunk), each commit as
    // `prefillParts` makes it: the armed hook taken and run.
    var kept: usize = 0;
    for ([_]usize{ 2, 3, 0, 1 }, 0..) |chunks, gi| {
        p.begin();
        const k0 = kept;
        for (0..chunks) |c| {
            events.add(10000 + gi * 100 + c);
            kept += 1;
            if (takeLagHook()) |h| try h.run(h.ctx);
        }
        try p.end(gi, k0, kept);
    }
    try p.flush();
    try t.expect(takeLagHook() == null);
    try t.expectEqualSlices(u32, &.{
        10000, 10001, // group 0: nothing pending
        10100, 20002, 30000, 10101, 10102, // group 0 at group 1's first commit
        20205, 30001, // group 1 at group 2's end: group 2 committed nothing
        10300, 20505, 30002, // group 2 at group 3's first commit
        20506, 30003, // group 3 at the call's end
    }, events.evs[0..events.n]);
    try t.expectEqual(lagged0 + 2, lag_lagged);
    try t.expectEqual(flushed0 + 1, lag_flushed);
}
