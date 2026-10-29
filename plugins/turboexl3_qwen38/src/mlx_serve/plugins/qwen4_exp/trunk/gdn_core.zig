//! copied_from: src/plugins/qwen4_exp/trunk/gdn_core.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! The GDN core: a prefill chunk's packed prework and pipelined recurrence on kernels shaped for the GPU, each output
//! the replaced kernel's bytes. `prework` (kernels/gdn_core_prework.metal) walks a run of rows per simdgroup where the
//! packed prework runs a threadgroup per (row, head slot); `recur` (kernels/gdn_core_recur.metal) is the pipelined
//! kernel with its block and threadgroup sizes as template ints. Decode, verify and the stock route's short calls
//! (T < 64) never take them.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const log = @import("../../../sdk/util/log.zig");
const decode_fused = @import("../../../sdk/blocks/gdn/decode_fused.zig");
const sigmoidTableFor = @import("../../../sdk/blocks/mlp/fused_activations.zig").sigmoidTableFor;
const GdnPrework = decode_fused.GdnPrework;
const GdnPreworkArgs = decode_fused.GdnPreworkArgs;

pub const PREWORK_SOURCE = kernelSource("plugins/qwen4_exp/kernels/gdn_core_prework.metal");
pub const PREWORK_HEADER = kernelSource("kernels/gdn/log1p.metal");
pub const RECUR_SOURCE = kernelSource("plugins/qwen4_exp/kernels/gdn_core_recur.metal");
pub const RECUR_HEADER = kernelSource("plugins/qwen4_exp/kernels/gdn_core_recur_header.metal");

/// The prework's geometry: rows a run (whole 4-row groups) and head slots a threadgroup.
pub const PW_ROWS: c_int = 4;
pub const PW_NSG: c_int = 8;
/// The recurrence's: tokens a staged block (whole 4-token groups), value rows a threadgroup, k and q staged in the input
/// dtype (KQF 0), one value row a lane (RPT 1).
pub const TB: c_int = 16;
pub const DB: c_int = 32;
pub const KQF: c_int = 0;
pub const RPT: c_int = 1;
/// Threads a recurrence threadgroup: 8 lanes a row group.
pub const NT: c_int = @divExact(8 * DB, RPT);
const METAL_TG_BYTES: usize = 32 * 1024;

var recur_kernel: ?mlx.mlx_fast_metal_kernel = null;
var pw_kernel: ?mlx.mlx_fast_metal_kernel = null;
const PwKey = struct { hk: c_int, hv: c_int, seq: c_int, batch: c_int, c: c_int, qs: c_int, qo: c_int, bs: c_int, bo: c_int, as: c_int, ao: c_int, dt: mlx.mlx_dtype, gt: mlx.mlx_dtype };
var pw_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var pw_cfg_key: PwKey = std.mem.zeroes(PwKey);
var core_said = false;

/// Threadgroup bytes of one recurrence threadgroup: the k and q stages, v, g and beta, y.
pub fn recurTgBytes(dk: usize, in_size: usize, out_size: usize) usize {
    const tb: usize = TB;
    const db: usize = DB;
    const ks: usize = if (KQF >= 1) 4 else in_size;
    const qs: usize = if (KQF >= 2) 4 else in_size;
    return tb * dk * ks + tb * dk * qs + tb * db * 4 + 2 * tb * 4 + tb * db * out_size;
}

/// Whether a pipelined-route call (Dk 128) of value width `dv` takes the core recurrence: whole DB-row threadgroups,
/// the stages within 32 KiB at the input width, a 16-bit y (the forward's); else the pipelined kernel runs.
pub fn recurServes(dv: c_int, in_dtype: mlx.mlx_dtype, out_dtype: mlx.mlx_dtype) bool {
    if (dv <= 0 or @rem(dv, DB) != 0) return false;
    const in_size: usize = switch (in_dtype) {
        .bfloat16, .float16 => 2,
        .float32 => 4,
        else => return false,
    };
    if (out_dtype != .bfloat16 and out_dtype != .float16) return false;
    return recurTgBytes(128, in_size, 2) <= METAL_TG_BYTES;
}

var declined_said = false;

/// The declined line, once: the first prefill chunk whose recurrence the core kernel does not serve.
pub fn sayDeclined(why: []const u8) void {
    if (declined_said) return;
    declined_said = true;
    log.info("[gdn-core] declined: {s}; the pipelined recurrence serves\n", .{why});
}

/// The engaged line, once: the first call that ran a part.
pub fn say(seq_len: c_int, pw_ran: bool, rec: bool) void {
    if (core_said or (!pw_ran and !rec)) return;
    core_said = true;
    log.info("[gdn-core] engaged: S={d} prework={} recurrence={}\n", .{ seq_len, pw_ran, rec });
}

fn recurKernel() !mlx.mlx_fast_metal_kernel {
    if (recur_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "q", "k", "v", "g", "beta", "state_in", "T" };
    const output_names = [_][*:0]const u8{ "y", "state_out" };
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new("msv_gdn_core_recur", in_vec, out_vec, RECUR_SOURCE, RECUR_HEADER, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    recur_kernel = k;
    return k;
}

pub const RecurOut = struct { y: mlx.mlx_array, state: mlx.mlx_array };

/// One launch of the core recurrence over q, k [B, T, Hk, Dk], v [B, T, Hv, Dv], g, beta [B, T, Hv] from `state`
/// [B, Hv, Dv, Dk]: the pipelined route's contract and bytes (y [B, T, Hv, Dv] in `out_dtype`, the state after the last
/// row in the state's dtype). The caller checks `recurServes`.
pub fn recur(s: mlx.mlx_stream, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, g: mlx.mlx_array, beta: mlx.mlx_array, state: mlx.mlx_array, hk: c_int, hv: c_int, dk: c_int, dv: c_int, out_dtype: mlx.mlx_dtype) !RecurOut {
    if (dk != 128 or dv <= 0 or @rem(dv, DB) != 0) return error.GdnCoreRecurGeometry;
    const qsh = mlx.getShape(q);
    if (qsh.len != 4) return error.GdnCoreRecurGeometry;
    const batch = qsh[0];
    const t = qsh[1];
    const st_dtype = mlx.mlx_array_dtype(state);
    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &[_]c_int{ batch, t, hv, dv }, 4, out_dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &[_]c_int{ batch, hv, dv, dk }, 4, st_dtype));
    // NT threads own DB value rows (8 lanes a row group, RPT rows a lane); Dv / DB threadgroups per value head
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, NT * @divExact(dv, DB), hv, batch));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, NT, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "InT", mlx.mlx_array_dtype(q)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "StT", st_dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "OutT", out_dtype));
    const ints = .{ .{ "Dk", dk }, .{ "Dv", dv }, .{ "Hk", hk }, .{ "Hv", hv }, .{ "TB", TB }, .{ "DB", DB }, .{ "KQF", KQF }, .{ "RPT", RPT } };
    inline for (ints) |kv| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, kv[0], kv[1]));
    const t_scalar = mlx.mlx_array_new_int(t);
    defer _ = mlx.mlx_array_free(t_scalar);
    const inputs_arr = [_]mlx.mlx_array{ q, k, v, g, beta, state, t_scalar };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    const kernel = try recurKernel();
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, config, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 2) return error.MetalKernelBadOutputCount;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, outputs_vec, 0));
    var st = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(st);
    try mlx.check(mlx.mlx_vector_array_get(&st, outputs_vec, 1));
    return .{ .y = y, .state = st };
}

fn preworkKernel() !mlx.mlx_fast_metal_kernel {
    if (pw_kernel) |k| return k;
    // the packed prework's inputs and outputs, in its order
    const input_names = [_][*:0]const u8{ "qkv", "conv_state", "conv_w", "q_scale", "k_scale", "b_in", "a_in", "A_log", "dt_bias", "sigtab", "seq", "qk_eps" };
    const output_names = [_][*:0]const u8{ "q_out", "k_out", "v_out", "conv_out", "g_out", "beta_out" };
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new("msv_gdn_core_prework", in_vec, out_vec, PREWORK_SOURCE, PREWORK_HEADER, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    pw_kernel = k;
    return k;
}

/// The packed prework's six outputs (`gdnPreworkFused`'s contract and bytes) from the row stream, at the widths the
/// packed kernel's prefill arm takes (17 <= S, B <= 2) with a per-head gate; null where it declines (nothing launched):
/// the caller runs the packed kernel. The caller owns the outputs.
pub fn prework(s: mlx.mlx_stream, in: GdnPreworkArgs) !?GdnPrework {
    if (!(decode_fused.gdn_prework_override orelse true) or !mlx.streamIsGpu(s)) return null;
    if (!decode_fused.gdnPrefillFusedFor(in.seq, in.batch) or !in.gate) return null;
    // the packed kernel's own checks (`gdnPreworkFused`)
    if (in.dk != 128 or in.dv != 128) return null;
    const dt = mlx.mlx_array_dtype(in.qkv);
    if (dt != .bfloat16 and dt != .float16) return null;
    inline for (.{ in.b, in.a, in.conv_state, in.conv_w, in.q_scale, in.k_scale, in.dt_bias }) |arr| {
        if (mlx.mlx_array_dtype(arr) != dt) return null;
    }
    const wsh = mlx.getShape(in.conv_w);
    if (wsh.len < 2 or wsh[1] != 4) return null;
    const c_dim: c_int = in.hk * in.dk * 2 + in.hv * in.dv;
    const qsh = mlx.getShape(in.qkv);
    if (qsh.len != 3 or qsh[0] != in.batch or qsh[1] != in.seq or qsh[2] != in.qkv_stride or in.qkv_off + c_dim > in.qkv_stride) return null;
    const bsh = mlx.getShape(in.b);
    if (bsh.len != 3 or bsh[0] != in.batch or bsh[1] != in.seq or bsh[2] != in.b_stride or in.b_off + in.hv > in.b_stride) return null;
    const ash = mlx.getShape(in.a);
    if (ash.len != 3 or ash[0] != in.batch or ash[1] != in.seq or ash[2] != in.a_stride or in.a_off + in.hv > in.a_stride) return null;
    const csh = mlx.getShape(in.conv_state);
    if (csh.len != 3 or csh[0] != in.batch or csh[1] != 3 or csh[2] != c_dim) return null;
    // 8-byte row reads and stores: qkv's row stride and column offset, and the channel count, on 4-element boundaries
    if (@rem(in.qkv_stride, 4) != 0 or @rem(in.qkv_off, 4) != 0 or @rem(c_dim, 4) != 0) return null;

    const nkeep: c_int = 3;
    const gt = in.g_dtype orelse dt;
    const nslot: c_int = 2 * in.hk + in.hv;
    const key = PwKey{ .hk = in.hk, .hv = in.hv, .seq = in.seq, .batch = in.batch, .c = c_dim, .qs = in.qkv_stride, .qo = in.qkv_off, .bs = in.b_stride, .bo = in.b_off, .as = in.a_stride, .ao = in.a_off, .dt = dt, .gt = gt };
    if (pw_cfg == null or !std.meta.eql(pw_cfg_key, key)) {
        if (pw_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        pw_cfg = null;
        const config = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
        const q_shape = [_]c_int{ in.batch, in.seq, in.hk, in.dk };
        const v_shape = [_]c_int{ in.batch, in.seq, in.hv, in.dv };
        const st_shape = [_]c_int{ in.batch, nkeep, c_dim };
        const g_shape = [_]c_int{ in.batch, in.seq, in.hv };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &q_shape, 4, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &q_shape, 4, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &v_shape, 4, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &st_shape, 3, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &g_shape, 3, gt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &g_shape, 3, dt));
        // NSG head slots (simdgroups) per threadgroup, one threadgroup row per run of R rows of a batch row
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32 * PW_NSG, in.batch * @divTrunc(in.seq + PW_ROWS - 1, PW_ROWS), @divTrunc(nslot + PW_NSG - 1, PW_NSG)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32 * PW_NSG, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "GT", gt));
        const ints = .{ .{ "HK", in.hk }, .{ "HV", in.hv }, .{ "DK", in.dk }, .{ "DV", in.dv }, .{ "NKEEP", nkeep }, .{ "C", c_dim }, .{ "QSTRIDE", in.qkv_stride }, .{ "QOFF", in.qkv_off }, .{ "BSTRIDE", in.b_stride }, .{ "BOFF", in.b_off }, .{ "ASTRIDE", in.a_stride }, .{ "AOFF", in.a_off }, .{ "GATE", @as(c_int, 1) }, .{ "R", PW_ROWS }, .{ "NSG", PW_NSG } };
        inline for (ints) |kv| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, kv[0], kv[1]));
        pw_cfg = config;
        pw_cfg_key = key;
    }

    const kernel = try preworkKernel();
    const seq_arr = mlx.mlx_array_new_int(in.seq);
    defer _ = mlx.mlx_array_free(seq_arr);
    const eps_arr = mlx.mlx_array_new_float(in.qk_eps);
    defer _ = mlx.mlx_array_free(eps_arr);
    // the packed kernel's sigmoid source at these widths: the table (its TAB arm from 17 rows)
    const inputs_arr = [_]mlx.mlx_array{ in.qkv, in.conv_state, in.conv_w, in.q_scale, in.k_scale, in.b, in.a, in.A_log, in.dt_bias, try sigmoidTableFor(s, dt), seq_arr, eps_arr };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, pw_cfg.?, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 6) return error.MetalKernelBadOutputCount;
    var out = GdnPrework{
        .q = mlx.mlx_array_new(),
        .k = mlx.mlx_array_new(),
        .v = mlx.mlx_array_new(),
        .conv_state = mlx.mlx_array_new(),
        .g = mlx.mlx_array_new(),
        .beta = mlx.mlx_array_new(),
    };
    errdefer inline for (.{ &out.q, &out.k, &out.v, &out.conv_state, &out.g, &out.beta }) |p| {
        if (p.*.ctx != null) _ = mlx.mlx_array_free(p.*);
    };
    try mlx.check(mlx.mlx_vector_array_get(&out.q, outputs_vec, 0));
    try mlx.check(mlx.mlx_vector_array_get(&out.k, outputs_vec, 1));
    try mlx.check(mlx.mlx_vector_array_get(&out.v, outputs_vec, 2));
    try mlx.check(mlx.mlx_vector_array_get(&out.conv_state, outputs_vec, 3));
    try mlx.check(mlx.mlx_vector_array_get(&out.g, outputs_vec, 4));
    try mlx.check(mlx.mlx_vector_array_get(&out.beta, outputs_vec, 5));
    return out;
}

/// Same dtype, shape and bit patterns, any strides (`array_equal` would take -0 for +0 and fail NaN).
pub fn sameBits(x: mlx.mlx_array, y: mlx.mlx_array, s: mlx.mlx_stream) !bool {
    if (mlx.mlx_array_dtype(x) != mlx.mlx_array_dtype(y) or !std.mem.eql(c_int, mlx.getShape(x), mlx.getShape(y))) return false;
    const u: mlx.mlx_dtype = switch (mlx.mlx_array_itemsize(x)) {
        2 => .uint16,
        4 => .uint32,
        else => .uint8,
    };
    var views: [4]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
    defer for (views) |v| {
        _ = mlx.mlx_array_free(v);
    };
    try mlx.check(mlx.mlx_contiguous(&views[0], x, false, s));
    try mlx.check(mlx.mlx_contiguous(&views[1], y, false, s));
    try mlx.check(mlx.mlx_view(&views[2], views[0], u, s));
    try mlx.check(mlx.mlx_view(&views[3], views[1], u, s));
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_array_equal(&eq, views[2], views[3], false, s));
    try mlx.check(mlx.mlx_array_eval(eq));
    var same = false;
    try mlx.check(mlx.mlx_array_item_bool(&same, eq));
    return same;
}

// ── tests ──

const testing = std.testing;
const kernels = @import("../../../sdk/blocks/gdn/kernels.zig");
const gdnParityRand = decode_fused.gdnParityRand;

/// One prework case at Flash Next's GDN geometry (Hk 16, Hv 48, 128 x 128 heads): the row stream against the packed
/// kernel on the same arguments. `joined`: qkv read from a wider row and a / b from one [a | 0 | b] row, as the joined
/// projections hand them over.
fn preworkCase(s: mlx.mlx_stream, seq: c_int, batch: c_int, cold: bool, joined: bool, gt: mlx.mlx_dtype) !void {
    var prng = std.Random.DefaultPrng.init(0xC0_2E + @as(u64, @intCast(seq * 7 + batch)));
    const rnd = prng.random();
    const hk: c_int = 16;
    const hv: c_int = 48;
    const c_dim: c_int = 2 * hk * 128 + hv * 128;
    const q_stride: c_int = if (joined) c_dim + 6144 + 128 else c_dim;
    const q_off: c_int = if (joined) 64 else 0;
    const ab_stride: c_int = if (joined) 112 else hv;
    const b_off: c_int = if (joined) 64 else 0;
    const dt: mlx.mlx_dtype = .bfloat16;
    var arrs: [9]mlx.mlx_array = undefined;
    const shapes = [_][]const c_int{ &.{ batch, seq, q_stride }, &.{ batch, seq, ab_stride }, &.{ batch, seq, ab_stride }, &.{hv}, &.{hv}, &.{ batch, 3, c_dim }, &.{ c_dim, 4, 1 }, &.{}, &.{} };
    const scales = [_]f32{ 1, 16, 16, 1, 1, 1, 1, 1, 1 };
    for (&arrs, shapes, scales, 0..) |*o, sh, sc, i| o.* = switch (i) {
        3 => try gdnParityRand(rnd, sh, sc, .float32, s),
        5 => if (cold) blk: {
            var z = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(&z, sh.ptr, sh.len, dt, s));
            break :blk z;
        } else try gdnParityRand(rnd, sh, sc, dt, s),
        7 => try @import("../../../mlx/array.zig").scalarOf(1.0 / 128.0, dt, s),
        8 => try @import("../../../mlx/array.zig").scalarOf(@sqrt(1.0 / 128.0), dt, s),
        else => try gdnParityRand(rnd, sh, sc, dt, s),
    };
    defer for (arrs) |x| {
        _ = mlx.mlx_array_free(x);
    };
    const args: GdnPreworkArgs = .{ .qkv = arrs[0], .qkv_off = q_off, .qkv_stride = q_stride, .b = arrs[1], .b_off = b_off, .b_stride = ab_stride, .a = arrs[2], .a_off = 0, .a_stride = ab_stride, .A_log = arrs[3], .dt_bias = arrs[4], .conv_state = arrs[5], .conv_w = arrs[6], .q_scale = arrs[7], .k_scale = arrs[8], .hk = hk, .hv = hv, .dk = 128, .dv = 128, .seq = seq, .batch = batch, .gate = true, .g_dtype = gt };
    const want = (try decode_fused.gdnPreworkFused(s, args)) orelse return error.PackedDeclined;
    defer inline for (.{ want.q, want.k, want.v, want.conv_state, want.g, want.beta }) |x| {
        _ = mlx.mlx_array_free(x);
    };
    const got = (try prework(s, args)) orelse return error.CoreDeclined;
    defer inline for (.{ got.q, got.k, got.v, got.conv_state, got.g, got.beta }) |x| {
        _ = mlx.mlx_array_free(x);
    };
    inline for (.{ "q", "k", "v", "conv_state", "g", "beta" }) |f| try testing.expect(try sameBits(@field(want, f), @field(got, f), s));
}

/// The recurrence's inputs as the packed prework leaves them, `t` rows of `batch` slots, and a state of `st`.
const RecurIn = struct {
    a: [6]mlx.mlx_array,

    fn init(s: mlx.mlx_stream, seed: u64, batch: c_int, t: c_int, in_dt: mlx.mlx_dtype, st: mlx.mlx_dtype) !RecurIn {
        var prng = std.Random.DefaultPrng.init(seed);
        const rnd = prng.random();
        var r: RecurIn = undefined;
        r.a[0] = try gdnParityRand(rnd, &.{ batch, t, 16, 128 }, 0.02, in_dt, s);
        r.a[1] = try gdnParityRand(rnd, &.{ batch, t, 16, 128 }, 0.15, in_dt, s);
        r.a[2] = try gdnParityRand(rnd, &.{ batch, t, 48, 128 }, 2, in_dt, s);
        // g in (0.8, 1): a decaying state
        const lo = mlx.mlx_array_new_float(0.8);
        defer _ = mlx.mlx_array_free(lo);
        const hi = mlx.mlx_array_new_float(1.0);
        defer _ = mlx.mlx_array_free(hi);
        var key = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(key);
        try mlx.check(mlx.mlx_random_key(&key, seed));
        r.a[3] = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_random_uniform(&r.a[3], lo, hi, &[_]c_int{ batch, t, 48 }, 3, .float32, key, s));
        r.a[4] = try gdnParityRand(rnd, &.{ batch, t, 48 }, 0.5, in_dt, s);
        r.a[5] = try gdnParityRand(rnd, &.{ batch, 48, 128, 128 }, 1, st, s);
        return r;
    }

    fn deinit(r: *RecurIn) void {
        for (r.a) |x| _ = mlx.mlx_array_free(x);
    }
};
