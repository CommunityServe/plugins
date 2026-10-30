//! copied_from: src/sdk/blocks/gdn/decode.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! Fused single-token GatedDeltaNet step for Hadamard packs (B=1, S=1, swish
//! output gate; activations T, recurrent state StT): two dispatches instead of
//! prework + recurrence + norm-gate + rotation. K1 runs one head over SPLIT threadgroups, each
//! recomputing the conv/silu/q-k norm prework for its head (cheaper than a
//! barrier between kernels) before its slice of the recurrence rows. K2 does
//! the per-head gated RMS norm over one Hadamard block of heads and rotates it
//! for out_proj. Bit-identical to the composed chain.
const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
pub const HEADER = kernelSource("kernels/gdn/log1p.metal");
/// The dtype every kernel here (GT) rounds the decay gate a + dt_bias -> g to. Under bf16
/// activations it is the state's: a bf16 gate near one is exactly 1.0 and that head never
/// forgets. f16-activation (Hadamard) packs keep the f16 gate their fidelity guard measured.
pub fn gateDtype(act: mlx.mlx_dtype, state: mlx.mlx_dtype) mlx.mlx_dtype {
    return if (act == .bfloat16) state else act;
}
// K1 over TL tokens (verify widths): the same per-token prework and recurrence,
// token after token, plus the per-step state capture MTP rollback reads
// (state_seq[TL-1] is never written, as in the stock capture kernel).
// K1S at one threadgroup per head (NT=1024, SPLIT=1: the same 4 rows per
// simdgroup, so each row's recurrence is unchanged) with the verify epilogues
// folded in. A head's 128 y values stay in threadgroup
// memory, rounded to T as the stored y was, and simdgroup t runs the norm-gate
// kernel's exact reduction for token t. The conv-input rows rollback slices are copied out as well.
// A draft tree runs in two launches, one simdgroup per job: K1P does each
// row's conv/silu/q-k norm prework once per head (K1S's arithmetic), reading
// qkv (row stride QS) and a, b (row stride ABS, at AOFF/BOFF) in place, so
// the columns of the joined in-projection need no copies. Then K1TR
// walks the rows parents first, one dv row per simdgroup in 128-thread
// threadgroups, a restart row's state kept in registers.
// Row t restarts from its parent's state unless the parent is row t - 1.
// The recurrence along `path` (window rows, oldest first) from the round's
// input state, the tree kernel's ops and per-token rounding: the state serial
// decoding holds after the path's last row. K1TR's grid. The same launch
// copies the kept conv window: rows `conv_rows` of the conv input.
pub const K1R_SOURCE = kernelSource("kernels/gdn/decode_k1r.metal");
fn makeKernel(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [*:0]const u8, header: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, header, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    return k;
}
pub const Geometry = struct { hk: c_int, hv: c_int, dk: c_int, dv: c_int };
pub const Inputs = struct {
    qkv: mlx.mlx_array, // [1,1,C]
    z: mlx.mlx_array, // [1,1,Hv*Dv]
    a: mlx.mlx_array, // [1,1,Hv]
    b: mlx.mlx_array, // [1,1,Hv]
    conv_state: mlx.mlx_array, // [1,3,C]
    ssm_state: mlx.mlx_array, // [1,Hv,Dv,Dk]
    conv_w: mlx.mlx_array,
    A_log: mlx.mlx_array,
    dt_bias: mlx.mlx_array,
    q_scale: mlx.mlx_array, // 0-dim bf16
    k_scale: mlx.mlx_array, // 0-dim bf16
    norm_w: mlx.mlx_array,
    eps: mlx.mlx_array, // 0-dim f32
    signs: mlx.mlx_array, // [Hv*Dv] f32, out_proj's
};
const CfgKey = struct { g: Geometry, dt: mlx.mlx_dtype, st: mlx.mlx_dtype };
pub const Recur = struct { y: mlx.mlx_array, conv_state: mlx.mlx_array, ssm_state: mlx.mlx_array };
/// A tree round's per-row recurrence inputs (f32): k [T,Hk,Dk], v [T,Hv,Dv],
/// g|beta [T,Hv,2].
pub const Prework = struct { k: mlx.mlx_array = .{ .ctx = null }, v: mlx.mlx_array = .{ .ctx = null }, gb: mlx.mlx_array = .{ .ctx = null } };
pub const RecurSeq = struct { y: mlx.mlx_array, conv_state: mlx.mlx_array, ssm_state: mlx.mlx_array, state_seq: mlx.mlx_array };
/// The fold keeps a head's y rows in threadgroup memory beside K1S's q/k/v rows; past 8 they pass 32 KB.
pub const FOLD_MAX_SEQ: c_int = 8;
var k1r_cache: ?mlx.mlx_fast_metal_kernel = null;
var replay_cfg: mlx.mlx_fast_metal_kernel_config = .{ .ctx = null };
var replay_key: ?CfgKey = null;
pub const Commit = struct { ssm_state: mlx.mlx_array, conv_state: mlx.mlx_array };
/// The state after the window rows `path` (oldest first), recurred from
/// `state_in` with a tree round's `prework` (the state serial decoding holds
/// there), and the conv state: rows `conv_rows` of `conv_input` [1,3+T,C].
/// Null outside the kernel's geometry.
pub fn replay(g: Geometry, state_in: mlx.mlx_array, pw: Prework, path: []const i32, conv_input: mlx.mlx_array, conv_rows: [3]i32, s: mlx.mlx_stream) !?Commit {
    if (!mlx.streamIsGpu(s) or pw.k.ctx == null or path.len == 0) return null;
    if (g.dk != 128 or g.dv != 128 or @rem(g.hv, g.hk) != 0) return null;
    const c = 2 * g.hk * g.dk + g.hv * g.dv;
    const csh = mlx.getShape(conv_input);
    if (csh.len != 3 or csh[0] != 1 or csh[2] != c or 3 * c > 32 * g.dv * g.hv) return null;
    for (conv_rows) |r| if (r < 0 or r >= csh[1]) return null;
    const st = mlx.mlx_array_dtype(state_in);
    const dt = mlx.mlx_array_dtype(conv_input);
    if (k1r_cache == null) k1r_cache = try makeKernel("msv_gdn_decode_replay", &.{ "state_in", "pk", "pv", "pg", "path", "n_path", "conv_in", "conv_rows" }, &.{ "state_out", "conv_out" }, K1R_SOURCE, HEADER);
    const key = CfgKey{ .g = g, .dt = dt, .st = st };
    if (replay_key == null or !std.meta.eql(replay_key.?, key)) {
        if (replay_cfg.ctx != null) _ = mlx.mlx_fast_metal_kernel_config_free(replay_cfg);
        replay_cfg = mlx.mlx_fast_metal_kernel_config_new();
        replay_key = null;
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(replay_cfg, &[_]c_int{ 1, g.hv, g.dv, g.dk }, 4, st));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(replay_cfg, &[_]c_int{ 1, 3, c }, 3, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(replay_cfg, 32, g.dv, g.hv));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(replay_cfg, 32, 4, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(replay_cfg, "StT", st));
        inline for (.{ .{ "HK", g.hk }, .{ "HV", g.hv }, .{ "DK", g.dk }, .{ "DV", g.dv }, .{ "C", c } }) |kv|
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(replay_cfg, kv[0], kv[1]));
        replay_key = key;
    }
    const path_arr = mlx.mlx_array_new_data(path.ptr, &[_]c_int{@intCast(path.len)}, 1, .int32);
    defer _ = mlx.mlx_array_free(path_arr);
    const n: i32 = @intCast(path.len);
    const n_arr = mlx.mlx_array_new_data(&n, &[_]c_int{1}, 1, .int32);
    defer _ = mlx.mlx_array_free(n_arr);
    const rows_arr = mlx.mlx_array_new_data(&conv_rows, &[_]c_int{3}, 1, .int32);
    defer _ = mlx.mlx_array_free(rows_arr);
    const ins = [_]mlx.mlx_array{ state_in, pw.k, pw.v, pw.gb, path_arr, n_arr, conv_input, rows_arr };
    const v = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&o, k1r_cache.?, v, replay_cfg, s));
    var st_out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(st_out);
    try mlx.check(mlx.mlx_vector_array_get(&st_out, o, 0));
    var conv_out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&conv_out, o, 1));
    return .{ .ssm_state = st_out, .conv_state = conv_out };
}
// Whether this GPU's pipeline runs each width's 1024-thread fold; null = not dispatched yet.
// The fold without the per-step state capture. In its place each threadgroup writes the rows' recurrence inputs K1S holds
// in threadgroup memory (`Prework`'s layout, what K1R reads), so a partial accept replays the kept rows from the round's
// input state (`replay`) instead of slicing a capture. Every other statement is the fold's.
/// `recurSeqFold`'s outputs but the per-step states, plus the rows' recurrence inputs `replay` reads.
pub const RecurSeqFoldReplay = struct { gated: mlx.mlx_array, conv_state: mlx.mlx_array, ssm_state: mlx.mlx_array, conv_input: mlx.mlx_array, prework: Prework };
// As `fold_ok`, for this pipeline (its thread limit is its own).
