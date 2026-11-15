//! A KDA layer's prompt recurrence and its bounded gate as the chunkwise delta rule (GLM53F_KDA_PROMPT_CHUNK,
//! rounding-class): fla's chunk_kda form over 16-row chunks, each chunk's rows through one triangular solve in f32
//! simdgroup matrices and the state read and written once a chunk, not once a row; the gate built in the launch from
//! the gate input, dt_bias and A_log by kdaGateChain's statements. Its sums run in another order than the per-token
//! route's, so its outputs are that route's within rounding, not its bits. It takes GLM's prompt rows where the per-core
//! route declines, after the prologue, the packed prework or the chain.
//! copied_from: mlx-serve src/plugins/glm5_next/kda_chunk.zig@4bab19a5 (#1196), its tests on the GPU in src/gpu_tests.zig.

const std = @import("std");
const kit = @import("mlx_serve");
const mlx = kit.mlx;
const kernelSource = @import("kernels.zig").source;
const options = @import("options.zig");
const log = kit.log;
const config = kit.bases.kda_hybrid.config;

const Array = mlx.mlx_array;

const CHUNK_SOURCE = kernelSource("kda_chunk.metal");
/// Metal's simdgroup matrices and the prologue's `kda_sigmoid` (unary_ops.h `Sigmoid`, the chain's).
const HEADER = "#include <metal_simdgroup_matrix>\n" ++ kernelSource("kda_prologue_header.metal");

/// Rows a chunk, value rows a threadgroup (8 simdgroups of 8) and threads a threadgroup.
pub const chunk_rows: c_int = 16;
pub const value_rows: c_int = 64;
pub const threads: c_int = 256;
/// The head width the text is written for (16 key-channel blocks of 8).
const head_dim: c_int = 128;
/// The most negative gate bound the text takes: a chunk's factors, taken against its middle row, stay within e^(8 x 5)
/// and e^-(8 x 5), where f32 neither overflows nor flushes a term that counts.
pub const min_bound: f32 = -5.0;

/// The prompt switches as set.
pub const Switches = struct { chunk: bool = false };

/// The switches as set.
pub fn switches() !Switches {
    return .{ .chunk = options.on(.kda_prompt_chunk) };
}

/// Whether the text takes a layer of `hk` x `dk` key and `hv` x `dv` value heads: heads alike and 128 wide (two
/// threadgroups of 64 value rows a head).
pub fn layoutServes(hk: c_int, hv: c_int, dk: c_int, dv: c_int) bool {
    return hk == hv and hk >= 1 and dk == head_dim and dv == head_dim;
}

/// Whether the text takes the gate's bound: in [min_bound, 0) (a bounded gate; GLM-5.3-Flash's is -5).
pub fn boundServes(lower_bound: f32) bool {
    return lower_bound < 0 and lower_bound >= min_bound;
}

/// The launch's threadgroups for `h` value heads of 128.
pub fn groupsFor(h: c_int) c_int {
    return h * @divExact(head_dim, value_rows);
}

/// Whether the launch takes a GLM layer's prompt rows: a GPU stream, the layout and bound, 16-bit activations, an f32
/// state, and the per-head A_log and f32 dt_bias trunk.zig loads for GLM (`a_log_h`, `dt_f32`).
pub fn serves(s: mlx.mlx_stream, hk: c_int, hv: c_int, dk: c_int, dv: c_int, act: mlx.mlx_dtype, state: mlx.mlx_dtype, a_log_h: Array, dt_f32: Array, lower_bound: f32) bool {
    if (!mlx.streamIsGpu(s) or !layoutServes(hk, hv, dk, dv) or !boundServes(lower_bound)) return false;
    if ((act != .bfloat16 and act != .float16) or state != .float32) return false;
    if (a_log_h.ctx == null or dt_f32.ctx == null) return false;
    return mlx.mlx_array_size(a_log_h) == @as(usize, @intCast(hv)) and mlx.mlx_array_size(dt_f32) == @as(usize, @intCast(hv * dk));
}

/// The load's checks for a GLM pack: the switches' refusals, and with the chunkwise route on, the layer's heads and gate
/// bound and one launch over zeros (Metal caps a threadgroup by its register use, and MLX reports that only at
/// evaluation). Refuses by name.
pub fn loadCheck(s: mlx.mlx_stream, c: *const config.Config) !void {
    const sw = try switches();
    if (!sw.chunk) return;
    if (!mlx.streamIsGpu(s)) return error.KdaChunkStream;
    const hk: c_int = @intCast(c.linear_num_key_heads);
    const hv: c_int = @intCast(c.linear_num_value_heads);
    if (!layoutServes(hk, hv, @intCast(c.linear_key_head_dim), @intCast(c.linear_value_head_dim))) return error.KdaChunkLayout;
    if (!boundServes(c.kda_gate_lower_bound)) return error.KdaChunkBound;
    if (!try probe(s, hv, c.kda_gate_lower_bound)) return error.KdaChunkThreads;
}

/// One launch over zeros for `h` heads (bf16 activations, as a prompt's), evaluated: false where this GPU refuses the
/// threadgroup (the load's check; the GPU tests skip there).
pub fn probe(s: mlx.mlx_stream, h: c_int, lower_bound: f32) !bool {
    var arrs: [6]Array = @splat(.{ .ctx = null });
    defer for (arrs) |x| free(x);
    const shapes = [_][]const c_int{ &.{ 1, 1, h, head_dim }, &.{ 1, 1, h * head_dim }, &.{ 1, 1, h }, &.{h}, &.{h * head_dim}, &.{ 1, h, head_dim, head_dim } };
    const dts = [_]mlx.mlx_dtype{ .bfloat16, .bfloat16, .bfloat16, .float32, .float32, .float32 };
    for (&arrs, shapes, dts) |*x, sh, d| {
        x.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(x, sh.ptr, sh.len, d, s));
    }
    const r = try launch(s, arrs[0], arrs[0], arrs[0], arrs[1], arrs[2], arrs[3], arrs[4], lower_bound, arrs[5]);
    defer {
        free(r.y);
        free(r.state);
    }
    mlx.check(mlx.mlx_array_eval(r.state)) catch |e| {
        if (e == error.MlxError and mlx.takeErrorIf("Thread group size")) return false;
        return e;
    };
    return true;
}

/// y [B, T, H, 128] in q's dtype and the new f32 state [B, H, 128, 128], each a new array.
pub const Recur = struct { y: Array, state: Array };

/// Launches since the load (the tests read that the route ran).
pub var runs: u64 = 0;
var engaged_said = false;
var chunk_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// The chunkwise recurrence over a prompt span (where `serves`): q, k [B, T, H, 128] normed and scaled, v [B, T, H,
/// 128], the gate input `a` [B, T, H x 128], beta [B, T, H], the per-head A_log, the f32 dt_bias, the bound and the f32
/// state [B, H, 128, 128].
pub fn recur(s: mlx.mlx_stream, q: Array, k: Array, v: Array, a: Array, beta: Array, a_log_h: Array, dt_f32: Array, lower_bound: f32, state: Array) !Recur {
    const r = try launch(s, q, k, v, a, beta, a_log_h, dt_f32, lower_bound, state);
    runs += 1;
    if (!engaged_said) {
        engaged_said = true;
        const qs = mlx.getShape(q);
        log.info("[kda] prompt chunk engaged: S={d} Hv={d} chunk={d} threadgroups={d}\n", .{ qs[1], qs[2], chunk_rows, groupsFor(qs[2]) });
    }
    return r;
}

/// One launch: `a` and beta in q's dtype, A_log and dt_bias f32, the bound an f32 [1]; grid (256 x the threadgroups,
/// 1, B).
fn launch(s: mlx.mlx_stream, q: Array, k: Array, v: Array, a: Array, beta: Array, a_log: Array, dt_bias: Array, lower_bound: f32, state: Array) !Recur {
    const qs = mlx.getShape(q);
    const b = qs[0];
    const t = qs[1];
    const h = qs[2];
    const dt = mlx.mlx_array_dtype(q);
    var cast: [4]Array = @splat(.{ .ctx = null });
    defer for (cast) |x| free(x);
    for (&cast, [_]Array{ a, beta, a_log, dt_bias }, [_]mlx.mlx_dtype{ dt, dt, .float32, .float32 }) |*o, x, d| {
        o.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(o, x, d, s));
    }
    const lbv = [1]f32{lower_bound};
    const lb = mlx.mlx_array_new_data(&lbv, &[_]c_int{1}, 1, .float32);
    defer _ = mlx.mlx_array_free(lb);
    const t_arr = mlx.mlx_array_new_int(t);
    defer _ = mlx.mlx_array_free(t_arr);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ b, t, h, head_dim }, 4, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ b, h, head_dim, head_dim }, 4, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, threads * groupsFor(h), 1, b));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, threads, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "InT", dt));
    inline for (.{ .{ "Dk", head_dim }, .{ "Dv", head_dim }, .{ "H", h } }) |ta| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, ta[0], ta[1]));
    }
    const ins = [_]Array{ q, k, v, cast[0], cast[1], cast[2], cast[3], lb, state, t_arr };
    const names = [_][*:0]const u8{ "q", "k", "v", "a", "beta", "A_log", "dt_bias", "lower_bound", "state_in", "T" };
    const kern = try kernelOf(&chunk_kernel, "msv_glm_kda_chunk", &names, CHUNK_SOURCE);
    const outs = try apply(s, kern, cfg, &ins);
    return .{ .y = outs[0], .state = outs[1] };
}

fn free(a: Array) void {
    if (a.ctx != null) _ = mlx.mlx_array_free(a);
}

/// The kernel in `slot`, built once (its library compiles at the first launch); outputs y and state_out.
fn kernelOf(slot: *?mlx.mlx_fast_metal_kernel, name: [*:0]const u8, ins: []const [*:0]const u8, source: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    if (slot.*) |k| return k;
    const outs = [_][*:0]const u8{ "y", "state_out" };
    const iv = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new(name, iv, ov, source, HEADER, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    slot.* = k;
    return k;
}

/// One launch of `kern`: its two outputs, new arrays.
fn apply(s: mlx.mlx_stream, kern: mlx.mlx_fast_metal_kernel, cfg: mlx.mlx_fast_metal_kernel_config, ins: []const Array) ![2]Array {
    const iv = mlx.mlx_vector_array_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, s));
    if (mlx.mlx_vector_array_size(ov) != 2) return error.MetalKernelBadOutputCount;
    var outs: [2]Array = @splat(.{ .ctx = null });
    errdefer for (outs) |x| free(x);
    for (&outs, 0..) |*o, i| {
        o.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(o, ov, i));
    }
    return outs;
}
