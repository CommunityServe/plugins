//! `TURBOEXL3_GLM53F_DSA_VSEL=1` (opt-in, exact): a DSA layer's verify-window selection (glm5_next `selectRows`, 2 to
//! 16 rows) in two launches, through patches/turboexl3-model.patch. The stock path runs the
//! one-query texts once a row, the indexer scores (kernels/dec_score.metal) and the top-k with its expansion
//! (kernels/topk_expand.metal), then joins the rows: 8 launches at 3 rows. Here each text runs as it is with its row from
//! the grid (the scores' threadgroup z, the top-k's y), every row's pointers and position offset by it, so a row computes
//! what it computes alone: the same tokens. src/dsa_gpu_test.zig checks them bit for bit at 1 to 4 rows.
const std = @import("std");
const kit = @import("mlx_serve");
const mlx = kit.mlx;
const log = @import("mlx_serve/sdk/util/log.zig");
const kernelSource = @import("mlx_serve/mlx/kernels.zig").source;
const Array = mlx.mlx_array;

/// `TURBOEXL3_GLM53F_DSA_VSEL=1`, set by the plugin at load, before any forward.
pub var vsel_on: bool = false;
/// Launch pairs since the load.
pub var vsel_runs: u64 = 0;
/// The switch's load line (the A/B step reads "<switch>=1: ").
pub const vsel_line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_DSA_VSEL=1: a DSA layer's verify window selects in two launches, every row's indexer scores in one and their top-k and expansion in one (the one-query texts with the row in the grid: the same tokens)\n";

/// `text` with each edit `.{ from, to, n }` applied, `from` found exactly `n` times: a stock text that changes fails the
/// build here (also src/dsa_wfold.zig's).
pub fn edited(comptime text: []const u8, comptime edits: anytype) [:0]const u8 {
    comptime {
        @setEvalBranchQuota(400_000);
        var cur: []const u8 = text;
        for (edits) |e| {
            const from: []const u8 = e[0];
            var res: []const u8 = "";
            var rest: []const u8 = cur;
            var count: usize = 0;
            while (std.mem.indexOf(u8, rest, from)) |at| {
                res = res ++ rest[0..at] ++ e[1];
                rest = rest[at + from.len ..];
                count += 1;
            }
            if (count != e[2]) @compileError(std.fmt.comptimePrint("dsa_verify: the stock text holds `{s}` {d} times, not {d}", .{ from, count, e[2] }));
            cur = res ++ rest;
        }
        var buf: [cur.len:0]u8 = undefined;
        @memcpy(buf[0..cur.len], cur);
        buf[buf.len] = 0;
        const final = buf;
        return &final;
    }
}

/// kernels/dec_score.metal with its row in threadgroup z: row r reads q and the head weights at r and writes its scores
/// at r, as one query at position q_start + r (src/dsa_wfold.zig edits it further).
pub const scores_source = edited(kernelSource("plugins/glm5_next/kernels/dec_score.metal"), .{
    .{ "const int q_pos = dims[0];", "const uint vrow = threadgroup_position_in_grid.z;\nconst int q_pos = dims[0] + int(vrow);", 1 },
    .{ "const int P = dims[1];", "const int P = dims[1];\nconst device T* q_r = q + size_t(vrow) * (32 * DH);\nconst device float* w_r = w + size_t(vrow) * 32;\ndevice float* out_r = out + size_t(vrow) * size_t(P);", 1 },
    .{ "float(q[i])", "float(q_r[i])", 1 },
    .{ "w[lane]", "w_r[lane]", 1 },
    .{ "out[p] =", "out_r[p] =", 1 },
});
/// kernels/topk_expand.metal with its row in threadgroup y: row r reads its scores at r and writes its tokens at r, as
/// one query at position q_start + r (src/dsa_wfold.zig runs it too).
pub const topk_source = edited(kernelSource("plugins/glm5_next/kernels/topk_expand.metal"), .{
    .{ "const int q_pos = dims[0];", "const uint vrow = threadgroup_position_in_grid.y;\nconst int q_pos = dims[0] + int(vrow);", 1 },
    .{ "const int P = dims[1];", "const int P = dims[1];\nconst device float* scores_r = scores + size_t(vrow) * size_t(P);\ndevice int* out_r = out + size_t(vrow) * size_t(OUT_W);", 1 },
    .{ "scores[i]", "scores_r[i]", 3 },
    .{ "out[", "out_r[", 2 },
});
pub const topk_header = kernelSource("plugins/glm5_next/kernels/topk_expand_header.metal");
/// The top-k text's threadgroup (its bin search and compaction are written for 1024 threads).
pub const topk_tg: c_int = 1024;

var scores_kernel: ?mlx.mlx_fast_metal_kernel = null;
var topk_kernel: ?mlx.mlx_fast_metal_kernel = null;
var engaged = false;

fn kernel(slot: *?mlx.mlx_fast_metal_kernel, name: [*:0]const u8, ins: []const [*:0]const u8, src: [:0]const u8, header: [:0]const u8) !mlx.mlx_fast_metal_kernel {
    if (slot.*) |k| return k;
    const outs = [_][*:0]const u8{"out"};
    const iv = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new(name, iv, ov, src, header, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    slot.* = k;
    return k;
}

fn apply(s: mlx.mlx_stream, kern: mlx.mlx_fast_metal_kernel, cfg: mlx.mlx_fast_metal_kernel_config, inputs: []const Array) !Array {
    const iv = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
    return out;
}

/// The verify window's selection, `selectRows`' per-row launches and join in two: `q_idx` [1, S, 32, `head_dim`] in the
/// cache's dtype, `ws` f32 [1, S, 32] (the head weights with the Dh^-0.5), `idx_cache` [1, T, 2 * `head_dim`], the rows
/// at positions `q_start..`, `p` pools of `kpool` tokens with `select_k` of them selected, `sgs` and `pps` the stock
/// scores launch's simdgroups and pools a simdgroup. int32 [1, S, `topk` + `kpool` - 1], a new array; null with the
/// switch off, off the GPU, or for one row (the decode's own launches).
pub fn selectRows(s: mlx.mlx_stream, q_idx: Array, ws: Array, idx_cache: Array, q_start: c_int, p: c_int, select_k: c_int, kpool: c_int, topk: c_int, head_dim: c_int, sgs: c_int, pps: c_int) !?Array {
    if (!vsel_on or !mlx.streamIsGpu(s)) return null;
    const qs = mlx.getShape(q_idx);
    if (qs.len != 4 or qs[0] != 1 or qs[1] < 2 or qs[1] > 16 or qs[2] != 32 or qs[3] != head_dim) return null;
    const ss = mlx.getShape(ws);
    if (ss.len != 3 or ss[0] != 1 or ss[1] != qs[1] or ss[2] != 32 or mlx.mlx_array_dtype(ws) != .float32) return null;
    if (select_k < 1 or select_k >= p) return null;
    const rows = qs[1];
    const dims = mlx.mlx_array_new_data(&[_]i32{ q_start, p }, &[_]c_int{2}, 1, .int32);
    defer _ = mlx.mlx_array_free(dims);
    const sk = try kernel(&scores_kernel, "tx3g_dsa_vsel_scores", &.{ "cache", "q", "w", "dims" }, scores_source, "");
    const scfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(scfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(scfg, &[_]c_int{ rows, p }, 2, .float32));
    const per_tg = sgs * pps;
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(scfg, sgs * 32, @divTrunc(p + per_tg - 1, per_tg), rows));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(scfg, sgs * 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(scfg, "T", mlx.mlx_array_dtype(idx_cache)));
    inline for (.{ .{ "DH", head_dim }, .{ "KPOOL", kpool }, .{ "SGS", sgs }, .{ "PPS", pps } }) |kv| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(scfg, kv[0], kv[1]));
    }
    const scores = try apply(s, sk, scfg, &.{ idx_cache, q_idx, ws, dims });
    defer _ = mlx.mlx_array_free(scores);
    const tk = try kernel(&topk_kernel, "tx3g_dsa_vsel_topk", &.{ "scores", "dims" }, topk_source, topk_header);
    const out_w = topk + kpool - 1;
    const tcfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(tcfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(tcfg, &[_]c_int{ 1, rows, out_w }, 3, .int32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(tcfg, topk_tg, rows, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(tcfg, topk_tg, 1, 1));
    inline for (.{ .{ "SEL_K", select_k }, .{ "KPOOL", kpool }, .{ "TOPK_W", topk }, .{ "TAIL_W", kpool - 1 }, .{ "OUT_W", out_w } }) |kv| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(tcfg, kv[0], kv[1]));
    }
    const out = try apply(s, tk, tcfg, &.{ scores, dims });
    vsel_runs += 1;
    if (!engaged) {
        engaged = true;
        log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_DSA_VSEL engaged: a {d}-row verify window over {d} pools, {d} selected, in two launches\n", .{ rows, p, select_k });
    }
    return out;
}
