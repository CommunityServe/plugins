//! GLM53F_DSA_PROMPT_INDEXER: a GLM-5.3-Flash DSA layer's prompt pass, each piece the stock bytes.
//! - The indexer (`glm5_next.selectRows`' prompt branch, 2 rows or more, on NAX): the stock path writes the head-summed
//!   scores f32 [S, P] (`dsa_nax.indexerScores`), negates them, argpartitions them (MLX's GPU merge sort and its
//!   temporaries), slices, casts and sorts the 512 best pools and expands them to tokens. Here one launch
//!   (kernels/dsa_prompt_index.metal) scores each query row with the NAX text's MMAs in its order, keeps them in a
//!   scratch row of its threadgroup's and takes the 512 best by radix select in MLX's sort order (ties to the lower
//!   pool): the same tokens. A threadgroup takes rows g, g + G, ..., so the scratch is G rows (`groupsFor`).
//! - The output (`dsaAttn`'s dense and sparse parts, [1, 64, rows, 256] each): the stock path concatenates them along
//!   the rows, then its transpose's reshape copies again. Here each part's transpose is concatenated along the rows:
//!   one copy into the output projection's layout, the same values.
//! Each piece declines (null or false: the stock path runs) outside GLM-5.3-Flash's shapes.
//! GLM53F_DSA_PROMPT_ONEP (default off, rounding-class): a prompt pass's sparse latent attention (the NAX text,
//! past 8 rows) meets the values with P as one fp16 piece (`dsa_nax.sparseLatentOneP`), not the stock hi and lo
//! pieces: half the PV MMAs, P rounded once, so not the stock bytes. Off, nothing changes.
//! copied_from: mlx-serve src/plugins/glm5_next/dsa_prompt.zig@4bab19a5 (#1196), its tests on the GPU in src/gpu_tests.zig.

const std = @import("std");
const kit = @import("mlx_serve");
const mlx = kit.mlx;
const kernelSource = @import("kernels.zig").source;
const options = @import("options.zig");
const log = kit.log;
const naxAvailable = kit.blocks.linear.verify_qmm_nax.verifyQmmNaxAvailable;
const Indexer = @import("glm5_next.zig").Indexer;
const dsa_nax = @import("dsa_nax.zig");

const Array = mlx.mlx_array;

/// The indexer: 32 heads of 128, a cache row its key and gate halves, pools of 4 tokens, 2,048 token slots (512 pools).
pub const ix_heads: c_int = 32;
pub const ix_dim: c_int = 128;
pub const ix_kpool: c_int = 4;
pub const ix_topk: c_int = 2048;
pub const sel_k: c_int = @divExact(ix_topk, ix_kpool);
/// The token slots a row: the selection's, then the row's unfinished pool's.
pub const out_w: c_int = ix_topk + ix_kpool - 1;
/// The attention's output: 64 heads, a value head of 256.
pub const attn_heads: c_int = 64;
pub const attn_v: c_int = 256;
/// The indexer's threadgroup: 8 simdgroups (the kernel's NSG).
pub const threads: c_int = 256;
/// The scratch rows' budget, and the fewest rows (groups) a launch keeps in flight.
pub const scratch_bytes: usize = 32 << 20;
pub const min_groups: c_int = 256;

pub const INDEX_HEADER = kernelSource("dsa_sparse_nax_header.metal") ++ "\n" ++
    kernelSource("dsa_prompt_index_header.metal");
pub const INDEX_SOURCE = kernelSource("dsa_prompt_index.metal");
var index_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// Indexer launches since the process started (tests read them).
pub var index_runs: u64 = 0;
var said = false;

pub fn on() bool {
    return options.on(.dsa_prompt_indexer);
}

fn engaged(what: []const u8) void {
    if (said) return;
    said = true;
    log.info("[glm5_next] DSA prompt indexer engaged: first piece {s}\n", .{what});
}

/// The groups a launch over `rows` query rows of `pools` pools takes: the scratch budget's rows (at least
/// `min_groups`), at most one a row.
pub fn groupsFor(rows: c_int, pools: c_int) c_int {
    const per: usize = @as(usize, @intCast(pools)) * @sizeOf(f32);
    const budget: c_int = @intCast(@min(@as(usize, std.math.maxInt(c_int)), scratch_bytes / per));
    return @max(1, @min(rows, @max(min_groups, budget)));
}

pub const Plan = struct { rows: c_int, pools: c_int, groups: c_int };

/// Whether the indexer launch serves these inputs: `q` [1, S, 32, 128] bf16 (S >= 2), `w` f32 [1, S, 32], the indexer
/// cache [1, T, 256] bf16 with more pools than the selection takes. MLX-free.
pub fn indexPlan(q: []const c_int, q_bf16: bool, w: []const c_int, w_f32: bool, cache: []const c_int, cache_bf16: bool) ?Plan {
    if (!q_bf16 or !w_f32 or !cache_bf16) return null;
    if (q.len != 4 or q[0] != 1 or q[1] < 2 or q[2] != ix_heads or q[3] != ix_dim) return null;
    if (w.len != 3 or w[0] != 1 or w[1] != q[1] or w[2] != ix_heads) return null;
    if (cache.len != 3 or cache[0] != 1 or cache[2] != 2 * ix_dim) return null;
    const pools = @divTrunc(cache[1], ix_kpool);
    if (pools <= sel_k) return null;
    return .{ .rows = q[1], .pools = pools, .groups = groupsFor(q[1], pools) };
}

/// Whether `ix` is GLM-5.3-Flash's indexer, the one the kernel's constants hold. MLX-free.
pub fn indexerServes(ix: Indexer) bool {
    return ix.heads == ix_heads and ix.head_dim == ix_dim and ix.kpool == ix_kpool and ix.topk == ix_topk;
}

/// The launch's outputs: the tokens int32 [1, S, 2051] and the scratch rows f32 [G, P] (with one row a group, G = S,
/// each row's scores of the pools it can see; the rest unwritten).
pub const Index = struct { tokens: Array, scores: Array };

/// The indexer over queries at positions `q_start ..`: `q_idx` [1, S, 32, 128], `ws` f32 [1, S, 32] (the head weights
/// with both scales, as `selectRows` makes them), `idx_cache` [1, T, 256]; `groups` the launch's (null: `groupsFor`).
/// Null outside `indexPlan`.
pub fn index(s: mlx.mlx_stream, q_idx: Array, ws: Array, idx_cache: Array, q_start: c_int, groups: ?c_int) !?Index {
    if (!mlx.streamIsGpu(s)) return null;
    const plan = indexPlan(mlx.getShape(q_idx), mlx.mlx_array_dtype(q_idx) == .bfloat16, mlx.getShape(ws), mlx.mlx_array_dtype(ws) == .float32, mlx.getShape(idx_cache), mlx.mlx_array_dtype(idx_cache) == .bfloat16) orelse return null;
    const g = groups orelse plan.groups;
    if (g < 1 or g > plan.rows) return null;
    const kern = index_kernel orelse blk: {
        const ins = [_][*:0]const u8{ "q", "w", "cache", "dims" };
        const outs = [_][*:0]const u8{ "out", "scr" };
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const k = mlx.mlx_fast_metal_kernel_new("glm5_dsa_prompt_index", iv, ov, INDEX_SOURCE, INDEX_HEADER, true, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        index_kernel = k;
        break :blk k;
    };
    const dims = mlx.mlx_array_new_data(&[_]i32{ plan.rows, q_start, plan.pools, g }, &[_]c_int{4}, 1, .int32);
    defer _ = mlx.mlx_array_free(dims);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, plan.rows, out_w }, 3, .int32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ g, plan.pools }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, threads * g, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, threads, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    const inputs = [_]Array{ q_idx, ws, idx_cache, dims };
    const iv = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, s));
    var tokens = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(tokens);
    try mlx.check(mlx.mlx_vector_array_get(&tokens, ov, 0));
    var scores = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(scores);
    try mlx.check(mlx.mlx_vector_array_get(&scores, ov, 1));
    return .{ .tokens = tokens, .scores = scores };
}

/// `glm5_next.selectRows`' prompt branch for queries at positions `q_start ..` (`w` [1, S, 32] f32, scaled by the
/// heads' count, as `dsaAttn` makes it) over `p` pools of which `select_k` are taken: the tokens int32 [1, S, 2051], a
/// new array; null with the switch off, off NAX, at one row, or outside GLM-5.3-Flash's indexer.
pub fn selectRows(s: mlx.mlx_stream, q_idx: Array, w: Array, idx_cache: Array, q_start: c_int, p: c_int, select_k: c_int, ix: Indexer) !?Array {
    if (!mlx.streamIsGpu(s) or !indexerServes(ix) or select_k != sel_k or p <= select_k) return null;
    const qs = mlx.getShape(q_idx);
    if (qs.len != 4 or qs[1] < 2 or !on() or !naxAvailable()) return null;
    // the head weights as `selectRows` scales them (by head_dim^-0.5, the same multiply)
    const scale = mlx.mlx_array_new_float(1.0 / @sqrt(@as(f32, @floatFromInt(ix.head_dim))));
    defer _ = mlx.mlx_array_free(scale);
    var ws = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ws);
    try mlx.check(mlx.mlx_multiply(&ws, w, scale, s));
    const cs = mlx.getShape(idx_cache);
    if (cs.len != 3 or @divTrunc(cs[1], ix_kpool) != p) return null;
    const r = (try index(s, q_idx, ws, idx_cache, q_start, null)) orelse return null;
    _ = mlx.mlx_array_free(r.scores);
    index_runs += 1;
    engaged("the indexer launch");
    return r.tokens;
}

/// One-P launches since the process started (tests read them).
pub var onep_runs: u64 = 0;
var onep_said = false;

pub fn onepOn() bool {
    return options.on(.dsa_prompt_onep);
}

/// `dsa_nax.sparseLatentOneP` for a prompt pass's sparse rows (the caller's gate: more than `HC_PRE_MAX_ROWS` rows in
/// the pass, more than 8 sparse rows) with the switch on and NAX on the device; null otherwise, the switch read first
/// (the caller runs the stock `sparseLatentNax`).
pub fn sparseOneP(s: mlx.mlx_stream, q_abs: Array, latent: Array, idx: Array, scale: f32) !?Array {
    if (!onepOn() or !naxAvailable()) return null;
    const out = (try dsa_nax.sparseLatentOneP(s, q_abs, latent, idx, scale)) orelse return null;
    onep_runs += 1;
    if (!onep_said) {
        onep_said = true;
        log.info("[glm5_next] DSA one-pass engaged\n", .{});
    }
    return out;
}

/// `dsaAttn`'s output from its dense rows' `dense` and sparse rows' `sparse` (each [1, 64, rows, 256], one dtype) in
/// the output projection's row order, [1, S, 64, 256] in one copy, into `o`; false (nothing set) with the switch off or
/// outside GLM-5.3-Flash's attention (the stock concatenate and transpose run).
pub fn assemble(o: *Array, s: mlx.mlx_stream, dense: Array, sparse: Array) !bool {
    const ds = mlx.getShape(dense);
    const ss = mlx.getShape(sparse);
    if (ds.len != 4 or ss.len != 4 or ds[0] != 1 or ss[0] != 1) return false;
    if (ds[1] != attn_heads or ss[1] != attn_heads or ds[3] != attn_v or ss[3] != attn_v) return false;
    if (mlx.mlx_array_dtype(dense) != mlx.mlx_array_dtype(sparse) or !on()) return false;
    var td = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(td);
    try mlx.check(mlx.mlx_transpose_axes(&td, dense, &[_]c_int{ 0, 2, 1, 3 }, 4, s));
    var ts = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ts);
    try mlx.check(mlx.mlx_transpose_axes(&ts, sparse, &[_]c_int{ 0, 2, 1, 3 }, 4, s));
    const two = [_]Array{ td, ts };
    const vec = mlx.mlx_vector_array_new_data(&two, two.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_concatenate_axis(o, vec, 1, s));
    engaged("the one-copy output");
    return true;
}
