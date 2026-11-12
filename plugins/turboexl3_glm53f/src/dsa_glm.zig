//! `TURBOEXL3_GLM53F_DSA_GLM=1` (opt-in, exact): a DSA layer's prompt pass on GLM-5.3-Flash kernels, the model's dims
//! fixed at compile time, through patches/turboexl3-model.patch. The generic path stays the
//! fallback: each part declines any other shape, and the stock path runs.
//!
//! - **The indexer** (glm5_next `selectRows`' prompt branch, 2 rows or more, on NAX): the stock path writes the
//!   indexer's head-summed scores f32 [S, P] (dsa_nax `indexerScores`), negates them, argpartitions them (MLX's GPU
//!   merge sort and its temporaries), slices, casts and sorts the 512 best pools and expands them to tokens: about
//!   1.7 GB of fresh buffers a DSA layer at 16K rows, 0.94 GB of them the merge sort's own. Here one launch
//!   (src/kernels/dsa_glm_index.metal: 32 heads of 128, pools of 4, 512 pools, 2,051 slots) scores each query row with
//!   the NAX text's MMAs and statement in its order (the row's heads its MMA rows), keeps the row's scores in a scratch
//!   row of its threadgroup's, takes the 512 best with a radix select and ordered compaction (MLX's sort
//!   order: ties to the lower pool) and writes the tokens: the same tokens, from the same scores. A threadgroup takes
//!   rows g, g + G, ... so the scratch is G rows of the pools (`groupsFor`: the rows 32 MiB holds, at least 256), not
//!   one a query row.
//! - **The output** (`dsaAttn`'s dense and sparse rows' outputs, both [1, 64, rows, 256]): the stock path concatenates
//!   them along the rows ([1, 64, S, 256], a copy), then its transpose's reshape copies again into [1, S, 64 * 256].
//!   Here each part's transpose is concatenated along the rows, so one copy writes the output projection's layout: the
//!   same values (536 MB a DSA layer less to write at a 16K prompt's main call).
//!
//! Each part counts its runs and prints an engaged line at its first; src/dsa_glm_gpu_test.zig holds both to the stock
//! path bit for bit.
const std = @import("std");
const kit = @import("mlx_serve");
const mlx = kit.mlx;
const log = @import("mlx_serve/sdk/util/log.zig");
const kernelSource = @import("mlx_serve/mlx/kernels.zig").source;
const naxCheck = @import("mlx_serve/sdk/blocks/linear/verify_qmm_nax.zig").verifyQmmNaxAvailable;
const Array = mlx.mlx_array;

/// `TURBOEXL3_GLM53F_DSA_GLM=1`, set by the plugin at load, before any forward.
pub var on: bool = false;
/// The indexer's launches since the load.
pub var index_runs: u64 = 0;
/// The one-copy outputs since the load.
pub var assemble_runs: u64 = 0;
/// The switch's load line (the A/B step reads "<switch>=1: ").
pub const line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_DSA_GLM=1: a DSA layer's prompt indexer (scores, top-k and expansion) as one GLM-5.3-Flash launch and its output in one copy (the same tokens and values)\n";

// ── GLM-5.3-Flash's DSA dims (the kernels' constants) ──

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

const index_header = kernelSource("plugins/glm5_next/kernels/dsa_sparse_nax_header.metal") ++ "\n" ++ @embedFile("kernels/dsa_topk_header.metal");
const index_source = @embedFile("kernels/dsa_glm_index.metal");
var index_kernel: ?mlx.mlx_fast_metal_kernel = null;
var index_engaged = false;
var assemble_engaged = false;

/// The groups a launch over `rows` query rows of `pools` pools takes: the scratch budget's rows (at least
/// `min_groups`), at most one a row.
pub fn groupsFor(rows: c_int, pools: c_int) c_int {
    const per: usize = @as(usize, @intCast(pools)) * @sizeOf(f32);
    const budget: c_int = @intCast(@min(@as(usize, std.math.maxInt(c_int)), scratch_bytes / per));
    return @max(1, @min(rows, @max(min_groups, budget)));
}

pub const Plan = struct { rows: c_int, pools: c_int, groups: c_int };

/// Whether the indexer launch serves these inputs: `q` [1, S, 32, 128] bf16 (S >= 2), `w` f32 [1, S, 32], the indexer
/// cache [1, T, 256] bf16 with more pools than the selection takes. GLM-5.3-Flash's shapes only.
pub fn indexPlan(q: []const c_int, q_bf16: bool, w: []const c_int, w_f32: bool, cache: []const c_int, cache_bf16: bool) ?Plan {
    if (!q_bf16 or !w_f32 or !cache_bf16) return null;
    if (q.len != 4 or q[0] != 1 or q[1] < 2 or q[2] != ix_heads or q[3] != ix_dim) return null;
    if (w.len != 3 or w[0] != 1 or w[1] != q[1] or w[2] != ix_heads) return null;
    if (cache.len != 3 or cache[0] != 1 or cache[2] != 2 * ix_dim) return null;
    const pools = @divTrunc(cache[1], ix_kpool);
    if (pools <= sel_k) return null;
    return .{ .rows = q[1], .pools = pools, .groups = groupsFor(q[1], pools) };
}

/// The launch's two outputs: the tokens int32 [1, S, 2051] and the scratch rows f32 [G, P] (with one row a group, G =
/// S, each row's scores of the pools it can see; the rest unwritten).
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
        const k = mlx.mlx_fast_metal_kernel_new("tx3g_glm_dsa_index", iv, ov, index_source, index_header, true, false);
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
    if (!index_engaged) {
        index_engaged = true;
        log.info("[turboexl3_glm53f] DSA GLM indexer engaged ({d} rows of {d} pools, {d} groups)\n", .{ plan.rows, plan.pools, g });
    }
    return .{ .tokens = tokens, .scores = scores };
}

/// glm5_next `selectRows`' prompt branch for queries at positions `q_start ..` (`w` [1, S, 32] f32, scaled by the
/// heads' count, as `dsaAttn` makes it), the indexer `kpool`, `topk`, `heads` and `head_dim` the model's: the tokens
/// int32 [1, S, 2051], a new array; null with the switch off, off NAX, at one row (the stock path scores it off NAX) or
/// outside GLM-5.3-Flash's indexer (the stock path runs).
pub fn selectRows(s: mlx.mlx_stream, q_idx: Array, w: Array, idx_cache: Array, q_start: c_int, p: c_int, select_k: c_int, kpool: c_int, topk: c_int, heads: c_int, head_dim: c_int) !?Array {
    if (!on or !mlx.streamIsGpu(s)) return null;
    if (heads != ix_heads or head_dim != ix_dim or kpool != ix_kpool or topk != ix_topk or select_k != sel_k or p <= select_k) return null;
    const qs = mlx.getShape(q_idx);
    if (qs.len != 4 or qs[1] < 2 or !naxCheck()) return null;
    // the head weights as `selectRows` scales them (by head_dim^-0.5, the same multiply)
    const scale = mlx.mlx_array_new_float(1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))));
    defer _ = mlx.mlx_array_free(scale);
    var ws = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ws);
    try mlx.check(mlx.mlx_multiply(&ws, w, scale, s));
    if (mlx.getShape(idx_cache).len != 3 or @divTrunc(mlx.getShape(idx_cache)[1], ix_kpool) != p) return null;
    const r = (try index(s, q_idx, ws, idx_cache, q_start, null)) orelse return null;
    _ = mlx.mlx_array_free(r.scores);
    index_runs += 1;
    return r.tokens;
}

/// `dsaAttn`'s output from its dense rows' `dense` and sparse rows' `sparse` (each [1, 64, rows, 256], one dtype) in
/// the output projection's row order, [1, S, 64, 256] in one copy, into `o`; false (nothing set) with the switch off or
/// outside GLM-5.3-Flash's attention (the stock concatenate and transpose run).
pub fn assemble(o: *Array, s: mlx.mlx_stream, dense: Array, sparse: Array) !bool {
    if (!on) return false;
    const ds = mlx.getShape(dense);
    const ss = mlx.getShape(sparse);
    if (ds.len != 4 or ss.len != 4 or ds[0] != 1 or ss[0] != 1) return false;
    if (ds[1] != attn_heads or ss[1] != attn_heads or ds[3] != attn_v or ss[3] != attn_v) return false;
    if (mlx.mlx_array_dtype(dense) != mlx.mlx_array_dtype(sparse)) return false;
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
    assemble_runs += 1;
    if (!assemble_engaged) {
        assemble_engaged = true;
        log.info("[turboexl3_glm53f] DSA GLM one-copy output engaged ({d} dense and {d} sparse rows)\n", .{ ds[2], ss[2] });
    }
    return true;
}

test "turboexl3_glm53f stream: the DSA GLM indexer serves GLM-5.3-Flash's indexer shapes from 2 rows and more pools than it selects, its scratch 32 MiB of rows at most and at least 256 rows" {
    const t = std.testing;
    // a 16K prompt's main call and tail, a 64K and a 128K context
    try t.expectEqual(Plan{ .rows = 14305, .pools = 4089, .groups = 2051 }, indexPlan(&.{ 1, 14305, 32, 128 }, true, &.{ 1, 14305, 32 }, true, &.{ 1, 16356, 256 }, true).?);
    try t.expectEqual(Plan{ .rows = 31, .pools = 4096, .groups = 31 }, indexPlan(&.{ 1, 31, 32, 128 }, true, &.{ 1, 31, 32 }, true, &.{ 1, 16387, 256 }, true).?);
    try t.expectEqual(Plan{ .rows = 4096, .pools = 16384, .groups = 512 }, indexPlan(&.{ 1, 4096, 32, 128 }, true, &.{ 1, 4096, 32 }, true, &.{ 1, 65536, 256 }, true).?);
    try t.expectEqual(Plan{ .rows = 4096, .pools = 65536, .groups = 256 }, indexPlan(&.{ 1, 4096, 32, 128 }, true, &.{ 1, 4096, 32 }, true, &.{ 1, 262144, 256 }, true).?);
    // one row, another indexer, other dtypes, a batch, a cache within the selection
    try t.expect(indexPlan(&.{ 1, 1, 32, 128 }, true, &.{ 1, 1, 32 }, true, &.{ 1, 16356, 256 }, true) == null);
    try t.expect(indexPlan(&.{ 1, 64, 64, 128 }, true, &.{ 1, 64, 64 }, true, &.{ 1, 16356, 256 }, true) == null);
    try t.expect(indexPlan(&.{ 1, 64, 32, 64 }, true, &.{ 1, 64, 32 }, true, &.{ 1, 16356, 128 }, true) == null);
    try t.expect(indexPlan(&.{ 1, 64, 32, 128 }, false, &.{ 1, 64, 32 }, true, &.{ 1, 16356, 256 }, true) == null);
    try t.expect(indexPlan(&.{ 1, 64, 32, 128 }, true, &.{ 1, 64, 32 }, false, &.{ 1, 16356, 256 }, true) == null);
    try t.expect(indexPlan(&.{ 2, 64, 32, 128 }, true, &.{ 2, 64, 32 }, true, &.{ 2, 16356, 256 }, true) == null);
    try t.expect(indexPlan(&.{ 1, 64, 32, 128 }, true, &.{ 1, 64, 32 }, true, &.{ 1, 2051, 256 }, true) == null);
    try t.expect(indexPlan(&.{ 1, 64, 32, 128 }, true, &.{ 1, 63, 32 }, true, &.{ 1, 16356, 256 }, true) == null);
    // the scratch: budget rows, at least min_groups, at most one a query row
    try t.expectEqual(@as(c_int, 2051), groupsFor(100000, 4089));
    try t.expectEqual(@as(c_int, 17), groupsFor(17, 4089));
    try t.expectEqual(@as(c_int, 256), groupsFor(100000, 1 << 20));
    try t.expect(@as(usize, @intCast(groupsFor(100000, 4089))) * 4089 * 4 <= scratch_bytes);
}
