//! This plugin's own, in place of mlx-serve's `src/engines/mlx/forward_ctx.zig` (a cut edge of
//! scripts/sync-mlx-serve.sh): the forward context, with the fields the K packs' forward reads.

const mlx = @import("../../mlx/ffi.zig");
const row_attn = @import("../../sdk/blocks/attention/row_attn.zig");
const PlePending = @import("../../plugins/qwen4_exp/ple.zig").PlePending;
const KVCache = @import("kv/cache.zig").KVCache;
const SSMCacheEntry = @import("ssm/entry.zig").SSMCacheEntry;

/// A draft tree's verify rows.
pub const SpecTree = struct {
    attn: row_attn.Tree,
};

/// The layer outputs a forward captures: `out[k]` is h after layer `ids[k]` (ascending), at every position.
pub const CaptureLayers = struct {
    ids: []const u32,
    out: []mlx.mlx_array,
};

pub const ForwardCtx = struct {
    cache: *KVCache,
    stream_in: ?mlx.mlx_array = null,
    moe_seq_offset: *usize,
    ssm_entries: ?[]SSMCacheEntry,
    ssm_member_gen: u64 = 0,
    capture_hidden: ?*mlx.mlx_array,
    capture_hidden_all: ?*mlx.mlx_array = null,
    capture_stream_all: ?*mlx.mlx_array = null,
    capture_layers: ?*CaptureLayers = null,
    prefill_mask_add: ?mlx.mlx_array = null,
    capture_ssm_seq: bool = false,
    tree: ?*const SpecTree = null,
    spec_verify: bool = false,
    pipeline_build: u32 = 0,
    vision_embeddings: ?mlx.mlx_array,
    mrope_pos: ?[]const i32 = null,
    mrope_cos_cur: ?mlx.mlx_array = null,
    qsa_mask: mlx.mlx_array = .{ .ctx = null },
    qsa_blocks: mlx.mlx_array = .{ .ctx = null },
    key_pad_mask: ?mlx.mlx_array = null,
    skip_lm_head: bool = false,
    batch_slots: ?[]const *ForwardCtx = null,
    batch_rope_offsets: ?mlx.mlx_array = null,
    qsa_entry: ?*SSMCacheEntry = null,
    qsa_pos_base: c_int = 0,
    ple_defer: bool = false,
    ple_pending: ?PlePending = null,
    ple_ids_src: ?[]const mlx.mlx_array = null,
};
