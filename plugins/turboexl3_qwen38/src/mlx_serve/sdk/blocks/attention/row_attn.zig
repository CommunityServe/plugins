//! copied_from: src/sdk/blocks/attention/row_attn.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! Decode attention whose every row has the bits of the one-row step at its
//! position, ported from TensorFold's `row_attention.py` (MIT, see NOTICE).
//! One threadgroup per (kv head, window row, key chunk). Keys are cut into
//! CK-position chunks counted from 0 and a chunk's keys
//! interleave over SPLIT simdgroups; per simdgroup an online softmax in
//! ascending position, the simdgroups merged in order, then the chunks. What
//! rides in the window changes nothing a row computes, so serial decoding
//! through this kernel is the reference a verify window reproduces.
const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");

const CK = 128;
const SPLIT = 4;
const BLK = 4;
pub const MAX_ROWS = 16;

// K/V are the KV cache's views [1, HKV, L, D]: indexed through their strides,
// never copied. The window's rows sit at cache rows P .. P + W - 1 (P = L - W);
// row w is at position P + depth[w] and its keys past P are the window rows
// path[w][0 .. depth[w]] (a chain: depth w, path 0..w; a tree: its ancestors).
pub const PARTIAL = kernelSource("kernels/attention/row_attn_partial.metal");

pub const MERGE = kernelSource("kernels/attention/row_attn_merge.metal");

var partial_kernel: ?mlx.mlx_fast_metal_kernel = null;
var merge_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// A draft tree over a window's rows: `depth [W]` int32 and `path [W, MAXD]`
/// int32 (row w's window rows at depths 0 .. depth[w]), `max_depth` = MAXD - 1.
pub const Tree = struct { depth: mlx.mlx_array, path: mlx.mlx_array, max_depth: c_int };

// ── tests ──

const testing = std.testing;
