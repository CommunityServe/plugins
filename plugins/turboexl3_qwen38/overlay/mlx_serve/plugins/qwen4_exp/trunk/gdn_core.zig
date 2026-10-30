//! copied_from: src/plugins/qwen4_exp/trunk/gdn_core.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! The GDN core: a prefill chunk's packed prework and pipelined recurrence on kernels shaped for the GPU, each output
//! the replaced kernel's bytes. `prework` (kernels/gdn_core_prework.metal) walks a run of rows per simdgroup where the
//! packed prework runs a threadgroup per (row, head slot); `recur` (kernels/gdn_core_recur.metal) is the pipelined
//! kernel with its block and threadgroup sizes as template ints. Decode, verify and the stock route's short calls
//! (T < 64) never take them.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../sdk/util/log.zig");
const decode_fused = @import("../../../sdk/blocks/gdn/decode_fused.zig");
const sigmoidTableFor = @import("../../../sdk/blocks/mlp/fused_activations.zig").sigmoidTableFor;
const GdnPrework = decode_fused.GdnPrework;
const GdnPreworkArgs = decode_fused.GdnPreworkArgs;




pub const RecurOut = struct { y: mlx.mlx_array, state: mlx.mlx_array };

// ── tests ──

const testing = std.testing;
const kernels = @import("../../../sdk/blocks/gdn/kernels.zig");
