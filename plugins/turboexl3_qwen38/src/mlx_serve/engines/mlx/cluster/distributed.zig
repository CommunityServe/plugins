//! This plugin's own, in place of mlx-serve's `src/engines/mlx/cluster/distributed.zig` (a cut edge of
//! scripts/sync-mlx-serve.sh): the engine's Thunderbolt cluster. This plugin serves one Mac, so nothing is planned and
//! no peer exists: the copies' cluster branches never run.

const ModelConfig = @import("../../../sdk/config/model.zig").ModelConfig;
const mlx = @import("../../../mlx/ffi.zig");

pub const LayerSpan = @import("../../../sdk/weights/shard.zig").LayerSpan;
pub const Shard = @import("../../../sdk/weights/shard.zig").Shard;
pub const headOnly = @import("../../../sdk/weights/shard.zig").headOnly;

pub const Part = union(enum) { span: LayerSpan, shard: Shard, remote: LayerSpan };
/// A cluster peer, which never exists here (`peer` is null, so is every trunk's `dist_peer` and `dist_tp`): its calls
/// refuse by name.
pub const Peer = struct {
    pub fn hop(self: *Peer, h: mlx.mlx_array, offset: usize) !mlx.mlx_array {
        _ = self;
        _ = h;
        _ = offset;
        return error.NoClusterPeer;
    }

    pub fn sendMoe(self: *Peer, x: mlx.mlx_array, logits: mlx.mlx_array, layer: u32) !void {
        _ = self;
        _ = x;
        _ = logits;
        _ = layer;
        return error.NoClusterPeer;
    }

    pub fn recvLike(self: *Peer, x: mlx.mlx_array) !mlx.mlx_array {
        _ = self;
        _ = x;
        return error.NoClusterPeer;
    }
};

pub fn planFor(model_dir: []const u8, config: *const ModelConfig) !?Part {
    _ = model_dir;
    _ = config;
    return null;
}

pub fn peer() ?*Peer {
    return null;
}
