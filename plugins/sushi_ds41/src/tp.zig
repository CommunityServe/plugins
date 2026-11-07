//! Tensor parallel across Macs (`SUSHI_DS41_TP=jaccl|ring`, off by default). Each rank serves a pack repacked to its
//! half of every expert and attention head (`bench/studio-cluster/stq-tp-split.py` in local-llm), and the two partial
//! sums of a layer (attention out, MoE out) are summed over MLX's distributed group before they join the stream. The
//! peers come from MLX's own environment: `MLX_RANK` plus `MLX_IBV_DEVICES` and `MLX_JACCL_COORDINATOR` for jaccl, or
//! `MLX_HOSTFILE` for ring. Every rank must see the same requests in the same order.
const std = @import("std");
const mlx = @import("sushi/mlx.zig");
const log = @import("sushi/log.zig");
const sc_mod = @import("sushi/dsv41/dsv41_scope.zig");

const A = mlx.mlx_array;
const Group = extern struct { ctx: ?*anyopaque = null };

extern "c" fn mlx_distributed_group_new() Group;
extern "c" fn mlx_distributed_is_available(bk: ?[*:0]const u8) bool;
extern "c" fn mlx_distributed_init(res: *Group, strict: bool, bk: ?[*:0]const u8) c_int;
extern "c" fn mlx_distributed_group_rank(group: Group) c_int;
extern "c" fn mlx_distributed_group_size(group: Group) c_int;
extern "c" fn mlx_distributed_all_sum(res: *A, x: A, group: Group, s: mlx.mlx_stream) c_int;

var group: Group = .{};
var cpu: mlx.mlx_stream = .{};
pub var rank: u32 = 0;
pub var size: u32 = 1;
pub var sums: u64 = 0;

pub fn on() bool {
    return size > 1;
}

pub fn init() !void {
    const raw = std.c.getenv("SUSHI_DS41_TP") orelse return;
    const backend = std.mem.span(raw);
    if (!std.mem.eql(u8, backend, "jaccl") and !std.mem.eql(u8, backend, "ring")) {
        log.err("[sushi_ds41] SUSHI_DS41_TP=\"{s}\": use jaccl (RDMA) or ring (TCP)\n", .{backend});
        return error.SushiTpBackend;
    }
    if (!mlx_distributed_is_available(raw)) {
        log.err("[sushi_ds41] tensor parallel: MLX has no {s} backend here\n", .{backend});
        return error.SushiTpBackend;
    }
    var g = mlx_distributed_group_new();
    try mlx.check(mlx_distributed_init(&g, true, raw));
    group = g;
    rank = @intCast(mlx_distributed_group_rank(g));
    size = @intCast(mlx_distributed_group_size(g));
    cpu = mlx.mlx_default_cpu_stream_new();
    log.info("[sushi_ds41] tensor parallel: rank {d} of {d} over {s}\n", .{ rank, size, backend });
}

/// `x` summed over every rank: lazy, on the CPU stream, owned by `sc`. `x` itself when tensor parallel is off.
pub fn sum(sc: *sc_mod.Scope, x: A) !A {
    if (!on()) return x;
    sums += 1;
    var o = mlx.mlx_array_new();
    return sc.res(mlx_distributed_all_sum(&o, x, group, cpu), &o);
}
