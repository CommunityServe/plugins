//! A0: the decode's first DSpark cycle against its warm cycles, in profile builds only (`dsv41_decode_timers.enabled`;
//! in every other build each entry below compiles to nothing and the launch paths carry no call).
//!
//! What it counts:
//!   - per routed layer call: the routing barrier's wait, the route, the MoE's build and commit (hit, hoist and miss
//!     waves), and the distinct experts the call missed; the first cycle apart, the warm cycles summed;
//!   - per layer, the demand reads' drive time: the stream's read gauge from one barrier's end to the next layer's
//!     (the reads a route issues land before the next barrier returns), against that barrier;
//!   - the prompt's last `tail_rows` rows' routed experts per layer, and in the first cycle how many of each layer's
//!     misses they hold and how many reads they would cost (A0's first-verify reads at the grow);
//!   - the decode timers' buckets at the first cycle's end;
//!   - A0 (c)'s D1: the first cycle's draft block run twice (`Loop.droppedDraft`), the first dropped, each block's build,
//!     wait and the bytes MLX took for it, after the state the draft's eval would realise (evaluated apart);
//!   - the first dispatches of each phase (construction, the prompt pass, the first cycle, the warm cycles): a
//!     registry kernel's variant (MLX builds one library per kernel, template and input binding: scalar, `constant`
//!     under 8 elements, else `device`) and a compiled region's input signature (MLX traces one per shapes and dtypes).
//! Printed once after the decode (`DECODE_FIRST`).
const std = @import("std");
const mlx = @import("sdk").mlx;
const dt = @import("decode_timers.zig");

pub const enabled = dt.enabled;

/// The process's phase: construction (its self-checks and warm-up), the prompt pass (through the phase change), the
/// decode's first cycle, its later cycles.
pub const Phase = enum(u8) { build, prompt, cycle1, warm };
pub var phase: Phase = .build;

pub const max_layers = @import("decode_timers.zig").max_layers;
pub const max_experts = 512;
pub const Layer = struct { calls: u64 = 0, barrier_ns: u64 = 0, route_ns: u64 = 0, moe_ns: u64 = 0, misses: u64 = 0, read_ns: u64 = 0 };
/// Per routed layer: [0] the first cycle, [1] the warm cycles summed.
pub var layers: [2][max_layers]Layer = @splat(@splat(.{}));

var tail: [max_layers]std.StaticBitSet(max_experts) = @splat(.empty);

/// First dispatches counted per phase; the first cycle's named.
pub var new_per_phase: [4]u32 = @splat(0);
pub const Named = struct { what: []const u8, key: u64 };
pub var cycle1_new: [96]Named = undefined;
pub var n_cycle1_new: usize = 0;
var seen: [16384]u64 = @splat(0);

/// A launch config's template key (`exl3_kernels.launchKey`, kept by `Bound.prepare`).
pub const TKey = if (enabled) u64 else void;

/// A registry kernel dispatched (`Bound.applyPrepared`, through the observer the kernel set installs from the
/// backend's profile hook): its variant counted in the phase that first dispatches it.
pub fn kernel(name: []const u8, tkey: TKey, inputs: []const mlx.mlx_array) void {
    if (comptime !enabled) return;
    var h = std.hash.Wyhash.init(tkey);
    h.update(name);
    for (inputs) |x| {
        const d: c_int = @backingInt(mlx.mlx_array_dtype(x));
        const bind: u8 = if (mlx.mlx_array_ndim(x) == 0) 's' else if (mlx.mlx_array_size(x) < 8) 'c' else 'd';
        h.update(std.mem.asBytes(&d));
        h.update(&.{bind});
    }
    note(name, h.final());
}

fn note(name: []const u8, hash: u64) void {
    const key = hash | 1;
    var i: usize = @intCast(key % seen.len);
    for (0..seen.len) |_| {
        if (seen[i] == key) return;
        if (seen[i] == 0) break;
        i = (i + 1) % seen.len;
    } else return;
    seen[i] = key;
    new_per_phase[@backingInt(phase)] += 1;
    if (phase == .cycle1 and n_cycle1_new < cycle1_new.len) {
        cycle1_new[n_cycle1_new] = .{ .what = name, .key = key };
        n_cycle1_new += 1;
    }
}
