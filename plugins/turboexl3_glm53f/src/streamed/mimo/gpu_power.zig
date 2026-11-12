//! X12 measurement (mimo/gpu-warm; csrc/mimo_gpu_power.inc): the GPU's energy, its DVFS-state residency (IOReport's
//! "GPU Performance States" GPUPH: OFF, P1..P15, 24 MHz ticks) mapped to MHz through pmgr's "voltage-states9" table,
//! and its closed-loop thermal-limit residency ("CLTM-induced GPU Performance States": any state but NO_CLTM), between
//! two samples. System-wide counters, read without root and without any GPU work. Used by the `gpuwarm` bench (per
//! train) and the MTP profile (per request: `MIMO_GPU_POWER {...}`); no served path calls it.
const std = @import("std");
const builtin = @import("builtin");

pub const max_states = 32;

/// csrc/mimo_gpu_power.inc `MimoPowerDelta`.
pub const Delta = extern struct {
    gpu_energy_j: f64 = -1,
    seconds: f64 = 0,
    n_states: u32 = 0,
    has_cltm: u32 = 0,
    state_s: [max_states]f64 = @splat(0),
    cltm_s: f64 = 0,
    state_names: [max_states][8]u8 = @splat(@splat(0)),
};

comptime {
    if (@sizeOf(Delta) != 8 + 8 + 4 + 4 + 8 * max_states + 8 + 8 * max_states) @compileError("gpu_power: Delta is not the shim's layout");
}

const c = if (builtin.os.tag == .macos) struct {
    extern fn mimo_power_open() c_int;
    extern fn mimo_power_sample() ?*anyopaque;
    extern fn mimo_power_free(s: ?*anyopaque) void;
    extern fn mimo_power_delta(a: ?*anyopaque, b: ?*anyopaque, out: *Delta) c_int;
    extern fn mimo_power_freqs(mhz: [*]u32, cap: u32) c_int;
    extern fn mimo_power_last_error() [*:0]const u8;
} else struct {
    fn mimo_power_free(_: ?*anyopaque) void {}
};

/// The process's subscription (opened once) and the DVFS table (MHz per GPUPH state index; 0 = OFF or unknown).
pub const Sampler = struct {
    mhz: [max_states]u32 = @splat(0),
    n_mhz: u32 = 0,

    pub fn free(_: *const Sampler, s: ?*anyopaque) void {
        c.mimo_power_free(s);
    }

};

const testing = std.testing;

const table = [_]u32{ 0, 338, 486, 636, 796, 888, 988, 1084, 1182, 1278, 1374, 1470, 1578, 1620 };
