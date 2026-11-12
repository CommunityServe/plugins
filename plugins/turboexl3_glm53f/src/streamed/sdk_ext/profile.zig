//! G7, profile builds: a plugin's profile instruments reach the code of its other kinds as a hook its arch injects at
//! comptime, never as an import, so no kind imports another's profiling. The SDK declares the probes' shapes, the
//! hooks that compile to nothing and the probes' one clock (`Clock`); it carries no timer or counter of its own.
//!
//! The backend type an arch binds its quant and its source with declares the hook (`pub const profile_hook: Hook`);
//! the generic code they share reads it with `of(Backend)`, once, at comptime. A backend that declares none (a host
//! test backend, another arch's) gets `off`, and so does every probe of a build without the plugin's profile options:
//! a served build carries no clock read, counter, field or branch for them.

const std = @import("std");
const mlx = @import("sdk").mlx;

/// Where the prompt pass's routed call spends its host time: the routing barrier (the ids to the host), the source's
/// route (reads issued), the read waits, the waves' encode (graphs, host tables, submission), the drains (the host
/// blocked on GPU compute), the join (the outputs' concatenate / take).
pub const PrefillBucket = enum { barrier, route, read_wait, encode, drain, join };

/// One plugin's probes, each a namespace.
pub const Hook = struct {
    /// The prompt pass's routed-call split: `enabled: bool`, `Stamp: type`, `now() Stamp`,
    /// `charge(PrefillBucket, Stamp) void`, `count(waves: u64, launches: u64) void`, `countCall() void`.
    prefill: type = prefill_off,
    /// A pinned kernel's dispatch, seen by the kernel registry's launcher: `enabled: bool`,
    /// `kernel(name: []const u8, key: u64, inputs: []const mlx.mlx_array) void` (`key`: the launch config's template).
    launch: type = launch_off,
};

/// The hook of a backend that declares none.
pub const off: Hook = .{};

/// A probe file's clock: `Stamp` is the monotonic clock in ns where its probes are compiled in (`enabled`), else
/// `void`, and `now` reads nothing.
pub fn Clock(comptime enabled: bool) type {
    return struct {
        pub const Stamp = if (enabled) u64 else void;

        pub inline fn now() Stamp {
            if (comptime !enabled) return {};
            var ts: std.c.timespec = undefined;
            _ = std.c.clock_gettime(.MONOTONIC, &ts);
            return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
        }
    };
}

pub const prefill_off = struct {
    pub const enabled = false;
    pub const Stamp = void;
    pub inline fn now() Stamp {}
    pub inline fn charge(_: PrefillBucket, _: Stamp) void {}
    pub inline fn count(_: u64, _: u64) void {}
    pub inline fn countCall() void {}
};

pub const launch_off = struct {
    pub const enabled = false;
    pub fn kernel(_: []const u8, _: u64, _: []const mlx.mlx_array) void {}
};

const testing = std.testing;
