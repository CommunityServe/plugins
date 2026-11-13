//! The kernel set over a pinned registry `R` (G5): one manifest, its texts, its pin, built and bound once per
//! model load. The host's load context owns it. Its consumers take it at their accept (mlx-stream's EXL3 registry:
//! the routed-expert quant, `exl3_quant`, and the V4.1 trunk, `dsv41_kernel_routes`). Each consumer self-checks only
//! its own subset, and the set is freed after the last consumer's `deinit`.
//!
//! The consumers' subsets partition the registry (`checkPartition`, at comptime; `partitionError`
//! names a violator at run time). So the manifest sha, the self-check pins and every device receipt
//! stay the set's, whichever consumers load. With one consumer loaded, the whole set is still pinned
//! (every text verified) and only that consumer's subset is self-checked.

const std = @import("std");
const mlx = @import("sdk").mlx;
/// MLX custom kernels through the plugin's launch cache (sdk_ext/metal_launch.zig).
const mtl = @import("metal_launch.zig");
const profile = @import("profile.zig");

/// A load's kernel set as a quant's accept receives it (`sdk.quant.Context.kernels`): the set, erased, and the pin
/// of the registry it is built over. A quant takes it back as its own registry's set (`KernelSet(R).Set.of`); a set
/// over another registry is refused by name there, never cast.
pub const SetRef = struct {
    set: *const anyopaque,
    manifest_sha256: []const u8,
};

// ── What every pinned registry shares (`exl3_kernels`, `glm53_exl3_kernels`) ──
// The texts' sha256 pins against a manifest, the manifest's headers and kernels by name, the mlx fast-kernel objects,
// launch configs and launches over them with the decode-timers build's launch probe, and the self-check's results,
// report, depth, verdict and array scope. A registry keeps its own manifest schema, checks and refusal set; these take
// its message prefix.

const bo = @import("../build_flags.zig");
const Sha256 = std.crypto.hash.sha2.Sha256;

/// A template argument of a launch: an int or a dtype.
pub const TemplateValue = union(enum) { int: i32, dtype: mlx.mlx_dtype };
pub const TemplateArg = struct { name: [:0]const u8, value: TemplateValue };

/// One launch's outputs (shapes and dtypes), grid (threads), threadgroup and template arguments.
pub const Launch = struct {
    grid: [3]u32,
    threadgroup: [3]u32,
    out_shapes: []const []const c_int,
    out_dtypes: []const mlx.mlx_dtype,
    template: []const TemplateArg,
};

/// G7: the package's decode-timers build observes each launch of a bound set (its first dispatches per phase, the
/// observer `Bound.observe` installs); every other build has no observer field, launch key or call.
pub const launch_observed: bool = if (@hasDecl(bo, "dsv41_decode_timers")) bo.dsv41_decode_timers else false;
/// A launch config's template key (`launchKey`), kept with its prepared config.
pub const LaunchKey = if (launch_observed) u64 else void;
/// The observer a profile hook installs: the kernel's name, the launch key and the inputs.
pub const Observer = if (launch_observed) ?*const fn ([]const u8, u64, []const mlx.mlx_array) void else void;

/// The kernels of registry `R` built on one GPU stream (`R.Registry.bind`); freed after that stream has drained. `R`
/// supplies `Registry`, `Kernel`, `n_kernels`, `max_outputs`, `LaunchConfig` and
///     launchOf(reg, k, cfg, shapes: *[max_outputs][]const c_int, template: *[16]TemplateArg) Launch
/// (the registry's config as the mlx launch; the slices may point into `shapes` and `template`).
pub fn BoundOf(comptime R: type) type {
    return struct {
        const Self = @This();
        reg: *const R.Registry,
        stream: mlx.mlx_stream,
        kernels: [R.n_kernels]mlx.mlx_fast_metal_kernel,
        /// the profile hook's launch probe (`observe`); void outside the decode-timers build
        observer: Observer = if (launch_observed) null else {},

        pub fn deinit(self: *Self) void {
            for (self.kernels) |k| {
                if (k.ctx != null) _ = mtl.mlx_fast_metal_kernel_free(k);
            }
            self.* = undefined;
        }

    };
}

/// The self-check's depth:
/// - `compile` (the construction's acceptance): every kernel compiles, binds and completes one launch; no numeric
///   comparison (the server's startup runs no numeric probes). A kernel that does not refuses the load by name.
/// - `full` (the device test and the window's smoke tests): every check the manifest lists.
pub const CheckDepth = enum { compile, full };

/// MLX arrays one check creates, freed together.
pub const Scope = struct {
    a: std.mem.Allocator,
    arrays: std.ArrayList(mlx.mlx_array) = .empty,

    /// `arr` kept until `deinit` (freed now when it cannot be kept).
    pub fn keep(sc: *Scope, arr: mlx.mlx_array) !mlx.mlx_array {
        sc.arrays.append(sc.a, arr) catch |e| {
            _ = mlx.mlx_array_free(arr);
            return e;
        };
        return arr;
    }

    /// A new array of `bytes` (copied), kept.
    pub fn host(sc: *Scope, bytes: []const u8, shape: []const c_int, dt: mlx.mlx_dtype) !mlx.mlx_array {
        const x = mlx.mlx_array_new_data(bytes.ptr, shape.ptr, @intCast(shape.len), dt);
        if (x.ctx == null) return error.MlxError;
        return sc.keep(x);
    }

    pub fn deinit(sc: *Scope) void {
        for (sc.arrays.items) |x| _ = mlx.mlx_array_free(x);
        sc.arrays.deinit(sc.a);
    }
};

const testing = std.testing;

/// A three-kernel registry without Metal: texts are names, the pin is their join, binding counts, the self-check plan
/// is the stub device's. Enough of `R` for the set's own machinery.
const FakeReg = struct {
    pub const Kernel = enum { k0, k1, k2 };
    pub const Header = enum { h0, h1 };
    pub const Check = enum { parity, bounds };
    pub const n_kernels = 3;
    pub const n_headers = 2;
    pub const Texts = [n_kernels][]const u8;
    pub const embedded: Texts = .{ "k0", "k1", "k2" };
    pub const manifest_sha256 = "fake-manifest";
    pub const Diag = struct { msg: []const u8 = "" };
    pub const Entry = struct { kernel: Kernel, checks: std.EnumSet(Check), header: ?Header };

    pub const Registry = struct {
        entries: [n_kernels]Entry,
        texts: *const Texts,
        live: *u32,

        var live_count: u32 = 0;
        var refuse_bind = false;

        pub fn init(a: std.mem.Allocator, texts: *const Texts, pin: []const u8, diag: *Diag) !Registry {
            _ = a;
            if (!std.mem.eql(u8, pin, manifest_sha256)) {
                diag.msg = "manifest not pinned";
                return error.ManifestNotPinned;
            }
            live_count += 1;
            return .{
                .entries = .{
                    .{ .kernel = .k0, .checks = .initMany(&.{ .parity, .bounds }), .header = .h0 },
                    .{ .kernel = .k1, .checks = .initOne(.parity), .header = .h0 },
                    .{ .kernel = .k2, .checks = .initOne(.bounds), .header = null },
                },
                .texts = texts,
                .live = &live_count,
            };
        }
        pub fn deinit(r: *Registry) void {
            r.live.* -= 1;
        }
    };

    pub const Bound = struct {
        reg: *const Registry,
        stream: mlx.mlx_stream,
        kernels: [n_kernels]struct { built: bool = false },
        observed: bool = false,
        pub fn deinit(_: *Bound) void {}
    };

    pub const selfcheck = struct {
        pub const Result = struct { kernel: Kernel, check: Check, words: u32, ok: bool, err: []const u8 };
        pub const Report = struct {
            results: std.ArrayList(Result) = .empty,
            fn deinit(r: *Report, a: std.mem.Allocator) void {
                r.results.deinit(a);
            }
        };
        pub const Depth = enum { compile, full };
        /// The fake's plan at either depth: every check of every wanted kernel.
        pub fn plan(reg: *const Registry, want: std.EnumSet(Kernel), _: Depth) std.EnumArray(Kernel, std.EnumSet(Check)) {
            var out: std.EnumArray(Kernel, std.EnumSet(Check)) = .initFill(.empty);
            for (&reg.entries) |*e| if (want.contains(e.kernel)) out.set(e.kernel, e.checks);
            return out;
        }
    };
};
