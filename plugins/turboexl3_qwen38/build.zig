//! Builds this plugin with the mlx-serve plugin kit and adds turboexl3 (build.zig.zon, pinned by URL and hash): its
//! library (`turboexl3`), its plugin-kit launcher (`turboexl3_kit`) and its quant table (`turboexl3_table`), over this
//! plugin's `sdk` for them (src/turboexl3_sdk.zig), which reaches the plugin's mlx-c layer through `mlx_host`, the
//! plugin's root, so no file is in two modules. `zig build test` runs config.zig's MLX-free tests, and `zig build
//! mlx-free-test` runs those and the ones rooted at the plugin (plugin.json's `test_step`: scripts/build.sh runs it).
//! `zig build gpu-test -Dmlx-lib=<dir>` runs #1005's EXL3 tests (src/gpu_tests.zig) on the Metal device, against the
//! server's own MLX (the directory holding its libmlx.dylib and libmlxc.dylib); `gpu-test-build` only compiles them.

const std = @import("std");

pub fn build(b: *std.Build) void {
    // src/mlx_serve is generated (scripts/prepare.sh): no build goes on with a tree other than the reviewed one.
    _ = b.run(&.{ "bash", b.root.joinString(b.allocator, "scripts/check-mlx-serve.sh") catch @panic("OOM") });
    const lib = @import("mlx_serve_kit").plugin(b, .{ .name = "turboexl3_qwen38" });
    const root = lib.root_module.import_table.get("plugin").?;
    const dep = b.dependency("turboexl3", .{ .target = root.resolved_target.?, .optimize = root.optimize.? });
    wire(b, root, dep);
    // The kit's test step roots config.zig, which has none of wire's modules: a test that reaches the plugin's own code
    // runs rooted at the plugin, by name, with MLX left undefined (it never calls it). mlx-free-test runs both.
    const tests = b.addTest(.{ .name = "mlx-free-test", .root_module = root, .filters = &mlx_free_filters });
    tests.linker_allow_shlib_undefined = true;
    const mlx_free = b.step("mlx-free-test", "Run the MLX-free tests: the kit's and those rooted at the plugin");
    mlx_free.dependOn(&b.top_level_steps.get("test").?.step);
    mlx_free.dependOn(&b.addRunArtifact(tests).step);
    gpuTests(b, root, dep);
}

/// turboexl3's modules and MLX's steel headers over `root` (a module rooted at plugin.zig).
fn wire(b: *std.Build, root: *std.Build.Module, dep: *std.Build.Dependency) void {
    const target = root.resolved_target.?;
    const optimize = root.optimize.?;
    const kit = root.import_table.get("mlx_serve").?;
    // MLX's steel headers at the server's MLX (src/mlx_steel/): lane_qmm's tile matmul and turboexl3's prefill GEMMs.
    const steel = b.createModule(.{ .root_source_file = b.path("src/mlx_steel/root.zig"), .target = target, .optimize = optimize });
    root.addImport("mlx_steel_sources", steel);
    const sdk = b.createModule(.{
        .root_source_file = b.path("src/turboexl3_sdk.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{ .{ .name = "mlx_host", .module = root }, .{ .name = "mlx_serve", .module = kit }, .{ .name = "mlx_steel_sources", .module = steel } },
    });
    const tkit = b.createModule(.{
        .root_source_file = dep.path("kit/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "sdk", .module = sdk }, .{ .name = "mlx_serve", .module = kit } },
    });
    sdk.addImport("turboexl3_kit", tkit);
    const t3 = b.createModule(.{
        .root_source_file = dep.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "sdk", .module = sdk }},
    });
    const table = b.createModule(.{
        .root_source_file = dep.path("table/table.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "sdk", .module = sdk },
            .{ .name = "mlx_serve", .module = kit },
            .{ .name = "turboexl3", .module = t3 },
            .{ .name = "turboexl3_kit", .module = tkit },
        },
    });
    // The copies import the EXL3 format's parse and bounds as `sushi_exl3`: turboexl3 serves those names.
    root.addImport("sushi_exl3", t3);
    root.addImport("turboexl3", t3);
    root.addImport("turboexl3_kit", tkit);
    root.addImport("turboexl3_table", table);
}

/// The tests `mlx-free-test` runs rooted at the plugin, by name: the MTP round's pure parts (src/spec/drafter.zig),
/// the server's flags as the plugin reads them (src/host_flags.zig), the disk tier's snapshot header
/// (src/cache/codec.zig), #1005's trunk_quant block lookup (block_test.zig), and the direct model's trellis n-gram
/// table, its warm and the resident bill (src/direct). The kit's test step runs config.zig's and the EXL3 tests of the
/// copies that build from there.
const mlx_free_filters = [_][]const u8{
    "drafter:",
    "host flags:",
    "aux codec:",
    "quant block: blocks are found under their keys",
    "direct ple:",
    "direct load: an EXL3 trunk pack warms",
    "direct serve:",
};

/// The tests `zig build gpu-test` runs, by name: #1005's tests of the EXL3 path the direct model still calls in the
/// copies, in the files src/gpu_tests.zig imports, and the direct load's trellis head bias (an MLX array). The copies
/// keep no other named test (scripts/sync-mlx-serve.sh takes them out).
const gpu_filters = [_][]const u8{
    "direct load: a trellis n-gram table takes its per-head bias",
    "pack keys: trunk_quant is a format's block",
    "qwen4_exp formats: the numerics are reordered on the EXL3 trunk profile",
    "qwen4_exp formats: an unclaimed trunk block",
    "side stream: a part takes a call only on an EXL3 trunk",
};

/// The GPU tests' build: a module graph of their own over plugin.zig (the dylib never links MLX, the tests link the
/// server's), whose test block imports src/gpu_tests.zig, filtered to the tests it names.
fn gpuTests(b: *std.Build, root: *std.Build.Module, dep: *std.Build.Dependency) void {
    const run_step = b.step("gpu-test", "Run #1005's EXL3 tests (src/gpu_tests.zig) against the server's MLX: -Dmlx-lib=<dir>");
    const build_step = b.step("gpu-test-build", "Compile #1005's EXL3 tests without running them: -Dmlx-lib=<dir>");
    const mlx_lib = b.option([]const u8, "mlx-lib", "The directory holding the server's libmlx.dylib and libmlxc.dylib (gpu-test)") orelse {
        const fail = b.addFail("gpu-test needs -Dmlx-lib=<the directory holding the server's libmlx.dylib and libmlxc.dylib>");
        run_step.dependOn(&fail.step);
        build_step.dependOn(&fail.step);
        return;
    };
    const mod = b.createModule(.{
        .root_source_file = b.path("plugin.zig"),
        .target = root.resolved_target,
        .optimize = root.optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "mlx_serve", .module = root.import_table.get("mlx_serve").? },
            .{ .name = "mlx_c", .module = root.import_table.get("mlx_c").? },
        },
    });
    wire(b, mod, dep);
    mod.addLibraryPath(.{ .cwd_relative = mlx_lib });
    mod.addRPath(.{ .cwd_relative = mlx_lib });
    mod.linkSystemLibrary("mlxc", .{});
    mod.linkSystemLibrary("mlx", .{});
    const t = b.addTest(.{ .name = "gpu-test", .root_module = mod, .filters = &gpu_filters });
    run_step.dependOn(&b.addRunArtifact(t).step);
    build_step.dependOn(&b.addInstallArtifact(t, .{}).step);
}
