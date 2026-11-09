//! Builds the plugin with the mlx-serve kit and wires the generated copy of mlx-serve's expert streamer
//! (src/mlx_serve, scripts/prepare.sh) as mlx-serve's own build does: the `mlx_stream` module and its `sdk`, both
//! reaching mlx, log, env and shared through `mlx_host`, which is the plugin's root; the streamer's read pool is C.

const std = @import("std");

pub fn build(b: *std.Build) void {
    _ = b.run(&.{ "bash", b.root.joinString(b.allocator, "scripts/check-mlx-serve.sh") catch @panic("OOM") });
    const lib = @import("mlx_serve_kit").plugin(b, .{ .name = "minimax_m3" });
    const root = lib.root_module.import_table.get("plugin").?;
    wire(b, root);
    const tests = b.addTest(.{ .name = "mlx-free-test", .root_module = root, .filters = &mlx_free_filters });
    tests.linker_allow_shlib_undefined = true;
    const mlx_free = b.step("mlx-free-test", "Run the MLX-free tests: the kit's and those rooted at the plugin");
    mlx_free.dependOn(&b.top_level_steps.get("test").?.step);
    mlx_free.dependOn(&b.addRunArtifact(tests).step);
    mlx_free.dependOn(&b.addSystemCommand(&.{ "python3", "-m", "unittest", "discover", "-s", "scripts", "-p", "test_*.py" }).step);
    gpuTests(b, root);
}

/// The Metal tests (gpu_filters), linked to the server's own MLX: `zig build gpu-test -Dmlx-lib=<dir>`. The parity tests
/// skip without their float64 reference (MINIMAX_M3_REFERENCE, tests/dump_minimax_m3_fixtures.py).
fn gpuTests(b: *std.Build, root: *std.Build.Module) void {
    const run_step = b.step("gpu-test", "Run the Metal tests against the server's MLX: -Dmlx-lib=<dir>");
    const mlx_lib = b.option([]const u8, "mlx-lib", "The directory holding the server's libmlx.dylib and libmlxc.dylib (gpu-test)") orelse {
        run_step.dependOn(&b.addFail("gpu-test needs -Dmlx-lib=<the directory holding the server's libmlx.dylib and libmlxc.dylib>").step);
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
    wire(b, mod);
    mod.addLibraryPath(.{ .cwd_relative = mlx_lib });
    mod.addRPath(.{ .cwd_relative = mlx_lib });
    mod.linkSystemLibrary("mlxc", .{});
    mod.linkSystemLibrary("mlx", .{});
    const t = b.addTest(.{ .name = "gpu-test", .root_module = mod, .filters = &gpu_filters });
    run_step.dependOn(&b.addRunArtifact(t).step);
}

fn wire(b: *std.Build, root: *std.Build.Module) void {
    const target = root.resolved_target.?;
    const optimize = root.optimize.?;
    const options = b.addOptions();
    // The copies gate the expert streamer on this, as mlx-serve's macOS build sets it.
    options.addOption(bool, "mlx_stream", true);
    root.addImport("build_options", options.createModule());
    const sdk = b.createModule(.{
        .root_source_file = b.path("src/mlx_serve/sdk/weights/streamed/sdk/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "mlx_host", .module = root }},
    });
    const stream = b.createModule(.{
        .root_source_file = b.path("src/mlx_serve/mlx_stream.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "mlx_host", .module = root }, .{ .name = "sdk", .module = sdk } },
    });
    const csrc = "src/mlx_serve/sdk/weights/streamed/csrc";
    stream.addCSourceFile(.{ .file = b.path(csrc ++ "/q3_lookahead4_exl3.c"), .flags = &.{ "-O2", "-std=c11", "-Wall", "-Wextra", "-Werror", "-pthread" } });
    stream.addIncludePath(b.path(csrc));
    root.addImport("mlx_stream", stream);
    root.addImport("sdk", sdk);
}

const mlx_free_filters = [_][]const u8{
    "MiniMax claims",
    "MiniMax parse",
    "MiniMax default context",
    "MiniMax prepared settings",
    "MiniMax bill prices",
    "MiniMax state bill",
    "MiniMax preflight bound",
    "MiniMax bank fill",
    "MiniMax prepared expert_rows",
    "MiniMax load settings",
    "MiniMax load facts",
    "MiniMax settings refuse",
    "MiniMax forwards shared usage",
    "MiniMax load refuses a device",
    "MiniMax OQ3 prepared config",
    "released Flash config",
    "config refuses changed math",
    "config reads a smaller model",
};

/// The Metal tests `gpu-test` runs (the attention, cache and expert arms against their float64 oracles).
const gpu_filters = [_][]const u8{
    "MiniMax GPU",
    "minimax_m3 parity: decode block selection",
    "minimax_m3 parity: expert routing",
};
