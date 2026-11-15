//! Builds glm53f with the mlx-serve plugin kit. The plugin's root reads MLX's steel headers (the NAX router's one-row
//! text embeds MLX's gemv.h) as the kit's own `mlx_steel_sources` module.
//! - `zig build test`: config.zig's MLX-free tests (the kit's step).
//! - `zig build mlx-free-test` (plugin.json's `test_step`): those and the MLX-free tests rooted at the plugin.
//! - `zig build gpu-test -Dmlx-lib=<dir>`: the speed changes' bytes tests (src/gpu_tests.zig) on the Metal device,
//!   against the server's own MLX (the directory holding its libmlx.dylib and libmlxc.dylib); `gpu-test-build` only
//!   compiles them.
//! - `zig build check`: the dylib's root and the GPU tests analysed, nothing emitted (a compile check off macOS).

const std = @import("std");

pub fn build(b: *std.Build) void {
    const lib = @import("mlx_serve_kit").plugin(b, .{ .name = "glm53f" });
    const root = lib.root_module.import_table.get("plugin").?;
    wire(root);

    const tests = b.addTest(.{ .name = "mlx-free-test", .root_module = root, .filters = &mlx_free_filters });
    tests.linker_allow_shlib_undefined = true;
    const mlx_free = b.step("mlx-free-test", "Run the MLX-free tests: the kit's and those rooted at the plugin");
    mlx_free.dependOn(&b.top_level_steps.get("test").?.step);
    mlx_free.dependOn(&b.addRunArtifact(tests).step);

    const check = b.step("check", "Analyse the dylib's root and the GPU tests, emitting nothing");
    // The dylib's own graph, not installed: nothing asks for its binary, so it is analysed and not emitted.
    check.dependOn(&b.addLibrary(.{ .name = "glm53f-check", .linkage = .dynamic, .root_module = lib.root_module }).step);
    const gpu_mod = gpuModule(b, root);
    check.dependOn(&b.addTest(.{ .name = "gpu-check", .root_module = gpu_mod, .filters = &gpu_filters }).step);
    gpuTests(b, gpu_mod);
}

/// The kit module's `mlx_steel_sources` over `m`.
fn wire(m: *std.Build.Module) void {
    const kit = m.import_table.get("mlx_serve").?;
    m.addImport("mlx_steel_sources", kit.import_table.get("mlx_steel_sources").?);
}

/// The tests `mlx-free-test` runs rooted at the plugin, by name.
const mlx_free_filters = [_][]const u8{ "glm53f", "router_nax:" };

/// The tests `gpu-test` runs, by name (src/gpu_tests.zig).
const gpu_filters = [_][]const u8{"glm53f gpu"};

/// A module graph of the GPU tests' own over plugin.zig, whose test block imports src/gpu_tests.zig (the dylib never
/// links MLX; the tests link the server's).
fn gpuModule(b: *std.Build, root: *std.Build.Module) *std.Build.Module {
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
    wire(mod);
    return mod;
}

fn gpuTests(b: *std.Build, mod: *std.Build.Module) void {
    const run_step = b.step("gpu-test", "Run the speed changes' bytes tests on the Metal device: -Dmlx-lib=<dir>");
    const build_step = b.step("gpu-test-build", "Compile the speed changes' bytes tests without running them: -Dmlx-lib=<dir>");
    const mlx_lib = b.option([]const u8, "mlx-lib", "The directory holding the server's libmlx.dylib and libmlxc.dylib (gpu-test)") orelse {
        const fail = b.addFail("gpu-test needs -Dmlx-lib=<the directory holding the server's libmlx.dylib and libmlxc.dylib>");
        run_step.dependOn(&fail.step);
        build_step.dependOn(&fail.step);
        return;
    };
    const linked = b.createModule(.{
        .root_source_file = mod.root_source_file,
        .target = mod.resolved_target,
        .optimize = mod.optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "mlx_serve", .module = mod.import_table.get("mlx_serve").? },
            .{ .name = "mlx_c", .module = mod.import_table.get("mlx_c").? },
        },
    });
    wire(linked);
    linked.addLibraryPath(.{ .cwd_relative = mlx_lib });
    linked.addRPath(.{ .cwd_relative = mlx_lib });
    linked.linkSystemLibrary("mlxc", .{});
    linked.linkSystemLibrary("mlx", .{});
    const t = b.addTest(.{ .name = "gpu-test", .root_module = linked, .filters = &gpu_filters });
    run_step.dependOn(&b.addRunArtifact(t).step);
    build_step.dependOn(&b.addInstallArtifact(t, .{}).step);
}
