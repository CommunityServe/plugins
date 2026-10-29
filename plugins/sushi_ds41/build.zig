//! Builds the plugin with the mlx-serve kit and adds the vendored sushi slice: `sushi_exl3` is a module of its own that
//! reaches mlx, log and io_util through `mlx_host`, which is the plugin's root.

const std = @import("std");

pub fn build(b: *std.Build) void {
    const lib = @import("mlx_serve_kit").plugin(b, .{ .name = "sushi_ds41" });
    const root = lib.root_module.import_table.get("plugin").?;
    const exl3 = b.createModule(.{
        .root_source_file = b.path("src/sushi/exl3/root.zig"),
        .target = root.resolved_target,
        .optimize = root.optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "mlx_host", .module = root }},
    });
    root.addImport("sushi_exl3", exl3);
    // The kit's test step roots config.zig, which needs the vendored module; this one roots the plugin and leaves MLX
    // undefined (the tests never call it).
    const tests = b.addTest(.{ .root_module = root, .filters = &.{"sushi_ds41"} });
    tests.linker_allow_shlib_undefined = true;
    const run = b.addRunArtifact(tests);
    b.step("claims", "Run the claim and parse tests").dependOn(&run.step);

    // The Metal test of src/attn_nax.zig, linked to an MLX the caller names (the server's own, so its pin), never to
    // the dylib: `zig build gpu-test -Dmlx-lib=<dir holding libmlxc.dylib>`. It skips without an M5.
    const mlx_lib = b.option([]const u8, "mlx-lib", "Directory holding the libmlxc.dylib the Metal test links") orelse return;
    const gpu = b.createModule(.{ .root_source_file = b.path("src/attn_nax_test.zig"), .target = root.resolved_target, .optimize = root.optimize, .link_libc = true });
    gpu.addLibraryPath(.{ .cwd_relative = mlx_lib });
    gpu.linkSystemLibrary("mlxc", .{ .use_pkg_config = .no });
    gpu.addRPath(.{ .cwd_relative = mlx_lib });
    const gpu_run = b.addRunArtifact(b.addTest(.{ .root_module = gpu, .filters = &.{"sushi_ds41 gpu"} }));
    gpu_run.setEnvironmentVariable("SUSHI_DS41_GPU_TEST", "1");
    b.step("gpu-test", "Run the Metal test (needs -Dmlx-lib; skips without an M5)").dependOn(&gpu_run.step);
}
