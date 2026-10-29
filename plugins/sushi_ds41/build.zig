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
}
