//! Builds this plugin with the mlx-serve plugin kit: turboexl3 (build.zig.zon, pinned by URL and hash) over this
//! plugin's `sdk` for it (src/turboexl3_sdk.zig), and the kit's expert stream read pool (C, `streamSources`) compiled into
//! the plugin. `zig build test` runs the MLX-free tests: config.zig (the kit's step), src/clamps.zig, src/bank_rows.zig,
//! src/ling_switches.zig, src/ling_manifest.zig, src/stream_stats.zig and src/prompt_plan.zig.

const std = @import("std");

pub fn build(b: *std.Build) void {
    const kit_build = @import("mlx_serve_kit");
    const lib = kit_build.plugin(b, .{ .name = "turboexl3_ling" });
    const root = lib.root_module.import_table.get("plugin").?;
    const target = root.resolved_target.?;
    const optimize = root.optimize.?;
    const kit = root.import_table.get("mlx_serve").?;
    const steel = kit.import_table.get("mlx_steel_sources").?;
    const dep = b.dependency("turboexl3", .{ .target = target, .optimize = optimize });
    const sdk = b.createModule(.{
        .root_source_file = b.path("src/turboexl3_sdk.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{ .{ .name = "mlx_serve", .module = kit }, .{ .name = "mlx_steel_sources", .module = steel } },
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
    root.addImport("turboexl3", t3);
    root.addImport("turboexl3_kit", tkit);
    // The kit stream's read pool; its MLX internals are this plugin's C symbols over the host's table (src/host_shims.zig).
    kit_build.streamSources(b, root, false);

    // The MLX-free tests, each a root of its own.
    const test_step = b.top_level_steps.get("test").?;
    for ([_][]const u8{ "clamps", "bank_rows", "ling_switches", "ling_manifest", "stream_stats", "prompt_plan" }) |name| {
        const m = b.createModule(.{ .root_source_file = b.path(b.fmt("src/{s}.zig", .{name})), .target = target, .optimize = optimize });
        test_step.step.dependOn(&b.addRunArtifact(b.addTest(.{ .name = name, .root_module = m })).step);
    }
}
