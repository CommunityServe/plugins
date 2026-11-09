//! Builds this plugin with the mlx-serve plugin kit, with turboexl3 (build.zig.zon, pinned by URL and hash) over this
//! plugin's `sdk` for it (src/turboexl3_sdk.zig), and the EXL3 lane over the kit's expert stream (`kit.streamed`) as a
//! module of its own (`mlx_stream`, src/stream.zig, over its own `sdk` in src/streamed/sdk/), the kit's read pool C
//! compiled in. `zig build test` runs config.zig's MLX-free tests and the lane's (src/*_test.zig).

const std = @import("std");

pub fn build(b: *std.Build) void {
    const kit_build = @import("mlx_serve_kit");
    const lib = kit_build.plugin(b, .{ .name = "turboexl3_ds41" });
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

    // The EXL3 lane over the kit's expert stream: its own `sdk`, the module, and the kit's read pool C. The stream's MLX
    // internals (events, aliases, the launch cache) are this plugin's C symbols over the host's table (src/host_shims.zig).
    const ssdk = b.createModule(.{
        .root_source_file = b.path("src/streamed/sdk/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "mlx_serve", .module = kit }},
    });
    const stream = b.createModule(.{
        .root_source_file = b.path("src/stream.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "sdk", .module = ssdk }, .{ .name = "mlx_serve", .module = kit }, .{ .name = "turboexl3", .module = t3 } },
    });
    kit_build.streamSources(b, stream, false);
    root.addImport("mlx_stream", stream);

    // The lane's MLX-free tests (src/*_test.zig), each a root of its own: the a4k record layout through the kit's read
    // pool, the half-integer-rate tiles against turboexl3's format.
    const format = b.createModule(.{ .root_source_file = dep.path("src/core/format.zig"), .target = target, .optimize = optimize });
    for ([_][]const u8{ "frac", "a4k" }) |name| {
        const m = b.createModule(.{ .root_source_file = b.path(b.fmt("src/{s}_test.zig", .{name})), .target = target, .optimize = optimize, .link_libc = true });
        if (std.mem.eql(u8, name, "frac")) {
            m.addImport("turboexl3_format", format);
            m.addImport("sdk", ssdk);
        }
        if (std.mem.eql(u8, name, "a4k")) {
            m.addImport("mlx_serve", kit);
            kit_build.streamSources(b, m, false);
        }
        b.top_level_steps.get("test").?.step.dependOn(&b.addRunArtifact(b.addTest(.{ .name = name, .root_module = m })).step);
    }
}
