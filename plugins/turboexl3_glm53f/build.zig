//! Builds this plugin with the mlx-serve plugin kit. GLM-5.3-Flash's EXL3 bank is a module of its own (`mlx_stream`:
//! src/mlx_stream.zig over src/streamed/ and src/exl3/, with its own `sdk` in src/streamed/sdk/) over the kit's expert
//! stream (`kit.streamed`), the kit's read pool C compiled in (`streamSources`); turboexl3 (build.zig.zon, pinned by URL
//! and hash) runs the EXL3 math over this plugin's `sdk` for it (src/turboexl3_sdk.zig). Each reaches the plugin's root
//! (plugin.zig) as `mlx_host`. `zig build test` runs the MLX-free tests: config.zig's (the kit's step), the tile gate,
//! and the plugin-rooted ones (options, bank rows, the MTP bank's manifest), which never call MLX.

const std = @import("std");

pub fn build(b: *std.Build) void {
    const kit_build = @import("mlx_serve_kit");
    const lib = kit_build.plugin(b, .{ .name = "turboexl3_glm53f" });
    const root = lib.root_module.import_table.get("plugin").?;
    const target = root.resolved_target.?;
    const optimize = root.optimize.?;
    const kit = root.import_table.get("mlx_serve").?;
    const steel = kit.import_table.get("mlx_steel_sources").?;
    const dep = b.dependency("turboexl3", .{ .target = target, .optimize = optimize });

    // GLM-5.3-Flash's EXL3 bank over the kit's expert stream and its own `sdk`: MLX's event, alloc and address shims are
    // this plugin's, over the host's GPU table (src/host_shims.zig).
    const ssdk = b.createModule(.{
        .root_source_file = b.path("src/streamed/sdk/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "mlx_host", .module = root }},
    });
    const stream = b.createModule(.{
        .root_source_file = b.path("src/mlx_stream.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "mlx_host", .module = root }, .{ .name = "sdk", .module = ssdk }, .{ .name = "mlx_serve", .module = kit } },
    });
    kit_build.streamSources(b, stream, false);
    // TURBOEXL3_GLM53F_KEEP_WARM: the keep-warm engine's Metal shim (src/csrc/glm53f_gpu_warm.m), its symbols hidden in
    // the dylib.
    if (target.result.os.tag == .macos) {
        stream.addCSourceFile(.{ .file = b.path("src/csrc/glm53f_gpu_warm.m"), .flags = &.{ "-O2", "-fobjc-arc", "-Wall", "-Wextra" } });
        stream.linkFramework("Metal", .{});
        stream.linkFramework("Foundation", .{});
    }
    root.addImport("mlx_stream", stream);

    // turboexl3: its library and its plugin-kit launcher over this plugin's `sdk` for them.
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
    root.addImport("turboexl3", t3);
    root.addImport("turboexl3_kit", tkit);
    root.addImport("mlx_steel_sources", steel);

    const test_step = &b.top_level_steps.get("test").?.step;
    // The tile gate: turboexl3's reference decode of real GLM-5.3-Flash routed tiles at every rate the SC packs route at,
    // against exllamav3's rule (src/frac_test.zig, src/glm_tiles.json), turboexl3's MLX-free src/core/format.zig as a
    // module of its own.
    const format = b.createModule(.{ .root_source_file = dep.path("src/core/format.zig"), .target = target, .optimize = optimize });
    const tiles = b.addTest(.{ .name = "glm_tiles", .root_module = b.createModule(.{
        .root_source_file = b.path("src/frac_test.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "turboexl3_format", .module = format }},
    }) });
    test_step.dependOn(&b.addRunArtifact(tiles).step);
    // The plugin-rooted tests (filtered to this plugin's names): MLX left undefined, as none of them calls it.
    const own = b.addTest(.{ .name = "turboexl3_glm53f", .root_module = root, .filters = &.{"turboexl3_glm53f"} });
    own.linker_allow_shlib_undefined = true;
    test_step.dependOn(&b.addRunArtifact(own).step);
}
