//! Builds this plugin with the mlx-serve plugin kit. The root module (plugin.zig) holds the generated copy of mlx-serve's
//! DeepSeek-V4.1 model (src/mlx_serve, scripts/prepare.sh); the expert streamer is a module of its own (`mlx_stream`, over
//! its own `sdk`), and turboexl3 (build.zig.zon, pinned by URL and hash) runs the EXL3 math over this plugin's `sdk` for it
//! (src/turboexl3_sdk.zig). Each reaches the plugin's root as `mlx_host`, so no file is in two modules. `zig build test`
//! runs config.zig's MLX-free tests (plugin.json's test_step).

const std = @import("std");

pub fn build(b: *std.Build) void {
    // src/mlx_serve is generated (scripts/prepare.sh): no build goes on with a tree other than the reviewed one.
    _ = b.run(&.{ "bash", b.root.joinString(b.allocator, "scripts/check-mlx-serve.sh") catch @panic("OOM") });
    const lib = @import("mlx_serve_kit").plugin(b, .{ .name = "turboexl3_ds41" });
    const root = lib.root_module.import_table.get("plugin").?;
    const target = root.resolved_target.?;
    const optimize = root.optimize.?;
    const kit = root.import_table.get("mlx_serve").?;
    // What the copies read from mlx-serve's build: the streamer is built in, no slow tests.
    const opts = b.addOptions();
    opts.addOption(bool, "mlx_stream", true);
    opts.addOption(bool, "slow_tests", false);
    // The decode route recorder (src/route_rec.zig, `MLX_SERVE_DSV41_ROUTES`): a diagnostic build's, off in every served one.
    opts.addOption(bool, "route_rec", b.option(bool, "route-rec", "Compile the decode route recorder in (MLX_SERVE_DSV41_ROUTES); a diagnostic build, never served") orelse false);
    // The copied stream's profile probes and the decode verify's GPU timeline (src/verify_tl.zig, `MLX_SERVE_DSV41_TIMELINE`):
    // a profile build's, off in every served one.
    const profile = b.option(bool, "plugin-profile", "Compile the streamer's profile probes and the verify timeline in (MLX_SERVE_DSV41_TIMELINE); a diagnostic build, never served") orelse false;
    opts.addOption(bool, "plugin_profile", profile);
    root.addOptions("build_options", opts);
    // MLX's steel headers as the kit ships them (lane_qmm's tile matmul, turboexl3's prefill GEMMs): the kit module's own
    // import, so the file is in one module.
    const steel = kit.import_table.get("mlx_steel_sources").?;
    root.addImport("mlx_steel_sources", steel);
    // The expert streamer and its own `sdk`, as mlx-serve's build.zig makes them (build/libs.zig mlxStreamPkg): the read
    // pool is C; MLX's event and alloc shims are this plugin's, over the host's GPU table (src/host_shims.zig).
    const csrc = "src/mlx_serve/sdk/weights/streamed/csrc";
    const ssdk = b.createModule(.{
        .root_source_file = b.path("src/mlx_serve/sdk/weights/streamed/sdk/root.zig"),
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
        .imports = &.{ .{ .name = "mlx_host", .module = root }, .{ .name = "sdk", .module = ssdk } },
    });
    // A profile build also keeps the read pool's event-signal log, links the timeline's shim (non-ARC Objective-C++, as
    // mlx-serve's profile build compiles it) and the buffer it hooks (src/tl_buffer.m), over mlx-c's headers as the kit
    // ships them.
    const pool: []const []const u8 = if (profile)
        &.{ "-O2", "-std=c11", "-Wall", "-Wextra", "-Werror", "-pthread", "-DQ3LD_EVSIG" }
    else
        &.{ "-O2", "-std=c11", "-Wall", "-Wextra", "-Werror", "-pthread" };
    stream.addCSourceFile(.{ .file = b.path(csrc ++ "/q3_lookahead4_exl3.c"), .flags = pool });
    stream.addIncludePath(b.path(csrc));
    if (profile) {
        const objc: []const []const u8 = &.{ "-O2", "-Wall", "-Wextra", "-Werror", "-fno-objc-arc" };
        stream.addCSourceFile(.{ .file = b.path(csrc ++ "/dsv41_cb_timeline.mm"), .flags = objc });
        stream.addCSourceFile(.{ .file = b.path("src/tl_buffer.m"), .flags = objc });
        stream.addIncludePath(b.dependency("mlx_serve_kit", .{}).path("include"));
        stream.link_libcpp = true;
        stream.linkSystemLibrary("objc", .{});
        stream.linkFramework("Foundation", .{});
        stream.linkFramework("Metal", .{});
    }
    root.addImport("mlx_stream", stream);
    // turboexl3: its library and its plugin-kit launcher over this plugin's `sdk` for them.
    const dep = b.dependency("turboexl3", .{ .target = target, .optimize = optimize });
    // ROUTED_ONE=check's exact reference (src/one_exact.zig) against turboexl3's own reference projection: its MLX-free
    // src/format.zig as a module of its own, for a test in `zig build test` (the served build reaches it as `t3.format`).
    const format = b.createModule(.{ .root_source_file = dep.path("src/format.zig"), .target = target, .optimize = optimize });
    const exact = b.addTest(.{ .name = "one_exact", .root_module = b.createModule(.{
        .root_source_file = b.path("src/one_exact_test.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "turboexl3_format", .module = format }},
    }) });
    b.top_level_steps.get("test").?.step.dependOn(&b.addRunArtifact(exact).step);
    // MLX_SERVE_DSV41_SEED_FREE on the streamer's own policy, MLX-free: src/seed_free_test.zig as a test of its own (the
    // policy is the `mlx_stream` module's file, never the plugin root's).
    const free = b.addTest(.{ .name = "seed_free", .root_module = b.createModule(.{
        .root_source_file = b.path("src/seed_free_test.zig"),
        .target = target,
        .optimize = optimize,
    }) });
    b.top_level_steps.get("test").?.step.dependOn(&b.addRunArtifact(free).step);
    // MLX_SERVE_DSV41_STAGE's transient rows on the same policy (`PlanOpts.held_rows`): src/stage_test.zig, likewise.
    const stage_rows = b.addTest(.{ .name = "stage_rows", .root_module = b.createModule(.{
        .root_source_file = b.path("src/stage_test.zig"),
        .target = target,
        .optimize = optimize,
    }) });
    b.top_level_steps.get("test").?.step.dependOn(&b.addRunArtifact(stage_rows).step);
    // MLX_SERVE_DSV41_DIRECT_PARTS's byte gate: the copied read pool's C file alone (no MLX) reading the pack's record
    // layout through the PARTS route and the staging route, the same slot bytes (src/direct_parts_test.zig).
    const parts_gate = b.addTest(.{ .name = "direct_parts", .root_module = b.createModule(.{
        .root_source_file = b.path("src/direct_parts_test.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    }) });
    parts_gate.root_module.addCSourceFile(.{ .file = b.path(csrc ++ "/q3_lookahead4_exl3.c"), .flags = &.{ "-O2", "-std=c11", "-Wall", "-Wextra", "-Werror", "-pthread" } });
    parts_gate.root_module.addIncludePath(b.path(csrc));
    b.top_level_steps.get("test").?.step.dependOn(&b.addRunArtifact(parts_gate).step);
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
}
