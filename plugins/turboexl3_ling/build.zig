//! Builds this plugin with the mlx-serve plugin kit. The root module (plugin.zig) holds the generated copy of mlx-serve's
//! Ling model (src/mlx_serve, scripts/prepare.sh); the expert streamer is a module of its own (`mlx_stream`, over its own
//! `sdk`), and turboexl3 (build.zig.zon, pinned by URL and hash) runs the EXL3 math over this plugin's `sdk` for it
//! (src/turboexl3_sdk.zig). Each reaches the plugin's root as `mlx_host`, so no file is in two modules. `zig build test`
//! runs the plugin's MLX-free tests (config.zig, src/clamps.zig, src/bill.zig, src/ling_switches.zig,
//! src/ling_embed_rows.zig, ...).

const std = @import("std");

pub fn build(b: *std.Build) void {
    // src/mlx_serve is generated (scripts/prepare.sh): no build goes on with a tree other than the reviewed one.
    _ = b.run(&.{ "bash", b.root.joinString(b.allocator, "scripts/check-mlx-serve.sh") catch @panic("OOM") });
    const lib = @import("mlx_serve_kit").plugin(b, .{ .name = "turboexl3_ling" });
    const root = lib.root_module.import_table.get("plugin").?;
    const target = root.resolved_target.?;
    const optimize = root.optimize.?;
    const kit = root.import_table.get("mlx_serve").?;
    // What the copies read from mlx-serve's build: the streamer is built in, no slow tests.
    const opts = b.addOptions();
    opts.addOption(bool, "mlx_stream", true);
    opts.addOption(bool, "slow_tests", false);
    root.addOptions("build_options", opts);
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
    stream.addCSourceFile(.{ .file = b.path(csrc ++ "/q3_lookahead4_exl3.c"), .flags = &.{ "-O2", "-std=c11", "-Wall", "-Wextra", "-Werror", "-pthread" } });
    stream.addIncludePath(b.path(csrc));
    root.addImport("mlx_stream", stream);
    // The per-layer clamps, MLX-free, as a test of their own.
    const clamps = b.addTest(.{ .name = "clamps", .root_module = b.createModule(.{
        .root_source_file = b.path("src/clamps.zig"),
        .target = target,
        .optimize = optimize,
    }) });
    b.top_level_steps.get("test").?.step.dependOn(&b.addRunArtifact(clamps).step);
    // The memory bill beside the bank, MLX-free.
    const bill = b.addTest(.{ .name = "bill", .root_module = b.createModule(.{
        .root_source_file = b.path("src/bill.zig"),
        .target = target,
        .optimize = optimize,
    }) });
    b.top_level_steps.get("test").?.step.dependOn(&b.addRunArtifact(bill).step);
    // The MTP head's MLA layer on its latent cache (src/ling_mtp_latent.zig), MLX-free: which MLA calls take it.
    const mtp_latent = b.addTest(.{ .name = "mtp_latent", .root_module = b.createModule(.{
        .root_source_file = b.path("src/ling_mtp_latent.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    }) });
    b.top_level_steps.get("test").?.step.dependOn(&b.addRunArtifact(mtp_latent).step);
    // LAYER_MAJOR's parity-instrument option, PF_DFIN's whole-call rule and counts, and the served cache's layout
    // namespace (src/ling_switches.zig), MLX-free (the kit's module for the option's type).
    const switches = b.addTest(.{ .name = "switches", .root_module = b.createModule(.{
        .root_source_file = b.path("src/ling_switches.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "mlx_serve", .module = kit }},
    }) });
    b.top_level_steps.get("test").?.step.dependOn(&b.addRunArtifact(switches).step);
    // The run-path line's names, in the stack's order (src/ling_ran.zig), MLX-free.
    const ling_ran = b.addTest(.{ .name = "ling_ran", .root_module = b.createModule(.{
        .root_source_file = b.path("src/ling_ran.zig"),
        .target = target,
        .optimize = optimize,
    }) });
    b.top_level_steps.get("test").?.step.dependOn(&b.addRunArtifact(ling_ran).step);
    // PF_MOE_TAIL's span bookkeeping (src/ling_pf_tail_plan.zig), MLX-free.
    const pf_tail_plan = b.addTest(.{ .name = "pf_tail_plan", .root_module = b.createModule(.{
        .root_source_file = b.path("src/ling_pf_tail_plan.zig"),
        .target = target,
        .optimize = optimize,
    }) });
    b.top_level_steps.get("test").?.step.dependOn(&b.addRunArtifact(pf_tail_plan).step);
    // The copied streamer's routed calls on the trace backend, MLX-free (src/ling_streamer_test.zig). The streamer's `sdk`
    // is a module of its own here, over a stand-in plugin root holding the names plugin.zig gives the streamer
    // (src/ling_streamer_host.zig).
    const streamer_host = b.createModule(.{ .root_source_file = b.path("src/ling_streamer_host.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "mlx_serve", .module = kit }} });
    const streamer_sdk = b.createModule(.{ .root_source_file = b.path("src/mlx_serve/sdk/weights/streamed/sdk/root.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "mlx_host", .module = streamer_host }} });
    const streamer = b.addTest(.{ .name = "streamer", .root_module = b.createModule(.{
        .root_source_file = b.path("src/ling_streamer_test.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "mlx_serve", .module = kit }, .{ .name = "mlx_host", .module = streamer_host }, .{ .name = "sdk", .module = streamer_sdk } },
    }) });
    b.top_level_steps.get("test").?.step.dependOn(&b.addRunArtifact(streamer).step);
    // Ling's read-ahead predictor (its scores' form and the copied streamer's group-limited pick), MLX-free.
    const predictor = b.addTest(.{ .name = "predictor", .root_module = b.createModule(.{
        .root_source_file = b.path("src/ling_predictor_test.zig"),
        .target = target,
        .optimize = optimize,
    }) });
    b.top_level_steps.get("test").?.step.dependOn(&b.addRunArtifact(predictor).step);
    // EMBED_ROWS' table in its shard and its row reads (src/ling_embed_rows.zig), MLX-free: test packs in /tmp.
    const embed_rows = b.addTest(.{ .name = "embed_rows", .root_module = b.createModule(.{
        .root_source_file = b.path("src/ling_embed_rows.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    }) });
    b.top_level_steps.get("test").?.step.dependOn(&b.addRunArtifact(embed_rows).step);
    // RQ's capacity replay, plan, prompt refusal and slot table (src/ling_rq.zig), MLX-free.
    const rq = b.addTest(.{ .name = "rq", .root_module = b.createModule(.{
        .root_source_file = b.path("src/ling_rq.zig"),
        .target = target,
        .optimize = optimize,
    }) });
    b.top_level_steps.get("test").?.step.dependOn(&b.addRunArtifact(rq).step);
    // turboexl3: its library and its plugin-kit launcher over this plugin's `sdk` for them.
    const dep = b.dependency("turboexl3", .{ .target = target, .optimize = optimize });
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
    // The copied EXL3 format seam (sdk/quant/exl3/format.zig) names Sushi's module: turboexl3 keeps its API.
    root.addImport("sushi_exl3", t3);
    root.addImport("turboexl3_kit", tkit);
    // turboexl3_ling: the GPU byte tests (src/ling_gpu_tests.zig: the KDA paths, the exact verify's dense rows), apart
    // from the MLX-free `test` step: a test
    // binary over the copied blocks (the kit's `mlx_serve` and the build options only: a module that imports the plugin
    // root would claim the copied files too), linked against a staged MLX (`-Dmlx-lib`: the server's lib/mlx/lib,
    // holding libmlxc, libmlx and mlx.metallib). Its tests skip unless MLX_SERVE_LING_GPU_TESTS=1. `zig build gpu-test
    // -Dmlx-lib=<dir>` builds and runs it; `gpu-test-build` only builds it (zig-out/tests/ling_kda_gpu).
    if (b.option([]const u8, "mlx-lib", "a staged MLX for `zig build gpu-test` (libmlxc, libmlx, mlx.metallib)")) |mlx_lib| {
        const gmod = b.createModule(.{ .root_source_file = b.path("src/ling_gpu_tests.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "mlx_serve", .module = kit }} });
        gmod.addOptions("build_options", opts);
        // turboexl3 for the SwiGLU epilogue's test (src/ling_swiglu_epi_test.zig): its library and launcher over an
        // `sdk` of this binary's own, whose `mlx_host` is the test root (src/ling_gpu_tests.zig names mlx, shared and
        // quant_abi), so no copied file is in two modules. With no host table bound, turboexl3 runs mlx-c's own kernels.
        const gsdk = b.createModule(.{
            .root_source_file = b.path("src/turboexl3_sdk.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{ .{ .name = "mlx_host", .module = gmod }, .{ .name = "mlx_serve", .module = kit }, .{ .name = "mlx_steel_sources", .module = steel } },
        });
        const gkit = b.createModule(.{
            .root_source_file = dep.path("kit/root.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{ .{ .name = "sdk", .module = gsdk }, .{ .name = "mlx_serve", .module = kit } },
        });
        gsdk.addImport("turboexl3_kit", gkit);
        gmod.addImport("turboexl3", b.createModule(.{
            .root_source_file = dep.path("src/root.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{.{ .name = "sdk", .module = gsdk }},
        }));
        gmod.addLibraryPath(.{ .cwd_relative = mlx_lib });
        gmod.linkSystemLibrary("mlxc", .{ .use_pkg_config = .no });
        gmod.addRPath(.{ .cwd_relative = mlx_lib });
        const gt = b.addTest(.{ .name = "ling_kda_gpu", .root_module = gmod });
        const gt_build = b.step("gpu-test-build", "Build the plugin's GPU byte tests (zig-out/tests/ling_kda_gpu, ling_restore_gpu)");
        gt_build.dependOn(&b.addInstallArtifact(gt, .{ .dest_dir = .{ .override = .{ .custom = "tests" } } }).step);
        const run = b.addRunArtifact(gt);
        run.has_side_effects = true;
        const gt_run = b.step("gpu-test", "Run the plugin's GPU byte tests (with MLX_SERVE_LING_GPU_TESTS=1)");
        gt_run.dependOn(&run.step);
        // turboexl3_ling: part A of the restore byte check (src/ling_restore_test.zig), a binary of its own beside
        // the suite (zig-out/tests/ling_restore_gpu), built and run by the same steps. It skips unless
        // MLX_SERVE_LING_RESTORE_OUT names the file it writes for part B.
        const rt = restoreTest(b, target, optimize, kit, steel, dep, mlx_lib);
        gt_build.dependOn(&b.addInstallArtifact(rt, .{ .dest_dir = .{ .override = .{ .custom = "tests" } } }).step);
        const rrun = b.addRunArtifact(rt);
        rrun.has_side_effects = true;
        gt_run.dependOn(&rrun.step);
    }
}

/// turboexl3_ling: part A of the restore byte check (src/ling_restore_test.zig), its own test binary. It runs the
/// copied MLA layer (`mlaAttn`) with the copied trunk's model, so its build options keep that trunk off the expert
/// streamer (trunk.zig `Bank`), which this binary does not link. Otherwise it is wired as the suite's binary: the copied
/// blocks over the kit's `mlx_serve`, and turboexl3 over an `sdk` whose `mlx_host` is this binary's root, which names
/// mlx, shared and quant_abi.
fn restoreTest(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, kit: *std.Build.Module, steel: *std.Build.Module, dep: *std.Build.Dependency, mlx_lib: []const u8) *std.Build.Step.Compile {
    const mod = b.createModule(.{ .root_source_file = b.path("src/ling_restore_test.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "mlx_serve", .module = kit }, .{ .name = "mlx_steel_sources", .module = steel } } });
    const opts = b.addOptions();
    opts.addOption(bool, "mlx_stream", false);
    opts.addOption(bool, "slow_tests", false);
    mod.addOptions("build_options", opts);
    const sdk = b.createModule(.{
        .root_source_file = b.path("src/turboexl3_sdk.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{ .{ .name = "mlx_host", .module = mod }, .{ .name = "mlx_serve", .module = kit }, .{ .name = "mlx_steel_sources", .module = steel } },
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
    mod.addImport("turboexl3", t3);
    mod.addImport("sushi_exl3", t3);
    mod.addImport("turboexl3_kit", tkit);
    mod.addLibraryPath(.{ .cwd_relative = mlx_lib });
    mod.linkSystemLibrary("mlxc", .{ .use_pkg_config = .no });
    mod.addRPath(.{ .cwd_relative = mlx_lib });
    // C4's byte test (src/ling_lm_pass_test.zig) runs the trunk's KDA layers, whose recurrence plans its kernel by the
    // GPU's core count (the copied sdk/blocks/kda/recurrence.zig `serves` calls sdk/util/system.zig `gpuCoreCount`:
    // IOKit and CoreFoundation). The host links both frameworks (mlx-serve build/macos.zig), and the plugin's dylib
    // leaves those symbols to the process that loads it, but a test executable must link them itself: without them
    // this binary failed to link (_CFRelease, _IOObjectRelease and 7 more undefined; c4.FAIL, 2026-11-11 18:11:09Z).
    for ([_][]const u8{ "IOKit", "CoreFoundation" }) |f| mod.linkFramework(f, .{});
    return b.addTest(.{ .name = "ling_restore_gpu", .root_module = mod });
}
