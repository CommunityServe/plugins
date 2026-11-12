//! Builds this plugin with the mlx-serve plugin kit. The root module (plugin.zig) holds the generated copy of mlx-serve's
//! GLM-5.3-Flash model (src/mlx_serve, scripts/prepare.sh); GLM-5.3-Flash's EXL3 bank is a module of its own (`mlx_stream`:
//! src/mlx_stream.zig over src/streamed/ and src/exl3/, with its own `sdk` in src/streamed/sdk/) over the kit's expert
//! stream (`kit.streamed`), the kit's read pool C compiled in (`streamSources`), and turboexl3 (build.zig.zon, pinned by
//! URL and hash) runs the EXL3 math over this plugin's `sdk` for it (src/turboexl3_sdk.zig). Each reaches the plugin's
//! root as `mlx_host`, so no file is in two modules. `zig build test` runs the MLX-free tests (plugin.json's test_step):
//! config.zig's, the tile gate and the bank's construction checks.
//! `zig build gpu-test -Dmlx-lib=<dir>` runs turboexl3's own GLM-5.3-Flash tests (its src/models/glm53f, through its
//! `testModule` root) over this plugin's `sdk`, on the Metal device, against the server's MLX (the directory holding its
//! libmlx.dylib and libmlxc.dylib); `gpu-test-build` only compiles them. `--fork=<turboexl3 checkout>` builds them from
//! that tree in place of the pinned one.

const std = @import("std");

pub fn build(b: *std.Build) void {
    // src/mlx_serve is generated (scripts/prepare.sh): no build goes on with a tree other than the reviewed one.
    _ = b.run(&.{ "bash", b.root.joinString(b.allocator, "scripts/check-mlx-serve.sh") catch @panic("OOM") });
    const lib = @import("mlx_serve_kit").plugin(b, .{ .name = "turboexl3_glm53f" });
    const root = lib.root_module.import_table.get("plugin").?;
    const target = root.resolved_target.?;
    const optimize = root.optimize.?;
    const dep = b.dependency("turboexl3", .{ .target = target, .optimize = optimize });
    _ = wire(b, root, dep);
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
    b.top_level_steps.get("test").?.step.dependOn(&b.addRunArtifact(tiles).step);
    // The expert stream's construction checks over the SC packs' banks (src/stream_options_test.zig, by name). The kit's
    // test step roots config.zig, which has none of wire's modules, so this one roots the plugin and leaves MLX undefined
    // (the checks never call it). -Dglm53f-packs=<pack dir>[,...]: the same checks over those packs' own manifests and
    // experts.bin (opened, no record read; always run when given).
    const stream = b.addTest(.{ .name = "stream_options", .root_module = root, .filters = &.{"turboexl3_glm53f stream"} });
    stream.linker_allow_shlib_undefined = true;
    const stream_run = b.addRunArtifact(stream);
    if (b.option([]const u8, "glm53f-packs", "Pack dirs (a comma list): src/stream_options_test.zig's checks over their own banks")) |packs| {
        stream_run.setEnvironmentVariable("GLM53F_PACKS", packs);
        stream_run.has_side_effects = true;
    }
    b.top_level_steps.get("test").?.step.dependOn(&stream_run.step);
    const mlx_lib = b.option([]const u8, "mlx-lib", "The directory holding the server's libmlx.dylib and libmlxc.dylib (gpu-test, kda-gpu-test, router-gpu-test, dsa-gpu-test, smallop-gpu-test, draft-gpu-test, moe-combine-gpu-test, dec-prep-gpu-test)");
    gpuTests(b, root, dep, mlx_lib);
    kdaGpuTests(b, root, dep, mlx_lib);
    routerGpuTests(b, root, dep, mlx_lib);
    dsaGpuTests(b, root, dep, mlx_lib);
    smallopGpuTests(b, root, dep, mlx_lib);
    decPrepGpuTests(b, root, dep, mlx_lib);
    draftGpuTests(b, root, dep, mlx_lib);
    moeCombineGpuTests(b, root, dep, mlx_lib);
}

/// The copies' build options, MLX's steel headers, the GLM bank over the kit's stream and turboexl3 over `root` (a module
/// rooted at plugin.zig); returns turboexl3's `sdk`.
fn wire(b: *std.Build, root: *std.Build.Module, dep: *std.Build.Dependency) *std.Build.Module {
    const target = root.resolved_target.?;
    const optimize = root.optimize.?;
    const kit = root.import_table.get("mlx_serve").?;
    // What the copies read from mlx-serve's build: the bank is built in, no slow tests.
    const opts = b.addOptions();
    opts.addOption(bool, "mlx_stream", true);
    opts.addOption(bool, "slow_tests", false);
    root.addOptions("build_options", opts);
    // MLX's steel headers as the kit ships them (turboexl3's prefill GEMMs): the kit module's own import, so the file is
    // in one module.
    const steel = kit.import_table.get("mlx_steel_sources").?;
    root.addImport("mlx_steel_sources", steel);
    // GLM-5.3-Flash's EXL3 bank over the kit's expert stream (`kit.streamed`) and its own `sdk`: the kit's read pool is C,
    // compiled in (`streamSources`); MLX's event, alloc and address shims are this plugin's, over the host's GPU table
    // (src/host_shims.zig).
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
    @import("mlx_serve_kit").streamSources(b, stream, false);
    // TURBOEXL3_GLM53F_KEEP_WARM: the keep-warm engine's Metal shim, this plugin's own (src/csrc/glm53f_gpu_warm.m: the
    // server's csrc/mimo_gpu_warm.inc is not in the host table), its symbols hidden in the dylib; Metal and Foundation
    // linked for it (the engine's other targets take the copy's stubs).
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
    // The copied EXL3 format seam (sdk/quant/exl3/format.zig, a Sushi pack's experts) imports Sushi's module by name;
    // turboexl3's root has Sushi's names, so it stands in (no pack this plugin claims takes that arm).
    root.addImport("sushi_exl3", t3);
    root.addImport("turboexl3_kit", tkit);
    return sdk;
}

/// turboexl3's tests at GLM-5.3-Flash's shapes and rates (its `testModule` root, src/tests.zig, filtered to `glm53f`),
/// built as turboexl3's docs say a consumer builds them: over this plugin's `sdk` for it, with src/t3_test_host.zig as
/// `mlx_host` (the plugin's MLX layer and globals, without the plugin's copies), linking the server's MLX. Their GPU
/// half runs under TURBOEXL3_GPU=1.
fn gpuTests(b: *std.Build, root: *std.Build.Module, dep: *std.Build.Dependency, mlx_lib_opt: ?[]const u8) void {
    const run_step = b.step("gpu-test", "Run turboexl3's GLM-5.3-Flash tests against the server's MLX (TURBOEXL3_GPU=1 for the GPU ones): -Dmlx-lib=<dir>");
    const build_step = b.step("gpu-test-build", "Compile turboexl3's GLM-5.3-Flash tests without running them: -Dmlx-lib=<dir>");
    const mlx_lib = mlx_lib_opt orelse {
        const fail = b.addFail("gpu-test needs -Dmlx-lib=<the directory holding the server's libmlx.dylib and libmlxc.dylib>");
        run_step.dependOn(&fail.step);
        build_step.dependOn(&fail.step);
        return;
    };
    const target = root.resolved_target.?;
    const optimize = root.optimize.?;
    const kit = root.import_table.get("mlx_serve").?;
    const host = b.createModule(.{
        .root_source_file = b.path("src/t3_test_host.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "mlx_serve", .module = kit }},
    });
    const steel = kit.import_table.get("mlx_steel_sources").?;
    const sdk = b.createModule(.{
        .root_source_file = b.path("src/turboexl3_sdk.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{ .{ .name = "mlx_host", .module = host }, .{ .name = "mlx_serve", .module = kit }, .{ .name = "mlx_steel_sources", .module = steel } },
    });
    const tkit = b.createModule(.{
        .root_source_file = dep.path("kit/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "sdk", .module = sdk }, .{ .name = "mlx_serve", .module = kit } },
    });
    sdk.addImport("turboexl3_kit", tkit);
    const mod = b.createModule(.{
        .root_source_file = dep.path("src/tests.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "sdk", .module = sdk }, .{ .name = "mlx_host", .module = host } },
    });
    mod.addLibraryPath(.{ .cwd_relative = mlx_lib });
    mod.addRPath(.{ .cwd_relative = mlx_lib });
    mod.linkSystemLibrary("mlxc", .{});
    mod.linkSystemLibrary("mlx", .{});
    // `-Dgpu-test-filter=<words>`: only the GLM tests whose names hold them (default: every GLM test)
    const filter = b.option([]const u8, "gpu-test-filter", "gpu-test: only turboexl3's GLM-5.3-Flash tests whose names hold these words (default: glm53f, every one)") orelse "glm53f";
    const t = b.addTest(.{ .name = "gpu-test", .root_module = mod, .filters = &.{filter} });
    const run = b.addRunArtifact(t);
    run.has_side_effects = true; // the GPU is not a cached input
    run_step.dependOn(&run.step);
    build_step.dependOn(&b.addInstallArtifact(t, .{}).step);
}

/// The KDA prompt texts' device parity (src/kda_gpu_test.zig: kda.zig's forward with each text on against the op chain,
/// bit for bit), on the Metal device against the server's MLX: a test rooted at plugin.zig with wire's modules, as the
/// plugin is, linking the server's libmlx and libmlxc and the frameworks the copy's GPU core count reads (IOKit,
/// CoreFoundation). The test runs under TURBOEXL3_GLM53F_KDA_GPU=1; `kda-gpu-test-build` only compiles it.
fn kdaGpuTests(b: *std.Build, root: *std.Build.Module, dep: *std.Build.Dependency, mlx_lib_opt: ?[]const u8) void {
    const run_step = b.step("kda-gpu-test", "Run the KDA prompt texts' bitwise parity with the op chain against the server's MLX (TURBOEXL3_GLM53F_KDA_GPU=1): -Dmlx-lib=<dir>");
    const build_step = b.step("kda-gpu-test-build", "Compile the KDA prompt texts' parity test without running it: -Dmlx-lib=<dir>");
    const mlx_lib = mlx_lib_opt orelse {
        const fail = b.addFail("kda-gpu-test needs -Dmlx-lib=<the directory holding the server's libmlx.dylib and libmlxc.dylib>");
        run_step.dependOn(&fail.step);
        build_step.dependOn(&fail.step);
        return;
    };
    const mod = b.createModule(.{
        .root_source_file = b.path("plugin.zig"),
        .target = root.resolved_target.?,
        .optimize = root.optimize.?,
        .link_libc = true,
        .imports = &.{
            .{ .name = "mlx_serve", .module = root.import_table.get("mlx_serve").? },
            .{ .name = "mlx_c", .module = root.import_table.get("mlx_c").? },
        },
    });
    _ = wire(b, mod, dep);
    mod.addLibraryPath(.{ .cwd_relative = mlx_lib });
    mod.addRPath(.{ .cwd_relative = mlx_lib });
    mod.linkSystemLibrary("mlxc", .{});
    mod.linkSystemLibrary("mlx", .{});
    mod.linkFramework("IOKit", .{});
    mod.linkFramework("CoreFoundation", .{});
    // the KDA prompt texts' parity and the hyper-connection text's (src/hc_gpu_test.zig)
    const t = b.addTest(.{ .name = "kda-gpu-test", .root_module = mod, .filters = &.{ "turboexl3_glm53f kda gpu", "turboexl3_glm53f hc gpu" } });
    // as the plugin's own tests: the host's symbols the root names resolve at load, and none is reached here
    t.linker_allow_shlib_undefined = true;
    const run = b.addRunArtifact(t);
    run.has_side_effects = true; // the GPU is not a cached input
    run_step.dependOn(&run.step);
    build_step.dependOn(&b.addInstallArtifact(t, .{}).step);
}

/// TURBOEXL3_GLM53F_ROUTER's device parity (src/router_gpu_test.zig: the two launches against the stock route,
/// bit for bit), on the Metal device against the server's MLX: a test rooted at plugin.zig with wire's modules, as the
/// plugin is, linking the server's libmlx and libmlxc and the frameworks the copy's GPU core count reads (IOKit,
/// CoreFoundation). The test runs under TURBOEXL3_GLM53F_ROUTER_GPU=1; `router-gpu-test-build` only compiles it.
fn routerGpuTests(b: *std.Build, root: *std.Build.Module, dep: *std.Build.Dependency, mlx_lib_opt: ?[]const u8) void {
    const run_step = b.step("router-gpu-test", "Run TURBOEXL3_GLM53F_ROUTER's bitwise parity with the stock route against the server's MLX (TURBOEXL3_GLM53F_ROUTER_GPU=1): -Dmlx-lib=<dir>");
    const build_step = b.step("router-gpu-test-build", "Compile TURBOEXL3_GLM53F_ROUTER's parity test without running it: -Dmlx-lib=<dir>");
    const mlx_lib = mlx_lib_opt orelse {
        const fail = b.addFail("router-gpu-test needs -Dmlx-lib=<the directory holding the server's libmlx.dylib and libmlxc.dylib>");
        run_step.dependOn(&fail.step);
        build_step.dependOn(&fail.step);
        return;
    };
    const mod = b.createModule(.{
        .root_source_file = b.path("plugin.zig"),
        .target = root.resolved_target.?,
        .optimize = root.optimize.?,
        .link_libc = true,
        .imports = &.{
            .{ .name = "mlx_serve", .module = root.import_table.get("mlx_serve").? },
            .{ .name = "mlx_c", .module = root.import_table.get("mlx_c").? },
        },
    });
    _ = wire(b, mod, dep);
    mod.addLibraryPath(.{ .cwd_relative = mlx_lib });
    mod.addRPath(.{ .cwd_relative = mlx_lib });
    mod.linkSystemLibrary("mlxc", .{});
    mod.linkSystemLibrary("mlx", .{});
    mod.linkFramework("IOKit", .{});
    mod.linkFramework("CoreFoundation", .{});
    const t = b.addTest(.{ .name = "router-gpu-test", .root_module = mod, .filters = &.{"turboexl3_glm53f router gpu"} });
    // as the plugin's own tests: the host's symbols the root names resolve at load, and none is reached here
    t.linker_allow_shlib_undefined = true;
    const run = b.addRunArtifact(t);
    run.has_side_effects = true; // the GPU is not a cached input
    run_step.dependOn(&run.step);
    build_step.dependOn(&b.addInstallArtifact(t, .{}).step);
}

/// A device test step `name` (and `name`-build, which only compiles it): the tests whose names hold `filter`, rooted at
/// plugin.zig with wire's modules, as the plugin is, linking the server's libmlx and libmlxc and the frameworks the
/// copy's GPU core count reads (IOKit, CoreFoundation).
fn deviceTests(b: *std.Build, root: *std.Build.Module, dep: *std.Build.Dependency, mlx_lib_opt: ?[]const u8, comptime name: []const u8, comptime filter: []const u8, comptime run_what: []const u8, comptime build_what: []const u8) void {
    const run_step = b.step(name, run_what);
    const build_step = b.step(name ++ "-build", build_what);
    const mlx_lib = mlx_lib_opt orelse {
        const fail = b.addFail(name ++ " needs -Dmlx-lib=<the directory holding the server's libmlx.dylib and libmlxc.dylib>");
        run_step.dependOn(&fail.step);
        build_step.dependOn(&fail.step);
        return;
    };
    const mod = b.createModule(.{
        .root_source_file = b.path("plugin.zig"),
        .target = root.resolved_target.?,
        .optimize = root.optimize.?,
        .link_libc = true,
        .imports = &.{
            .{ .name = "mlx_serve", .module = root.import_table.get("mlx_serve").? },
            .{ .name = "mlx_c", .module = root.import_table.get("mlx_c").? },
        },
    });
    _ = wire(b, mod, dep);
    mod.addLibraryPath(.{ .cwd_relative = mlx_lib });
    mod.addRPath(.{ .cwd_relative = mlx_lib });
    mod.linkSystemLibrary("mlxc", .{});
    mod.linkSystemLibrary("mlx", .{});
    mod.linkFramework("IOKit", .{});
    mod.linkFramework("CoreFoundation", .{});
    const t = b.addTest(.{ .name = name, .root_module = mod, .filters = &.{filter} });
    // as the plugin's own tests: the host's symbols the root names resolve at load, and none is reached here
    t.linker_allow_shlib_undefined = true;
    const run = b.addRunArtifact(t);
    run.has_side_effects = true; // the GPU is not a cached input
    run_step.dependOn(&run.step);
    build_step.dependOn(&b.addInstallArtifact(t, .{}).step);
}

/// The DSA prompt texts' device checks (src/dsa_gpu_test.zig: the one-P sparse attention against the plain form,
/// bounded), as `kdaGpuTests`. The test runs under TURBOEXL3_GLM53F_DSA_GPU=1;
/// `dsa-gpu-test-build` only compiles it.
fn dsaGpuTests(b: *std.Build, root: *std.Build.Module, dep: *std.Build.Dependency, mlx_lib_opt: ?[]const u8) void {
    deviceTests(b, root, dep, mlx_lib_opt, "dsa-gpu-test", "turboexl3_glm53f dsa gpu", "Run the DSA prompt texts' checks against the chain and the plain form on the server's MLX (TURBOEXL3_GLM53F_DSA_GPU=1): -Dmlx-lib=<dir>", "Compile the DSA prompt texts' checks without running them: -Dmlx-lib=<dir>");
}

/// TURBOEXL3_GLM53F_HEAD_LAST_ROW's device check (src/draft_gpu_test.zig: the MTP bank's routed experts over one row
/// against that row of the call over every row, bit for bit), as `kdaGpuTests`. The test runs under
/// TURBOEXL3_GLM53F_DRAFT_GPU=1; `draft-gpu-test-build` only compiles it.
fn draftGpuTests(b: *std.Build, root: *std.Build.Module, dep: *std.Build.Dependency, mlx_lib_opt: ?[]const u8) void {
    deviceTests(b, root, dep, mlx_lib_opt, "draft-gpu-test", "turboexl3_glm53f draft gpu", "Run TURBOEXL3_GLM53F_HEAD_LAST_ROW's one-row routed experts against the call over every row, bit for bit, on the server's MLX (TURBOEXL3_GLM53F_DRAFT_GPU=1): -Dmlx-lib=<dir>", "Compile TURBOEXL3_GLM53F_HEAD_LAST_ROW's device check without running it: -Dmlx-lib=<dir>");
}

/// TURBOEXL3_GLM53F_MOE_COMBINE1's and TURBOEXL3_GLM53F_ROUTE_VIEW's device check (src/moe_combine_gpu_test.zig: the one
/// launch against the stock join and combine, and the ids' view against the cast, bit for bit), as `kdaGpuTests`, and
/// TURBOEXL3_GLM53F_PF_MOE_HOLD's (src/pf_hold_gpu_test.zig: the hold's arrays against the routed chain's own, and
/// PF_MOE_HOLD_TOP's hand-back from a cache over its limit). The tests run under TURBOEXL3_GLM53F_MOE_COMBINE_GPU=1;
/// `moe-combine-gpu-test-build` only compiles them.
fn moeCombineGpuTests(b: *std.Build, root: *std.Build.Module, dep: *std.Build.Dependency, mlx_lib_opt: ?[]const u8) void {
    deviceTests(b, root, dep, mlx_lib_opt, "moe-combine-gpu-test", "turboexl3_glm53f moe combine gpu", "Run TURBOEXL3_GLM53F_MOE_COMBINE1's one launch against the stock join and combine and TURBOEXL3_GLM53F_ROUTE_VIEW's view against the cast, bit for bit, on the server's MLX (TURBOEXL3_GLM53F_MOE_COMBINE_GPU=1): -Dmlx-lib=<dir>", "Compile TURBOEXL3_GLM53F_MOE_COMBINE1's and TURBOEXL3_GLM53F_ROUTE_VIEW's device check without running it: -Dmlx-lib=<dir>");
}

/// The prepared-input producers' device check (src/dec_prep_gpu_test.zig: KDA_OPREP and DSA_OPREP against the stock GLM
/// chain, bit for bit, under DEC-ONE's default GEMV and g1), as `kdaGpuTests`. The test runs under
/// TURBOEXL3_GLM53F_DEC_PREP_GPU=1; `dec-prep-gpu-test-build` only compiles it.
fn decPrepGpuTests(b: *std.Build, root: *std.Build.Module, dep: *std.Build.Dependency, mlx_lib_opt: ?[]const u8) void {
    deviceTests(b, root, dep, mlx_lib_opt, "dec-prep-gpu-test", "turboexl3_glm53f dec_prep gpu", "Run the prepared-input producers against the stock GLM chain, bit for bit, on the server's MLX (TURBOEXL3_GLM53F_DEC_PREP_GPU=1): -Dmlx-lib=<dir>", "Compile the prepared-input producers' device check without running it: -Dmlx-lib=<dir>");
}

/// TURBOEXL3_GLM53F_SMALLOP_FUSE's device check (src/smallop_gpu_test.zig: each merged piece against the stock ops it
/// replaces, bit for bit, at decode and verify rows), as `kdaGpuTests`. The test runs under
/// TURBOEXL3_GLM53F_SMALLOP_GPU=1; `smallop-gpu-test-build` only compiles it.
fn smallopGpuTests(b: *std.Build, root: *std.Build.Module, dep: *std.Build.Dependency, mlx_lib_opt: ?[]const u8) void {
    deviceTests(b, root, dep, mlx_lib_opt, "smallop-gpu-test", "turboexl3_glm53f smallop gpu", "Run TURBOEXL3_GLM53F_SMALLOP_FUSE's merged pieces against the stock ops, bit for bit, on the server's MLX (TURBOEXL3_GLM53F_SMALLOP_GPU=1): -Dmlx-lib=<dir>", "Compile TURBOEXL3_GLM53F_SMALLOP_FUSE's device check without running it: -Dmlx-lib=<dir>");
}
