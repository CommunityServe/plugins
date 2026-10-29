//! copied_from: src/plugins/qwen4_exp/contract_gdn.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! qwen4_exp's GDN kernels under the kernel contract: the core prework and recurrence register through the gdn area's
//! test types (`sdk/blocks/gdn/contract.zig`) as replacements, bit-equal to the packed prework and the pipelined
//! kernel on the same inputs.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const kc = @import("../../sdk/testing/kernel_contract/root.zig");
const gc = @import("../../sdk/blocks/gdn/contract.zig");
const gdn_core = @import("trunk/gdn_core.zig");

const spec = kc.spec;
const metal = kc.metal;

const core_recur_launch: metal.Launch = .{
    .name = "msv_gdn_core_recur",
    .inputs = &.{ "q", "k", "v", "g", "beta", "state_in", "T" },
    .outputs = &.{ "y", "state_out" },
    .source = gdn_core.RECUR_SOURCE,
    .header = gdn_core.RECUR_HEADER,
    // k_s, q_s [16][128] bf16, v_s [16][32] f32, g_s, b_s [16] f32, y_s [16][32] bf16
    .tg_bytes = gdn_core.recurTgBytes(128, 2, 2),
    .m1 = 1024,
    .m2 = 1024,
};

const core_prework_launch: metal.Launch = .{
    .name = "msv_gdn_core_prework",
    .inputs = &.{ "qkv", "conv_state", "conv_w", "q_scale", "k_scale", "b_in", "a_in", "A_log", "dt_bias", "sigtab", "seq", "qk_eps" },
    .outputs = &.{ "q_out", "k_out", "v_out", "conv_out", "g_out", "beta_out" },
    .source = gdn_core.PREWORK_SOURCE,
    .header = gdn_core.PREWORK_HEADER,
    .tg_bytes = 0,
    .m1 = 1024,
    .m2 = 1024,
};

const core_geo: gc.CoreGeo = .{ .tb = gdn_core.TB, .db = gdn_core.DB, .kqf = gdn_core.KQF, .rpt = gdn_core.RPT, .nt = gdn_core.NT };

pub fn register(set: *kc.Set) !void {
    const a = set.a;
    const f32s = mlx.mlx_dtype.float32;
    const replaces_recur = "gated_delta_pipelined_tb12 (the pipelined recurrence)";
    {
        var sp = gc.recurSpec("msv_gdn_core_recur bf16, f32 state (y)", .y, f32s, gc.prefill_t, gdn_core.TB);
        sp.replaces = replaces_recur;
        try set.add(gc.Recur, .initCore(a, &core_recur_launch, core_geo, f32s, .y), sp, null);
    }
    {
        var sp = gc.recurSpec("msv_gdn_core_recur bf16, f32 state (state)", .states, f32s, gc.prefill_t, gdn_core.TB);
        sp.replaces = replaces_recur;
        try set.add(gc.Recur, .initCore(a, &core_recur_launch, core_geo, f32s, .states), sp, null);
    }
    {
        var sp = gc.recurSpec("msv_gdn_core_recur bf16, bf16 state", .both, .bfloat16, gc.prefill_t, gdn_core.TB);
        sp.replaces = replaces_recur;
        try set.add(gc.Recur, .initCore(a, &core_recur_launch, core_geo, .bfloat16, .both), sp, null);
    }
    const pw = struct {
        const act: gc.Cfg = .{ .kind = .prework, .st = .float32, .reads = .act, .tab = true, .core = true, .run_rows = gdn_core.PW_ROWS, .nsg = gdn_core.PW_NSG };
        const wide: gc.Cfg = .{ .kind = .prework, .st = .float32, .reads = .wide, .tab = true, .core = true, .run_rows = gdn_core.PW_ROWS, .nsg = gdn_core.PW_NSG };
        const bf16: gc.Cfg = .{ .kind = .prework, .tab = true, .core = true, .run_rows = gdn_core.PW_ROWS, .nsg = gdn_core.PW_NSG };
    };
    inline for (.{ .{ "msv_gdn_core_prework bf16, f32 gate (bf16 outputs)", pw.act }, .{ "msv_gdn_core_prework bf16, f32 gate (g)", pw.wide }, .{ "msv_gdn_core_prework bf16, bf16 gate", pw.bf16 } }) |r| {
        var sp = gc.decSpec(r[0], r[1], gc.tokens(&.{ 17, 33, 64 }));
        sp.replaces = "gdn_prework2 (the packed prework)";
        try set.add(gc.Dec, .init(a, &core_prework_launch, r[1]), sp, null);
    }
}

// ── tests ──

const testing = std.testing;

/// Runs a fixture's declared launch with every output kept.
fn launchAll(rig: *metal.Rig, l: *const metal.Launch, geo: metal.Geometry, args: []const mlx.mlx_array) !void {
    var g = geo;
    g.unread = 0;
    try rig.launch(l, g, args);
}

fn expectSameBytes(rig: *metal.Rig, ours: mlx.mlx_array, theirs: mlx.mlx_array) !void {
    const a = testing.allocator;
    const x = try a.dupe(u8, try rig.bytesOf(&.{ours}));
    defer a.free(x);
    try testing.expectEqualSlices(u8, x, try rig.bytesOf(&.{theirs}));
}

test "kernel contract qwen4_exp gdn: each declared launch is the shipped wrapper's, byte for byte" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const a = testing.allocator;
    var set = kc.Set.init(a);
    defer set.deinit();
    try register(&set);
    for (set.kernels.items) |k| {
        const shape = k.spec.served[k.spec.served.len - 1];
        if (std.mem.startsWith(u8, k.spec.name, "msv_gdn_core_recur")) {
            const t: *gc.Recur = @ptrCast(@alignCast(k.ctx));
            try testing.expect(try t.prepare(shape, .random, 7));
            try launchAll(&t.rig, t.launch, t.geometry(shape).?, t.args[0..7]);
            const g = t.args;
            const r = try gdn_core.recur(t.rig.stream(), g[0], g[1], g[2], g[3], g[4], g[5], 16, 16, 128, 128, .bfloat16);
            defer for ([_]mlx.mlx_array{ r.y, r.state }) |x| {
                _ = mlx.mlx_array_free(x);
            };
            try expectSameBytes(&t.rig, t.rig.output(0), r.y);
            try expectSameBytes(&t.rig, t.rig.output(1), r.state);
        } else {
            const t: *gc.Dec = @ptrCast(@alignCast(k.ctx));
            try testing.expect(try t.prepare(shape, .random, 7));
            try launchAll(&t.rig, t.launch, t.geometry(shape).?, t.args[0..t.n_args]);
            const g = t.args;
            const pre = (try gdn_core.prework(t.rig.stream(), .{ .qkv = g[0], .qkv_off = 0, .qkv_stride = mlx.getShape(g[0])[2], .b = g[5], .b_off = 0, .b_stride = mlx.getShape(g[5])[2], .a = g[6], .a_off = 0, .a_stride = mlx.getShape(g[6])[2], .A_log = g[7], .dt_bias = g[8], .conv_state = g[1], .conv_w = g[2], .q_scale = g[3], .k_scale = g[4], .hk = 16, .hv = 32, .dk = 128, .dv = 128, .seq = @intCast(shape.rows), .gate = true, .g_dtype = t.cfg.gt() })) orelse return error.WrapperDeclined;
            const theirs = [_]mlx.mlx_array{ pre.q, pre.k, pre.v, pre.conv_state, pre.g, pre.beta };
            defer for (theirs) |x| {
                _ = mlx.mlx_array_free(x);
            };
            for (theirs, 0..) |x, i| try expectSameBytes(&t.rig, t.rig.output(i), x);
        }
    }
}
