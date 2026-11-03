//! The Neural Engine test of `src/ane_shared.zig` (`SUSHI_DS41_ANE_SHARED`'s programs) against a host SwiGLU over the same int8
//! weights. A root of `zig build gpu-test` only (through `gpu_test.zig`); skipped where the runtime has no ANE. It needs no
//! model: the load-time self-check (`aneSelfCheck`) compares the real pack's layer 0 with the GPU chain.
const std = @import("std");
const ane = @import("ane_shared.zig");

const H = 256;
const F = 128;
const CHUNK = 128;
const LIMIT: f32 = 10;

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

fn fill(r: std.Random, q: []i8, s: []f32, scale: f32) void {
    for (q) |*v| v.* = r.intRangeAtMost(i8, -127, 127);
    for (s) |*v| v.* = scale * (0.5 + r.float(f32));
}

/// Row `row` of the int8 SwiGLU, `x` `[H]`, into `y` `[H]`, as `ane_mlp.m` builds it: gate <= limit, |up| <= limit.
fn reference(x: []const f32, gq: []const i8, gs: []const f32, uq: []const i8, us: []const f32, dq: []const i8, ds: []const f32, y: []f32) void {
    var act: [F]f32 = undefined;
    for (0..F) |f| {
        var g: f32 = 0;
        var u: f32 = 0;
        for (0..H) |c| {
            g += @as(f32, @floatFromInt(gq[f * H + c])) * x[c];
            u += @as(f32, @floatFromInt(uq[f * H + c])) * x[c];
        }
        g = @min(g * gs[f], LIMIT);
        u = std.math.clamp(u * us[f], -LIMIT, LIMIT);
        act[f] = g / (1 + @exp(-g)) * u;
    }
    for (0..H) |o| {
        var v: f32 = 0;
        for (0..F) |f| v += @as(f32, @floatFromInt(dq[o * F + f])) * act[f];
        y[o] = v * ds[o];
    }
}

test "sushi_ds41 gpu: the ANE shared-expert programs give the host's int8 SwiGLU on both units, through run and start/join" {
    if (std.c.getenv("SUSHI_DS41_GPU_TEST") == null) return error.SkipZigTest;
    if (!ane.available()) return error.SkipZigTest;
    // The bridge's compile cache lives under NSHomeDirectory()/.mlx-serve/ane-cache: point that home at the temp dir so
    // the test never writes into the user's server cache. CF reads the override when it is asked for the home.
    if (std.c.getenv("CFFIXED_USER_HOME") == null) {
        const tmp = std.c.getenv("TMPDIR") orelse "/private/tmp";
        var buf: [1024]u8 = undefined;
        const home = try std.fmt.bufPrintSentinel(&buf, "{s}/sushi-ds41-ane-test", .{std.mem.span(tmp)}, 0);
        if (setenv("CFFIXED_USER_HOME", home.ptr, 1) != 0) return error.SkipZigTest;
    }
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(41);
    const r = prng.random();
    var gq: [F * H]i8 = undefined;
    var uq: [F * H]i8 = undefined;
    var dq: [H * F]i8 = undefined;
    var gs: [F]f32 = undefined;
    var us: [F]f32 = undefined;
    var ds: [H]f32 = undefined;
    fill(r, &gq, &gs, 1.0 / (127.0 * @sqrt(@as(f32, H))));
    fill(r, &uq, &us, 1.0 / (127.0 * @sqrt(@as(f32, H))));
    fill(r, &dq, &ds, 1.0 / (127.0 * @sqrt(@as(f32, F))));

    var progs = try ane.Programs.init(2, CHUNK, H, F);
    defer progs.deinit();
    // Layer 3, so the procedure lookup by layer is exercised, not just index 0.
    try progs.addLayer(3, &gq, &gs, &uq, &us, &dq, &ds, LIMIT);
    try progs.finish();
    const R = progs.rows;

    // Rows of the chunk, and the units' channel-major slabs (unit u takes rows u * R ..).
    const x = try gpa.alloc(f32, CHUNK * H);
    defer gpa.free(x);
    for (x) |*v| v.* = @floatCast(@as(f16, @floatCast(r.floatNorm(f32))));
    const slabs = try gpa.alloc(f16, CHUNK * H);
    defer gpa.free(slabs);
    for (0..CHUNK) |row| for (0..H) |c| {
        const u = row / R;
        slabs[u * H * R + c * R + (row - u * R)] = @floatCast(x[row * H + c]);
    };
    const want = try gpa.alloc(f32, CHUNK * H);
    defer gpa.free(want);
    for (0..CHUNK) |row| reference(x[row * H ..][0..H], &gq, &gs, &uq, &us, &dq, &ds, want[row * H ..][0..H]);

    for (0..2) |pass| {
        if (pass == 0) {
            try std.testing.expect(progs.run(3, slabs.ptr).ok);
        } else {
            // The early path's order: the planes written first (as the GPU pack does), then a null input.
            // Pass 0's outputs cleared, so a stale plane cannot pass.
            for (0..progs.n_units) |u| {
                @memcpy(progs.input(u)[0 .. H * R], slabs[u * H * R ..][0 .. H * R]);
                @memset(@constCast(progs.output(u))[0 .. H * R], 0);
            }
            var pend: ane.Programs.Pending = .{};
            progs.start(&pend, 3, null);
            try std.testing.expect(pend.join().ok);
        }
        var dot: f64 = 0;
        var nw: f64 = 0;
        var ng: f64 = 0;
        for (0..CHUNK) |row| for (0..H) |c| {
            const u = row / R;
            const got: f64 = @floatCast(progs.output(u)[c * R + (row - u * R)]);
            const w: f64 = want[row * H + c];
            dot += got * w;
            nw += w * w;
            ng += got * got;
        };
        const cos = dot / @sqrt(nw * ng);
        // M5 Max, 2026-11-03: cos 0.999885 and a norm ratio of 1.0000 on both passes.
        if (!(cos > 0.999)) {
            std.debug.print("ane shared programs (pass {d}): cos {d:.6}, |ane|/|host| {d:.4}\n", .{ pass, cos, @sqrt(ng / nw) });
            return error.TestUnexpectedResult;
        }
    }
}
