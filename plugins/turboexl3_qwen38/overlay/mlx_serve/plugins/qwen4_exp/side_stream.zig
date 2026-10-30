//! copied_from: src/plugins/qwen4_exp/side_stream.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! The side stream: independent prefill work recorded on a second MLX GPU stream (its own MTLCommandQueue). The ops and
//! the graph are the one-stream chain's, so the bytes are too; MLX fences each edge between the two queues with an
//! MTLSharedEvent. Parts: `shared`, the gated shared expert beside the routed experts (`addGatedShared`); `gdnz`, a GDN
//! layer's z projection beside its qkv chain (`trunk/gdn.zig`). On an EXL3 trunk pack at prefill rows only (`MIN_ROWS`),
//! decided per trunk: an MLX trunk loaded in the same process stays on one stream.
const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../../sdk/util/log.zig");
const shared = @import("../../sdk/util/shared.zig");
const Trunk = @import("trunk.zig").Trunk;
/// Calls recorded on the side stream (the tests' proof that a part engaged).
pub var side_stream_calls: u64 = 0;
pub const SideStream = struct {
    pub const Parts = packed struct(u8) { shared: bool = false, gdnz: bool = false, _pad: u6 = 0 };
    pub const Part = enum { shared, gdnz };
    /// The narrowest call that takes the side stream: a prefill chunk. A decode or verify width stays on one stream: an
    /// event round trip per layer costs more than it hides.
    pub const MIN_ROWS: usize = 512;
    /// Test seams: the parts in force (every trunk's) and the row floor.
    pub var override: ?Parts = null;
    pub var min_rows_override: ?usize = null;
    var engaged = false;
    /// While the wavefront's later half is built (`wave.zig`): its own side stream (`own`), or none (`main`: the part
    /// stays on that half's stream). `off` everywhere else.
    pub const WaveRoute = enum { off, own, main };
    /// A thread's streams: MLX registers a stream's command encoder on the thread that creates it, so each inference
    /// thread makes its own side stream, and the wavefront's later half its own. `failed`: MLX made none on this thread
    /// (it is not asked again).
    pub const Tls = struct { stream: mlx.mlx_stream = .{ .ctx = null }, wave_route: WaveRoute = .off, wave_side: mlx.mlx_stream = .{ .ctx = null }, failed: bool = false };
    pub const tls_shared = shared.SharedThreadLocal(Tls, "qwen4_side_tls", .{});
    /// The parts a trunk's pack serves: both on an EXL3 trunk pack (`Formats.profile`), the configuration they were
    /// measured on; none on any other.
    pub fn partsFor(profile: bool) Parts {
        return if (profile) .{ .shared = true, .gdnz = true } else .{};
    }
    /// The parts in force for trunk `t`: its own pack's, so two models in one process each keep theirs.
    pub fn active(t: *const Trunk) Parts {
        return override orelse partsFor(t.formats.profile);
    }
    /// The second stream on `main`'s device (made once per thread), the wavefront half's own where one is building, or
    /// null when MLX makes none.
    pub fn get(main: mlx.mlx_stream) ?mlx.mlx_stream {
        const tls = tls_shared.get();
        switch (tls.wave_route) {
            .off => {},
            .own => return if (tls.wave_side.ctx != null) tls.wave_side else null,
            .main => return null,
        }
        if (tls.stream.ctx != null) return tls.stream;
        if (tls.failed) return null;
        tls.stream = newStream(main) orelse {
            tls.failed = true;
            log.warn("[side-stream] MLX made no second GPU stream; every part stays on one stream\n", .{});
            return null;
        };
        return tls.stream;
    }
    /// The side stream for `part` on a call of `rows` rows recorded on trunk `t`'s stream, or null: the part off for
    /// `t` (`active`), the call narrower than the floor, or `t.s` not a GPU stream. A decision only: `note` counts what
    /// was recorded on it.
    pub fn forPart(part: Part, t: *const Trunk, rows: usize) ?mlx.mlx_stream {
        if (rows < (min_rows_override orelse MIN_ROWS)) return null;
        const p = active(t);
        const on = switch (part) {
            .shared => p.shared,
            .gdnz => p.gdnz,
        };
        if (!on or !mlx.streamIsGpu(t.s)) return null;
        return get(t.s);
    }
    /// A call recorded a part on the side stream (`how` names the chain): the count and, once, the line.
    pub fn note(how: []const u8, rows: usize) void {
        side_stream_calls +%= 1;
        if (engaged) return;
        engaged = true;
        log.info("[side-stream] engaged: {s} on a second GPU stream from {d} rows (first call {d} rows)\n", .{ how, min_rows_override orelse MIN_ROWS, rows });
    }
    /// The rows of `x`: every dim but the last.
    pub fn rowsOf(x: mlx.mlx_array) usize {
        const sh = mlx.getShape(x);
        var rows: usize = 1;
        if (sh.len > 1) {
            for (sh[0 .. sh.len - 1]) |d| rows *= @intCast(d);
        }
        return rows;
    }
};
/// A new stream on `on`'s device, or null.
pub fn newStream(on: mlx.mlx_stream) ?mlx.mlx_stream {
    var dev = mlx.mlx_device{ .ctx = null };
    if (mlx.mlx_stream_get_device(&dev, on) != 0) return null;
    defer _ = mlx.mlx_device_free(dev);
    const st = mlx.mlx_stream_new_device(dev);
    return if (st.ctx != null) st else null;
}
// ── tests ──
const testing = std.testing;
test "side stream: a part takes a call only on an EXL3 trunk at a prefill width, per trunk; the stream is made once" {
    defer {
        SideStream.override = null;
        SideStream.min_rows_override = null;
    }
    const gpu = mlx.gpuStream();
    // an MLX trunk and an EXL3 trunk in one process: each keeps its own parts, whichever loaded last
    var plain: Trunk = undefined;
    plain.s = gpu;
    plain.formats = .{};
    var exl3: Trunk = undefined;
    exl3.s = gpu;
    exl3.formats = .{ .profile = true };
    try testing.expectEqual(SideStream.Parts{}, SideStream.active(&plain));
    try testing.expectEqual(SideStream.Parts{ .shared = true, .gdnz = true }, SideStream.active(&exl3));
    try testing.expect(SideStream.forPart(.shared, &plain, 8192) == null);
    try testing.expect(SideStream.forPart(.gdnz, &plain, 8192) == null);
    try testing.expect(SideStream.forPart(.shared, &exl3, SideStream.MIN_ROWS - 1) == null);
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    exl3.s = cpu;
    try testing.expect(SideStream.forPart(.gdnz, &exl3, 8192) == null);
    exl3.s = gpu;
    SideStream.override = .{ .gdnz = true };
    try testing.expect(SideStream.forPart(.shared, &exl3, 8192) == null);
    if (!mlx.streamIsGpu(gpu)) return;
    const side = SideStream.forPart(.gdnz, &exl3, 8192) orelse return error.SideStreamMissing;
    try testing.expect(!mlx.mlx_stream_equal(side, gpu));
    const again = SideStream.forPart(.gdnz, &exl3, SideStream.MIN_ROWS) orelse return error.SideStreamMissing;
    try testing.expect(mlx.mlx_stream_equal(side, again));
}
