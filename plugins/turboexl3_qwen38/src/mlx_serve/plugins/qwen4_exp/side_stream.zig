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
const MoeMlpWeights = @import("../../sdk/blocks/moe/weights.zig").MoeMlpWeights;
const lastDim = @import("../../engines/mlx/load/weights.zig").lastDim;
const qmatmulBits = @import("../../sdk/blocks/linear/qmatmul.zig").qmatmulBits;
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

/// `moeAddGatedShared` where the `shared` part takes the call: the shared expert, its gate's logit, sigmoid and product
/// on the side stream beside the routed launches, the Add that joins the routed sum on the trunk's stream (one fence
/// back). Its ops on its operands, so its bytes.
pub fn addGatedShared(t: *Trunk, expert_sum: mlx.mlx_array, expert_x: mlx.mlx_array, mw: *const MoeMlpWeights) !mlx.mlx_array {
    const rows = SideStream.rowsOf(expert_x);
    const side = SideStream.forPart(.shared, t, rows) orelse return t.moeAddGatedShared(expert_sum, expert_x, mw, null);
    const main = t.s;
    const gated = blk: {
        t.s = side;
        defer t.s = main;
        const down = (try t.moeSharedDown(.qmatmul, expert_x, mw, &.{})).?;
        defer _ = mlx.mlx_array_free(down);
        const seg_w = mw.shared_expert_gate_w.?;
        const seg_qp = t.quantParamsHinted(seg_w, mw.shared_expert_gate_s.?, lastDim(expert_x));
        const logit = try qmatmulBits(expert_x, seg_w, mw.shared_expert_gate_s.?, mw.shared_expert_gate_b.?, seg_qp.bits, seg_qp.group_size, seg_qp.mode, side);
        defer _ = mlx.mlx_array_free(logit);
        var sig = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sig);
        try mlx.check(mlx.mlx_sigmoid(&sig, logit, side));
        var g = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(g);
        try mlx.check(mlx.mlx_multiply(&g, sig, down, side));
        break :blk g;
    };
    defer _ = mlx.mlx_array_free(gated);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_add(&out, expert_sum, gated, main));
    SideStream.note("shared", rows);
    return out;
}

// ── tests ──

const testing = std.testing;

/// `shape` of bf16 values uniform in [-scale, scale), evaluated.
fn randBf16(rnd: std.Random, shape: []const c_int, scale: f32, s: mlx.mlx_stream) !mlx.mlx_array {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const buf = try testing.allocator.alloc(f32, n);
    defer testing.allocator.free(buf);
    for (buf) |*v| v.* = (rnd.float(f32) * 2 - 1) * scale;
    const f = mlx.mlx_array_new_data(buf.ptr, shape.ptr, @intCast(shape.len), .float32);
    defer _ = mlx.mlx_array_free(f);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_astype(&out, f, .bfloat16, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

fn sameBits(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !bool {
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_array_equal(&eq, a, b, false, s));
    try mlx.check(mlx.mlx_array_eval(eq));
    var same = false;
    try mlx.check(mlx.mlx_array_item_bool(&same, eq));
    return same;
}
