//! copied_from: src/plugins/qwen4_exp/hc_fused.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! Hyper-connection reads and writes fused into custom kernels (the deferred write folded into the next read, the
//! prefill-width pending write), with their kill switches.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const hc_decode2 = @import("../../sdk/blocks/hc/hc_decode2.zig");
const KVCache = @import("../../engines/mlx/kv/cache.zig").KVCache;
const testing = @import("../../engines/mlx/kv/cache.zig").testing;
const Trunk = @import("trunk.zig").Trunk;
const attn256MaxDiff = @import("../../sdk/blocks/attention/attn256.zig").attn256MaxDiff;
const attn256RandBf16 = @import("../../sdk/blocks/attention/attn256.zig").attn256RandBf16;
const qmatmulBits = @import("../../sdk/blocks/linear/qmatmul.zig").qmatmulBits;
const verifySharedHardware = @import("../../sdk/blocks/linear/verify_projection.zig").verifySharedHardware;
const standinOnes = @import("../../sdk/diag/standin.zig").standinOnes;
const standinRef = @import("../../sdk/diag/standin.zig").standinRef;
const bf16BitEqual = @import("../../sdk/testing/parity.zig").bf16BitEqual;
const testReadF32 = @import("../../sdk/testing/parity.zig").testReadF32;
const HcWeights = @import("weights.zig").HcWeights;

// ── qwen4_exp fused hyper-connection READ at decode and verify width ──
// The read is two launches (down, then up/mix) in hc_decode2.zig for 1..HC_FUSED_MAX_ROWS rows; anything outside its
// geometry keeps the unfused chain.
var hc_fused_eps: ?mlx.mlx_array = null;
/// Test seam: false keeps the unfused chain.
pub var hc_fused_override: ?bool = null;
var hc_fused_eps_val: f32 = 0;

pub fn hcFusedEnabled() bool {
    return hc_fused_override orelse true;
}

pub const HcFusedOut = struct { mixed: mlx.mlx_array, inj: mlx.mlx_array, stream: mlx.mlx_array };

/// A deferred `hcWrite`: the next read's N kernel applies `stream + out*inj`
/// itself (one dispatch fewer per block). Handles are retained copies.
pub const HcPending = struct {
    out: mlx.mlx_array, // [B,S,hidden]
    inj: mlx.mlx_array, // [B,S,hc,1]
    pub fn deinit(self: *HcPending) void {
        _ = mlx.mlx_array_free(self.out);
        _ = mlx.mlx_array_free(self.inj);
    }
};

/// Fused decode-width hyper-connection read over `batch*seq` rows (1..
/// HC_FUSED_MAX_ROWS: decode, verify widths, batched slots — rows are
/// independent, the grid carries them, weights are read once per row).
/// `x` holds `batch*seq*hc*hidden` elements; `iw` is the row-major dense
/// `[hc*hidden, hc]` inject weight or null-ctx (mixer). With `pend`, the read
/// first applies that pending write and returns the written stream in
/// `.stream` (null-ctx otherwise). Returns `mixed [B,S,hidden]` +
/// `inj [B,S,hc,1]`, or null when the geometry/quant is outside the kernel
/// (caller keeps the chain).
pub const HC_FUSED_MAX_ROWS: c_int = 16;
pub fn hcReadJoined(a: std.mem.Allocator, s: mlx.mlx_stream, inputs: []const mlx.mlx_array, w: HcWeights, width: c_int, eps: f32, bits: u32, gs: u32, pending: ?[]const HcPending, trunk: bool) !?[]HcFusedOut {
    if (!mlx.streamIsGpu(s) or !verifySharedHardware() or inputs.len < 2 or inputs.len > 8 or width < 2) return null;
    const total = @as(c_int, @intCast(inputs.len)) * width;
    if (total > HC_FUSED_MAX_ROWS or w.inject_flat.ctx == null) return null;
    for (inputs) |input| {
        if (input.ctx == null or mlx.mlx_array_dtype(input) != .bfloat16 or !std.mem.eql(c_int, mlx.getShape(input), &.{ 1, width, 10240 })) return null;
    }
    if (pending) |values| {
        if (values.len != inputs.len) return null;
        for (values) |p| {
            if (p.out.ctx == null or p.inj.ctx == null or mlx.mlx_array_dtype(p.out) != .bfloat16 or mlx.mlx_array_dtype(p.inj) != .bfloat16 or
                !std.mem.eql(c_int, mlx.getShape(p.out), &.{ 1, width, 2560 }) or !std.mem.eql(c_int, mlx.getShape(p.inj), &.{ 1, width, 4, 1 })) return null;
        }
    }
    const Join = struct {
        fn array(st: mlx.mlx_stream, values: []const mlx.mlx_array) !mlx.mlx_array {
            const vec = mlx.mlx_vector_array_new_data(values.ptr, values.len);
            defer _ = mlx.mlx_vector_array_free(vec);
            var out = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(out);
            try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 1, st));
            return out;
        }
    };
    const joined = try Join.array(s, inputs);
    defer _ = mlx.mlx_array_free(joined);
    var joined_pending: ?HcPending = null;
    defer if (joined_pending) |p| {
        _ = mlx.mlx_array_free(p.out);
        _ = mlx.mlx_array_free(p.inj);
    };
    if (pending) |values| {
        var outs: [8]mlx.mlx_array = undefined;
        var injs: [8]mlx.mlx_array = undefined;
        for (values, 0..) |p, i| {
            outs[i] = p.out;
            injs[i] = p.inj;
        }
        const out = try Join.array(s, outs[0..values.len]);
        errdefer _ = mlx.mlx_array_free(out);
        joined_pending = .{ .out = out, .inj = try Join.array(s, injs[0..values.len]) };
    }
    const dense = w.down_rows.ctx != null;
    const full = (try hcReadFusedPack(s, joined, 1, total, w.norm_w, if (dense) w.down_rows else w.down_w, w.down_s, w.down_b, if (dense) w.up_rows else w.up_w, w.up_s, w.up_b, w.inject_flat, eps, 4, 2560, bits, gs, joined_pending, trunk)) orelse return null;
    defer inline for (.{ "mixed", "inj", "stream" }) |name| {
        if (@field(full, name).ctx != null) _ = mlx.mlx_array_free(@field(full, name));
    };
    const result = try a.alloc(HcFusedOut, inputs.len);
    @memset(result, .{ .mixed = .{}, .inj = .{}, .stream = .{} });
    errdefer {
        for (result) |r| inline for (.{ "mixed", "inj", "stream" }) |name| {
            if (@field(r, name).ctx != null) _ = mlx.mlx_array_free(@field(r, name));
        };
        a.free(result);
    }
    for (result, 0..) |*r, i| inline for (.{ "mixed", "inj", "stream" }) |name| {
        const value = @field(full, name);
        if (value.ctx != null) {
            const shape = mlx.getShape(value);
            var start: [4]c_int = @splat(0);
            var end: [4]c_int = @splat(1);
            @memcpy(end[0..shape.len], shape);
            start[1] = @as(c_int, @intCast(i)) * width;
            end[1] = start[1] + width;
            @field(r, name) = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(&@field(r, name), value, &start, shape.len, &end, shape.len, &[_]c_int{ 1, 1, 1, 1 }, shape.len, s));
        }
    };
    return result;
}

/// The epsilon operand the hc kernels bind, cached per value.
fn hcEpsArray(eps: f32) !mlx.mlx_array {
    if (hc_fused_eps == null or hc_fused_eps_val != eps) {
        if (hc_fused_eps) |e| _ = mlx.mlx_array_free(e);
        const esh = [_]c_int{1};
        var ev = eps;
        hc_fused_eps = mlx.mlx_array_new_data(&ev, &esh, 1, .float32);
        hc_fused_eps_val = eps;
    }
    return hc_fused_eps.?;
}

/// `hcReadFusedPack` on a pack whose trunk is not an EXL3 format's.
pub fn hcReadFused(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    batch: c_int,
    seq: c_int,
    nw: mlx.mlx_array,
    dw: mlx.mlx_array,
    ds: mlx.mlx_array,
    db: mlx.mlx_array,
    uw: mlx.mlx_array,
    us: mlx.mlx_array,
    ub: mlx.mlx_array,
    iw: mlx.mlx_array,
    eps: f32,
    hc: c_int,
    hidden: c_int,
    bits: u32,
    group_size: u32,
    pend: ?HcPending,
) !?HcFusedOut {
    return hcReadFusedPack(s, x, batch, seq, nw, dw, ds, db, uw, us, ub, iw, eps, hc, hidden, bits, group_size, pend, false);
}

/// The fused read on a pack whose trunk is (`trunk`, the config's `trunk_format`) or is not an EXL3 format's: an EXL3
/// trunk's decode read is `hc_decode2.readTrunk`, every other pack's `hc_decode2.read`.
pub fn hcReadFusedPack(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    batch: c_int,
    seq: c_int,
    nw: mlx.mlx_array,
    dw: mlx.mlx_array,
    ds: mlx.mlx_array,
    db: mlx.mlx_array,
    uw: mlx.mlx_array,
    us: mlx.mlx_array,
    ub: mlx.mlx_array,
    iw: mlx.mlx_array,
    eps: f32,
    hc: c_int,
    hidden: c_int,
    bits: u32,
    group_size: u32,
    pend: ?HcPending,
    trunk: bool,
) !?HcFusedOut {
    if (!hcFusedEnabled()) return null;
    if (!mlx.streamIsGpu(s) and !mlx.cudaAvailable()) return null;
    const vpw = hc_decode2.packValues(bits) orelse return null;
    const rows = batch * seq;
    if (rows < 1 or rows > HC_FUSED_MAX_ROWS) return null;
    if (@rem(@as(c_int, @intCast(group_size)), vpw) != 0) return null;
    if (bits != 16 and (ds.ctx == null or db.ctx == null or us.ctx == null or ub.ctx == null)) return null;
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16) return null;
    if (mlx.mlx_array_dtype(nw) != xd) return null;
    const K: c_int = hc * hidden;
    if (hc < 1 or hc > 8 or hidden < 8 or @rem(hidden, 8) != 0 or @rem(hidden, vpw) != 0) return null;
    if (mlx.mlx_array_size(x) != @as(usize, @intCast(rows * K)) or mlx.mlx_array_size(nw) != @as(usize, @intCast(K))) return null;
    const dsh = mlx.getShape(dw);
    const ush = mlx.getShape(uw);
    if (dsh.len != 2 or ush.len != 2) return null;
    const R: c_int = dsh[0];
    const bi: c_int = @intCast(bits);
    if (dsh[1] * 32 != K * bi) return null;
    if (ush[0] != K or ush[1] * 32 != R * bi) return null;
    const gsi: c_int = @intCast(group_size);
    if (@rem(K, gsi) != 0 or @rem(R, gsi) != 0) return null;
    // N: 256 threads per stream; D: 8 simdgroups × 32 lanes split each row's packs.
    if (@rem(hidden, 256) != 0 or @rem(@divTrunc(K, vpw), 256) != 0 or @rem(R, vpw) != 0) return null;
    const inj: c_int = @intFromBool(iw.ctx != null);
    if (pend) |pd| {
        if (mlx.mlx_array_dtype(pd.out) != xd or mlx.mlx_array_dtype(pd.inj) != xd) return null;
        if (mlx.mlx_array_size(pd.out) != @as(usize, @intCast(rows * hidden)) or mlx.mlx_array_size(pd.inj) != @as(usize, @intCast(rows * hc))) return null;
    }
    if (inj == 1) {
        if (mlx.mlx_array_dtype(iw) != xd) return null;
        const ish = mlx.getShape(iw);
        if (ish.len != 2 or ish[0] != K or ish[1] != hc) return null;
    }

    const pending: ?hc_decode2.Pending = if (pend) |pd| .{ .out = pd.out, .inj = pd.inj } else null;
    const read = if (mlx.cudaAvailable())
        try @import("../../sdk/blocks/hc/hc_cuda.zig").read(s, x, nw, dw, ds, db, uw, us, ub, iw, try hcEpsArray(eps), rows, hc, hidden, bits, group_size, pending)
    else if (trunk and rows <= hc_decode2.maxRows())
        try hc_decode2.readTrunk(s, x, nw, dw, ds, db, uw, us, ub, iw, try hcEpsArray(eps), rows, hc, hidden, bits, group_size, pending)
    else if (rows <= hc_decode2.maxRows())
        try hc_decode2.read(s, x, nw, dw, ds, db, uw, us, ub, iw, try hcEpsArray(eps), rows, hc, hidden, bits, group_size, pending)
    else
        null;
    {
        if (read) |two| {
            defer inline for (.{ "mixed", "inj", "stream" }) |name| {
                if (@field(two, name).ctx != null) _ = mlx.mlx_array_free(@field(two, name));
            };
            const xsh = mlx.getShape(x);
            var out: HcFusedOut = .{ .mixed = mlx.mlx_array_new(), .inj = .{ .ctx = null }, .stream = .{ .ctx = null } };
            errdefer inline for (.{ "mixed", "inj", "stream" }) |name| {
                if (@field(out, name).ctx != null) _ = mlx.mlx_array_free(@field(out, name));
            };
            try mlx.check(mlx.mlx_reshape(&out.mixed, two.mixed, &[_]c_int{ batch, seq, hidden }, 3, s));
            if (two.inj.ctx != null) {
                out.inj = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_reshape(&out.inj, two.inj, &[_]c_int{ batch, seq, hc, 1 }, 4, s));
            }
            if (two.stream.ctx != null) {
                out.stream = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_reshape(&out.stream, two.stream, xsh.ptr, @intCast(xsh.len), s));
            }
            return out;
        }
    }
    return null;
}

// ── tests ──
