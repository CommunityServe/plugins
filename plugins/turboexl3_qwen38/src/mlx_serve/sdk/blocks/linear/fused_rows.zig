//! copied_from: src/sdk/blocks/linear/fused_rows.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! sdk/blocks/linear/fused_rows.
//! Owns FusedQkv, fused_qkv_engaged, fused_qkv_override, fused_qkv_env, fusedQkvEnabled, FusedRows, and 7 more.

const std = @import("std");
const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");
const Weights = @import("../../weights/resident.zig").Weights;
const testing = std.testing;
const transposeBf16Weight = @import("../../../mlx/array.zig").transposeBf16Weight;
const QuantParams = @import("../../quant/params.zig").QuantParams;
const testReadF32 = @import("../../testing/parity.zig").testReadF32;

/// Quantized projections that read the same input, joined along their
/// output rows so one matmul serves them all. `widths[i]` is part i's row
/// count. A null `w` means the group was not built.
pub const FusedRows = struct {
    w: mlx.mlx_array = .{ .ctx = null },
    s: mlx.mlx_array = .{ .ctx = null },
    b: mlx.mlx_array = .{ .ctx = null },
    widths: [4]c_int = @splat(0),
    count: u8 = 0,

    /// Rows [start, start+width) of the last axis of a matmul result.
    pub fn part(self: *const FusedRows, out: *mlx.mlx_array, y: mlx.mlx_array, i: usize, s: mlx.mlx_stream) !void {
        var start: c_int = 0;
        for (self.widths[0..i]) |w| start += w;
        try sliceQkvPart(out, y, start, self.widths[i], s);
    }
};

/// A fused row group serves DECODE width only: at M == 1 every row is the
/// same per-row dot product whatever the output width, while from M == 2 the
/// verify lanes (`vqmmLaneForTile`) and MLX's own split-K pick kernels by N,
/// so the joined matrix would reduce K in another order. Pinned by the
/// `fused row projections` test.
pub const FUSED_ROWS_MAX_M: c_int = 1;

/// Groups joined by the loader in flight (reported once per load).
pub var fused_row_groups: usize = 0;

/// Join up to four quantized `{w, s, b}` triples along axis 0 into one
/// evaluated buffer and rewrite each triple to a zero-copy row VIEW of it,
/// so no weight is resident twice. Null (nothing changed) unless every part
/// shares one packed geometry: same packed width and dtype, same scales
/// width and dtype, same bias presence — the single matmul must dequantize
/// them all the same way.
pub fn fuseRowGroup(parts: []const [3]*mlx.mlx_array, s: mlx.mlx_stream) !?FusedRows {
    if (parts.len < 2 or parts.len > 4) return null;
    const w0 = parts[0][0].*;
    const s0 = parts[0][1].*;
    if (w0.ctx == null or s0.ctx == null) return null;
    const have_b = parts[0][2].ctx != null;
    const wsh0 = mlx.getShape(w0);
    const ssh0 = mlx.getShape(s0);
    if (wsh0.len != 2 or ssh0.len != 2) return null;
    var widths: [4]c_int = @splat(0);
    for (parts, 0..) |p, i| {
        const w = p[0].*;
        const sc = p[1].*;
        if (w.ctx == null or sc.ctx == null) return null;
        if ((p[2].ctx != null) != have_b) return null;
        const wsh = mlx.getShape(w);
        const ssh = mlx.getShape(sc);
        if (wsh.len != 2 or ssh.len != 2) return null;
        if (wsh[1] != wsh0[1] or ssh[1] != ssh0[1] or wsh[0] != ssh[0]) return null;
        if (mlx.mlx_array_dtype(w) != mlx.mlx_array_dtype(w0) or mlx.mlx_array_dtype(sc) != mlx.mlx_array_dtype(s0)) return null;
        if (have_b) {
            const bsh = mlx.getShape(p[2].*);
            if (bsh.len != 2 or bsh[0] != ssh[0] or bsh[1] != ssh[1]) return null;
            if (mlx.mlx_array_dtype(p[2].*) != mlx.mlx_array_dtype(parts[0][2].*)) return null;
        }
        widths[i] = wsh[0];
    }
    const join = struct {
        fn run(ps: []const [3]*mlx.mlx_array, which: usize, str: mlx.mlx_stream) !mlx.mlx_array {
            var arrs: [4]mlx.mlx_array = undefined;
            for (ps, 0..) |p, i| arrs[i] = p[which].*;
            const vec = mlx.mlx_vector_array_new_data(&arrs, ps.len);
            defer _ = mlx.mlx_vector_array_free(vec);
            var joined = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(joined);
            try mlx.check(mlx.mlx_concatenate_axis(&joined, vec, 0, str));
            var out = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(out);
            try mlx.check(mlx.mlx_contiguous(&out, joined, false, str));
            try mlx.check(mlx.mlx_array_eval(out));
            return out;
        }
    }.run;
    var fused = FusedRows{ .widths = widths, .count = @intCast(parts.len) };
    fused.w = try join(parts, 0, s);
    errdefer _ = mlx.mlx_array_free(fused.w);
    fused.s = try join(parts, 1, s);
    errdefer _ = mlx.mlx_array_free(fused.s);
    if (have_b) fused.b = try join(parts, 2, s);
    errdefer if (have_b) {
        _ = mlx.mlx_array_free(fused.b);
    };
    // Row views: the map's originals are released with the map, the views
    // keep the joined buffer alive.
    var start: c_int = 0;
    for (parts, 0..) |p, i| {
        const stop = start + widths[i];
        const srcs = [_]mlx.mlx_array{ fused.w, fused.s, fused.b };
        for (srcs, 0..) |src, which| {
            if (src.ctx == null) continue;
            const cols = mlx.getShape(src)[1];
            const lo = [_]c_int{ start, 0 };
            const hi = [_]c_int{ stop, cols };
            const st = [_]c_int{ 1, 1 };
            var view = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(&view, src, &lo, 2, &hi, 2, &st, 2, s));
            p[which].* = view;
        }
        start = stop;
    }
    return fused;
}

/// `fuseRowGroup` plus ownership: the joined buffers go to `owned`; each
/// part's original is released — map parts (`names`, from index `name_from`)
/// have their entries swapped for the view, handle parts (Spark's split
/// copies, `parts[0..name_from]`) are dropped from `owned` and their views
/// parked there instead. Returns an empty group when nothing was fused.
pub fn fuseRowsInPlace(
    parts: [][3]*mlx.mlx_array,
    name_from: usize,
    names: []const []const u8,
    weights: *Weights,
    name_buf: *[256]u8,
    prefix: []const u8,
    li: u32,
    owned: *std.ArrayList(mlx.mlx_array),
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
) !FusedRows {
    var originals: [4][3]mlx.mlx_array = undefined;
    for (parts, 0..) |p, i| originals[i] = .{ p[0].*, p[1].*, p[2].* };
    const leaves = [_][]const u8{ "weight", "scales", "biases" };
    // Only map-owned handles are fused: a part the map does not hold under
    // its name (a materialized copy, another spelling) would stay resident
    // beside the joined buffer.
    for (parts[name_from..], name_from..) |p, i| {
        for (0..3) |which| {
            if (p[which].*.ctx == null) continue;
            const name = std.fmt.bufPrint(name_buf, "{s}.layers.{d}.{s}.{s}", .{ prefix, li, names[i - name_from], leaves[which] }) catch unreachable;
            const held = weights.get(name) orelse return .{};
            if (held.ctx != p[which].*.ctx) return .{};
        }
    }
    const fused = (try fuseRowGroup(parts, s)) orelse return .{};
    fused_row_groups += 1;
    try owned.append(allocator, fused.w);
    try owned.append(allocator, fused.s);
    if (fused.b.ctx != null) try owned.append(allocator, fused.b);
    for (parts, 0..) |p, i| {
        for (0..3) |which| {
            const view = p[which].*;
            if (view.ctx == null) continue;
            if (i < name_from) {
                try owned.append(allocator, view);
                for (owned.items, 0..) |a, j| {
                    if (a.ctx == originals[i][which].ctx) {
                        _ = mlx.mlx_array_free(a);
                        _ = owned.swapRemove(j);
                        break;
                    }
                }
            } else {
                const name = std.fmt.bufPrint(name_buf, "{s}.layers.{d}.{s}.{s}", .{ prefix, li, names[i - name_from], leaves[which] }) catch unreachable;
                weights.replace(name, view);
            }
        }
    }
    return fused;
}


/// Slice `[B, S, total]` back into the three projections. At decode (B*S == 1)
/// every slice is contiguous memory; at prefill they are strided views whose
/// later reshape to [B,S,H,D] splits the stride-1 axis, which MLX expresses as
/// a view rather than a copy.
pub fn sliceQkvPart(out: *mlx.mlx_array, qkv: mlx.mlx_array, start: c_int, width: c_int, s: mlx.mlx_stream) !void {
    const sh = mlx.getShape(qkv);
    var lo: [3]c_int = .{ 0, 0, start };
    var hi: [3]c_int = .{ sh[0], sh[1], start + width };
    const st = [_]c_int{ 1, 1, 1 };
    try mlx.check(mlx.mlx_slice(out, qkv, &lo, 3, &hi, 3, &st, 3, s));
}

/// A projection triple (`scales`/`biases` null-ctx when dense). Borrowed handles.
pub const QW = struct {
    w: mlx.mlx_array = .{ .ctx = null },
    s: mlx.mlx_array = .{ .ctx = null },
    b: mlx.mlx_array = .{ .ctx = null },
};

// ── tests ──

const qmatmulBits = @import("qmatmul.zig").qmatmulBits;
