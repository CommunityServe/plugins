//! copied_from: src/plugins/qwen4_exp/trunk/gdn_ab.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! The GDN layer's in_proj_a and in_proj_b as one GEMM: at load the bf16 pair becomes views of one buffer [a | 0 | b]
//! (`joinInPlace`), and prefill rows project both in one MLX matmul (`project`) whose a and b columns are the two
//! products' bits. Each part starts at a multiple of 32 with zero rows to its slot's end (where its own product loads
//! zeros past N), so every slot of the joined product holds the operands that slot holds in the part's own product,
//! and the bits match wherever MLX walks K alike for both widths: the same split-K decision (it reads max(M, N)) and,
//! when it splits, a partition K alone fixes (K <= 4096). `sameChains` is that test; without NAX (SIMD split-K
//! partitions sized by N) the pair keeps its two products.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../sdk/util/log.zig");
const model = @import("../../../sdk/config/model.zig");
const Trunk = @import("../trunk.zig").Trunk;
const LinearAttnWeights = @import("../../../sdk/blocks/gdn/gated_delta_net.zig").LinearAttnWeights;
const GdnAb = @import("../../../sdk/blocks/gdn/gated_delta_net.zig").GdnAb;
const transposeBf16Weight = @import("../../../mlx/array.zig").transposeBf16Weight;
const sliceQkvPart = @import("../../../sdk/blocks/linear/fused_rows.zig").sliceQkvPart;
const naxAvailable = @import("../../../mlx/device.zig").naxAvailable;

var engaged_logged = false;
var declined_logged = false;

/// Where a join of parts `wa` and `wb` wide puts b, and its width: b at the first multiple of 32 at or past a's end
/// (MLX's NAX simdgroup tile is 32 columns wide), zero columns between.
pub fn layout(wa: c_int, wb: c_int) struct { b_off: c_int, n: c_int } {
    const b_off = @divTrunc(wa + 31, 32) * 32;
    return .{ .b_off = b_off, .n = b_off + wb };
}

/// MLX's NAX split-K pick for [m, k] x [k, n] (steel_matmul_axpby's Case 2), batch collapsed into m.
pub fn splitK(m: c_int, n: c_int, k: c_int) bool {
    const mx = @max(m, n);
    return k >= 3 * mx or (mx <= 1024 and k > 2 * mx);
}

/// Whether MLX walks K alike for the joined product [m, k] x [k, n] and each part's [m, k] x [k, w] (bf16 on a NAX
/// GPU): past the gemv routes (m >= 16), the same split-K decision, and a partition K alone fixes when it splits.
pub fn sameChains(m: c_int, k: c_int, widths: []const c_int, n: c_int) bool {
    if (m < 16 or k < 1) return false;
    const split = splitK(m, n, k);
    if (split and k > 4096) return false;
    for (widths) |w| if (splitK(m, w, k) != split) return false;
    return true;
}

pub const Joined = struct { joined: mlx.mlx_array, a: mlx.mlx_array, b: mlx.mlx_array, b_off: c_int, n: c_int };

/// The joined buffer of row-major parts a [wa, k] and b [wb, k] (one 16-bit float dtype): [a | 0 | b] at `layout`'s
/// offsets, contiguous and evaluated, and each part as a row view of it (the caller owns the three arrays).
pub fn joinRows(a_rows: mlx.mlx_array, b_rows: mlx.mlx_array, s: mlx.mlx_stream) !Joined {
    const ash = mlx.getShape(a_rows);
    const bsh = mlx.getShape(b_rows);
    const dt = mlx.mlx_array_dtype(a_rows);
    if (ash.len != 2 or bsh.len != 2 or ash[1] != bsh[1] or ash[0] < 1 or bsh[0] < 1) return error.GdnAbShape;
    if ((dt != .bfloat16 and dt != .float16) or mlx.mlx_array_dtype(b_rows) != dt) return error.GdnAbShape;
    const wa = ash[0];
    const k = ash[1];
    const lay = layout(wa, bsh[0]);
    var pad = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pad);
    var pieces: [3]mlx.mlx_array = .{ a_rows, b_rows, .{ .ctx = null } };
    var count: usize = 2;
    if (lay.b_off > wa) {
        try mlx.check(mlx.mlx_zeros(&pad, &[_]c_int{ lay.b_off - wa, k }, 2, dt, s));
        pieces = .{ a_rows, pad, b_rows };
        count = 3;
    }
    const vec = mlx.mlx_vector_array_new_data(&pieces, count);
    defer _ = mlx.mlx_vector_array_free(vec);
    var cat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cat);
    try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, 0, s));
    var joined = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(joined);
    try mlx.check(mlx.mlx_contiguous(&joined, cat, false, s));
    try mlx.check(mlx.mlx_array_eval(joined));
    var a = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(a);
    try mlx.check(mlx.mlx_slice(&a, joined, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ wa, k }, 2, &[_]c_int{ 1, 1 }, 2, s));
    var b = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(b);
    try mlx.check(mlx.mlx_slice(&b, joined, &[_]c_int{ lay.b_off, 0 }, 2, &[_]c_int{ lay.n, k }, 2, &[_]c_int{ 1, 1 }, 2, s));
    return .{ .joined = joined, .a = a, .b = b, .b_off = lay.b_off, .n = lay.n };
}

/// At load, after the layer binder transposed the bf16 parts: a layer's in_proj_a / in_proj_b become views of one
/// `joinRows` buffer (`la.a_w` / `la.b_w` the parts' transposed views, `la.ab.w` the whole buffer's), the map's entries
/// the row views, and the binder's transposed views leave `owned`, so the originals are released (no weight resident
/// twice). Nothing changes for a combined projection, a quantized part, or parts the map does not hold under their names.
pub fn joinInPlace(la: *LinearAttnWeights, weights: *model.Weights, name_buf: *[256]u8, prefix: []const u8, li: u32, owned: *std.ArrayList(mlx.mlx_array), allocator: std.mem.Allocator, s: mlx.mlx_stream) !void {
    if (la.combined_proj or la.a_w.ctx == null or la.b_w.ctx == null) return;
    if (la.a_s.ctx != null or la.b_s.ctx != null) {
        if (!declined_logged) {
            declined_logged = true;
            log.info("[gdn-ab] declined at load: in_proj_a and in_proj_b are quantized; the joined GEMM reads unquantized rows, so the two matmuls serve\n", .{});
        }
        return;
    }
    const names = [_][]const u8{ "linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight" };
    const old = [_]mlx.mlx_array{ la.a_w, la.b_w };
    var rows: [2]mlx.mlx_array = undefined;
    for (names, old, 0..) |nm, tv, i| {
        const name = std.fmt.bufPrint(name_buf, "{s}.layers.{d}.{s}", .{ prefix, li, nm }) catch return;
        rows[i] = weights.get(name) orelse return;
        // the map's [w, k] rows under the binder's transposed [k, w] view
        const rs = mlx.getShape(rows[i]);
        const ts = mlx.getShape(tv);
        if (rs.len != 2 or ts.len != 2 or rs[0] != ts[1] or rs[1] != ts[0]) return;
    }
    const j = joinRows(rows[0], rows[1], s) catch |e| switch (e) {
        error.GdnAbShape => return,
        else => return e,
    };
    try owned.append(allocator, j.joined);
    const a_t = try transposeBf16Weight(j.a, s);
    try owned.append(allocator, a_t);
    const b_t = try transposeBf16Weight(j.b, s);
    try owned.append(allocator, b_t);
    const w_t = try transposeBf16Weight(j.joined, s);
    try owned.append(allocator, w_t);
    // the binder's transposed views go; the map takes the row views (freeing the originals' handles)
    for (old) |o| {
        for (owned.items, 0..) |arr, idx| {
            if (arr.ctx == o.ctx) {
                _ = mlx.mlx_array_free(arr);
                _ = owned.swapRemove(idx);
                break;
            }
        }
    }
    const views = [_]mlx.mlx_array{ j.a, j.b };
    for (names, views) |nm, view| {
        const name = std.fmt.bufPrint(name_buf, "{s}.layers.{d}.{s}", .{ prefix, li, nm }) catch unreachable;
        weights.replace(name, view);
    }
    la.a_w = a_t;
    la.b_w = b_t;
    la.ab = .{ .w = w_t, .n = j.n, .b_off = j.b_off, .widths = .{ mlx.getShape(j.a)[0], mlx.getShape(j.b)[0] } };
}

/// x [B, S, hidden] times the joined [a | 0 | b] view in one MLX matmul, [B, S, n] (a at column 0, b at `ab.b_off`);
/// null where the pair keeps its two GEMMs: not joined at load, a non-GPU stream, no NAX, another dtype, a projection
/// path other than the plain matmul (`ternary`: the 2-bit kernels see the parts first), or rows where MLX would not run
/// both widths through the same chains.
pub fn project(x: mlx.mlx_array, ab: GdnAb, ternary: bool, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (ab.w.ctx == null or ternary) return null;
    const xsh = mlx.getShape(x);
    if (xsh.len != 3) return null;
    if (!mlx.streamIsGpu(s) or !naxAvailable() or mlx.mlx_array_dtype(x) != mlx.mlx_array_dtype(ab.w)) return null;
    if (!sameChains(xsh[0] * xsh[1], xsh[2], &ab.widths, ab.n)) return null;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_matmul(&y, x, ab.w, s));
    if (!engaged_logged) {
        engaged_logged = true;
        log.info("[gdn-ab] engaged: rows={d} K={d} N={d} (in_proj_a | in_proj_b as one GEMM)\n", .{ xsh[0] * xsh[1], xsh[2], ab.n });
    }
    return y;
}

/// in_proj_a and in_proj_b into `a_proj` / `b_proj`: column views of the one joined product where `project` serves
/// (`ab_all` takes it), else the two GEMMs.
pub fn parts(self: *Trunk, x: mlx.mlx_array, la: *const LinearAttnWeights, ab_all: *mlx.mlx_array, a_proj: *mlx.mlx_array, b_proj: *mlx.mlx_array) !void {
    if (try project(x, la.ab, self.ternary_2bit, self.s)) |ab| {
        ab_all.* = ab;
        a_proj.* = mlx.mlx_array_new();
        b_proj.* = mlx.mlx_array_new();
        try sliceQkvPart(a_proj, ab, 0, la.ab.widths[0], self.s);
        try sliceQkvPart(b_proj, ab, la.ab.b_off, la.ab.widths[1], self.s);
    } else {
        a_proj.* = try self.qmatmul(x, la.a_w, la.a_s, la.a_b);
        b_proj.* = try self.qmatmul(x, la.b_w, la.b_s, la.b_b);
    }
}

// ── tests ──

const testing = std.testing;
