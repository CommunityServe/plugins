//! A wide prompt call's DIG-X waves (the EXL3 stream quant's prefill lane) write their intermediates into per-prompt
//! regions (src/scratch.zig's `Region`: one MTLBuffer each, from the host's `alloc_uninit`) instead of fresh MLX
//! buffers.
//!
//! A wave of n rows runs five launches, each output a fresh array in the stock lane: take2's x0 and x1 (f16
//! [n,1,5120]), the gate|up GEMM's zg and zu (f32 [n,2304]), onepass's hd (f16 [n,1,2304]), the down GEMM's z (f32
//! [n,5120]) and rot_widen1's y (bf16 [n,1,5120]). Two waves are in flight (the tier shape's `inflight`): a wave is
//! submitted once all but one older wave have been drained to completion on the host. So two sets of the six
//! intermediates (by the wave's parity, sized for the shape's 7168-row budget, 2 x 459 MB) carry every wave, a set
//! reused only after the wave two before drained. y stays MLX's own output (the combine's part, 10,240 B of a wave
//! row's 74,240 B).
//!
//! Each launch writing a set is its stock text's write-into twin (src/wave_twins.zig, derived at construction from the
//! registry's pinned stock texts and compiled then, by the copied quant's `IntoTwins`): its outputs come from the set's
//! regions, bound whole as inputs (evaluated arrays, so no `contiguous` copy stands in for them), and its only MLX
//! output is a one-word carrier; each launch binds the previous launch's carrier (`dep`), so MLX's barrier or fence
//! orders the wave's launches as their data flow. rot_widen1 is a dep twin: the stock text with `dep` bound, its y
//! MLX's output. The kernels index only the wave's rows (the grid and the wave table), so a whole region serves every
//! wave size. A carried call (its last waves left in flight past it), a solo wave (rows above a set's) and a call the
//! regions do not fit (`scratch.cap_bytes`) take the fresh path. Exact: the same texts' words, only where they land
//! changes.

const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const scratch = @import("scratch.zig");
/// The twins' text derivation (std only), for the copied quant.
pub const twins = @import("wave_twins.zig");

const A = mlx.mlx_array;
const none: A = .{ .ctx = null };

/// A set's rows: the served tier shape's row budget (exl3 stream quant `PrefillShape.tier.row_budget`); a wave above it
/// (solo) takes the fresh path.
pub const set_rows: u32 = 7168;
const hidden: c_int = 5120;
const inter: c_int = 2304;

/// One wave's six intermediates' regions.
pub const Set = struct {
    x0: scratch.Region = .{},
    x1: scratch.Region = .{},
    zg: scratch.Region = .{},
    zu: scratch.Region = .{},
    hd: scratch.Region = .{},
    z: scratch.Region = .{},

    fn ensure(s: *Set) !bool {
        const r: c_int = @intCast(set_rows);
        const wide = [_]c_int{ r, 1, hidden };
        const gu = [_]c_int{ r, inter };
        const mid = [_]c_int{ r, 1, inter };
        const dn = [_]c_int{ r, hidden };
        if ((try s.x0.ensure(&wide, .float16)) == null) return false;
        if ((try s.x1.ensure(&wide, .float16)) == null) return false;
        if ((try s.zg.ensure(&gu, .float32)) == null) return false;
        if ((try s.zu.ensure(&gu, .float32)) == null) return false;
        if ((try s.hd.ensure(&mid, .float16)) == null) return false;
        if ((try s.z.ensure(&dn, .float32)) == null) return false;
        return true;
    }
    fn free(s: *Set) void {
        inline for (.{ "x0", "x1", "zg", "zu", "hd", "z" }) |f| @field(s, f).free();
    }
};

pub var sets: [2]Set = .{ .{}, .{} };
/// The bank pool (src/bank_pool.zig): the twelve regions the copied stream's pool holds, set by set in `Set`'s field
/// order (x0, x1, zg, zu, hd, z), at `set_rows` rows: `Set.ensure`'s shapes.
pub const pool_shapes: [12]scratch.Shape = blk: {
    const r: c_int = @intCast(set_rows);
    const one = [6]scratch.Shape{
        .{ .shape = .{ r, 1, hidden, 0 }, .ndim = 3, .dtype = .float16 },
        .{ .shape = .{ r, 1, hidden, 0 }, .ndim = 3, .dtype = .float16 },
        .{ .shape = .{ r, inter, 0, 0 }, .ndim = 2, .dtype = .float32 },
        .{ .shape = .{ r, inter, 0, 0 }, .ndim = 2, .dtype = .float32 },
        .{ .shape = .{ r, 1, inter, 0 }, .ndim = 3, .dtype = .float16 },
        .{ .shape = .{ r, hidden, 0, 0 }, .ndim = 2, .dtype = .float32 },
    };
    break :blk one ++ one;
};
/// The bank pool: the pool's arrays for the two sets' regions (`pool_shapes`' order), each region keeping a reference
/// until `dropPool`.
pub fn adoptPool(arrays: []const A) !void {
    if (arrays.len != pool_shapes.len) return error.ScratchWavesPool;
    for (0..2) |k| {
        inline for (.{ "x0", "x1", "zg", "zu", "hd", "z" }, 0..) |f, j| try @field(sets[k], f).adoptPooled(arrays[k * 6 + j], pool_shapes[k * 6 + j]);
    }
}
pub fn dropPool() void {
    for (&sets) |*s| {
        inline for (.{ "x0", "x1", "zg", "zu", "hd", "z" }) |f| @field(s, f).dropPooled();
    }
}
/// Whether the call's waves may take the regions.
var call_live: bool = false;
/// A `[1]` uint32 zeros the first launch of a wave binds as its `dep` (nothing to order: the set's last user drained).
var no_dep: A = none;

/// A wide routed call begins (the source, before its routes). The regions are made at their first use and kept while
/// their shapes hold; false when they would pass the regions' cap (the call's waves take the fresh path).
pub fn callBegin() !bool {
    call_live = false;
    if (!try sets[0].ensure() or !try sets[1].ensure()) return false;
    if (no_dep.ctx == null) {
        no_dep = mlx.mlx_array_new_data(&[_]u32{0}, &[_]c_int{1}, 1, .uint32);
        if (no_dep.ctx == null) return error.ScratchWavesDep;
    }
    call_live = true;
    return true;
}

/// A call's end (the source, after its combine is built): no wave takes the regions until the next call begins.
pub fn callEnd() void {
    call_live = false;
}

/// Whether a wave of `rows` takes the regions now (`views`' check, pure): a live call, not a solo wave (rows above a
/// set's).
fn fits(rows: u32) bool {
    return call_live and rows > 0 and rows <= set_rows;
}

/// One arena wave's destinations: set `parity`'s whole regions (new handles the caller owns, of evaluated arrays: a
/// row-contiguous kernel binds them as they are) and the first launch's `dep`. Null: the fresh path (no live call, a
/// solo wave).
pub const Views = struct { x0: A, x1: A, zg: A, zu: A, hd: A, z: A, dep: A };
pub fn views(parity: u1, rows: u32) !?Views {
    if (!fits(rows)) return null;
    const st = &sets[parity];
    var v: Views = .{ .x0 = none, .x1 = none, .zg = none, .zu = none, .hd = none, .z = none, .dep = no_dep };
    errdefer inline for (.{ "x0", "x1", "zg", "zu", "hd", "z" }) |f| {
        if (@field(v, f).ctx != null) _ = mlx.mlx_array_free(@field(v, f));
    };
    inline for (.{ "x0", "x1", "zg", "zu", "hd", "z" }) |f| @field(v, f) = try handle(@field(st, f).arr);
    return v;
}
fn handle(x: A) !A {
    var h = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(h);
    try mlx.check(mlx.mlx_array_set(&h, x));
    return h;
}

/// The decode phase change (or the model's end): every region let go.
pub fn releaseAll() void {
    sets[0].free();
    sets[1].free();
    if (no_dep.ctx != null) _ = mlx.mlx_array_free(no_dep);
    no_dep = none;
    call_live = false;
}

test "scratch_waves fits: none without a live call or for a solo wave" {
    // (pure: the plugin's tests link no MLX, so `views` itself is not called here)
    const saved = call_live;
    defer call_live = saved;
    call_live = false;
    try std.testing.expect(!fits(16));
    call_live = true;
    try std.testing.expect(fits(16));
    try std.testing.expect(fits(set_rows));
    try std.testing.expect(!fits(0));
    try std.testing.expect(!fits(set_rows + 1));
}
