//! `TURBOEXL3_GLM53F_TRUNK_F16_FREE=1` (opt-in; a residency change, not a numerics one: every array the model reads keeps
//! its bytes): the trunk's plain f16 matrices let go of their f16 sources once their bf16 casts are evaluated
//! (patches/turboexl3-model.patch: trunk.zig's `Loader.lin`).
//!
//! The pack ships the trunk's small matrices in f16 (a KDA layer's b, f_a, g_a, f_b and g_b; a DSA layer's kv_b, split
//! into embed_q and unembed_out; the indexer's wk and weights_proj), and the loader serves them as bf16 copies. Without
//! the switch the loader also keeps the f16 matrix it cast (the `[in, out]` view `transposed` owns and the reference
//! `kept` adds), so the shard's f16 buffer stays allocated beside its bf16 copy for as long as the model is loaded
//! (`mlx_load_safetensors` reads into MLX buffers, not a file mapping): about 0.65 GB at sc_3.10. With the switch the
//! loader hands each such source here instead (`take`, `hold`); once the shard's source map is released after the load,
//! `release` frees them, clears MLX's cache and measures what came back (MLX's active memory before less after: a
//! source something else still reads, such as a router's raw table, is not freed and not counted). The freed bytes go to
//! the bank's rows: TURBOEXL3_GLM53F_BILL_FIT's measured bases held these sources, so each phase's base is taken less
//! them (src/stream_room.zig `fitPlanFreed`); without the fit the memory available when the bank opens includes them.
const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const Array = mlx.mlx_array;
const gpa = std.heap.c_allocator;

/// `TURBOEXL3_GLM53F_TRUNK_F16_FREE`, set by the plugin at load, before the trunk loads
pub var on: bool = false;
/// The switch's load line (an A/B step reads "<switch>=1: ")
pub const line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_TRUNK_F16_FREE=1: each plain f16 matrix's source let go once its bf16 cast is evaluated, freed after the load and its measured bytes given to the bank's rows\n";

/// The sources let go and not yet released, one handle each
var held: std.ArrayList(Array) = .empty;
/// The sources let go since the load began, their bytes as stored, and the bytes `release` measured
pub var tensors: usize = 0;
pub var nominal: u64 = 0;
pub var freed: u64 = 0;

fn counted(a: Array) void {
    tensors += 1;
    // the kit's ffi declares size and itemsize (no mlx_array_nbytes)
    nominal += mlx.mlx_array_size(a) * mlx.mlx_array_itemsize(a);
}

/// Takes `a`, a handle the load owned (an f16 source it let go), until `release`
pub fn take(a: Array) !void {
    try held.append(gpa, a);
    counted(a);
}

/// Holds a new reference on `a` (an f16 source the load read but did not own) until `release`
pub fn hold(a: Array) !void {
    var h = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(h);
    try mlx.check(mlx.mlx_array_set(&h, a));
    try held.append(gpa, h);
    counted(a);
}

/// Frees the held sources (after the shard's source map is released) and clears MLX's cache: `freed` is MLX's active
/// memory before less after, the bytes the release gave back
pub fn release() u64 {
    var before: usize = 0;
    _ = mlx.mlx_get_active_memory(&before);
    for (held.items) |h| _ = mlx.mlx_array_free(h);
    held.clearAndFree(gpa);
    _ = mlx.mlx_clear_cache();
    var after: usize = 0;
    _ = mlx.mlx_get_active_memory(&after);
    freed = before -| after;
    return freed;
}

/// Drops what is held and the counts (the model's unload, or a load refused before `release`)
pub fn reset() void {
    for (held.items) |h| _ = mlx.mlx_array_free(h);
    held.clearAndFree(gpa);
    tensors = 0;
    nominal = 0;
    freed = 0;
}

/// The load line after `release`
pub fn released(buf: []u8, n: usize, stored: u64, got: u64) []const u8 {
    return std.fmt.bufPrint(buf, "[turboexl3_glm53f] TURBOEXL3_GLM53F_TRUNK_F16_FREE: {d} f16 sources let go after their bf16 casts ({d} MiB as stored): {d} MiB freed (MLX's active memory before less after the release), given to both phases' rows\n", .{ n, stored >> 20, got >> 20 }) catch "[turboexl3_glm53f] TURBOEXL3_GLM53F_TRUNK_F16_FREE: (line too long)\n";
}

const testing = std.testing;

test "turboexl3_glm53f stream f16 free: the load line names the sources let go, their stored bytes and the measured bytes freed" {
    var buf: [256]u8 = undefined;
    try testing.expectEqualStrings("[turboexl3_glm53f] TURBOEXL3_GLM53F_TRUNK_F16_FREE: 92 f16 sources let go after their bf16 casts (620 MiB as stored): 618 MiB freed (MLX's active memory before less after the release), given to both phases' rows\n", released(&buf, 92, 620 << 20, 618 << 20));
    try testing.expect(std.mem.startsWith(u8, line, "[turboexl3_glm53f] TURBOEXL3_GLM53F_TRUNK_F16_FREE=1: "));
}
