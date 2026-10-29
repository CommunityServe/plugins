//! What a restore point keys its images by. An image's span is the image token repeated, so two different images give
//! the same ids; the key is a hash of the soft-token rows the covered ids splice in. The host passes the request's rows
//! as one array `[1, rows, hidden]`, every image's rows in the order of its image tokens (the same array the forward
//! reads by image-token count), so the first `n` rows are the covered ones.
const std = @import("std");
const mlx = @import("sushi/mlx.zig");

/// How many of `ids` are image span positions (none when the pack has no image token).
pub fn imageRows(ids: []const u32, image_token_id: u32) usize {
    if (image_token_id == 0) return 0;
    return std.mem.count(u32, ids, &.{image_token_id});
}

/// The key of the prefix `ids` under the request's rows: 0 for a prefix with no image, or a request without rows (the
/// image token is then text).
pub fn prefixKey(ids: []const u32, image_token_id: u32, rows: ?mlx.mlx_array, s: mlx.mlx_stream) !u64 {
    return visionKey(rows, imageRows(ids, image_token_id), s);
}

/// A Wyhash over the first `n` rows as float32, so the rows of an image after them do not enter. Refuses rows shorter
/// than `n` rather than hashing the wrong ones.
pub fn visionKey(rows: ?mlx.mlx_array, n: usize, s: mlx.mlx_stream) !u64 {
    if (n == 0) return 0;
    const r = rows orelse return 0;
    const shape = mlx.getShape(r);
    if (shape.len == 0) return error.VisionRowsShort;
    const d = shape[shape.len - 1];
    if (d <= 0 or n * @as(usize, @intCast(d)) > mlx.mlx_array_size(r)) return error.VisionRowsShort;
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, r, &[_]c_int{ -1, d }, 2, s));
    var head = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(head);
    try mlx.check(mlx.mlx_slice(&head, flat, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ @intCast(n), d }, 2, &[_]c_int{ 1, 1 }, 2, s));
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, head, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f));
    const data = mlx.mlx_array_data_float32(f) orelse return error.VisionRowsUnreadable;
    return std.hash.Wyhash.hash(0x5e1f, std.mem.sliceAsBytes(data[0..mlx.mlx_array_size(f)]));
}

const testing = std.testing;

fn rowsOf(data: []const f32, n: c_int, d: c_int) mlx.mlx_array {
    return mlx.mlx_array_new_data(data.ptr, &[_]c_int{ 1, n, d }, 3, .float32);
}

test "sushi_ds41 gpu: an image prefix's key is its covered rows, not its ids" {
    if (std.c.getenv("SUSHI_DS41_GPU_TEST") == null) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const img: u32 = 17;
    // Two images of one row each (hidden 2) in image-token order; `b` differs from `a` only in its first image.
    const a = rowsOf(&.{ 1, 2, 3, 4 }, 2, 2);
    defer _ = mlx.mlx_array_free(a);
    const b = rowsOf(&.{ 9, 8, 3, 4 }, 2, 2);
    defer _ = mlx.mlx_array_free(b);
    const a_again = rowsOf(&.{ 1, 2, 3, 4 }, 2, 2);
    defer _ = mlx.mlx_array_free(a_again);
    const a_first = rowsOf(&.{ 1, 2 }, 1, 2);
    defer _ = mlx.mlx_array_free(a_first);
    const one = [_]u32{ 3, img, 4 };
    const two = [_]u32{ 3, img, 4, img };

    // Same ids, different images: different keys, so one image's restore point never serves the other.
    const ka = try prefixKey(&one, img, a, s);
    const kb = try prefixKey(&one, img, b, s);
    try testing.expect(ka != kb);
    try testing.expect(ka != 0);
    // Same ids, same rows (a separate array): equal keys, so the cache hits.
    try testing.expectEqual(ka, try prefixKey(&one, img, a_again, s));
    // An image after the boundary does not enter: the key over the first image's rows alone.
    try testing.expectEqual(ka, try prefixKey(&one, img, a_first, s));
    try testing.expect(try prefixKey(&two, img, a, s) != ka);
    // No rows, no image token in the ids, or a pack without an image token: 0.
    try testing.expectEqual(@as(u64, 0), try prefixKey(&one, img, null, s));
    try testing.expectEqual(@as(u64, 0), try prefixKey(&.{ 3, 4, 5 }, img, a, s));
    try testing.expectEqual(@as(u64, 0), try prefixKey(&one, 0, a, s));
    // Fewer rows than the covered image positions: a named refusal, not a key over the wrong rows.
    try testing.expectError(error.VisionRowsShort, prefixKey(&.{ img, img, img }, img, a, s));
}
