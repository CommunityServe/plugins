//! copied_from: src/engines/mlx/compiled.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! engines/mlx/compiled.
//! Owns compileGelu, compileGeglu, gegluClosureCallback, and 7 more.

const mlx = @import("../../mlx/ffi.zig");
const log = @import("../../sdk/util/log.zig");
const Transformer = @import("transformer.zig").Transformer;

pub fn closureIn(input: mlx.mlx_vector_array, i: usize) ?mlx.mlx_array {
    var a = mlx.mlx_array_new();
    if (mlx.mlx_vector_array_get(&a, input, i) != 0) return null;
    return a;
}

pub fn closureOut(res: *mlx.mlx_vector_array, a: mlx.mlx_array) c_int {
    const arr = [_]mlx.mlx_array{a};
    res.* = mlx.mlx_vector_array_new_data(&arr, 1);
    _ = mlx.mlx_array_free(a);
    return 0;
}

pub fn applyClosure(compiled: ?mlx.mlx_closure, inputs: []const mlx.mlx_array) !?mlx.mlx_array {
    const c = compiled orelse return null;
    const in_vec = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var out_vec = mlx.mlx_vector_array{ .ctx = null };
    defer _ = mlx.mlx_vector_array_free(out_vec);
    try mlx.check(mlx.mlx_closure_apply(&out_vec, c, in_vec));
    if (mlx.mlx_vector_array_size(out_vec) != 1) return null;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, out_vec, 0));
    return out;
}
