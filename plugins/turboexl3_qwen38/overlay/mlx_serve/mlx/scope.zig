//! This plugin's own, in place of mlx-serve's `src/mlx/scope.zig` (a cut edge of scripts/sync-mlx-serve.sh): the
//! arrays one block makes, freed together; only what the copies call of it.

const std = @import("std");
const mlx = @import("ffi.zig");

const A = mlx.mlx_array;

pub const Scope = struct {
    a: std.mem.Allocator,
    s: mlx.mlx_stream,
    items: std.ArrayList(A) = .empty,

    pub fn init(a: std.mem.Allocator, s: mlx.mlx_stream) Scope {
        return .{ .a = a, .s = s };
    }

    pub fn deinit(self: *Scope) void {
        for (self.items.items) |x| _ = mlx.mlx_array_free(x);
        self.items.deinit(self.a);
    }

    /// `x`, freed with the scope (at once when it cannot be kept).
    pub fn keep(self: *Scope, x: A) !A {
        self.items.append(self.a, x) catch |e| {
            _ = mlx.mlx_array_free(x);
            return e;
        };
        return x;
    }

    /// An op's output `o` after it returned `rc`, kept.
    pub fn res(self: *Scope, rc: c_int, o: *const A) !A {
        if (rc != 0) {
            _ = mlx.mlx_array_free(o.*);
            return error.MlxError;
        }
        return self.keep(o.*);
    }

    /// A new reference on `x` that outlives the scope (the caller's).
    pub fn out(_: *Scope, x: A) A {
        var o = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&o, x);
        return o;
    }

    pub fn astype(sc: *Scope, x: A, dt: mlx.mlx_dtype) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_astype(&o, x, dt, sc.s), &o);
    }
};
