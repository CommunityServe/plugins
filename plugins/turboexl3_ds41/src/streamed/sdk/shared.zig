//! mlx-serve's `Shared` (one copy of a global across its `-Dunits` units) as this one dylib needs it: a plain global.

pub fn Shared(comptime T: type, comptime name: []const u8, comptime init: T) type {
    _ = name;
    return struct {
        var storage: T = init;

        pub inline fn get() *T {
            return &storage;
        }
    };
}
