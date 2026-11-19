//! mlx-serve's `Shared` (one copy of a global across its `-Dunits` units) as this one dylib needs it: a plain global,
//! one per name.

pub fn Shared(comptime T: type, comptime name: []const u8, comptime init: T) type {
    return struct {
        // Named, so the type is this global's own: a body that captures only T and init is one type, and one storage,
        // for every global of that T and init (turboexl3's `Plain` states the same).
        const global_name = name;
        var storage: T = init;

        pub inline fn get() *T {
            return &storage;
        }
    };
}

test "turboexl3_glm53f shared: two names of one type and init are two globals" {
    const t = @import("std").testing;
    const Box = struct { v: bool };
    const a = Shared(Box, "turboexl3_a", .{ .v = false });
    const b = Shared(Box, "turboexl3_b", .{ .v = false });
    a.get().v = true;
    try t.expect(a.get() != b.get());
    try t.expect(!b.get().v);
    try t.expect(Shared(Box, "turboexl3_a", .{ .v = false }).get() == a.get());
}
