//! This plugin's own, in place of mlx-serve's `src/sdk/util/shared.zig` (a cut edge of scripts/sync-mlx-serve.sh):
//! mlx-serve keeps a process-wide global in its host, exported by name to the plugin units it compiles. A dylib holds
//! one copy of each of its globals and exports none of them, so here each is a plain global of its own.
//!
//! Each global's type keeps its `name` (`global_name`): Zig interns a struct a function returns by the values its body
//! captures, so one that captured only `T` and `init` would be one type, and one storage, for every name of that type.
//! turboexl3's `Global` takes this `Shared` too (the plugin's turboexl3 `sdk` hands it over), for the same reason.

pub fn Shared(comptime T: type, comptime name: []const u8, comptime init: T) type {
    return struct {
        pub const global_name = name;
        var storage: T = init;

        pub inline fn get() *T {
            return &storage;
        }
    };
}

pub fn SharedThreadLocal(comptime T: type, comptime name: []const u8, comptime init: T) type {
    return struct {
        pub const global_name = name;
        threadlocal var storage: T = init;

        pub inline fn get() *T {
            return &storage;
        }
    };
}

const testing = @import("std").testing;

test "shared: two globals of one type and one initial value are two globals" {
    const a = Shared(?u32, "shared_test_a", null);
    const b = Shared(?u32, "shared_test_b", null);
    a.get().* = 7;
    try testing.expectEqual(@as(?u32, null), b.get().*);
    const ta = SharedThreadLocal(u64, "shared_test_ta", 0);
    const tb = SharedThreadLocal(u64, "shared_test_tb", 0);
    ta.get().* = 3;
    try testing.expectEqual(@as(u64, 0), tb.get().*);
}
