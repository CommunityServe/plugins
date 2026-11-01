//! copied_from: src/plugins/minimax_m3/gather_groups.zig@90fb57282d374c0989a46503b12f4fd9be300963
const std = @import("std");
pub fn Grouping(comptime Ref: type) type {
    return GroupingCapacity(Ref, 4);
}
pub fn GroupingCapacity(comptime Ref: type, comptime capacity: usize) type {
    return struct {
        const Self = @This();
        const Group = struct {
            bank: @FieldType(Ref, "bank"),
            rows: [capacity]u32 = undefined,
            positions: [capacity]usize = undefined,
            n: usize = 0,
        };
        groups: [3]Group = undefined,
        n: usize = 0,
        pub fn init(refs: []const Ref) !Self {
            if (refs.len == 0 or refs.len > capacity or refs.len % 4 != 0) return error.MiniMaxRouteWidth;
            var out: Self = .{};
            for (refs, 0..) |ref, position| {
                var index: usize = 0;
                while (index < out.n and out.groups[index].bank != ref.bank) : (index += 1) {}
                if (index == out.n) {
                    if (out.n == out.groups.len) return error.MiniMaxSlotBankMissing;
                    out.groups[index] = .{ .bank = ref.bank };
                    out.n += 1;
                }
                const group = &out.groups[index];
                group.rows[group.n] = ref.row;
                group.positions[group.n] = position;
                group.n += 1;
            }
            return out;
        }
    };
}
