//! Test assertions that stay cheap to compile.
const std = @import("std");

/// `std.testing.expectError` that names a success payload's type instead of printing it:
/// `{any}` of an engine struct instantiates a printer of tens of KiB per payload type.
pub fn expectError(expected: anyerror, actual: anytype) !void {
    if (actual) |_| {
        const Payload = @typeInfo(@TypeOf(actual)).error_union.payload;
        std.debug.print("expected error.{s}, found a {s}\n", .{ @errorName(expected), @typeName(Payload) });
        return error.TestExpectedError;
    } else |err| if (err != expected) {
        std.debug.print("expected error.{s}, found error.{s}\n", .{ @errorName(expected), @errorName(err) });
        return error.TestUnexpectedError;
    }
}

test "expectError passes on the expected error" {
    const S = struct {
        fn get(e: ?anyerror) anyerror!u32 {
            return if (e) |err| err else 7;
        }
    };
    try expectError(error.Wanted, S.get(error.Wanted));
}
