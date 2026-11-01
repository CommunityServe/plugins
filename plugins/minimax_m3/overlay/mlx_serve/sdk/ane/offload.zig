//! This plugin's own, in place of mlx-serve's `src/sdk/ane/offload.zig` (a cut edge of scripts/sync-mlx-serve.sh):
//! the plugin offloads nothing to the ANE; #1005's MTP depth cap reads only the machine's chip brand from here.

const std = @import("std");
const builtin = @import("builtin");

var chip_brand_buf: [128]u8 = undefined;
var chip_brand_len: usize = 0;
var chip_brand_state = std.atomic.Value(u8).init(0);

/// `machdep.cpu.brand_string` ("Apple M5 Max"), read once; empty off Darwin or when the sysctl fails.
pub fn chipBrand() []const u8 {
    if (chip_brand_state.load(.acquire) != 2) {
        if (chip_brand_state.cmpxchgStrong(0, 1, .acquire, .monotonic) == null) {
            chip_brand_len = chipBrandString(&chip_brand_buf).len;
            chip_brand_state.store(2, .release);
        } else {
            while (chip_brand_state.load(.acquire) != 2) std.atomic.spinLoopHint();
        }
    }
    return chip_brand_buf[0..chip_brand_len];
}

pub fn chipBrandString(buf: []u8) []const u8 {
    if (comptime !builtin.os.tag.isDarwin()) return "";
    var len: usize = buf.len;
    if (std.c.sysctlbyname("machdep.cpu.brand_string", buf.ptr, &len, null, 0) != 0) return "";
    if (len > 0 and buf[len - 1] == 0) len -= 1;
    return buf[0..len];
}
