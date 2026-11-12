//! The box's non-file memory (`TURBOEXL3_GLM53F_BILL_FIT`'s load-time check, src/stream_room.zig `fitPlan`): its wired,
//! anonymous and compressor pages (Mach `host_statistics64`, HOST_VM_INFO64), the account the GPU window's guard trips
//! on; and with `TURBOEXL3_GLM53F_FIT_INUSE` the part in use (less the purgeable and reusable pages). The struct and
//! calls are the copy's (src/mlx_serve/sdk/util/system.zig), whose reader keeps them private.

const std = @import("std");

extern "c" fn mach_host_self() u32;
extern "c" fn host_statistics64(host: u32, flavor: u32, info: [*]i32, cnt: *u32) i32;
extern "c" fn host_page_size(host: u32, out: *usize) i32;

/// `vm_statistics64` (mach/vm_statistics.h), as sdk/util/system.zig reads it.
const VmStats64 = extern struct {
    free_count: u32,
    active_count: u32,
    inactive_count: u32,
    wire_count: u32,
    zero_fill_count: u64,
    reactivations: u64,
    pageins: u64,
    pageouts: u64,
    faults: u64,
    cow_faults: u64,
    lookups: u64,
    hits: u64,
    purges: u64,
    purgeable_count: u32,
    speculative_count: u32,
    decompressions: u64,
    compressions: u64,
    swapins: u64,
    swapouts: u64,
    compressor_page_count: u32,
    throttled_count: u32,
    external_page_count: u32,
    internal_page_count: u32,
    total_uncompressed_pages_in_compressor: u64,
};

/// The box's wired, anonymous (purgeable included) and compressor bytes now; 0 when the kernel does not answer.
pub fn nonFileBytes() u64 {
    var page: usize = 0;
    if (host_page_size(mach_host_self(), &page) != 0) return 0;
    var vm = std.mem.zeroes(VmStats64);
    var count: u32 = @sizeOf(VmStats64) / @sizeOf(i32);
    if (host_statistics64(mach_host_self(), 4, @ptrCast(&vm), &count) != 0) return 0;
    return (@as(u64, vm.wire_count) + vm.internal_page_count + vm.compressor_page_count) * page;
}

extern "c" fn sysctlbyname(name: [*:0]const u8, oldp: ?*anyopaque, oldlenp: ?*usize, newp: ?*const anyopaque, newlen: usize) c_int;

/// `TURBOEXL3_GLM53F_FIT_INUSE`: the box's non-file memory in use now: `nonFileBytes`' pages less the purgeable ones
/// (the same reading's `purgeable_count`) and the reusable ones (`vm.page_reusable_count`: freed malloc pages the
/// kernel takes back on demand), the A/B harness's serve gate measure; 0 when the kernel does not answer.
pub fn inUseNonFileBytes() u64 {
    var page: usize = 0;
    if (host_page_size(mach_host_self(), &page) != 0) return 0;
    var vm = std.mem.zeroes(VmStats64);
    var count: u32 = @sizeOf(VmStats64) / @sizeOf(i32);
    if (host_statistics64(mach_host_self(), 4, @ptrCast(&vm), &count) != 0) return 0;
    var reusable: u64 = 0;
    var len: usize = @sizeOf(u64);
    if (sysctlbyname("vm.page_reusable_count", @ptrCast(&reusable), &len, null, 0) != 0) return 0;
    // an int on today's kernels (4 bytes, the low ones on this little-endian box), a 64-bit count otherwise
    const pages: u64 = switch (len) {
        4 => @as(u32, @truncate(reusable)),
        8 => reusable,
        else => return 0,
    };
    return inUsePages(vm.wire_count, vm.internal_page_count, vm.compressor_page_count, vm.purgeable_count, pages) * page;
}

/// The pages of a reading in use: wired, anonymous and compressor pages less the purgeable and reusable ones, none
/// below 0.
pub fn inUsePages(wired: u64, anonymous: u64, compressor: u64, purgeable: u64, reusable: u64) u64 {
    return (wired + anonymous + compressor) -| purgeable -| reusable;
}

test "turboexl3_glm53f stream box memory: TURBOEXL3_GLM53F_FIT_INUSE's in-use pages are the non-file pages less the purgeable and reusable ones, never below 0" {
    try std.testing.expectEqual(@as(u64, 900), inUsePages(300, 600, 100, 40, 60));
    try std.testing.expectEqual(@as(u64, 1000), inUsePages(300, 600, 100, 0, 0));
    try std.testing.expectEqual(@as(u64, 0), inUsePages(10, 10, 0, 15, 15));
}
