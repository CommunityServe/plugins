//! Process and box memory readings from the kernel's own ledgers (task_vm_info rev3, rusage v4, vm_statistics64):
//! read-only probes a plugin's memory bill and construction checks compare against. Zero where the platform has no
//! ledger. The host's status bar reads the same functions (status.zig re-exports them).

const std = @import("std");
const builtin = @import("builtin");

extern "c" var mach_task_self_: u32;
extern "c" fn mach_host_self() u32;
extern "c" fn task_info(task: u32, flavor: u32, info: [*]i32, cnt: *u32) i32;
extern "c" fn host_statistics64(host: u32, flavor: u32, info: [*]i32, cnt: *u32) i32;
extern "c" fn host_page_size(host: u32, out: *usize) i32;
extern "c" fn sysctlbyname(name: [*:0]const u8, oldp: ?*anyopaque, oldlenp: ?*usize, newp: ?*const anyopaque, newlen: usize) c_int;

/// task_vm_info through the rev3 ledger block. Field order matches <mach/task_info.h>
/// exactly; @sizeOf(TaskVmInfo)/@sizeOf(i32) == 84 == TASK_VM_INFO_REV3_COUNT. The count
/// is in/out: a kernel below rev3 fills fewer fields and leaves the rest zero.
const TaskVmInfo = extern struct {
    virtual_size: u64,
    region_count: i32,
    page_size: i32,
    resident_size: u64,
    resident_size_peak: u64,
    device: u64,
    device_peak: u64,
    internal: u64,
    internal_peak: u64,
    external: u64,
    external_peak: u64,
    reusable: u64,
    reusable_peak: u64,
    purgeable_volatile_pmap: u64,
    purgeable_volatile_resident: u64,
    purgeable_volatile_virtual: u64,
    compressed: u64,
    compressed_peak: u64,
    compressed_lifetime: u64,
    phys_footprint: u64,
    // rev2
    min_address: u64,
    max_address: u64,
    // rev3: the ledger block
    ledger_phys_footprint_peak: i64,
    ledger_purgeable_nonvolatile: i64,
    ledger_purgeable_novolatile_compressed: i64,
    ledger_purgeable_volatile: i64,
    ledger_purgeable_volatile_compressed: i64,
    ledger_tag_network_nonvolatile: i64,
    ledger_tag_network_nonvolatile_compressed: i64,
    ledger_tag_network_volatile: i64,
    ledger_tag_network_volatile_compressed: i64,
    ledger_tag_media_footprint: i64,
    ledger_tag_media_footprint_compressed: i64,
    ledger_tag_media_nofootprint: i64,
    ledger_tag_media_nofootprint_compressed: i64,
    ledger_tag_graphics_footprint: i64,
    ledger_tag_graphics_footprint_compressed: i64,
    ledger_tag_graphics_nofootprint: i64,
    ledger_tag_graphics_nofootprint_compressed: i64,
    ledger_tag_neural_footprint: i64,
    ledger_tag_neural_footprint_compressed: i64,
    ledger_tag_neural_nofootprint: i64,
    ledger_tag_neural_nofootprint_compressed: i64,
};

comptime {
    std.debug.assert(@sizeOf(TaskVmInfo) / @sizeOf(i32) == 84); // TASK_VM_INFO_REV3_COUNT
}

/// rusage_info_v4 (<sys/resource.h>): the footprint's interval high-water mark, which
/// `proc_reset_footprint_interval` (libsystem_kernel) restarts: the kernel's own ledger, no sampling.
const RusageInfoV4 = extern struct {
    uuid: [16]u8,
    f: [35]u64,
    const lifetime_max_phys_footprint = 28;
    const interval_max_phys_footprint = 33;
};
extern "c" fn proc_pid_rusage(pid: c_int, flavor: c_int, buffer: *RusageInfoV4) c_int;
extern "c" fn proc_reset_footprint_interval(pid: c_int) c_int;
extern "c" fn getpid() c_int;

const testing = std.testing;
