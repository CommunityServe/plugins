//! copied_from: src/sdk/util/system.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! What the machine has and how loaded it is: process and system memory, the OS reserve, memory pressure, CPU and GPU
//! load, GPU cores (Mach, IOKit, /proc, Win32).

const std = @import("std");
const crt = @import("crt.zig");
const builtin = @import("builtin");
const mlx = @import("../../mlx/ffi.zig");
const is_macos = builtin.os.tag == .macos;

// ── macOS Mach externs ──

extern "c" var mach_task_self_: u32;
extern "c" fn mach_host_self() u32;
extern "c" fn task_info(task: u32, flavor: u32, info: [*]i32, cnt: *u32) i32;
extern "c" fn host_statistics(host: u32, flavor: u32, info: [*]i32, cnt: *u32) i32;
extern "c" fn host_statistics64(host: u32, flavor: u32, info: [*]i32, cnt: *u32) i32;
extern "c" fn host_page_size(host: u32, out: *usize) i32;
extern "c" fn sysctlbyname(name: [*:0]const u8, oldp: ?*anyopaque, oldlenp: ?*usize, newp: ?*const anyopaque, newlen: usize) c_int;

// ── IOKit / CoreFoundation externs ──

extern "c" fn IOServiceMatching(name: [*:0]const u8) ?*anyopaque;
extern "c" fn IOServiceGetMatchingServices(port: u32, matching: ?*anyopaque, iter: *u32) i32;
extern "c" fn IOIteratorNext(iter: u32) u32;
extern "c" fn IORegistryEntryCreateCFProperties(entry: u32, props: *?*anyopaque, alloc: ?*anyopaque, opts: u32) i32;
extern "c" fn IOObjectRelease(obj: u32) i32;
extern "c" fn CFDictionaryGetValue(dict: ?*const anyopaque, key: ?*const anyopaque) ?*const anyopaque;
extern "c" fn CFStringCreateWithCString(alloc: ?*anyopaque, s: [*:0]const u8, enc: u32) ?*const anyopaque;
extern "c" fn CFNumberGetValue(num: ?*const anyopaque, typ: u32, out: *anyopaque) u8;
extern "c" fn CFRelease(cf: ?*const anyopaque) void;

// ── Mach struct layouts (extern = C ABI) ──

const TaskBasicInfo = extern struct {
    virtual_size: u64,
    resident_size: u64,
    resident_size_max: u64,
    user_time_sec: i32,
    user_time_usec: i32,
    sys_time_sec: i32,
    sys_time_usec: i32,
    policy: i32,
    suspend_count: i32,
};

/// task_vm_info truncated through phys_footprint (rev1). Field order matches
/// <mach/task_info.h> exactly; @sizeOf(TaskVmInfo)/@sizeOf(i32) == 38 ==
/// TASK_VM_INFO_REV1_COUNT, so the kernel fills through phys_footprint without
/// overrunning the buffer.
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
};

const CpuLoadInfo = extern struct {
    ticks: [4]u32, // user, system, idle, nice
};

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

// ── CPU delta tracking (module-level state) ──
var prev_ticks: [4]u64 = @splat(0);

// ── Public metric helpers ──

extern "c" fn os_proc_available_memory() usize;

/// Where weights live in device memory (`mlx.weightsInDeviceMemory`: CUDA on Windows, WSL) the
/// GPU's memory is what every gate bills, as unified memory is on a Mac; elsewhere the host's.
pub fn memoryPool(device_weights: bool, device: u64, host: u64) u64 {
    return if (device_weights) device else host;
}

pub fn getTotalMemBytes() u64 {
    return memoryPool(mlx.weightsInDeviceMemory(), mlx.maxRecommendedWorkingSet(), hostTotalMemBytes());
}

/// Total physical RAM (hw.memsize on Darwin; /proc/meminfo on Linux). 0 on failure.
fn hostTotalMemBytes() u64 {
    if (comptime builtin.os.tag == .windows) return windowsMemStatus().ullTotalPhys;
    if (comptime !builtin.os.tag.isDarwin()) return linuxMemInfoKib("MemTotal:") * 1024;
    var total_mem: u64 = 0;
    var len: usize = @sizeOf(u64);
    if (sysctlbyname("hw.memsize", @ptrCast(&total_mem), &len, null, 0) != 0) return 0;
    return total_mem;
}

/// `--os-reserve-gib N` in bytes; 0 turns the reserve off. Null = the automatic eighth.
pub var os_reserve_override: ?u64 = null;

const MemoryStatusEx = extern struct {
    dwLength: u32 = @sizeOf(MemoryStatusEx),
    dwMemoryLoad: u32 = 0,
    ullTotalPhys: u64 = 0,
    ullAvailPhys: u64 = 0,
    ullTotalPageFile: u64 = 0,
    ullAvailPageFile: u64 = 0,
    ullTotalVirtual: u64 = 0,
    ullAvailVirtual: u64 = 0,
    ullAvailExtendedVirtual: u64 = 0,
};
extern "kernel32" fn GlobalMemoryStatusEx(status: *MemoryStatusEx) callconv(.winapi) i32;

/// Zeroed (callers' "unknown") when the call fails.
fn windowsMemStatus() MemoryStatusEx {
    var s: MemoryStatusEx = .{};
    if (GlobalMemoryStatusEx(&s) == 0) return .{};
    return s;
}

/// One `/proc/meminfo` field in KiB ("MemTotal:", "MemAvailable:", …).
/// 0 when missing or unreadable — callers treat 0 as "unknown".
fn linuxMemInfoKib(key: []const u8) u64 {
    var buf: [8192]u8 = undefined;
    const n = readProcFile("/proc/meminfo", &buf) orelse return 0;
    var it = std.mem.splitScalar(u8, buf[0..n], '\n');
    while (it.next()) |line| {
        if (!std.mem.startsWith(u8, line, key)) continue;
        const val = std.mem.trim(u8, line[key.len..], " \t");
        const end = std.mem.indexOf(u8, val, " kB") orelse val.len;
        return std.fmt.parseInt(u64, std.mem.trim(u8, val[0..end], " \t"), 10) catch 0;
    }
    return 0;
}

/// Whole-file read of a small procfs file. Returns null when missing/unreadable.
fn readProcFile(path: []const u8, buf: []u8) ?usize {
    if (path.len >= std.fs.max_path_bytes) return null;
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    @memcpy(pbuf[0..path.len], path);
    pbuf[path.len] = 0;
    const fd = crt.openRead(pbuf[0..path.len :0].ptr);
    if (fd < 0) return null;
    defer crt.close(fd);
    const n = crt.read(fd, buf);
    if (n <= 0) return null;
    return @intCast(n);
}

var gpu_cores: ?u32 = null;
