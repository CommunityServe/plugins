//! The slice of sushi's model.zig the DeepSeek-V4.1 files use: the weight map, the owned-region helpers that let MLX
//! adopt a mapping without a copy, and the safetensors reader. Ours; copied from sushi's model.zig, not generated.

const std = @import("std");
const mlx = @import("mlx.zig");
const expert_io = @import("expert_io.zig");
const dsv41_config = @import("dsv41/dsv41_config.zig");

/// The fields the vision tower reads of sushi's `ModelConfig`.
pub const ModelConfig = struct {
    dsv41: ?dsv41_config.Config = null,
    qv_heads: u32 = 0,
    qv_hidden: u32 = 0,
    qv_intermediate: u32 = 0,
    qv_out_hidden: u32 = 0,
    qv_merge: u32 = 0,
};

/// Holds all loaded weights as mlx arrays, keyed by name.
pub const Weights = struct {
    map: std.StringHashMap(mlx.mlx_array),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Weights {
        return .{
            .map = std.StringHashMap(mlx.mlx_array).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Weights) void {
        var it = self.map.iterator();
        while (it.next()) |entry| {
            _ = mlx.mlx_array_free(entry.value_ptr.*);
            self.allocator.free(entry.key_ptr.*);
        }
        self.map.deinit();
    }

    pub fn get(self: *const Weights, name: []const u8) ?mlx.mlx_array {
        return self.map.get(name);
    }

    /// Remove lazy tensors before the transformer binds or evaluates them.
    pub fn dropPrefix(self: *Weights, prefix: []const u8) !void {
        var keys: std.ArrayList([]const u8) = .empty;
        defer keys.deinit(self.allocator);
        var it = self.map.keyIterator();
        while (it.next()) |key| if (std.mem.startsWith(u8, key.*, prefix)) {
            try keys.append(self.allocator, key.*);
        };
        for (keys.items) |key| {
            const removed = self.map.fetchRemove(key).?;
            _ = mlx.mlx_array_free(removed.value);
            self.allocator.free(removed.key);
        }
    }

    /// Takes `name` out of the map: the caller owns the array, or null when it is absent.
    pub fn take(self: *Weights, name: []const u8) ?mlx.mlx_array {
        const removed = self.map.fetchRemove(name) orelse return null;
        self.allocator.free(removed.key);
        return removed.value;
    }

    pub fn count(self: *const Weights) u32 {
        return @intCast(self.map.count());
    }
};

/// MLX's safetensors dtype table (`dtype_from_safetensor_str`): FP8 codes load as `uint8`.
fn safetensorsDtype(name: []const u8) ?struct { mlx.mlx_dtype, u64 } {
    const table = [_]struct { []const u8, mlx.mlx_dtype, u64 }{
        .{ "F32", .float32, 4 },  .{ "F16", .float16, 2 }, .{ "BF16", .bfloat16, 2 }, .{ "I64", .int64, 8 },
        .{ "I32", .int32, 4 },    .{ "I16", .int16, 2 },   .{ "I8", .int8, 1 },       .{ "U64", .uint64, 8 },
        .{ "U32", .uint32, 4 },   .{ "U16", .uint16, 2 },  .{ "U8", .uint8, 1 },      .{ "BOOL", .bool_, 1 },
        .{ "C64", .complex64, 8 }, .{ "F8_E4M3", .uint8, 1 }, .{ "F8_E8M0", .uint8, 1 },
    };
    for (table) |row| if (std.mem.eql(u8, row[0], name)) return .{ row[1], row[2] };
    return null;
}


pub const TensorArray = struct {
    array: mlx.mlx_array,
    /// The array's own bytes; valid as long as `array`.
    bytes: []const u8,
    /// The mapping MLX imported as the array's buffer; null when it copied instead.
    mapped: ?[*]const u8 = null,
};

/// below it, the scratch copy into a fresh MLX buffer costs less than a page per tensor.
pub var import_min_bytes: usize = 1 << 20;

/// `len` bytes at `offset` as a `shape`/`dtype` array, read around the file cache with the read
/// and copy of its chunks spread over several threads (`expert_io.ParallelReader`). A large tensor
/// lands at offset 0 of an anonymous mapping MLX imports as its buffer: a view at an offset
/// inside a buffer is not safe in this engine. Caller frees `array`.
pub fn readTensorArray(reader: *expert_io.ParallelReader, fd: std.c.fd_t, offset: u64, len: usize, shape: []const c_int, dtype: mlx.mlx_dtype) !TensorArray {
    if (len < import_min_bytes) {
        const host = try reader.allocator.alloc(u8, @max(len, 1));
        defer reader.allocator.free(host);
        try reader.fill(fd, host[0..len], offset);
        const array = mlx.mlx_array_new_data(host.ptr, shape.ptr, @intCast(shape.len), dtype);
        if (array.ctx == null) return error.OutOfMemory;
        errdefer _ = mlx.mlx_array_free(array);
        try mlx.check(mlx.mlx_array_eval(array));
        return .{ .array = array, .bytes = if (len == 0) &.{} else mlx.mlx_array_data_uint8(array).?[0..len] };
    }
    const region = try ownedRegion(len);
    reader.fill(fd, region.ptr[0..len], offset) catch |err| {
        destroyRegion(region);
        return err;
    };
    const ptr = region.ptr;
    const wrapped = try wrapRegion(region, shape, dtype);
    return .{ .array = wrapped.array, .bytes = mlx.mlx_array_data_uint8(wrapped.array).?[0..len], .mapped = if (wrapped.aliased) ptr else null };
}

pub const MappedRegion = struct { ptr: [*]align(std.heap.page_size_min) u8, len: usize };

/// Bytes of the live regions MLX holds as array buffers (`wrapOwned`), the weights sushi owns.
pub var owned_bytes: std.atomic.Value(u64) = .init(0);
/// No-copy wraps MLX refused (it copied the region instead).
pub var refused_imports: std.atomic.Value(u64) = .init(0);

const Span = struct { ptr: usize, len: usize };
var span_lock: std.atomic.Value(bool) = .init(false);
var spans: std.ArrayList(Span) = .empty;

fn lockSpans() void {
    while (span_lock.swap(true, .acquire)) std.atomic.spinLoopHint();
}

fn trackRegion(region: *const MappedRegion) void {
    _ = owned_bytes.fetchAdd(region.len, .monotonic);
    lockSpans();
    defer span_lock.store(false, .release);
    spans.append(std.heap.c_allocator, .{ .ptr = @intFromPtr(region.ptr), .len = region.len }) catch {};
}

/// Does `ptr` lie inside a live owned region? A test helper and the load log's audit.
/// Every byte sushi maps and MLX sees without a copy: weight regions plus the expert slab pools.
pub fn ownedBytesTotal() u64 {
    return owned_bytes.load(.monotonic) + expert_io.slab_bytes.load(.monotonic);
}

pub fn ownsPointer(ptr: ?*const anyopaque) bool {
    const at = @intFromPtr(ptr orelse return false);
    lockSpans();
    defer span_lock.store(false, .release);
    for (spans.items) |sp| if (at >= sp.ptr and at < sp.ptr + sp.len) return true;
    return false;
}

pub fn ownedRegionCount() usize {
    lockSpans();
    defer span_lock.store(false, .release);
    return spans.items.len;
}

/// A zeroed, page-rounded anonymous mapping of at least `len` bytes (a mapping cannot be empty).
pub fn ownedRegion(len: usize) !*MappedRegion {
    const size: usize = @intCast(expert_io.pageRoundUp(@max(len, 1)));
    const mapped = std.c.mmap(null, size, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    if (mapped == std.c.MAP_FAILED) return error.OutOfMemory;
    const region = std.heap.c_allocator.create(MappedRegion) catch {
        _ = std.c.munmap(@alignCast(mapped), size);
        return error.OutOfMemory;
    };
    region.* = .{ .ptr = @ptrCast(@alignCast(mapped)), .len = size };
    return region;
}

/// Releases a region that was never wrapped.
pub fn destroyRegion(region: *MappedRegion) void {
    _ = std.c.munmap(region.ptr, region.len);
    std.heap.c_allocator.destroy(region);
}

/// `region` as an evaluated array; `aliased` is false when MLX copied the mapping instead of importing it, which
/// releases the region.
fn wrapRegion(region: *MappedRegion, shape: []const c_int, dtype: mlx.mlx_dtype) !struct { array: mlx.mlx_array, aliased: bool } {
    const ptr = region.ptr;
    trackRegion(region);
    const array = mlx.mlx_array_new_data_managed_payload(ptr, shape.ptr, @intCast(shape.len), dtype, region, unmapRegion);
    if (array.ctx == null) {
        // MLX throws from this constructor only before adopting the mapping (its copy
        // branch runs the deleter last), so this is the region's only release.
        unmapRegion(region);
        return error.OutOfMemory;
    }
    errdefer _ = mlx.mlx_array_free(array);
    try mlx.check(mlx.mlx_array_eval(array));
    return .{ .array = array, .aliased = mlx.mlx_array_data_uint8(array) == ptr };
}

/// `region` as an evaluated `shape`/`dtype` array whose buffer is the region itself; the region goes with the array.
/// A mapping Metal refuses is copied by MLX into its own allocator, which is `error.OwnedImportCopied`.
pub fn wrapOwned(region: *MappedRegion, shape: []const c_int, dtype: mlx.mlx_dtype) !mlx.mlx_array {
    const wrapped = try wrapRegion(region, shape, dtype);
    if (wrapped.aliased) return wrapped.array;
    _ = refused_imports.fetchAdd(1, .monotonic);
    _ = mlx.mlx_array_free(wrapped.array);
    return error.OwnedImportCopied;
}

pub fn unmapRegion(payload: ?*anyopaque) callconv(.c) void {
    const region: *MappedRegion = @ptrCast(@alignCast(payload.?));
    _ = owned_bytes.fetchSub(region.len, .monotonic);
    {
        lockSpans();
        defer span_lock.store(false, .release);
        for (spans.items, 0..) |sp, i| if (sp.ptr == @intFromPtr(region.ptr)) {
            _ = spans.swapRemove(i);
            break;
        };
    }
    destroyRegion(region);
}

/// One header entry's payload as an MLX array. Caller frees.
pub fn readSafetensor(reader: *expert_io.ParallelReader, fd: std.c.fd_t, header: *const expert_io.SafetensorsHeader, meta: std.json.Value) !mlx.mlx_array {
    return (try readSafetensorArray(reader, fd, header, meta)).array;
}

/// `readSafetensor` with the import's outcome: `mapped` is null for a tensor MLX copied.
pub fn readSafetensorArray(reader: *expert_io.ParallelReader, fd: std.c.fd_t, header: *const expert_io.SafetensorsHeader, meta: std.json.Value) !TensorArray {
    if (meta != .object) return error.InvalidSafetensorsTensor;
    const dtype_value = meta.object.get("dtype") orelse return error.InvalidSafetensorsTensor;
    const dims = meta.object.get("shape") orelse return error.InvalidSafetensorsTensor;
    const offsets = meta.object.get("data_offsets") orelse return error.InvalidSafetensorsTensor;
    if (dtype_value != .string or dims != .array or dims.array.items.len > 16 or offsets != .array or offsets.array.items.len != 2) return error.InvalidSafetensorsTensor;
    const dtype, const itemsize = safetensorsDtype(dtype_value.string) orelse return error.UnsupportedSafetensorsDtype;
    var shape: [16]c_int = undefined;
    var bytes: u64 = itemsize;
    for (dims.array.items, 0..) |dim, i| {
        if (dim != .integer or dim.integer < 0 or dim.integer > std.math.maxInt(c_int)) return error.InvalidSafetensorsTensor;
        shape[i] = @intCast(dim.integer);
        bytes = try std.math.mul(u64, bytes, @intCast(dim.integer));
    }
    const lo = offsets.array.items[0];
    const hi = offsets.array.items[1];
    if (lo != .integer or hi != .integer or lo.integer < 0 or hi.integer < lo.integer or @as(u64, @intCast(hi.integer - lo.integer)) != bytes) return error.InvalidSafetensorsTensor;
    return readTensorArray(reader, fd, header.data_start + @as(u64, @intCast(lo.integer)), @intCast(bytes), shape[0..dims.array.items.len], dtype);
}
