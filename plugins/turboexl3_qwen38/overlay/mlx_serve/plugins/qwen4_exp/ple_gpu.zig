//! copied_from: src/plugins/qwen4_exp/ple_gpu.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! qwen4_exp n-gram PLE on the GPU: the whole `ngram_table.bin` mapping wrapped as ONE no-copy
//! Metal buffer, and one kernel that hashes the token ids, gathers the rows and rounds them to
//! bf16 exactly like `NgramHash.rowIds` + `NgramTable.gather` + `bf16Rne`. The kernel reads the
//! ids on the GPU, so a forward built on lazy ids needs no host read before it runs.

const std = @import("std");
const kernelSource = @import("../../mlx/kernels.zig").source;
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../../sdk/util/log.zig");
const qwen4 = @import("qwen4_exp.zig");
const qwen4_ple = @import("ple_table.zig");

/// Why the loader picked its arm. Everything but `.gpu` serves the host gather.
pub const Arm = enum { gpu, off, bits, misaligned, too_large, low_memory };

/// Process-wide under `-Dunits` (sdk/util/shared.zig): one copy in the host.
const PleGpuState = struct {
    /// `--ple-gpu`, set once in `main()`. Off by default: the first GPU forward makes the whole
    /// table resident beside the weights, and the host gather faults in only the rows it reads.
    enabled: bool = false,
};
pub const pleGpu_shared = @import("../../sdk/util/shared.zig").Shared(PleGpuState, "ple_gpu", .{});
pub inline fn pleGpu() *PleGpuState {
    return pleGpu_shared.get();
}

pub const Budget = struct {
    page: usize,
    max_buffer: u64,
    working_set: u64,
    model_bytes: u64,
};

/// Room the working set keeps past weights + table for KV, transients and the buffer pool.
pub const HEADROOM: u64 = 16 << 30;

/// Kernel dispatches since start; the engagement counter the tests read.
pub var dispatches: u64 = 0;
/// Mappings unmapped by MLX dropping their buffer (MLX may drop it off the inference thread).
pub var unmaps = std.atomic.Value(u64).init(0);

pub fn chooseArm(on: bool, bits: u32, base: usize, len: usize, b: Budget) Arm {
    if (!on) return .off;
    if (!kernelBits(bits)) return .bits;
    // MLX falls back to a malloc + copy of the WHOLE table when Metal refuses the no-copy buffer.
    if (b.page == 0 or base % b.page != 0) return .misaligned;
    const bytes = std.mem.alignForward(u64, len, b.page);
    if (bytes > b.max_buffer) return .too_large;
    if (b.model_bytes +| bytes +| HEADROOM > b.working_set) return .low_memory;
    return .gpu;
}

fn kernelBits(bits: u32) bool {
    return switch (bits) {
        2, 3, 4, 5, 6, 8, 16 => true,
        else => false,
    };
}

pub const Table = struct {
    arr: mlx.mlx_array,
    /// Byte offsets of the three regions inside `arr`, and the embedded pack's global
    /// weight scale (1.0 for `ngram_table.bin`).
    w_off: u64,
    s_off: u64,
    b_off: u64,
    scale: f32 = 1.0,

    pub fn release(self: Table) void {
        _ = mlx.mlx_array_free(self.arr);
    }
};

/// What MLX's deleter unmaps. `armed` stays false until `wrap` proved the buffer is the
/// mapping: MLX's copy fallback calls the deleter at once, while the host gather still reads it.
const Mapping = struct {
    map: []align(std.heap.page_size_min) const u8,
    armed: bool = false,
    fired: bool = false,
};

fn onRelease(payload: ?*anyopaque) callconv(.c) void {
    const m: *Mapping = @ptrCast(@alignCast(payload.?));
    if (!m.armed) {
        m.fired = true;
        return;
    }
    std.posix.munmap(m.map);
    _ = unmaps.fetchAdd(1, .monotonic);
    std.heap.page_allocator.destroy(m);
}

const ROW: usize = 4096;

/// `table`'s mapping as one no-copy GPU buffer. On success MLX owns the munmap, and the
/// table stops unmapping it on close.
pub fn wrap(table: *qwen4.NgramTable) !Table {
    const arr = try wrapMap(table.map);
    table.gpu_owns_map = true;
    return .{ .arr = arr, .w_off = table.w_off, .s_off = table.s_off, .b_off = table.b_off };
}

/// An embedded pack's shards copied into one anonymous page-aligned mapping in the
/// `ngram_table.bin` layout, wrapped no-copy; MLX owns the munmap on success.
fn wrapEmbedded(src: *const qwen4_ple.EmbeddedTable) !Table {
    if (comptime @import("builtin").os.tag == .windows) return error.PleGpuUnsupportedOnWindows;
    const len: usize = @intCast(std.mem.alignForward(u64, src.repackedLen(), std.heap.pageSize()));
    const map = try std.posix.mmap(null, len, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    var owned = true;
    defer if (owned) std.posix.munmap(map);
    try src.repackInto(map);
    const arr = try wrapMap(map);
    owned = false;
    const wbytes = src.rows * @as(u64, src.wcols) * 4;
    const sbytes = src.rows * @as(u64, src.scols) * qwen4_ple.scaleBytes(src.fmt);
    return .{ .arr = arr, .w_off = 0, .s_off = wbytes, .b_off = wbytes + sbytes, .scale = src.scale };
}

/// `map` as one no-copy uint8 `[pages, 4096]` array (no dim past int32). The caller has
/// checked the base and length (`chooseArm`); on success MLX owns the munmap.
fn wrapMap(map: []const u8) !mlx.mlx_array {
    const page = std.heap.pageSize();
    if (@intFromPtr(map.ptr) % page != 0) return error.PleMapMisaligned;
    const len = std.mem.alignForward(usize, map.len, page);
    if (len / ROW > std.math.maxInt(c_int)) return error.PleMapTooLarge;
    const m = try std.heap.page_allocator.create(Mapping);
    errdefer std.heap.page_allocator.destroy(m);
    m.* = .{ .map = @alignCast(map) };
    const shape = [_]c_int{ @intCast(len / ROW), @intCast(ROW) };
    const arr = mlx.mlx_array_new_data_managed_payload(@constCast(map.ptr), &shape, 2, .uint8, m, onRelease);
    if (arr.ctx == null) return error.PleWrapFailed;
    if (m.fired) {
        _ = mlx.mlx_array_free(arr);
        return error.PleWrapCopied;
    }
    m.armed = true;
    return arr;
}

fn gb(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / 1073741824.0;
}

/// Load-time arm choice for `table`; logs one line naming the arm. `on` is `--ple-gpu`,
/// `model_bytes` the weights already resident. Null = the host gather. An embedded pack's
/// shards are copied into one buffer, so the gate runs before anything is allocated.
pub fn load(table: *qwen4.NgramTable, on: bool, model_bytes: u64) ?Table {
    if (mlx.noGpuBackend() or !mlx.metalKernelsAvailable()) {
        log.info("[qwen4] ple gather: cpu (no Metal kernels: the GPU gather is one)\n", .{});
        return null;
    }
    // The kernel reads affine and BF16 rows; a trellis table's rings decode on the host.
    if (table.trellis_k != 0) {
        log.info("[qwen4] ple gather: cpu (a trellis table: its rings decode on the host)\n", .{});
        return null;
    }
    const b: Budget = .{
        .page = std.heap.pageSize(),
        .max_buffer = mlx.maxBufferLength(),
        .working_set = mlx.maxRecommendedWorkingSet(),
        .model_bytes = model_bytes,
    };
    const src = table.embedded;
    // A fresh anonymous mapping is page-aligned by construction.
    const base: usize = if (src != null) 0 else @intFromPtr(table.map.ptr);
    const len: usize = if (src) |e| @intCast(e.repackedLen()) else table.map.len;
    const arm = chooseArm(on, table.bits, base, len, b);
    if (arm == .off) {
        log.info("[qwen4] ple gather: cpu (--ple-gpu keeps the {d:.1} GB table resident for the GPU gather)\n", .{gb(len)});
        return null;
    }
    if (arm != .gpu) {
        log.info("[qwen4] ple gather: cpu ({s}: weights {d:.1} GB + table {d:.1} GB + headroom {d:.0} GB vs working set {d:.1} GB, max buffer {d:.1} GB)\n", .{ @tagName(arm), gb(model_bytes), gb(len), gb(HEADROOM), gb(b.working_set), gb(b.max_buffer) });
        return null;
    }
    const io = std.Io.Threaded.global_single_threaded.io();
    const t0 = std.Io.Timestamp.now(io, .boot);
    const tbl = (if (src) |e| wrapEmbedded(e) else wrap(table)) catch |e| {
        log.warn("[qwen4] ple gather: cpu (no-copy wrap failed: {s})\n", .{@errorName(e)});
        return null;
    };
    if (src != null) log.info("[qwen4] ple gather: embedded shards repacked into one buffer in {d:.1} s\n", .{@as(f64, @floatFromInt(t0.untilNow(io, .boot).nanoseconds)) / 1e9});
    log.info("[qwen4] ple gather: gpu (no-copy {d:.1} GB table buffer, weights {d:.1} GB, working set {d:.1} GB)\n", .{ gb(len), gb(model_bytes), gb(b.working_set) });
    return tbl;
}

var kernel: ?mlx.mlx_fast_metal_kernel = null;

pub const SOURCE = kernelSource("plugins/qwen4_exp/kernels/ple_gather.metal");
pub const HEADER = kernelSource("plugins/qwen4_exp/kernels/ple_gather_header.metal");

fn getKernel() !mlx.mlx_fast_metal_kernel {
    if (kernel) |k| return k;
    const in_names = [_][*:0]const u8{ "table", "ids", "prev", "params" };
    const out_names = [_][*:0]const u8{"out"};
    const ins = mlx.mlx_vector_string_new_data(&in_names, in_names.len);
    defer _ = mlx.mlx_vector_string_free(ins);
    const outs = mlx.mlx_vector_string_new_data(&out_names, out_names.len);
    defer _ = mlx.mlx_vector_string_free(outs);
    const k = mlx.mlx_fast_metal_kernel_new("msv_ple_gather", ins, outs, SOURCE, HEADER, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    kernel = k;
    return k;
}

/// `params` slots the kernel reads; the three tables sit at their `P_*` offsets.
const P_MULT = 13;
const P_VOCAB = P_MULT + qwen4.MAX_NGRAM_SIZE;
const P_OFFSETS = P_VOCAB + qwen4.MAX_HEADS;
const P_SCALE = P_OFFSETS + qwen4.MAX_HEADS;
const P_NVFP4 = P_SCALE + 1;
pub const P_LEN = P_NVFP4 + 1;
comptime {
    // ple_gather.metal reads these slots by literal.
    std.debug.assert(P_VOCAB == 21 and P_OFFSETS == 53 and P_SCALE == 85 and P_NVFP4 == 86);
}

/// The kernel's `params` for `n` ids.
pub fn params(tbl: Table, h: *const qwen4.NgramHash, t: *const qwen4.NgramTable, n: usize) [P_LEN]i64 {
    var p: [P_LEN]i64 = @splat(0);
    const head = [_]i64{ @intCast(n), h.n_heads, h.heads_per_ngram, h.ngram_size, h.eos, t.dim, t.bits, t.group_size, t.wcols, t.scols, @intCast(tbl.w_off), @intCast(tbl.s_off), @intCast(tbl.b_off) };
    @memcpy(p[0..P_MULT], &head);
    @memcpy(p[P_MULT..P_VOCAB], &h.multipliers);
    @memcpy(p[P_VOCAB..P_OFFSETS], &h.vocab);
    @memcpy(p[P_OFFSETS..P_SCALE], &h.offsets);
    p[P_SCALE] = @as(u32, @bitCast(tbl.scale));
    p[P_NVFP4] = @intFromBool(t.fmt == .nvfp4);
    return p;
}

/// bf16 `[S, n_heads * dim]` for the `S` ids of `ids` (any integer dtype, may be lazy),
/// hashed against `prev` (the `ngram_size - 1` tokens before them). GPU stream only.
pub fn embed(s: mlx.mlx_stream, tbl: Table, h: *const qwen4.NgramHash, t: *const qwen4.NgramTable, ids: mlx.mlx_array, prev: []const u32) !mlx.mlx_array {
    std.debug.assert(prev.len == h.ngram_size - 1);
    const n: usize = mlx.mlx_array_size(ids);
    const width: usize = @as(usize, h.n_heads) * t.dim;
    if (n * width > std.math.maxInt(c_int)) return error.PleChunkTooWide;
    const p = params(tbl, h, t, n);
    const params_arr = mlx.mlx_array_new_data(&p, &[_]c_int{P_LEN}, 1, .int64);
    defer _ = mlx.mlx_array_free(params_arr);
    var prev_i: [qwen4.MAX_NGRAM_SIZE]i32 = undefined;
    for (prev, 0..) |v, k| prev_i[k] = @intCast(v);
    const prev_arr = mlx.mlx_array_new_data(&prev_i, &[_]c_int{@intCast(prev.len)}, 1, .int32);
    defer _ = mlx.mlx_array_free(prev_arr);
    var ids_i = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids_i);
    try mlx.check(mlx.mlx_astype(&ids_i, ids, .int32, s));

    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const shape = [_]c_int{ @intCast(n), @intCast(width) };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &shape, 2, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @intCast(n * width), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    const ins = mlx.mlx_vector_array_new_data(&.{ tbl.arr, ids_i, prev_arr, params_arr }, 4);
    defer _ = mlx.mlx_vector_array_free(ins);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, try getKernel(), ins, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outs, 0));
    dispatches += 1;
    return out;
}

// ── tests ──

const testing = std.testing;

/// A synthetic table file for tests; `deinit` closes the table and removes the file.
pub const Fixture = struct {
    td: std.testing.TmpDir,
    table: qwen4.NgramTable,

    pub fn deinit(self: *Fixture) void {
        self.table.close();
        self.td.cleanup();
    }
};

fn randBf16(r: std.Random) u16 {
    // Normal values only (exponent 2^-17 .. 2^13): real tables carry no NaN, Inf or subnormal.
    const sign: u16 = @as(u16, r.int(u1)) << 15;
    const exp: u16 = r.intRangeAtMost(u16, 110, 140);
    return sign | (exp << 7) | r.int(u7);
}

/// A `ngram_table.bin` with random packed words, scales and biases. The header length is odd so
/// every region starts off a 4-byte boundary, as nothing in the format promises alignment.
pub fn writeFixture(bits: u32, rows: u64, dim: u32, gs: u32, seed: u64) !Fixture {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const a = testing.allocator;
    const hlen: usize = 509;
    const raw = bits == 16;
    const wbytes: u64 = if (raw) rows * dim * 2 else rows * (dim * bits / 32) * 4;
    const sbytes: u64 = if (raw) 0 else rows * (dim / gs) * 2;
    var hbuf: [hlen]u8 = @splat(' ');
    if (raw)
        _ = try std.fmt.bufPrint(&hbuf, "{{\"__metadata__\":{{\"format\":\"mlx-serve-ngram\",\"bits\":\"16\",\"group_size\":\"{d}\"}},\"weight\":{{\"dtype\":\"BF16\",\"shape\":[{d},{d}],\"data_offsets\":[0,{d}]}}}}", .{ gs, rows, dim, wbytes })
    else
        _ = try std.fmt.bufPrint(&hbuf, "{{\"__metadata__\":{{\"format\":\"mlx-serve-ngram\",\"bits\":\"{d}\",\"group_size\":\"{d}\"}},\"weight\":{{\"dtype\":\"U32\",\"shape\":[{d},{d}],\"data_offsets\":[0,{d}]}},\"scales\":{{\"dtype\":\"BF16\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}},\"biases\":{{\"dtype\":\"BF16\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}}}}", .{ bits, gs, rows, dim * bits / 32, wbytes, rows, dim / gs, wbytes, wbytes + sbytes, rows, dim / gs, wbytes + sbytes, wbytes + 2 * sbytes });
    const total: usize = @intCast(8 + hlen + wbytes + 2 * sbytes);
    const buf = try a.alloc(u8, total);
    defer a.free(buf);
    std.mem.writeInt(u64, buf[0..8], hlen, .little);
    @memcpy(buf[8 .. 8 + hlen], &hbuf);
    const data = buf[8 + hlen ..];
    if (raw) {
        var i: usize = 0;
        while (i < wbytes) : (i += 2) std.mem.writeInt(u16, data[i..][0..2], randBf16(r), .little);
    } else {
        r.bytes(data[0..@intCast(wbytes)]);
        var i: usize = @intCast(wbytes);
        while (i < data.len) : (i += 2) std.mem.writeInt(u16, data[i..][0..2], randBf16(r), .little);
    }
    var td = std.testing.tmpDir(.{});
    errdefer td.cleanup();
    const io = std.Io.Threaded.global_single_threaded.io();
    try td.dir.writeFile(io, .{ .sub_path = "ngram_table.bin", .data = buf });
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try td.dir.realPath(io, &pbuf);
    var full: [std.fs.max_path_bytes]u8 = undefined;
    const path = try std.fmt.bufPrint(&full, "{s}/ngram_table.bin", .{pbuf[0..root_len]});
    qwen4.warm_override = false;
    defer qwen4.warm_override = null;
    return .{ .td = td, .table = try qwen4.NgramTable.open(path) };
}

pub fn cpuRows(h: *const qwen4.NgramHash, t: *const qwen4.NgramTable, prev: []const u32, ids: []const u32) ![]u16 {
    const a = testing.allocator;
    const rows = try a.alloc(i64, ids.len * h.n_heads);
    defer a.free(rows);
    h.rowIds(prev, ids, rows);
    const host = try a.alloc(f32, rows.len * t.dim);
    defer a.free(host);
    t.gather(rows, host, 0);
    const out = try a.alloc(u16, host.len);
    for (host, out) |v, *o| o.* = qwen4.bf16Rne(v);
    return out;
}

const TEST_VOCAB: u32 = 1000;
const TEST_EOS: u32 = 999;
