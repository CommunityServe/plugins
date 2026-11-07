//! Tensor parallel across Macs (`SUSHI_DS41_TP=jaccl|ring`, off by default). Each rank serves a pack repacked to its
//! half of every expert and attention head (`bench/studio-cluster/stq-tp-split.py` in local-llm), and the two partial
//! sums of a layer (attention out, MoE out) are summed over MLX's distributed group before they join the stream. The
//! peers come from MLX's own environment: `MLX_RANK` plus `MLX_IBV_DEVICES` and `MLX_JACCL_COORDINATOR` for jaccl, or
//! `MLX_HOSTFILE` for ring. Every rank must see the same requests in the same order.
const std = @import("std");
const mlx = @import("sushi/mlx.zig");
const log = @import("sushi/log.zig");
const sc_mod = @import("sushi/dsv41/dsv41_scope.zig");

const A = mlx.mlx_array;
const Group = extern struct { ctx: ?*anyopaque = null };

extern "c" fn mlx_distributed_group_new() Group;
extern "c" fn mlx_distributed_is_available(bk: ?[*:0]const u8) bool;
extern "c" fn mlx_distributed_init(res: *Group, strict: bool, bk: ?[*:0]const u8) c_int;
extern "c" fn mlx_distributed_group_rank(group: Group) c_int;
extern "c" fn mlx_distributed_group_size(group: Group) c_int;
extern "c" fn mlx_distributed_all_sum(res: *A, x: A, group: Group, s: mlx.mlx_stream) c_int;

var group: Group = .{};
var cpu: mlx.mlx_stream = .{};
pub var rank: u32 = 0;
pub var size: u32 = 1;
pub var sums: u64 = 0;
/// Set when a call failed after the other ranks were told to run it: the ranks' states may differ, so nothing more is
/// served until both restart.
pub var lost: bool = false;

pub fn on() bool {
    return size > 1;
}

pub fn init() !void {
    const raw = std.c.getenv("SUSHI_DS41_TP") orelse return;
    const backend = std.mem.span(raw);
    if (!std.mem.eql(u8, backend, "jaccl") and !std.mem.eql(u8, backend, "ring")) {
        log.err("[sushi_ds41] SUSHI_DS41_TP=\"{s}\": use jaccl (RDMA) or ring (TCP)\n", .{backend});
        return error.SushiTpBackend;
    }
    if (!mlx_distributed_is_available(raw)) {
        log.err("[sushi_ds41] tensor parallel: MLX has no {s} backend here\n", .{backend});
        return error.SushiTpBackend;
    }
    var g = mlx_distributed_group_new();
    try mlx.check(mlx_distributed_init(&g, true, raw));
    group = g;
    rank = @intCast(mlx_distributed_group_rank(g));
    size = @intCast(mlx_distributed_group_size(g));
    cpu = mlx.mlx_default_cpu_stream_new();
    log.info("[sushi_ds41] tensor parallel: rank {d} of {d} over {s}\n", .{ rank, size, backend });
}

/// `x` summed over every rank: lazy, on the CPU stream, owned by `sc`. `x` itself when tensor parallel is off.
pub fn sum(sc: *sc_mod.Scope, x: A) !A {
    if (!on()) return x;
    sums += 1;
    var o = mlx.mlx_array_new();
    return sc.res(mlx_distributed_all_sum(&o, x, group, cpu), &o);
}

/// Rank 0's `words` on every rank, in place: the others contribute zeros, and an integer sum with zeros is exact.
pub fn bcast(words: []u32) !void {
    if (words.len == 0) return;
    if (rank != 0) @memset(words, 0);
    const shape = [_]c_int{@intCast(words.len)};
    const x = mlx.mlx_array_new_data(words.ptr, &shape, 1, .uint32);
    defer _ = mlx.mlx_array_free(x);
    var o = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx_distributed_all_sum(&o, x, group, cpu));
    try mlx.check(mlx.mlx_array_eval(o));
    const data = mlx.mlx_array_data_uint32(o) orelse return error.SushiTpBcast;
    @memcpy(words, data[0..words.len]);
}

// ── the leader's stream of model calls ──
//
// Rank 0 decides everything and every other rank replays its model calls. A call that issues collectives (`Op`) goes
// in lock-step: a header, then the queued bookkeeping and the call's payload. Bookkeeping (`Book`) issues none, so it
// waits in a queue for the next call.

pub const Op = enum(u32) { prefill = 1, draft, verify, finish };
pub const Book = enum(u32) { reset = 1, arm, snap_take, snap_slice, snap_free, restore };

pub const Header = extern struct {
    seq: u32 = 0,
    pos: u32 = 0,
    op: u32 = 0,
    rows: u32 = 0,
    arg: u32 = 0,
    book_words: u32 = 0,
    payload_words: u32 = 0,
    want_all: u32 = 0,
    ablate: u32 = 0,
    strength_bits: u32 = 0,
    media_ndim: u32 = 0,
    media_shape: [3]u32 = .{ 0, 0, 0 },
    media_dtype: u32 = 0,
    media_offset: u32 = 0,

    const words = @sizeOf(Header) / 4;
};

comptime {
    std.debug.assert(@sizeOf(Header) == 64);
}

/// One bookkeeping entry: its kind and up to four arguments.
pub const record_words = 5;

var book: std.ArrayList(u32) = .empty;
var book_mu: std.c.pthread_mutex_t = .{};
var seq: u32 = 0;

/// Rank 0: queues a bookkeeping entry for the next call.
pub fn queue(kind: Book, args: [4]u32) !void {
    if (!on() or rank != 0) return;
    _ = std.c.pthread_mutex_lock(&book_mu);
    defer _ = std.c.pthread_mutex_unlock(&book_mu);
    try book.append(std.heap.c_allocator, @intFromEnum(kind));
    try book.appendSlice(std.heap.c_allocator, &args);
}

/// Calls sent (rank 0) or received so far: a failure after this moved has left the other ranks mid-call.
pub fn sent() u32 {
    return seq;
}

/// Rank 0: sends `h` (its sequence and the queued bookkeeping filled in), then the bookkeeping and `payload`.
pub fn lead(h: Header, payload: []const u32) !void {
    _ = std.c.pthread_mutex_lock(&book_mu);
    defer _ = std.c.pthread_mutex_unlock(&book_mu);
    var head = h;
    seq +%= 1;
    head.seq = seq;
    head.book_words = @intCast(book.items.len);
    head.payload_words = @intCast(payload.len);
    var hw: [Header.words]u32 = @bitCast(std.mem.toBytes(head));
    try bcast(&hw);
    const body = try std.heap.c_allocator.alloc(u32, book.items.len + payload.len);
    defer std.heap.c_allocator.free(body);
    @memcpy(body[0..book.items.len], book.items);
    @memcpy(body[book.items.len..], payload);
    try bcast(body);
    book.clearRetainingCapacity();
}

/// A follower's next call: its header and body (bookkeeping, then payload; the caller frees the body).
pub const Call = struct { h: Header, body: []u32 };

/// Every other rank: waits for rank 0's next call. A sequence gap means the ranks no longer follow one stream.
pub fn follow() !Call {
    var hw: [Header.words]u32 = undefined;
    try bcast(&hw);
    const h = std.mem.bytesToValue(Header, std.mem.asBytes(&hw));
    seq +%= 1;
    if (h.seq != seq) return error.SushiTpSequence;
    const body = try std.heap.c_allocator.alloc(u32, h.book_words + h.payload_words);
    errdefer std.heap.c_allocator.free(body);
    try bcast(body);
    return .{ .h = h, .body = body };
}
