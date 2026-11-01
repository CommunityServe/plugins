//! copied_from: src/plugins/minimax_m3/plugin.zig@37e90382b439f163474f99d3c78522ae88caaf12
//! MiniMax M3.1 Flash's prepared text expert banks behind the model table.
const std = @import("std");
const abi = @import("mlx_serve").abi.plugin;
const mlx = @import("../../mlx/ffi.zig");
/// The bank streams through the expert streamer (module `mlx_stream`); a build without it (Linux, Windows) claims and
/// parses a pack and refuses its load.
const streamer = @import("build_options").mlx_stream;
const sdk = if (streamer) @import("mlx_stream").sdk else struct {};
const no_streamer = "minimax_m3: its expert bank needs the expert streamer, which this build does not include";
const legacy = @import("legacy.zig");
const bank = @import("bank.zig");
const resident_header = @import("resident_header.zig");
const Resident = @import("../../sdk/weights/resident.zig").Resident;
pub const config = @import("config.zig");
pub const plugin = config.plugin;
const gpa = std.heap.c_allocator;
const Verdict = abi.Verdict;
fn plannedRows(b: sdk.MemoryBill, baseline: u64, room: u64, forced: ?u32) !u32 {
    return plannedRowsWithin(b, baseline, room, forced, null, 128);
}
/// `plannedRows` for a bank of `experts` a layer, the memory fill bounded by `cap` (the pack's prepared row count);
/// `forced` rows stay exact.
fn plannedRowsWithin(b: sdk.MemoryBill, baseline: u64, room: u64, forced: ?u32, cap: ?u32, experts: u32) !u32 {
    const filled = if (forced == null) (sdk.fill(b, baseline, room, experts, 4) catch return error.MiniMaxStreamRoomTooSmall).prompt else 0;
    const rows = forced orelse @min(filled, cap orelse experts);
    if (rows < 4 or rows > experts) return error.MiniMaxExpertRows;
    sdk.admit(b, baseline, .{ .prompt = rows, .decode = rows }, room) catch return error.MiniMaxStreamRoomTooSmall;
    return rows;
}
/// The parse's settings (the pack's prepared ones under the model settings): an `expert_rows` there was sized for
/// some box, so it bounds the memory fill instead of forcing it; returned as that bound.
fn applyPreparedSettings(c: *legacy.Config, raw: []const u8) !?u32 {
    try applyLoadSettings(c, raw);
    if (!c.expert_rows_forced) return null;
    c.expert_rows_forced = false;
    return c.expert_rows;
}
fn applyLoadSettings(c: *legacy.Config, raw: []const u8) !void {
    if (raw.len == 0) return;
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, raw, .{});
    defer parsed.deinit();
    c.applySettings(parsed.value);
    if (c.settings_error) return error.MiniMaxUnsupportedSettings;
    if (parsed.value == .object) {
        if (parsed.value.object.get("ctx_size")) |v| if (v != .null) {
            if (v != .integer or v.integer < 1 or v.integer > c.model.max_position_embeddings) return error.MiniMaxContextUnsupported;
            c.context = @intCast(v.integer);
        };
        if (parsed.value.object.get("kv_quant")) |v| if (v != .null) {
            const kv = @import("../../sdk/blocks/attention/kv_quant.zig").KVQuantConfig.fromJsonValue(v) orelse return error.MiniMaxKvPrecisionUnsupported;
            try legacy.applyLoadFacts(c, &.{ .kv_quant_bits = kv.bits, .wired_margin_bytes = 0 });
        };
        if (parsed.value.object.get("stream_experts")) |v| if (v != .null) {
            const setting = @import("../../sdk/weights/streamed.zig").Setting.of(v) orelse return error.MiniMaxStreamedExpertsRequired;
            try legacy.applyLoadFacts(c, &.{ .expert_rows = setting.slotRows(c.model.n_routed_experts), .wired_margin_bytes = 0 });
        };
    }
}
pub const table = abi.ModelTable{
    .name = abi.Str.of(plugin.name),
    .claims = &config.claims,
    .parse = &config.parse,
    .free_parsed = &config.freeParsed,
    .bill = &bill,
    .load = &load,
    .unload = &unload,
    .forward = &forward,
    .release_slot = &releaseSlot,
    .cache = &.{ .layout = &layout, .state_bytes = &stateBytes },
    .weights = &.{ .resident_bytes = &residentBytes },
};
/// The tiny pack the conformance suite runs: the released architecture at fixture size (`fixture.zig`).
pub fn conformanceFixture(a: std.mem.Allocator) !@import("mlx_serve").sdk.fixture.Fixture {
    return .{ .files = try @import("fixture.zig").files(a, 7) };
}
pub const Model = struct {
    module: *legacy.Module,
    weights: sdk.Weights,
    model_dir: []u8,
    reader_claimed: bool = true,
    slot: u32 = 0,
};
pub fn layout(p: *const abi.Parsed, span: ?*const abi.LayerSpan, out: [*]abi.LayerCache, cap: u32) callconv(.c) u32 {
    const c = config.of(p).model;
    const count = @min(if (span) |s| s.count else c.n_layers, cap);
    for (out[0..count]) |*l| l.* = .{ .kind = .full, .kv_heads = c.n_kv_heads, .k_dim = c.head_dim, .v_dim = c.head_dim };
    return count;
}
fn bill(p: *const abi.Parsed, shape: *const abi.Shape, out: *abi.Bill) callconv(.c) void {
    const c = config.of(p).model;
    out.* = .{ .kv_per_token = kvTokenBytes(&c, shape.kv_bits), .prefill_attn_keys = &attnKeys, .prefill_bytes = &prefillBytes };
}
/// A token's K, V and index key at `kv_bits`: the Q8 cache's codes and scales, else BF16.
fn kvTokenBytes(c: anytype, kv_bits: u8) u64 {
    if (kv_bits == 8) return @import("q8_reference.zig").cacheBytes(c.n_layers, c.n_kv_heads, c.head_dim, c.index_head_dim, 1) catch c.kvBytes(1);
    return c.kvBytes(1);
}
fn attnKeys(_: *const abi.Parsed, seq: u64) callconv(.c) u64 {
    return @min(seq, 128 * 17);
}
fn prefillBytes(p: *const abi.Parsed, seq: u64, _: u32, kv_bits: u8, _: u64) callconv(.c) u64 {
    const c = config.of(p).model;
    const kv = kvTokenBytes(&c, kv_bits) *| seq;
    const rows: u64 = @min(seq, 8);
    const scratch = 256 * @as(u64, c.hidden_size) + 8 * @as(u64, c.n_heads) * seq + rows * 4 * c.hidden_size * 4;
    return (kv +| scratch +| (1 << 30)) *| 5 / 4;
}
fn residentBytes(p: *const abi.Parsed, dir: abi.Str, _: abi.Str, shards: u64, out: *u64) callconv(.c) Verdict {
    if (comptime !streamer) return Verdict.refuse(.unsupported, no_streamer);
    const io = std.Io.Threaded.global_single_threaded.io();
    const geometry = bank.inspect(gpa, io, dir.slice()) catch |e| return Verdict.named(.unsupported, @errorName(e));
    bank.requireModel(&geometry, &config.of(p).model) catch |e| return Verdict.named(.unsupported, @errorName(e));
    const resident = resident_header.residentBytesBound(gpa, io, dir.slice(), 64 << 30) catch |e| return Verdict.named(.unsupported, @errorName(e));
    out.* = residentBound(&geometry, resident, shards);
    return .ok;
}
/// The preflight floor matches what the stream allocates: top-k rows per layer, its transient rows and the reader staging.
fn residentBound(g: anytype, resident: u64, shards: u64) u64 {
    const rows = @as(u64, bank.routed_top_k) * g.num_moe_layers + @import("experts.zig").chunk_routes;
    return @max(shards, resident) +| rows * g.slotRowBytes() +| legacy.readerStagingBytes(g);
}
/// Why this device cannot load the model, or null: the sparse attention and Q8 cache kernels are Metal kernels, and
/// their threadgroups exceed what a virtual GPU grants (there they abort the process).
fn deviceRefusal(metal: bool, virtual: bool) ?Verdict {
    if (!metal) return Verdict.refuse(.unsupported, "minimax_m3: its kernels run on Metal only");
    if (virtual) return Verdict.refuse(.unsupported, "minimax_m3: its kernels need a physical Apple GPU (a virtual GPU caps their threadgroups)");
    return null;
}
fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    if (comptime !streamer) {
        out.* = Verdict.refuse(.unsupported, no_streamer);
        return null;
    }
    if (deviceRefusal(mlx.metalKernelsAvailable(), mlx.virtualGpu())) |why| {
        out.* = why;
        return null;
    }
    // A CPU stream (the conformance suite's) runs on the GPU stream instead, as deepseek_v41 does.
    const s = gpuOf(.{ .ctx = ctx.stream.ctx }) orelse {
        out.* = Verdict.refuse(.unsupported, "minimax_m3: its kernels run on Metal only");
        return null;
    };
    return loadModel(ctx, s) catch |e| {
        out.* = Verdict.named(switch (e) {
            error.ExpertReaderInUse => .busy,
            error.MiniMaxStreamRoomTooSmall => .too_large,
            else => .failed,
        }, @errorName(e));
        return null;
    };
}
fn gpuOf(s: mlx.mlx_stream) ?mlx.mlx_stream {
    if (!mlx.metalKernelsAvailable()) return null;
    if (mlx.streamIsGpu(s)) return s;
    return mlx.gpuStream();
}
/// The slot rows a layer: the memory fill within the host's `room`, or with no room stated (0: a load outside a
/// server, as the conformance suite's) the stated rows, else the fewest the stream runs (top-k), as glm5_next does.
fn loadRows(memory: sdk.MemoryBill, baseline: u64, room: u64, forced: ?u32, cap: ?u32, experts: u32) !u32 {
    if (room == 0) return forced orelse bank.routed_top_k;
    return plannedRowsWithin(memory, baseline, room, forced, cap, experts);
}
fn loadModel(ctx: *const abi.LoadCtx, s: mlx.mlx_stream) !*Model {
    const cfg = config.of(ctx.parsed);
    try @import("mlx_stream").sdk_ext.expert.takeReader();
    errdefer @import("mlx_stream").sdk_ext.expert.giveReader();
    const source = ctx.weights orelse return error.MiniMaxResidentSourceRequired;
    var resident = Resident.take(source) orelse return error.MiniMaxResidentSourceRequired;
    if (resident.w8.map.count() != 0) {
        resident.deinit();
        return error.MiniMaxInt8OverlayUnsupported;
    }
    var weights = sdk.Weights.init(resident.allocator);
    weights.map = resident.map;
    resident.map = std.StringHashMap(mlx.mlx_array).init(resident.allocator);
    resident.deinit();
    errdefer weights.deinit();
    const dir = try gpa.dupe(u8, ctx.model_dir.slice());
    errdefer gpa.free(dir);
    var c: legacy.Config = .{
        .model = cfg.model,
        .model_dir = dir,
        .residents_format = cfg.residents_format,
        .expert_format = switch (cfg.bank_format) { .nvfp4 => .nvfp4, .affine => .affine, .bf16 => .bf16 },
    };
    const cap = try applyPreparedSettings(&c, cfg.settings_json);
    try applyLoadSettings(&c, ctx.settings_json.slice());
    const io = std.Io.Threaded.global_single_threaded.io();
    const memory = try legacy.memoryBill(gpa, io, &c);
    defer memory.free(gpa);
    const room = ctx.stream_room;
    const forced = if (c.expert_rows_forced) c.expert_rows else null;
    const experts = c.model.n_routed_experts;
    _ = try loadRows(memory, 0, room, forced, cap, experts);
    try weights.materialize();
    const system = @import("../../sdk/util/system.zig");
    var active: usize = 0;
    try mlx.check(mlx.mlx_get_active_memory(&active));
    const used = @max(active, system.getTotalMemBytes() -| system.getAvailableMemBytes());
    const residents = try resident_header.residentBytesBound(gpa, io, dir, 64 << 30);
    // Materialized residents are already in `used`; the bill adds them once beside the bank and transients.
    c.expert_rows = try loadRows(memory, used -| residents, room, forced, cap, experts);
    const old: sdk.LoadCtx = .{
        .gpa = gpa,
        .io = std.Io.Threaded.global_single_threaded.io(),
        .stream = .{ .ctx = s.ctx },
        .weights = &weights,
        .loader = &sdk.loader,
        .facts = .{ .wired_margin_bytes = ctx.wired_margin_bytes },
        .ceiling = room,
    };
    const module = try legacy.Module.init(&old, &c);
    errdefer module.deinit();
    const m = try gpa.create(Model);
    m.* = .{ .module = module, .weights = weights, .model_dir = dir };
    return m;
}
fn unload(mp: *anyopaque) callconv(.c) void {
    if (comptime !streamer) unreachable; // no load succeeds without it
    const m: *Model = @ptrCast(@alignCast(mp));
    m.module.deinit();
    m.weights.deinit();
    gpa.free(m.model_dir);
    if (m.reader_claimed) @import("mlx_stream").sdk_ext.expert.giveReader();
    gpa.destroy(m);
}
fn releaseSlot(mp: *anyopaque, slot: u32) callconv(.c) void {
    if (comptime !streamer) unreachable; // no load succeeds without it
    const m: *Model = @ptrCast(@alignCast(mp));
    if (slot == m.slot) {
        m.module.requestEnd();
        m.slot = 0;
    }
}
fn stateBytes(mp: *anyopaque) callconv(.c) u64 {
    if (comptime !streamer) unreachable; // no load succeeds without it
    const m: *Model = @ptrCast(@alignCast(mp));
    const c = m.module.c.model;
    var bytes: u64 = 0;
    for (m.module.caches) |cache| {
        bytes +|= switch (cache.bits) {
            0 => @as(u64, cache.capacity) * (2 * @as(u64, c.n_kv_heads) * c.head_dim + c.index_head_dim) * 2,
            8 => @import("q8_reference.zig").cacheBytes(1, c.n_kv_heads, c.head_dim, c.index_head_dim, cache.capacity) catch return std.math.maxInt(u64),
            else => return std.math.maxInt(u64),
        };
    }
    return bytes;
}
fn forward(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) Verdict {
    if (comptime !streamer) unreachable; // no load succeeds without it
    const m: *Model = @ptrCast(@alignCast(mp));
    if (f.slots.len != 1 or f.rows == 0 or f.capture_layers.len != 0 or f.want_hidden or f.media.ctx != null)
        return Verdict.refuse(.unsupported, "minimax_m3: one text slot with logits only");
    if (f.mode != .prefill and f.mode != .decode and f.mode != .verify) return Verdict.refuse(.unsupported, "minimax_m3: forward mode is unsupported");
    const ids = tokenIds(f) catch |e| return Verdict.named(.failed, @errorName(e));
    defer if (f.ids.ctx != null) gpa.free(ids);
    const pos: u32 = if (f.positions.len == 1) f.positions.slice()[0] else 0;
    const slot = f.slots.slice()[0];
    if (f.mode == .prefill and pos != 0) return Verdict.refuse(.unsupported, "minimax_m3: whole prompt prefill starts at position zero");
    // Position 0 starts the slot's sequence over, whatever the mode; later rows continue the active slot's.
    if (pos != 0 and pos != m.module.position()) return Verdict.refuse(.failed, "minimax_m3: decode position differs from module state");
    if (pos != 0 and m.slot != slot) return Verdict.refuse(.failed, "minimax_m3: decode slot differs from active slot");
    const logits = (if (f.mode == .prefill and f.logits != .all_rows)
        wholePrompt(m, slot, ids)
    else serialRows(m, slot, pos, ids, f.logits == .all_rows)) catch |e| return Verdict.named(.failed, @errorName(e));
    if (f.logits == .none) _ = mlx.mlx_array_free(logits) else out.logits = .{ .ctx = logits.ctx };
    return .ok;
}
fn wholePrompt(m: *Model, slot: u32, ids: []const u32) !mlx.mlx_array {
    m.slot = slot;
    return m.module.prefill(ids, .{ .prompt_tokens = ids.len, .max_tokens = 0, .host_context = m.module.c.context });
}
/// One row at a time through the serial decode (a verify, a decode from position 0, an all-row prompt): the module
/// has no batched row path, so each row is its own step; `all_rows` stacks every row's logits `[rows, vocab]`.
fn serialRows(m: *Model, slot: u32, pos: u32, ids: []const u32, all_rows: bool) !mlx.mlx_array {
    var rows: [16]mlx.mlx_array = undefined;
    if (all_rows and ids.len > rows.len) return error.MiniMaxVerifyRows;
    var n: usize = 0;
    errdefer for (rows[0..n]) |r| {
        _ = mlx.mlx_array_free(r);
    };
    var last: ?mlx.mlx_array = null;
    errdefer if (last) |l| {
        _ = mlx.mlx_array_free(l);
    };
    for (ids, 0..) |id, i| {
        const one = [_]u32{id};
        const row = if (i == 0 and pos == 0) try wholePrompt(m, slot, &one) else try m.module.step(&one);
        if (all_rows) {
            rows[n] = row;
            n += 1;
        } else {
            if (last) |l| _ = mlx.mlx_array_free(l);
            last = row;
        }
    }
    if (!all_rows) {
        const l = last.?;
        last = null;
        return l;
    }
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (rows[0..n]) |r| _ = mlx.mlx_vector_array_append_value(vec, r);
    var stacked = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(stacked);
    try mlx.check(mlx.mlx_concatenate_axis(&stacked, vec, 0, m.module.g.s));
    for (rows[0..n]) |r| _ = mlx.mlx_array_free(r);
    n = 0;
    return stacked;
}
fn tokenIds(f: *const abi.Forward) ![]const u32 {
    if (f.ids.ctx == null) {
        if (f.tokens.len != f.rows) return error.MiniMaxTokenRowsMismatch;
        return f.tokens.slice();
    }
    const arr: mlx.mlx_array = .{ .ctx = f.ids.ctx };
    var i32s = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(i32s);
    try mlx.check(mlx.mlx_astype(&i32s, arr, .int32, mlx.gpuStream()));
    try mlx.check(mlx.mlx_array_eval(i32s));
    if (mlx.mlx_array_size(i32s) != f.rows) return error.MiniMaxTokenRowsMismatch;
    const data = mlx.mlx_array_data_int32(i32s) orelse return error.MiniMaxTokenReadback;
    const out = try gpa.alloc(u32, f.rows);
    errdefer gpa.free(out);
    for (out, data[0..f.rows]) |*id, value| {
        if (value < 0) return error.MiniMaxTokenOutOfRange;
        id.* = @intCast(value);
    }
    return out;
}
test {
    _ = @import("fixture.zig");
    _ = @import("parity_test.zig");
    _ = @import("geometry.zig");
    _ = @import("bank.zig");
    _ = @import("affine_bank.zig");
    _ = @import("bf16_bank.zig");
    _ = @import("module.zig");
}
test "MiniMax bank fill leaves the residents, transient peak and other live memory funded" {
    const gib = 1 << 30;
    const b: sdk.MemoryBill = .{ .per_row = 2 * gib, .terms = &.{
        .{ .name = "residents", .bytes = .{ 40 * gib, 40 * gib }, .at_construction = true },
        .{ .name = "scratch", .bytes = .{ 4 * gib, 2 * gib }, .at_construction = false },
    } };
    try std.testing.expectEqual(@as(u32, 14), try plannedRows(b, 8 * gib, 80 * gib, null));
    try std.testing.expectEqual(@as(u32, 14), try plannedRows(b, 8 * gib, 80 * gib, 14));
    try std.testing.expectError(error.MiniMaxStreamRoomTooSmall, plannedRows(b, 8 * gib, 80 * gib, 15));
    try std.testing.expectError(error.MiniMaxStreamRoomTooSmall, plannedRows(b, 8 * gib, 58 * gib, null));
    try std.testing.expectError(error.MiniMaxExpertRows, plannedRows(b, 0, 80 * gib, 3));
}
test "MiniMax load refuses a device without Metal or a virtual GPU unsupported, by name" {
    try std.testing.expect(deviceRefusal(true, false) == null);
    for ([_][2]bool{ .{ false, false }, .{ false, true }, .{ true, true } }) |d| {
        const v = deviceRefusal(d[0], d[1]) orelse return error.NotRefused;
        try std.testing.expectEqual(abi.Verdict.Code.unsupported, v.code);
        try std.testing.expect(std.mem.indexOf(u8, v.why.slice(), if (d[0]) "physical Apple GPU" else "Metal only") != null);
    }
}
test "MiniMax prepared expert_rows bound the memory fill; a host stream_experts setting stays exact" {
    const gib = 1 << 30;
    const b: sdk.MemoryBill = .{ .per_row = 2 * gib, .terms = &.{
        .{ .name = "residents", .bytes = .{ 40 * gib, 40 * gib }, .at_construction = true },
        .{ .name = "scratch", .bytes = .{ 4 * gib, 2 * gib }, .at_construction = false },
    } };
    // A prepared count over what the bill fits bounds the fill and never refuses the load.
    var c: legacy.Config = .{};
    const cap = try applyPreparedSettings(&c, "{\"expert_rows\":86}");
    try std.testing.expectEqual(@as(?u32, 86), cap);
    try std.testing.expect(!c.expert_rows_forced);
    try std.testing.expectEqual(@as(u32, 14), try plannedRowsWithin(b, 8 * gib, 80 * gib, null, cap, 128));
    try std.testing.expectEqual(@as(u32, 10), try plannedRowsWithin(b, 8 * gib, 80 * gib, null, 10, 128));
    // A host setting after the prepared one is exact again, refused when it does not fit.
    try applyLoadSettings(&c, "{\"stream_experts\":16}");
    try std.testing.expect(c.expert_rows_forced);
    try std.testing.expectError(error.MiniMaxStreamRoomTooSmall, plannedRowsWithin(b, 8 * gib, 80 * gib, c.expert_rows, cap, 128));
    var fresh: legacy.Config = .{};
    try std.testing.expectEqual(@as(?u32, null), try applyPreparedSettings(&fresh, "{}"));
    try std.testing.expectError(error.MiniMaxUnsupportedSettings, applyPreparedSettings(&fresh, "{\"expert_rows\":2}"));
}
test "MiniMax load settings honor dense and Q8 caches and reject unsupported precision" {
    var c: legacy.Config = .{};
    try applyLoadSettings(&c, "{\"ctx_size\":8192,\"kv_quant\":\"off\"}");
    try std.testing.expectEqual(@as(u32, 8192), c.context);
    try std.testing.expectEqual(@as(u8, 0), c.kv_quant_bits);
    try applyLoadSettings(&c, "{\"kv_quant\":8}");
    try std.testing.expectEqual(@as(u8, 8), c.kv_quant_bits);
    try applyLoadSettings(&c, "{\"ctx_size\":null,\"kv_quant\":null}");
    try std.testing.expectEqual(@as(u32, 8192), c.context);
    try std.testing.expectEqual(@as(u8, 8), c.kv_quant_bits);
    try std.testing.expectError(error.MiniMaxKvPrecisionUnsupported, applyLoadSettings(&c, "{\"kv_quant\":4}"));
    try std.testing.expectError(error.MiniMaxKvPrecisionUnsupported, applyLoadSettings(&c, "{\"kv_quant\":true}"));
    try std.testing.expectError(error.MiniMaxContextUnsupported, applyLoadSettings(&c, "{\"ctx_size\":1048577}"));
}
test "MiniMax load settings use the shared stream_experts row policy" {
    var c: legacy.Config = .{};
    try applyLoadSettings(&c, "{\"stream_experts\":16}");
    try std.testing.expectEqual(@as(u32, 16), c.expert_rows);
    try std.testing.expect(c.expert_rows_forced);
    try applyLoadSettings(&c, "{\"stream_experts\":true}");
    try std.testing.expectEqual(@as(u32, 32), c.expert_rows);
    try std.testing.expectError(error.MiniMaxStreamedExpertsRequired, applyLoadSettings(&c, "{\"stream_experts\":false}"));
    try std.testing.expectError(error.MiniMaxExpertRows, applyLoadSettings(&c, "{\"stream_experts\":2}"));
}
test "MiniMax bill prices the configured KV precision: Q8 a token as its cache holds it, BF16 otherwise" {
    const m: @import("geometry.zig").Config = .{};
    var cfg: config.Config = .{ .model = m, .bank_format = .nvfp4, .residents_format = .bf16, .settings_json = &.{} };
    const parsed: abi.Parsed = .{ .private = &cfg };
    var out: abi.Bill = .{};
    bill(&parsed, &.{ .kv_bits = 8 }, &out);
    try std.testing.expectEqual(try @import("q8_reference.zig").cacheBytes(m.n_layers, m.n_kv_heads, m.head_dim, m.index_head_dim, 4096), out.kv_per_token * 4096);
    try std.testing.expect(out.kv_per_token < m.kvBytes(1));
    bill(&parsed, &.{}, &out);
    try std.testing.expectEqual(m.kvBytes(1), out.kv_per_token);
    try std.testing.expect(prefillBytes(&parsed, 4096, 0, 8, 0) < prefillBytes(&parsed, 4096, 0, 0, 0));
}
test "MiniMax state bill includes allocated capacity and Q8 scales" {
    const Cache = @import("attention.zig").Cache;
    var caches = [_]Cache{ .{ .len = 1, .capacity = 256 }, .{ .len = 1, .capacity = 512, .bits = 8 } };
    var module: legacy.Module = undefined;
    module.c = .{};
    module.pos = 1;
    module.caches = &caches;
    var model: Model = undefined;
    model.module = &module;
    try std.testing.expectEqual(@as(u64, 589_824 + 663_552), stateBytes(&model));
    for (&caches) |*cache| cache.capacity = 0;
    try std.testing.expectEqual(@as(u64, 0), stateBytes(&model));
}
test "MiniMax preflight bound funds the top-k rows, the stream's transient rows and the reader staging" {
    const g = try bank.Geometry.init(6144, 3072, 128, 3, 57);
    const resident: u64 = 20 << 30;
    const rows = @as(u64, bank.routed_top_k) * g.num_moe_layers + @import("experts.zig").chunk_routes;
    try std.testing.expectEqual(resident + rows * g.slotRowBytes() + legacy.readerStagingBytes(&g), residentBound(&g, resident, 0));
    try std.testing.expectEqual(residentBound(&g, resident, 0) + (1 << 30), residentBound(&g, 0, resident + (1 << 30)));
}
