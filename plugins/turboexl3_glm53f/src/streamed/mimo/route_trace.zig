//! `mimo-route-trace-v2` (design-hit-rate §4a; v2 adds the MTP lane's M-row decode calls): one request's per-token
//! per-layer top-8 routes for the prompt and decode, the plan decode returned for each call, the residency at the phase
//! change and at the end, and the next layer's predictor top-16. Armed only in profile builds
//! (`MIMO_ROUTE_TRACE=<prefix>`, read once at construction): every hook sits behind `profile.enabled`, so a served build
//! has no field, branch or call for it. The type itself is plain so CPU tests exercise it in any build.
//!
//! A decode call is one routed forward of M rows (M = 1: a serial step; M = 1 + drafts <= `max_call_rows`: the MTP
//! verify of one round). Files (little endian), for request k: `<prefix>-r<k>.json` (the header) and
//!   `.prompt.u8`       [layers][prompt_tokens][8]  ids as `seedPrefill` received them (router order);
//!   `.decode.u8`       per call c in order: [layers][M_c][8]  the call's ids per row (router order); with every
//!                      M_c = 1 this is v1's [decode_calls][layers][8], byte for byte;
//!   `.calls.u8`        [decode_calls]              M_c (v2);
//!   `.accepted.u8`     [decode_calls]              the round's accepted drafts for an MTP verify call, 0xFF for a
//!                      call that is no round's verify (a serial step) (v2);
//!   `.plan.u8`         [decode_calls][layers][3]   n_hits, n_misses, n_persistent of the returned plan (the union's);
//!   `.resident0.u16`   [layers][max(rows)]          slot_to_expert right after the phase change (0xFFFF empty);
//!   `.resident_end.u16`[layers][max(rows)]          slot_to_expert when the decode phase ends;
//!   `.pred.bin`        [decode_calls][layers-1] x (16 u8 ids + 16 f32 scores): the top-16 of the scores layer L's
//!                      call carried for layer L+1 (score descending, ties by id; 0xFF / NaN when the call had none;
//!                      an M-row call: each expert's max over the rows).
const std = @import("std");
const env = @import("mlx_host").env;
const kset = @import("../sdk_ext/kernel_set.zig");
const expert_policy = @import("../sdk_ext.zig").expert.policy;

pub const format = "mimo-route-trace-v2";
pub const top_k: u32 = 8;
pub const pred_k: u32 = 16;
/// The decode calls a trace keeps (4 x the 1,024-token cell); later calls set `overflow`.
pub const max_decode_calls: u32 = 4096;
/// The widest decode call a trace keeps (the MTP verify: t1 + 3 drafts); a wider decode-phase call is counted in
/// `wide_decode_calls`, not kept.
pub const max_call_rows: u32 = 4;
/// `.accepted.u8` of a call that is no round's verify.
pub const not_a_round: u8 = 0xFF;

pub const Io = struct { hits: u64 = 0, misses: u64 = 0, bytes: u64 = 0, records: u64 = 0 };

/// MiMo's trace: u8 expert ids (at most 256 experts), `mimo-route-trace-v2`'s files.
pub const Trace = TraceOf(u8, format);

/// The trace over expert ids of type `Id`: u8 (at most 256 experts: the v2 files above, byte for byte) or u16 (at most
/// 512 experts: the same layout with every expert id two bytes little endian, `.prompt.u16` and `.decode.u16`, and the
/// predictor's top-16 ids u16 before their f32 scores; GLM-5.3-Flash's 288 experts), named `format_name` in the header.
pub fn TraceOf(comptime Id: type, comptime format_name: []const u8) type {
    if (Id != u8 and Id != u16) @compileError("route trace: expert ids are u8 or u16");
    return struct {
        const Self = @This();
        pub const trace_format = format_name;
        /// The id files' extension (`.prompt.<ext>`, `.decode.<ext>`).
        pub const id_ext = if (Id == u8) "u8" else "u16";
        pub const max_experts: u32 = if (Id == u8) 256 else 512;
        /// One call's predictor record: `pred_k` ids then `pred_k` f32 scores.
        pub const pred_len: u32 = predBytes(Id);
        const no_id: Id = std.math.maxInt(Id);
        a: std.mem.Allocator,
        io: std.Io,
        prefix: []const u8,
        layers: u32,
        experts: u32,
        first_moe_layer: u32,
        record_bytes: u64,
        request: u32 = 0,
        active: bool = false,
        // One request's buffers (sized at `begin`).
        prompt_tokens: u32 = 0,
        decode_cap: u32 = 0,
        prompt: []Id = &.{},
        prompt_ids: []u32 = &.{},
        decode: []Id = &.{},
        plan: []u8 = &.{},
        pred: []u8 = &.{},
        decode_n: []u32 = &.{},
        /// v2: per call its rows (M), its first row's index in `.decode.u8`'s row sequence, its round's accepted drafts.
        call_rows: []u8 = &.{},
        row_off: []u32 = &.{},
        accepted: []u8 = &.{},
        calls_seen: u32 = 0,
        row_cap: u32 = 0,
        rounds: u32 = 0,
        accepted_total: u64 = 0,
        /// A layer saw a call another layer saw with a different M (or before the first layer): the replay refuses it.
        row_mismatch: bool = false,
        rows0: []u32 = &.{},
        rows_end: []u32 = &.{},
        resident0: []u16 = &.{},
        resident_end: []u16 = &.{},
        decode_ids: std.ArrayList(u32) = .empty,
        prompt_sha256: [64]u8 = @splat('0'),
        // Integrity flags the replay checks before use.
        wide_decode_calls: u32 = 0,
        overflow: bool = false,
        extended_prompt: bool = false,
        io_prefill: Io = .{},
        io_decode: Io = .{},

        pub fn init(a: std.mem.Allocator, io: std.Io, prefix: []const u8, layers: u32, experts: u32, first_moe_layer: u32, record_bytes: u64) !Self {
            if (experts == 0 or experts > max_experts or layers < 2) return error.MimoRouteTraceShape;
            return .{ .a = a, .io = io, .prefix = try a.dupe(u8, prefix), .layers = layers, .experts = experts, .first_moe_layer = first_moe_layer, .record_bytes = record_bytes };
        }

        pub fn deinit(t: *Self) void {
            t.freeRequest();
            t.a.free(t.prefix);
            t.* = undefined;
        }

        /// A request's prompt (`ids`) begins: buffers for its prompt and up to `max_decode` decode calls.
        pub fn begin(t: *Self, ids: []const u32, max_decode: u64) !void {
            t.freeRequest();
            const a = t.a;
            const L: usize = t.layers;
            const n: usize = ids.len;
            t.prompt_tokens = @intCast(n);
            t.decode_cap = @intCast(@min(max_decode, max_decode_calls));
            const T: usize = t.decode_cap;
            t.row_cap = t.decode_cap * max_call_rows;
            t.prompt = try a.alloc(Id, L * n * top_k);
            t.prompt_ids = try a.alloc(u32, L);
            t.decode = try a.alloc(Id, @as(usize, t.row_cap) * L * top_k);
            t.call_rows = try a.alloc(u8, T);
            t.row_off = try a.alloc(u32, T);
            t.accepted = try a.alloc(u8, T);
            t.plan = try a.alloc(u8, T * L * 3);
            t.pred = try a.alloc(u8, T * (L - 1) * pred_len);
            t.decode_n = try a.alloc(u32, L);
            t.rows0 = try a.alloc(u32, L);
            t.rows_end = try a.alloc(u32, L);
            t.resident0 = try a.alloc(u16, L * t.experts);
            t.resident_end = try a.alloc(u16, L * t.experts);
            @memset(t.prompt, no_id);
            @memset(t.prompt_ids, 0);
            @memset(t.decode, no_id);
            @memset(t.call_rows, 0);
            @memset(t.row_off, 0);
            @memset(t.accepted, not_a_round);
            t.calls_seen = 0;
            t.rounds = 0;
            t.accepted_total = 0;
            t.row_mismatch = false;
            @memset(t.plan, 0);
            for (0..T * (L - 1)) |i| emptyPredOf(Id, t.pred[i * pred_len ..][0..pred_len]);
            @memset(t.decode_n, 0);
            @memset(t.rows0, 0);
            @memset(t.rows_end, 0);
            @memset(t.resident0, expert_policy.no_expert);
            @memset(t.resident_end, expert_policy.no_expert);
            t.prompt_sha256 = kset.shaHex(std.mem.sliceAsBytes(ids));
            t.wide_decode_calls = 0;
            t.overflow = false;
            t.extended_prompt = false;
            t.io_prefill = .{};
            t.io_decode = .{};
            t.active = true;
        }

    };
}

/// The predictor bytes per call for ids of type `Id`: `pred_k` ids then `pred_k` f32 scores.
pub fn predBytes(comptime Id: type) u32 {
    return pred_k * (@sizeOf(Id) + 4);
}

fn emptyPredOf(comptime Id: type, out: []u8) void {
    const w = @sizeOf(Id);
    @memset(out[0 .. pred_k * w], 0xFF);
    for (0..pred_k) |j| std.mem.writeInt(u32, out[pred_k * w + 4 * j ..][0..4], @bitCast(std.math.nan(f32)), .little);
}
