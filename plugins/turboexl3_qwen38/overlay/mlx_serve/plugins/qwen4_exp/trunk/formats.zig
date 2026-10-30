//! copied_from: src/plugins/qwen4_exp/trunk/formats.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! The trunk's weights a pack's quant formats serve (the weight-encoding seam, `abi/quant.zig`), bound once at load.
//! The claimant of `trunk_quant` loads every trunk linear stored in neither of MLX's encodings (`load_tensor`) and runs
//! it from `qmatmul` (`matmul`); the claimant of `expert_quant` loads each MoE layer's routed banks (`load_bank`) and
//! runs the layer's SwiGLU (`swiglu`). A claimant is `sdk/quant/formats.zig`'s `claimantAmong`, over the built-in
//! formats and turboexl3's table (linked in, build.zig.zon): the built-in EXL3 table is Sushi's (banks only), and
//! turboexl3's outranks it on the blocks it claims. Also the numerics the trunk's reordering paths read.

const std = @import("std");
const abi = @import("../../../abi/quant.zig");
const host_abi = @import("mlx_serve").abi.host;
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../sdk/util/log.zig");
const model = @import("../../../sdk/config/model.zig");
const formats = @import("../../../sdk/quant/formats.zig");
const quant_resident = @import("../../../sdk/quant/resident.zig");
const MoeMlpWeights = @import("../../../sdk/blocks/moe/weights.zig").MoeMlpWeights;
const Trunk = @import("../trunk.zig").Trunk;

/// The arithmetic the trunk's own paths run in. `exact`: no path that reorders a sum (the reference). `reordered`: the
/// paths that do.
pub const Numerics = enum { exact, reordered };

/// The numerics: reordered on the profile (the configuration the reordered paths were measured on), exact elsewhere.
pub fn numericsFor(profile: bool) Numerics {
    return if (profile) .reordered else .exact;
}

/// What the load hands the formats beside the map: the pack's config text, its format blocks and the host table.
pub const Pack = struct {
    config_json: []const u8 = "{}",
    /// `trunk_quant` and `expert_quant` as JSON; null where the pack has none.
    trunk_quant: ?[]const u8 = null,
    expert_quant: ?[]const u8 = null,
    host: ?*const host_abi.Host = null,
    model_dir: []const u8 = "",
};

pub const Formats = struct {
    /// The format of the trunk's linears and of the routed banks; null where they are the trunk's own.
    trunk: ?*const abi.QuantTable = null,
    experts: ?*const abi.QuantTable = null,
    /// The linears `trunk` loaded, by their `.weight` array in the map.
    linears: std.AutoHashMapUnmanaged(usize, abi.Weight) = .empty,
    /// The banks `experts` loaded (gate, up, down), by the gate bank's array in the map.
    banks: std.AutoHashMapUnmanaged(usize, [3]abi.Weight) = .empty,
    numerics: Numerics = .exact,
    /// A Qwen3.8 model on an EXL3 trunk pack (`trunk_quant`): the configuration the reordered numerics run on.
    profile: bool = false,
    /// The bytes the trunk format keeps resident for this model (`armTrunkResident`; 0: per-call decode).
    resident_bytes: u64 = 0,

    pub fn deinit(self: *Formats, a: std.mem.Allocator) void {
        if (self.trunk) |t| {
            // The kept matrices first: they are keyed by the weights' handles.
            if (self.resident_bytes > 0) if (t.resident) |keep| {
                var held: u64 = 0;
                _ = keep(.{}, 0, .{}, &held);
            };
            var it = self.linears.valueIterator();
            while (it.next()) |w| t.release(w);
        }
        if (self.experts) |t| {
            var it = self.banks.valueIterator();
            while (it.next()) |b| for (b) |*w| t.release(w);
        }
        self.linears.deinit(a);
        self.banks.deinit(a);
        self.* = .{};
    }

    /// The trunk format's weight for the map's `.weight` array `w`; null for a linear the trunk runs itself.
    pub fn linearOf(self: *const Formats, w: mlx.mlx_array) ?*const abi.Weight {
        if (self.linears.count() == 0) return null;
        return self.linears.getPtr(@intFromPtr(w.ctx));
    }

    /// The expert format's banks of the MoE block `mw`; null for banks the trunk runs itself.
    pub fn banksOf(self: *const Formats, mw: *const MoeMlpWeights) ?*const [3]abi.Weight {
        if (self.banks.count() == 0) return null;
        return self.banks.getPtr(@intFromPtr(mw.switch_gate_w.ctx));
    }

    /// `x @ W^T` through the trunk format.
    pub fn matmul(self: *const Formats, w: *const abi.Weight, x: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
        const t = self.trunk orelse return error.TrunkFormatMissing;
        var out: abi.Array = .{};
        const v = t.matmul(w, .{ .ctx = x.ctx }, .{ .ctx = s.ctx }, &out);
        if (!v.isOk()) {
            log.err("[qwen4] {s}: a trunk linear's matmul refused ({s})\n", .{ t.name.slice(), v.why.slice() });
            return error.TrunkFormatMatmulRefused;
        }
        return .{ .ctx = out.ctx };
    }

    /// The routed SwiGLU of `x` [B, S, D] through banks `b`'s experts `inds` [B, S, K], weighted by `scores`.
    pub fn swiglu(self: *const Formats, b: *const [3]abi.Weight, x: mlx.mlx_array, inds: mlx.mlx_array, scores: mlx.mlx_array, opts: abi.Swiglu, s: mlx.mlx_stream) !mlx.mlx_array {
        const t = self.experts orelse return error.ExpertFormatMissing;
        var out: abi.Array = .{};
        const v = t.swiglu.?(&b[0], &b[1], &b[2], .{ .ctx = x.ctx }, .{ .ctx = inds.ctx }, .{ .ctx = scores.ctx }, &opts, .{ .ctx = s.ctx }, &out);
        if (!v.isOk()) {
            log.err("[qwen4] {s}: a routed call refused ({s}, {d} rows)\n", .{ t.name.slice(), v.why.slice(), rowsOf(x) });
            return error.ExpertFormatSwigluRefused;
        }
        return .{ .ctx = out.ctx };
    }
};

/// The reordered numerics: a path that reorders a sum may run.
pub fn reordered(self: *const Trunk) bool {
    return self.formats.numerics == .reordered;
}

/// One load's formats while they are open: `begin` claims and opens them and loads the trunk's linears, `banks` loads
/// one MoE layer's routed banks, `end` closes them (the weights outlive it, in `formats`). Stays at its address while
/// open: the load contexts point into it.
pub const Loading = struct {
    formats: Formats = .{},
    a: std.mem.Allocator,
    src: abi.Source,
    trunk_ctx: ?abi.LoadCtx = null,
    experts_ctx: ?abi.LoadCtx = null,

    /// Refused by name: a format block nobody claims, a format that refuses it, a linear or bank it does not serve.
    pub fn begin(self: *Loading, a: std.mem.Allocator, config: *const model.ModelConfig, weights: *const model.Weights, pack: Pack, s: mlx.mlx_stream) !void {
        self.* = .{ .a = a, .src = .{ .ctx = @constCast(weights), .get = tensorOf } };
        errdefer self.abort();
        const f = &self.formats;
        f.profile = config.trunk_format;
        f.numerics = numericsFor(f.profile);
        if (config.trunk_format) {
            const json = pack.trunk_quant orelse return error.TrunkFormatBlockMissing;
            const t = try claimed(extra_tables, json, pack.config_json, "trunk_quant");
            try self.open(&self.trunk_ctx, t, json, pack, s, "trunk_quant");
            f.trunk = t;
            // the bytes its resident matrices are owed and do not hold yet, which the GPU ceiling holds aside
            if (t.resident_pending) |owed| quant_resident.pending.get().f = owed;
            const n = try self.loadLinears(t, weights);
            if (n == 0) {
                log.err("[qwen4] the pack names a trunk_quant format, and its weights hold no linear outside MLX's encodings\n", .{});
                return error.TrunkFormatHasNoLinears;
            }
            log.info("[qwen4] trunk_quant: {d} linears on {s}\n", .{ n, t.name.slice() });
        }
        if (config.exl3 != null) {
            const json = pack.expert_quant orelse return error.ExpertFormatBlockMissing;
            // turboexl3 serves an EXL3 trunk pack's banks; an expert-only (Sushi) pack's stay the built-in table's
            const t = try claimed(if (config.trunk_format) extra_tables else &.{}, json, pack.config_json, "expert_quant");
            if (t.swiglu == null) {
                log.err("[qwen4] {s} runs routed banks per projection only; the trunk runs a layer's banks as one SwiGLU\n", .{t.name.slice()});
                return error.ExpertFormatHasNoSwiglu;
            }
            try self.open(&self.experts_ctx, t, json, pack, s, "expert_quant");
            f.experts = t;
        }
        if (f.profile) log.info("[qwen4] EXL3 numerics: {s}; trunk linears on {s}, routed experts on {s}\n", .{
            @tagName(f.numerics),
            if (f.trunk) |t| t.name.slice() else "the trunk's kernels",
            if (f.experts) |t| t.name.slice() else "their affine banks",
        });
    }

    /// Layer `layer`'s banks under `<prefix>.layers.<layer>.mlp.switch_mlp` through the expert format (none: nothing);
    /// `hidden` / `inter` pin their widths (0: unchecked).
    pub fn banks(self: *Loading, prefix: []const u8, layer: u32, mw: *const MoeMlpWeights, hidden: u32, inter: u32) !void {
        const t = self.formats.experts orelse return;
        var out: [3]abi.Weight = @splat(.{});
        errdefer for (&out) |*w| t.release(w);
        for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, [_]u32{ hidden, hidden, inter }, &out) |proj, in_dim, *w| {
            var nb: [192]u8 = undefined;
            const base = try std.fmt.bufPrint(&nb, "{s}.layers.{d}.mlp.switch_mlp.{s}", .{ prefix, layer, proj });
            const v = t.load_bank(&self.experts_ctx.?, abi.Str.of(base), in_dim, w);
            if (!v.isOk()) {
                log.err("[qwen4] {s}: {s} does not serve the bank ({s})\n", .{ base, t.name.slice(), v.why.slice() });
                return error.ExpertBankUnserved;
            }
        }
        const slot = try self.formats.banks.getOrPut(self.a, @intFromPtr(mw.switch_gate_w.ctx));
        if (slot.found_existing) {
            for (&out) |*w| t.release(w);
            return;
        }
        slot.value_ptr.* = out;
    }

    /// Closes the open formats; their weights stay in `formats`.
    pub fn end(self: *Loading) void {
        if (self.trunk_ctx) |*c| self.formats.trunk.?.close(c);
        if (self.experts_ctx) |*c| self.formats.experts.?.close(c);
        self.trunk_ctx = null;
        self.experts_ctx = null;
    }

    fn abort(self: *Loading) void {
        self.end();
        self.formats.deinit(self.a);
    }

    fn open(self: *Loading, slot: *?abi.LoadCtx, t: *const abi.QuantTable, block_json: []const u8, pack: Pack, s: mlx.mlx_stream, comptime key: []const u8) !void {
        slot.* = .{
            .model_dir = abi.Str.of(pack.model_dir),
            .block_json = abi.Str.of(block_json),
            .config_json = abi.Str.of(pack.config_json),
            .source = &self.src,
            .stream = .{ .ctx = s.ctx },
            .host = pack.host,
        };
        const v = t.open(&slot.*.?);
        if (v.isOk()) return;
        slot.* = null;
        log.err("[qwen4] {s} refuses the pack's " ++ key ++ " block ({s})\n", .{ t.name.slice(), v.why.slice() });
        return error.QuantFormatRefused;
    }

    /// Every `.weight` in neither of MLX's encodings, through the trunk format; the count.
    fn loadLinears(self: *Loading, t: *const abi.QuantTable, weights: *const model.Weights) !u32 {
        var n: u32 = 0;
        var it = weights.map.iterator();
        while (it.next()) |e| {
            const name = e.key_ptr.*;
            if (!std.mem.endsWith(u8, name, ".weight") or mlxEncoded(e.value_ptr.*)) continue;
            var w: abi.Weight = .{};
            const v = t.load_tensor(&self.trunk_ctx.?, abi.Str.of(name[0 .. name.len - ".weight".len]), 0, &w);
            if (!v.isOk()) {
                log.err("[qwen4] {s}: {s} does not serve it ({s})\n", .{ name, t.name.slice(), v.why.slice() });
                return error.TrunkLinearUnserved;
            }
            const slot = self.formats.linears.getOrPut(self.a, @intFromPtr(e.value_ptr.ctx)) catch |err| {
                t.release(&w);
                return err;
            };
            // Two names over one array (a tied table) are one linear.
            if (slot.found_existing) {
                t.release(&w);
                continue;
            }
            slot.value_ptr.* = w;
            n += 1;
        }
        return n;
    }
};

/// The quant tables the trunk weighs beside the built-in formats: turboexl3's, linked in (build.zig.zon). Tests swap
/// their own in.
var extra_tables: []const *const abi.QuantTable = &linked_tables;
const linked_tables = [_]*const abi.QuantTable{&@import("turboexl3_table").table};

/// The claimant of the pack's `key` block among `extra` and the built-ins, refused by name when nobody claims it.
fn claimed(extra: []const *const abi.QuantTable, block_json: []const u8, config_json: []const u8, comptime key: []const u8) !*const abi.QuantTable {
    const got = formats.claimantAmong(extra, block_json, config_json) catch {
        log.err("[qwen4] two formats claim the pack's " ++ key ++ " block alike\n", .{});
        return error.QuantFormatAmbiguous;
    };
    return got orelse {
        log.err("[qwen4] no quant format claims the pack's " ++ key ++ " block {s}\n", .{block_json});
        return error.QuantFormatUnclaimed;
    };
}

/// MLX's own encodings of a `.weight`: a float table, or the uint32 words of MLX's quantized modes.
fn mlxEncoded(w: mlx.mlx_array) bool {
    return switch (mlx.mlx_array_dtype(w)) {
        .float16, .bfloat16, .float32, .uint32 => true,
        else => false,
    };
}

fn tensorOf(ctx: ?*anyopaque, name: abi.Str) callconv(.c) abi.Array {
    const w: *const model.Weights = @ptrCast(@alignCast(ctx.?));
    return .{ .ctx = (w.get(name.slice()) orelse return .{}).ctx };
}

fn rowsOf(x: mlx.mlx_array) usize {
    const sh = mlx.getShape(x);
    var rows: usize = 1;
    for (sh, 0..) |d, i| {
        if (i + 1 < sh.len) rows *= @intCast(d);
    }
    return rows;
}

// ── tests ──

const testing = std.testing;
const exl3_format = @import("../../../sdk/quant/exl3/format.zig");

/// A quant table as the trunk reads it: it claims a `"fake"` block, loads any linear but one named `bad` (a
/// new handle on the map's array) and returns `x` from its matmul.
const Fake = struct {
    fn claims(block_json: abi.Str, _: abi.Str) callconv(.c) abi.Claim {
        return if (std.mem.indexOf(u8, block_json.slice(), "\"fake\"") != null) .by_marker else .no;
    }
    fn open(_: *abi.LoadCtx) callconv(.c) abi.Verdict {
        return .ok;
    }
    fn close(_: *abi.LoadCtx) callconv(.c) void {}
    fn loadTensor(ctx: *const abi.LoadCtx, name: abi.Str, _: u32, out: *abi.Weight) callconv(.c) abi.Verdict {
        if (std.mem.endsWith(u8, name.slice(), "bad")) return abi.Verdict.refuse(.unsupported, "FakeLayoutUnserved");
        var nb: [128]u8 = undefined;
        const key = std.fmt.bufPrint(&nb, "{s}.weight", .{name.slice()}) catch return abi.Verdict.refuse(.failed, "FakeName");
        var h = mlx.mlx_array_new();
        if (mlx.mlx_array_set(&h, .{ .ctx = ctx.source.tensor(key).ctx }) != 0) return abi.Verdict.refuse(.failed, "FakeShare");
        out.* = .{ .w = .{ .ctx = h.ctx } };
        return .ok;
    }
    fn release(w: *abi.Weight) callconv(.c) void {
        if (w.w.ctx != null) _ = mlx.mlx_array_free(.{ .ctx = w.w.ctx });
        w.* = .{};
    }
    fn matmul(_: *const abi.Weight, x: abi.Array, _: abi.Stream, out: *abi.Array) callconv(.c) abi.Verdict {
        var h = mlx.mlx_array_new();
        if (mlx.mlx_array_set(&h, .{ .ctx = x.ctx }) != 0) return abi.Verdict.refuse(.failed, "FakeShare");
        out.* = .{ .ctx = h.ctx };
        return .ok;
    }
    const table: abi.QuantTable = .{ .name = abi.Str.of("fake"), .claims = &claims, .validate = undefined, .bill = undefined, .open = &open, .close = &close, .load_tensor = &loadTensor, .load_bank = undefined, .release = &release, .matmul = &matmul, .moe = undefined };
};

/// Claims every block at `by_marker`, as turboexl3 claims an EXL3 pack's.
const Marker = struct {
    fn claims(_: abi.Str, _: abi.Str) callconv(.c) abi.Claim {
        return .by_marker;
    }
    const table: abi.QuantTable = .{ .name = abi.Str.of("marker"), .claims = &claims, .validate = undefined, .bill = undefined, .open = undefined, .close = undefined, .load_tensor = undefined, .load_bank = undefined, .release = undefined, .matmul = undefined, .moe = undefined };
};

/// `Fake` with the resident entries: each weight takes `per` bytes.
const Keep = struct {
    const per: u64 = 1024;
    var last: usize = 0;
    var dropped = false;

    fn resident(ws: abi.Span(abi.Weight), room: u64, _: abi.Stream, bytes: *u64) callconv(.c) abi.Verdict {
        last = ws.len;
        bytes.* = ws.len * per;
        if (ws.len == 0) dropped = true;
        if (room != 0 and bytes.* > room) return abi.Verdict.refuse(.too_large, "FakeRoom");
        return .ok;
    }
    const table: abi.QuantTable = blk: {
        var t = Fake.table;
        t.resident = &resident;
        break :blk t;
    };
};
