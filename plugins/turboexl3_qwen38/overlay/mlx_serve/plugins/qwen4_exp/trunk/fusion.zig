//! copied_from: src/plugins/qwen4_exp/trunk/fusion.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! The trunk's fused paths through its quant formats' optional entries (`abi/quant.zig`), each the bytes of the chain
//! it replaces: a gated attention layer's q|gate, k and v with their norms and RoPE and the indexer's prepare beside
//! them at prefill rows, o_proj over the gated product (with the fused sparse-attention core's merge at decode rows),
//! the decode rows' query planes, and the shared expert inside the routed launches (the router folded in at 2..4
//! rows). A format without the entry, or one that refuses a call, leaves the chain; each entry logs once.

const std = @import("std");
const abi = @import("../../../abi/quant.zig");
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../sdk/util/log.zig");
const Trunk = @import("../trunk.zig").Trunk;
const Formats = @import("formats.zig").Formats;
const ForwardCtx = @import("../../../engines/mlx/forward_ctx.zig").ForwardCtx;
const FullAttnWeights = @import("../../../engines/mlx/layer_weights.zig").FullAttnWeights;
const SSMCacheEntry = @import("../../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
const MoeMlpWeights = @import("../../../sdk/blocks/moe/weights.zig").MoeMlpWeights;
const MoeRouting = @import("../../../sdk/blocks/moe/routing.zig").MoeRouting;
const DenseKVView = @import("../../../sdk/blocks/attention/kv_view.zig").DenseKVView;
const lastDim = @import("../../../mlx/array.zig").lastDim;
const qmatmulBits = @import("../../../sdk/blocks/linear/qmatmul.zig").qmatmulBits;

/// The rows the decode merges serve: decode and verify rows.
pub const decode_rows_max: c_int = 16;
/// The prefill fusions start past the decode rows and the strided gate kernel's widths (`add_norm_max_rows`).
pub const prefill_min_rows: c_int = 17;

/// A full-attention layer's fused state, set and freed within one `qwen4AttnProjected` (`kv_only` within one MTP head
/// forward).
pub const Scratch = struct {
    /// q's, k's and v's prepared copies from the indexer's four-member prepare (`idxPrepMask`), for the fused attention
    /// input.
    prep: [3]mlx.mlx_array = @splat(.{ .ctx = null }),
    /// The QSA mask already is a placed draft row's key visibility (`qsa/merges.zig`).
    place_vis: bool = false,
    /// Only this forward's KV rows and QSA history are read: the MTP head's history append.
    kv_only: bool = false,

    pub fn releaseLayer(self: *Scratch) void {
        for (&self.prep) |*p| {
            if (p.ctx != null) _ = mlx.mlx_array_free(p.*);
            p.* = .{ .ctx = null };
        }
        self.place_vis = false;
    }
};

// ── the entries ──

pub const Entry = enum { attn_in, attn_prep, linear_prepared_idx, gated_out, merged_gated_out, q_planes, indexer, swiglu_shared };

/// Per entry: its first use and its first decline, each logged once.
var entries_said = std.EnumArray(Entry, [2]bool).initFill(.{ false, false });

pub fn served(e: Entry, comptime fmt: []const u8, args: anytype) void {
    const s = &entries_said.getPtr(e)[0];
    if (s.*) return;
    s.* = true;
    log.info("[fused] {s} engaged: " ++ fmt ++ "\n", .{@tagName(e)} ++ args);
}

pub fn refused(t: *const abi.QuantTable, e: Entry, v: abi.Verdict) void {
    const s = &entries_said.getPtr(e)[1];
    if (s.*) return;
    s.* = true;
    log.info("[fused] {s}: {s} refused it ({s}); the chain runs\n", .{ @tagName(e), t.name.slice(), v.why.slice() });
}

pub fn arr(a: mlx.mlx_array) abi.Array {
    return .{ .ctx = a.ctx };
}

pub fn strm(s: mlx.mlx_stream) abi.Stream {
    return .{ .ctx = s.ctx };
}

pub fn own(a: abi.Array) mlx.mlx_array {
    return .{ .ctx = a.ctx };
}

pub fn freeAll(xs: []const abi.Array) void {
    for (xs) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(own(a));
    };
}

/// The fused attention input's results for one layer (each the holder's): q and k normed and rotated, v as projected,
/// all [1, H, S, D]; the gate [1, S, H, D].
pub const AttnIn = struct {
    q: mlx.mlx_array,
    gate: mlx.mlx_array,
    k: mlx.mlx_array,
    v: mlx.mlx_array,

    pub fn deinit(self: AttnIn) void {
        inline for (.{ self.q, self.gate, self.k, self.v }) |a| _ = mlx.mlx_array_free(a);
    }
};

pub fn callAttnIn(f: *const Formats, ws: [3]*const abi.Weight, x: mlx.mlx_array, opts: *const abi.AttnIn, s: mlx.mlx_stream) ?AttnIn {
    const t = f.trunk orelse return null;
    const call = t.attn_in orelse return null;
    var out: abi.AttnInOut = .{};
    const v = call(ws[0], ws[1], ws[2], arr(x), opts, strm(s), &out);
    if (!v.isOk()) {
        freeAll(&.{ out.q, out.gate, out.k, out.v });
        refused(t, .attn_in, v);
        return null;
    }
    return .{ .q = own(out.q), .gate = own(out.gate), .k = own(out.k), .v = own(out.v) };
}

pub fn callAttnPrep(f: *const Formats, ws: [3]*const abi.Weight, w4: *const abi.Weight, x: mlx.mlx_array, s: mlx.mlx_stream) ?[4]mlx.mlx_array {
    const t = f.trunk orelse return null;
    const call = t.attn_prep orelse return null;
    var out: [4]abi.Array = @splat(.{});
    const v = call(ws[0], ws[1], ws[2], w4, arr(x), strm(s), &out);
    if (!v.isOk()) {
        freeAll(&out);
        refused(t, .attn_prep, v);
        return null;
    }
    return .{ own(out[0]), own(out[1]), own(out[2]), own(out[3]) };
}

pub fn callLinearPreparedIdx(f: *const Formats, w: *const abi.Weight, xp: mlx.mlx_array, x: mlx.mlx_array, s: mlx.mlx_stream) ?mlx.mlx_array {
    const t = f.trunk orelse return null;
    const call = t.linear_prepared_idx orelse return null;
    var out: abi.Array = .{};
    const v = call(w, arr(xp), arr(x), strm(s), &out);
    if (!v.isOk()) {
        refused(t, .linear_prepared_idx, v);
        return null;
    }
    return own(out);
}

pub fn callGatedOut(f: *const Formats, o: *const abi.Weight, attn: mlx.mlx_array, gate: *const abi.Gate, s: mlx.mlx_stream) ?mlx.mlx_array {
    const t = f.trunk orelse return null;
    const call = t.gated_out orelse return null;
    var out: abi.Array = .{};
    const v = call(o, arr(attn), gate, strm(s), &out);
    if (!v.isOk()) {
        refused(t, .gated_out, v);
        return null;
    }
    return own(out);
}

pub fn callQPlanes(f: *const Formats, q: *const abi.Weight, x: mlx.mlx_array, opts: *const abi.QPlanes, s: mlx.mlx_stream) ?[2]mlx.mlx_array {
    const t = f.trunk orelse return null;
    const call = t.q_planes orelse return null;
    var out: [2]abi.Array = @splat(.{});
    const v = call(q, arr(x), opts, strm(s), &out);
    if (!v.isOk()) {
        freeAll(&out);
        refused(t, .q_planes, v);
        return null;
    }
    return .{ own(out[0]), own(out[1]) };
}

/// The indexer projection with its epilogue (`indexer`): qk, then the index queries and the pooled key where asked.
pub fn callIndexer(f: *const Formats, w: *const abi.Weight, x: mlx.mlx_array, opts: *const abi.Indexer, s: mlx.mlx_stream) ?[3]mlx.mlx_array {
    const t = f.trunk orelse return null;
    const call = t.indexer orelse return null;
    var out: [3]abi.Array = @splat(.{});
    const v = call(w, arr(x), opts, strm(s), &out);
    if (!v.isOk()) {
        freeAll(&out);
        refused(t, .indexer, v);
        return null;
    }
    return .{ own(out[0]), own(out[1]), own(out[2]) };
}

// ── the fused attention input, the indexer's prepare beside it, the head's KV-only append ──

// ── o_proj over the gated product, the query planes ──

/// Where o_proj's prepare reads the gate in place: element (row, head h, d) at base[row row_stride + h head_stride +
/// off + d].
pub const GateSrc = struct { base: mlx.mlx_array, row_stride: u32, head_stride: u32, off: u32 };

// ── the shared expert inside the routed launches ──

/// The expert format, where it also loaded the shared expert's linears (the trunk's format) and has the fused entry.
fn sharedTable(f: *const Formats) ?*const abi.QuantTable {
    const t = f.experts orelse return null;
    if (f.trunk != t or t.swiglu_shared == null) return null;
    return t;
}

/// `quiet`: a refusal is not logged (the routed form declines the rows the format prepares in a launch of its own).
pub fn callSharedSwiglu(self: *Trunk, banks: *const [3]abi.Weight, expert_x: mlx.mlx_array, inds: mlx.mlx_array, scores: mlx.mlx_array, opts: *const abi.SharedSwiglu, quiet: bool) ?mlx.mlx_array {
    const t = sharedTable(&self.formats).?;
    var out: abi.Array = .{};
    const v = t.swiglu_shared.?(&banks[0], &banks[1], &banks[2], arr(expert_x), arr(inds), arr(scores), opts, strm(self.s), &out);
    if (!v.isOk()) {
        if (!quiet) refused(t, .swiglu_shared, v);
        return null;
    }
    return own(out);
}

// ── tests ──

const testing = std.testing;
