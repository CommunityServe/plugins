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
const residual_norm = @import("../../../sdk/blocks/norms/residual_norm.zig");
const router_kernel = @import("../../../sdk/blocks/moe/router_kernel.zig");
const sparse_attn = @import("../qsa/sparse_attn.zig");
const Trunk = @import("../trunk.zig").Trunk;
const Formats = @import("formats.zig").Formats;
const ForwardCtx = @import("../../../engines/mlx/forward_ctx.zig").ForwardCtx;
const FullAttnWeights = @import("../../../engines/mlx/layer_weights.zig").FullAttnWeights;
const SSMCacheEntry = @import("../../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
const MoeMlpWeights = @import("../../../sdk/blocks/moe/weights.zig").MoeMlpWeights;
const MoeRouting = @import("../../../sdk/blocks/moe/routing.zig").MoeRouting;
const DenseKVView = @import("../../../sdk/blocks/attention/kv_view.zig").DenseKVView;
const Qwen4AttnProf = @import("../prof.zig").Qwen4AttnProf;
const qwen4Standin = @import("../../../sdk/diag/standin.zig").qwen4Standin;
const dispatchProbeCount = @import("../../../engines/mlx/diag/profile.zig").dispatchProbeCount;
const lastDim = @import("../../../mlx/array.zig").lastDim;
const qmatmulBits = @import("../../../sdk/blocks/linear/qmatmul.zig").qmatmulBits;
const swigluClampLimit = @import("../../../sdk/blocks/mlp/activations.zig").swigluClampLimit;
const fused_activations = @import("../../../sdk/blocks/mlp/fused_activations.zig");
const FUSED256_MIN_Q_LEN = @import("../../../sdk/blocks/attention/attn256.zig").FUSED256_MIN_Q_LEN;

/// The rows the decode merges serve: decode and verify rows.
pub const decode_rows_max: c_int = 16;
/// The prefill fusions start past the decode rows and the strided gate kernel's widths (`add_norm_max_rows`).
pub const prefill_min_rows: c_int = 17;
/// Up to these rows the chain's q and k share the hd-256 QK-norm + RoPE launch, so a KV-only call computes q anyway.
const joint_rope_rows: c_int = 32;

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

pub fn declined(e: Entry, why: []const u8) void {
    const s = &entries_said.getPtr(e)[1];
    if (s.*) return;
    s.* = true;
    log.info("[fused] {s} declined: {s} (the chain runs)\n", .{ @tagName(e), why });
}

fn refused(t: *const abi.QuantTable, e: Entry, v: abi.Verdict) void {
    const s = &entries_said.getPtr(e)[1];
    if (s.*) return;
    s.* = true;
    log.info("[fused] {s}: {s} refused it ({s}); the chain runs\n", .{ @tagName(e), t.name.slice(), v.why.slice() });
}

pub fn arr(a: mlx.mlx_array) abi.Array {
    return .{ .ctx = a.ctx };
}

fn strm(s: mlx.mlx_stream) abi.Stream {
    return .{ .ctx = s.ctx };
}

fn own(a: abi.Array) mlx.mlx_array {
    return .{ .ctx = a.ctx };
}

fn freeAll(xs: []const abi.Array) void {
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

fn callAttnIn(f: *const Formats, ws: [3]*const abi.Weight, x: mlx.mlx_array, opts: *const abi.AttnIn, s: mlx.mlx_stream) ?AttnIn {
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

fn callAttnPrep(f: *const Formats, ws: [3]*const abi.Weight, w4: *const abi.Weight, x: mlx.mlx_array, s: mlx.mlx_stream) ?[4]mlx.mlx_array {
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

fn callLinearPreparedIdx(f: *const Formats, w: *const abi.Weight, xp: mlx.mlx_array, x: mlx.mlx_array, s: mlx.mlx_stream) ?mlx.mlx_array {
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

fn callGatedOut(f: *const Formats, o: *const abi.Weight, attn: mlx.mlx_array, gate: *const abi.Gate, s: mlx.mlx_stream) ?mlx.mlx_array {
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

fn callQPlanes(f: *const Formats, q: *const abi.Weight, x: mlx.mlx_array, opts: *const abi.QPlanes, s: mlx.mlx_stream) ?[2]mlx.mlx_array {
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

/// What the fused attention input needs of a call, as `attnInFused` and `idxPrepMask` read it.
pub const AttnInFacts = struct {
    output_gate: bool,
    batch: c_int,
    prefill: bool,
    rows: c_int,
    /// The scalar RoPE at the chunk's offset: no batched slots, spec tree, M-RoPE, head or batched positions, YaRN.
    plain_rope: bool,
    head_dim: u32,
    rope_dims: c_int,
    /// The q / k norm weights present in x's dtype, the eps array built.
    norms: bool,
    /// q, k and v are the trunk format's linears.
    linears: bool,
    /// x is bf16 or f16.
    dtype_ok: bool,
};

/// The first reason the fused attention input does not serve a call, or null.
pub fn attnInRefusal(f: AttnInFacts) ?[]const u8 {
    if (!f.output_gate) return "no attention output gate";
    if (f.batch != 1 or !f.prefill) return "not a one-sequence prefill";
    if (f.rows < prefill_min_rows) return "decode / verify rows";
    if (!f.plain_rope) return "not the scalar RoPE (batched, tree, M-RoPE, head positions or YaRN)";
    if (f.head_dim != 256) return "head_dim is not 256";
    if (f.rope_dims != 32 and f.rope_dims != 64 and f.rope_dims != 128 and f.rope_dims != 256) return "rotary width not served";
    if (!f.norms) return "no q/k norm in the activation dtype";
    if (!f.linears) return "q/k/v are not the trunk format's linears";
    if (!f.dtype_ok) return "activations neither bf16 nor f16";
    return null;
}

pub fn ropeDims(self: *const Trunk) c_int {
    return @intFromFloat(@as(f32, @floatFromInt(self.config.head_dim)) * self.config.partial_rotary_factor);
}

fn qkvOf(f: *const Formats, fa: *const FullAttnWeights) ?[3]*const abi.Weight {
    return .{ f.linearOf(fa.q_w) orelse return null, f.linearOf(fa.k_w) orelse return null, f.linearOf(fa.v_w) orelse return null };
}

pub fn attnInFactsOf(self: *const Trunk, ctx: *const ForwardCtx, x: mlx.mlx_array, fa: *const FullAttnWeights, batch: c_int, seq_len: c_int, is_prefill: bool) AttnInFacts {
    const cfg = &self.config;
    const xd = mlx.mlx_array_dtype(x);
    return .{
        .output_gate = cfg.attn_output_gate,
        .batch = batch,
        .prefill = is_prefill,
        .rows = seq_len,
        .plain_rope = ctx.batch_slots == null and ctx.tree == null and ctx.mrope_pos == null and ctx.mrope_cos_cur == null and
            ctx.head_rope_pos.ctx == null and ctx.batch_rope_offsets == null and !self.yarnActive(),
        .head_dim = cfg.head_dim,
        .rope_dims = ropeDims(self),
        .norms = fa.q_norm.ctx != null and fa.k_norm.ctx != null and self.rms_eps_arr.ctx != null and
            mlx.mlx_array_dtype(fa.q_norm) == xd and mlx.mlx_array_dtype(fa.k_norm) == xd,
        .linears = qkvOf(&self.formats, fa) != null,
        .dtype_ok = xd == .bfloat16 or xd == .float16,
    };
}

/// The fused attention input: this layer's q|gate, k and v at prefill rows through `attn_in`, the chain in
/// `gatedFullAttnProjected` byte for byte: q|gate, k and v projected, the split, the q / k RMSNorm and the partial RoPE
/// (MLX's rope's own angles, `qkAngleRowsFor`), q, k and v head-major. Over the indexer's prepared copies
/// (`Scratch.prep`) where this layer made them. Null: the chain.
pub fn attnInFused(self: *Trunk, ctx: *const ForwardCtx, x: mlx.mlx_array, fa: *const FullAttnWeights, offset: c_int, batch: c_int, seq_len: c_int, is_prefill: bool) !?AttnIn {
    // A trunk format's only: every other pack runs the chain, and says nothing.
    if (self.formats.trunk == null) return null;
    if (attnInRefusal(attnInFactsOf(self, ctx, x, fa, batch, seq_len, is_prefill))) |why| {
        if (is_prefill and seq_len >= prefill_min_rows) declined(.attn_in, why);
        return null;
    }
    const cfg = &self.config;
    const ws = qkvOf(&self.formats, fa).?;
    const rope_dims = ropeDims(self);
    const angles = try self.qkAngleRowsFor(0, rope_dims, .{ .value = cfg.rope_theta, .has_value = true }, .{ .ctx = null }, offset, seq_len, 1.0, 1.0);
    var opts: abi.AttnIn = .{
        .heads = cfg.num_attention_heads,
        .kv_heads = cfg.num_key_value_heads,
        .rope_dims = @intCast(rope_dims),
        .q_norm = arr(fa.q_norm),
        .k_norm = arr(fa.k_norm),
        .eps = arr(self.rms_eps_arr),
        .angles = arr(angles),
    };
    const sc = &self.fused;
    if (sc.prep[0].ctx != null) opts.prep = .{ arr(sc.prep[0]), arr(sc.prep[1]), arr(sc.prep[2]) };
    const out = callAttnIn(&self.formats, ws, x, &opts, self.s) orelse return null;
    served(.attn_in, "q|gate, k and v with the q/k norm and RoPE in one finish (rows={d})", .{seq_len});
    return out;
}

/// The indexer's prepare: where this call's fused attention input runs, the indexer projection's prepare is the fourth
/// member of its prepare launch (`attn_prep`: x read once); the indexer's product runs over its copy
/// (`linear_prepared_idx`), then `qsaMaskFromQk` as `qsaMask`'s one-sequence arm runs it, and the fused attention input
/// takes the other three copies. Null: `qsaMask`.
pub fn idxPrepMask(self: *Trunk, ctx: *ForwardCtx, x: mlx.mlx_array, fa: *const FullAttnWeights, entry: *SSMCacheEntry, layer: u32, cache_len: c_int, pos_base: c_int, batch: c_int, seq_len: c_int, is_prefill: bool) !?mlx.mlx_array {
    if (attnInRefusal(attnInFactsOf(self, ctx, x, fa, batch, seq_len, is_prefill)) != null) return null;
    const f = &self.formats;
    const ws = qkvOf(f, fa).?;
    const wi = f.linearOf(fa.idx_qk_w) orelse {
        declined(.attn_prep, "the indexer projection is not the trunk format's linear");
        return null;
    };
    const xp = callAttnPrep(f, ws, wi, x, self.s) orelse return null;
    defer _ = mlx.mlx_array_free(xp[3]);
    for (&self.fused.prep, xp[0..3]) |*dst, p| {
        if (dst.ctx != null) _ = mlx.mlx_array_free(dst.*);
        dst.* = p;
    }
    // a refusal here leaves the fused attention input the three copies (its own prepare's bytes) and `qsaMask` the
    // indexer
    const qk = callLinearPreparedIdx(f, wi, xp[3], x, self.s) orelse return null;
    defer _ = mlx.mlx_array_free(qk);
    served(.attn_prep, "the indexer's prepare in the attention input's prepare launch (rows={d})", .{seq_len});
    return try self.qsaMaskFromQk(ctx, qk, fa, entry, cache_len, pos_base, batch, seq_len, layer, true, null);
}

/// The head's KV-only append: a forward past the joint-rope widths leaves the fused attention input's q|gate out; the
/// chain evaluates k and v without q (its q chain is built, never evaluated). Its k and v are the fused input's bytes.
/// Without a trunk format there is no fused attention input to leave out: false, and nothing said.
pub fn kvOnlySkips(self: *const Trunk, rows: c_int) bool {
    if (self.formats.trunk == null or !self.fused.kv_only or rows <= joint_rope_rows) return false;
    if (!kv_only_said) {
        kv_only_said = true;
        log.info("[fused] kv-only append engaged: the MTP head's history append projects k and v without the attention input's q|gate (rows={d})\n", .{rows});
    }
    return true;
}

var kv_only_said = false;

// ── o_proj over the gated product, the query planes ──

/// Where o_proj's prepare reads the gate in place: element (row, head h, d) at base[row row_stride + h head_stride +
/// off + d].
pub const GateSrc = struct { base: mlx.mlx_array, row_stride: u32, head_stride: u32, off: u32 };

/// The gate as it lies: a gate plane [1, S, H, D] (the fused attention input's, or the query planes'), else the q_proj
/// [1, S, 2 H D] the layer projected itself (a fresh contiguous linear output; a joined or verify-supplied q is a view:
/// the tail keeps it). Null: the tail.
pub fn gateSrc(plane: ?mlx.mlx_array, own_q_proj: ?mlx.mlx_array, heads: c_int, hd: c_int) ?GateSrc {
    if (plane) |g| return .{ .base = g, .row_stride = @intCast(heads * hd), .head_stride = @intCast(hd), .off = 0 };
    if (own_q_proj) |q| return .{ .base = q, .row_stride = @intCast(2 * heads * hd), .head_stride = @intCast(2 * hd), .off = @intCast(hd) };
    return null;
}

/// The gate's operand for `gated_out`, with the host's sigmoid table; null where the tail would not run the gate on
/// these rows over a trunk-format o_proj on the GPU.
fn gateFor(self: *Trunk, dt: mlx.mlx_dtype, gs: GateSrc, rows: c_int, batch: c_int, flat_batch: c_int, decode: bool) !?abi.Gate {
    if (!self.config.attn_output_gate or batch != 1 or flat_batch != 1) return null;
    if (decode and (rows < 1 or rows > decode_rows_max or rows > residual_norm.add_norm_max_rows)) return null;
    if ((dt != .bfloat16 and dt != .float16) or mlx.mlx_array_dtype(gs.base) != dt or !mlx.streamIsGpu(self.s)) return null;
    const sigtab = try fused_activations.swigluSigTable(self.s, dt, std.heap.c_allocator);
    return .{ .base = arr(gs.base), .row_stride = gs.row_stride, .head_stride = gs.head_stride, .off = gs.off, .sigtab = arr(sigtab) };
}

/// The fused attention output (decode / verify rows at the strided gate kernel's widths, prefill chunks past them):
/// o_proj over
/// T(attn * sigmoid(gate)) through `gated_out`, the product formed in o_proj's prepare, byte for byte the tail's
/// sigmoid / multiply / reshape / `attnProj`. Null: the tail.
pub fn gatedOut(self: *Trunk, attn_out: mlx.mlx_array, gs: GateSrc, fa: *const FullAttnWeights, flat_shape: [3]c_int, batch: c_int, is_prefill: bool) !?mlx.mlx_array {
    const rows = flat_shape[1];
    const decode = rows <= decode_rows_max;
    if (!decode and (!is_prefill or rows < prefill_min_rows)) return null;
    const o = self.formats.linearOf(fa.o_w) orelse return null;
    const ash = mlx.getShape(attn_out);
    if (ash.len != 4 or ash[0] != 1 or ash[2] != rows or ash[1] * ash[3] != flat_shape[2]) return null;
    const gate = (try gateFor(self, mlx.mlx_array_dtype(attn_out), gs, rows, batch, flat_shape[0], decode)) orelse return null;
    const y = callGatedOut(&self.formats, o, attn_out, &gate, self.s) orelse return null;
    served(.gated_out, "the sigmoid gate's product inside o_proj's prepare (rows={d})", .{rows});
    return y;
}

/// The merged attention output: where `qsaSlotAttn`'s first arm (the fused sparse-attention core) serves the call, its
/// split pass runs as there and `merged_gated_out` runs the core's merge, the gate product and o_proj's prepare in one
/// launch: byte for byte the core's merge + the tail. The core's bookkeeping as `qsaSlotAttn` notes it. Null:
/// `qsaSlotAttn` and the tail.
pub fn mergedGatedOut(self: *Trunk, ctx: *ForwardCtx, q_rope: mlx.mlx_array, kv_view: *const DenseKVView, gs: GateSrc, fa: *const FullAttnWeights, flat_shape: [3]c_int, batch: c_int, seq_len: c_int, ratio: c_int, attn_scale: f32) !?mlx.mlx_array {
    const qsa_ok = ctx.qsa_blocks.ctx != null and ctx.qsa_mask.ctx == null and !qwen4Standin().attn_sdpa;
    if (!qsa_ok or seq_len < sparse_attn.qsaAttnMinS() or seq_len >= FUSED256_MIN_Q_LEN) return null;
    const t = self.formats.trunk orelse return null;
    const call = t.merged_gated_out orelse return null;
    const o = self.formats.linearOf(fa.o_w) orelse return null;
    const gate = (try gateFor(self, mlx.mlx_array_dtype(q_rope), gs, flat_shape[1], batch, flat_shape[0], true)) orelse return null;
    var parts = (try sparse_attn.qsaSparseAttnSplit(self.s, q_rope, kv_view, ctx.qsa_blocks, ratio, attn_scale)) orelse return null;
    defer parts.deinit();
    var out: abi.Array = .{};
    const v = call(o, arr(parts.pacc), arr(parts.pml), &gate, strm(self.s), &out);
    if (!v.isOk()) {
        refused(t, .merged_gated_out, v);
        return null;
    }
    const y = own(out);
    ctx.qsa_arms.noteGather(seq_len);
    if (self.cost_trace_active) self.cost_attention |= 1;
    if (Qwen4AttnProf.active) Qwen4AttnProf.lap(y, .qsa);
    served(.merged_gated_out, "the sparse-attention core's merge, the sigmoid gate and o_proj's prepare in one launch (rows={d})", .{seq_len});
    return y;
}

/// The query planes: q_proj at decode rows through `q_planes`, its finish writing each head's [query | gate] halves as
/// two contiguous planes [1, S, H, D]: the split's two views as their bytes, so the hd-256 QK-norm + RoPE kernel's copy
/// of the strided query view never runs. Null: q_proj and the split.
pub fn qPlanes(self: *Trunk, x: mlx.mlx_array, fa: *const FullAttnWeights, batch: c_int, seq_len: c_int) ?[2]mlx.mlx_array {
    const cfg = &self.config;
    if (!cfg.attn_output_gate or batch != 1 or seq_len < 1 or seq_len > decode_rows_max) return null;
    const q = self.formats.linearOf(fa.q_w) orelse return null;
    const xd = mlx.mlx_array_dtype(x);
    if ((xd != .bfloat16 and xd != .float16) or cfg.head_dim % 128 != 0) return null;
    const opts: abi.QPlanes = .{ .heads = cfg.num_attention_heads, .head_dim = cfg.head_dim };
    const planes = callQPlanes(&self.formats, q, x, &opts, self.s) orelse return null;
    served(.q_planes, "q_proj's finish writes the query and gate planes (rows={d})", .{seq_len});
    return planes;
}

// ── the shared expert inside the routed launches ──

/// The expert format, where it also loaded the shared expert's linears (the trunk's format) and has the fused entry.
fn sharedTable(f: *const Formats) ?*const abi.QuantTable {
    const t = f.experts orelse return null;
    if (f.trunk != t or t.swiglu_shared == null) return null;
    return t;
}

fn sharedLinears(f: *const Formats, mw: *const MoeMlpWeights) ?[3]*const abi.Weight {
    return .{ f.linearOf(mw.shared_gate_w) orelse return null, f.linearOf(mw.shared_up_w) orelse return null, f.linearOf(mw.shared_down_w) orelse return null };
}

fn rowsOf(x: mlx.mlx_array) c_int {
    const sh = mlx.getShape(x);
    var rows: c_int = 1;
    if (sh.len > 1) for (sh[0 .. sh.len - 1]) |d| {
        rows *= d;
    };
    return rows;
}

/// The shared gate's logit exactly as `moeAddSharedDown` computes it.
fn sharedGateLogit(self: *Trunk, expert_x: mlx.mlx_array, mw: *const MoeMlpWeights) !mlx.mlx_array {
    const seg_w = mw.shared_expert_gate_w.?;
    const qp = self.quantParamsHinted(seg_w, mw.shared_expert_gate_s.?, lastDim(expert_x));
    return qmatmulBits(expert_x, seg_w, mw.shared_expert_gate_s.?, mw.shared_expert_gate_b.?, qp.bits, qp.group_size, qp.mode, self.s);
}

/// The fused shared expert's rows. One row keeps the chain (`swiglu`, then `moeAddGatedShared`): its shared gate and up
/// read only x and so run beside the router, where the fused launches' shared heads wait for the router's slots. The
/// same bytes.
fn sharedRows(rows: c_int) bool {
    return rows >= 2 and rows <= 4;
}

/// The shared expert's chain the fused launches reproduce: a gated shared expert on the trunk format's linears, the
/// table SwiGLU (silu, unclamped), bf16 activations, 2..4 rows (`sharedRows`).
fn sharedServes(self: *const Trunk, expert_x: mlx.mlx_array, mw: *const MoeMlpWeights) bool {
    if (mw.shared_ungated or mw.shared_expert_gate_w == null or qwen4Standin().moe_shared) return false;
    const rows = rowsOf(expert_x);
    if (!sharedRows(rows) or mlx.mlx_array_dtype(expert_x) != .bfloat16 or !fused_activations.swigluFusedEnabled()) return false;
    const clamp = swigluClampLimit(self.config.swiglu_limit, self.config.swiglu_clamp) orelse return false;
    return clamp == 0 and sharedTable(&self.formats) != null and sharedLinears(&self.formats, mw) != null;
}

/// `quiet`: a refusal is not logged (the routed form declines the rows the format prepares in a launch of its own).
fn callSharedSwiglu(self: *Trunk, banks: *const [3]abi.Weight, expert_x: mlx.mlx_array, inds: mlx.mlx_array, scores: mlx.mlx_array, opts: *const abi.SharedSwiglu, quiet: bool) ?mlx.mlx_array {
    const t = sharedTable(&self.formats).?;
    var out: abi.Array = .{};
    const v = t.swiglu_shared.?(&banks[0], &banks[1], &banks[2], arr(expert_x), arr(inds), arr(scores), opts, strm(self.s), &out);
    if (!v.isOk()) {
        if (!quiet) refused(t, .swiglu_shared, v);
        return null;
    }
    return own(out);
}

/// The fused shared expert: the routed experts and the gated shared expert in one launch chain at 2..4 rows, byte for
/// byte `swiglu` + `moeAddGatedShared` (the shared linears, the table SwiGLU, the gate's logit by
/// `qmatmulBits`, MLX's sigmoid, multiply and add). `given_logit`: the gate's logit the router kernel computed in MLX's
/// gemv order (`gateRouting`). Null: that chain.
pub fn sharedFused(self: *Trunk, expert_x: mlx.mlx_array, mw: *const MoeMlpWeights, banks: *const [3]abi.Weight, inds: mlx.mlx_array, scores: mlx.mlx_array, given_logit: ?mlx.mlx_array) !?mlx.mlx_array {
    if (!sharedServes(self, expert_x, mw)) return null;
    const sh = sharedLinears(&self.formats, mw).?;
    const logit = given_logit orelse try sharedGateLogit(self, expert_x, mw);
    defer if (given_logit == null) {
        _ = mlx.mlx_array_free(logit);
    };
    const sigtab = try fused_activations.swigluSigTable(self.s, .bfloat16, std.heap.c_allocator);
    const opts: abi.SharedSwiglu = .{ .gate = sh[0], .up = sh[1], .down = sh[2], .sigtab = arr(sigtab), .logit = arr(logit) };
    const y = callSharedSwiglu(self, banks, expert_x, inds, scores, &opts, false) orelse return null;
    served(.swiglu_shared, "the shared expert inside the routed launches (rows={d}{s})", .{ rowsOf(expert_x), if (given_logit != null) ", the gate's logit from the router" else "" });
    return y;
}

/// Where the router is folded into the fused shared expert (2..4 rows on an EXL3 trunk pack): `moeMLP2WithRouter`'s path
/// to `sharedFused` (no expert bias or per-expert scale, the fused softmax router on the GPU, no stand-in or dispatch
/// probe), with the router's logits in x's dtype.
fn routedServes(self: *const Trunk, expert_x: mlx.mlx_array, mw: *const MoeMlpWeights, router_logits: mlx.mlx_array) bool {
    const rows = rowsOf(expert_x);
    if (!self.config.trunk_format or rows < 2 or rows > 4) return false;
    if (mw.expert_bias != null or mw.per_expert_scale != null or mw.gate_expert_scale.ctx != null) return false;
    if (!router_kernel.moeRouterFusedEnabled() or dispatchProbeCount() != 0 or qwen4Standin().moe_router) return false;
    if (!sharedServes(self, expert_x, mw) or mlx.mlx_array_dtype(router_logits) != mlx.mlx_array_dtype(expert_x)) return false;
    const lsh = mlx.getShape(router_logits);
    const k = self.config.num_experts_per_tok;
    return lsh.len >= 1 and k >= 1 and k <= 31 and lsh[lsh.len - 1] >= @as(c_int, @intCast(k)) and lsh[lsh.len - 1] <= router_kernel.ROUTER_MAX_EXPERTS;
}

/// The shared gate's weight vector where the router kernel sums its logit in MLX's gemv order (`gateGemvShape`), on the
/// folded path; else null.
fn gateVec(self: *const Trunk, expert_x: mlx.mlx_array, mw: *const MoeMlpWeights, router_logits: mlx.mlx_array) ?mlx.mlx_array {
    const v = mw.shared_gate_vec;
    if (v.ctx == null or !routedServes(self, expert_x, mw, router_logits)) return null;
    const rows = rowsOf(expert_x);
    if (mlx.mlx_array_size(v) * @as(usize, @intCast(rows)) != mlx.mlx_array_size(expert_x)) return null;
    const xd = mlx.mlx_array_dtype(expert_x);
    if (!router_kernel.gateGemvShape(rows, @intCast(mlx.mlx_array_size(v)), xd, mlx.mlx_array_dtype(v), xd)) return null;
    return v;
}

/// The routed shared expert: the call the router kernel + `sharedFused` make at 2..4 rows, with the router's top-k and
/// weights inside the fused shared expert's routed prepare launch (`SharedSwiglu.logits`), and the gate's logit in that
/// launch too where its order is MLX's gemv's (`gateRouting`): one dispatch and one dependent level fewer, the same
/// bytes. Null: the chain.
pub fn routedShared(self: *Trunk, expert_x: mlx.mlx_array, mw: *const MoeMlpWeights, router_logits: mlx.mlx_array) !?mlx.mlx_array {
    if (!routedServes(self, expert_x, mw, router_logits)) return null;
    const banks = self.formats.banksOf(mw) orelse return null;
    const sh = sharedLinears(&self.formats, mw).?;
    const gvec = gateVec(self, expert_x, mw, router_logits);
    const logit: ?mlx.mlx_array = if (gvec == null) try sharedGateLogit(self, expert_x, mw) else null;
    defer if (logit) |l| {
        _ = mlx.mlx_array_free(l);
    };
    const sigtab = try fused_activations.swigluSigTable(self.s, .bfloat16, std.heap.c_allocator);
    const opts: abi.SharedSwiglu = .{
        .gate = sh[0],
        .up = sh[1],
        .down = sh[2],
        .sigtab = arr(sigtab),
        .logit = if (logit) |l| arr(l) else .{},
        .gate_vec = if (gvec) |g| arr(g) else .{},
        .logits = arr(router_logits),
        .k = self.config.num_experts_per_tok,
    };
    const y = callSharedSwiglu(self, banks, expert_x, .{ .ctx = null }, .{ .ctx = null }, &opts, true) orelse return null;
    if (!routed_said) {
        routed_said = true;
        log.info("[fused] routed swiglu_shared engaged: the router's top-k inside the routed prepare launch (rows={d}{s})\n", .{ rowsOf(expert_x), if (gvec != null) ", the gate's logit there too" else "" });
    }
    return y;
}

var routed_said = false;

/// The gemv gate logit where the routed shared expert did not run: the router kernel in `softmax_gate_gemv` (the
/// softmax mode's selection and weights, the gate's logit summed in MLX's gemv order) for `sharedFused`, whose gemv
/// launch then goes. Null: the router as before.
pub fn gateRouting(self: *Trunk, expert_x: mlx.mlx_array, mw: *const MoeMlpWeights, router_logits: mlx.mlx_array) !?MoeRouting {
    const v = gateVec(self, expert_x, mw, router_logits) orelse return null;
    const k: c_int = @intCast(self.config.num_experts_per_tok);
    const r = (try router_kernel.moeRouterTopK(self.s, router_logits, .{ .ctx = null }, k, .softmax_gate_gemv, true, 1.0, mlx.mlx_array_dtype(router_logits), 0, 0, .{ .x = expert_x, .w = v })) orelse return null;
    if (!gate_gemv_said) {
        gate_gemv_said = true;
        log.info("[fused] gate-gemv routing engaged: the router kernel sums the shared gate's logit in MLX's gemv order for the fused shared expert (rows={d})\n", .{rowsOf(expert_x)});
    }
    return r;
}

var gate_gemv_said = false;

// ── tests ──

const testing = std.testing;

fn testFacts() AttnInFacts {
    return .{ .output_gate = true, .batch = 1, .prefill = true, .rows = 512, .plain_rope = true, .head_dim = 256, .rope_dims = 64, .norms = true, .linears = true, .dtype_ok = true };
}
