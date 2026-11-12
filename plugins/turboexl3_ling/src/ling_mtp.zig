//! Ling-3.1's NextN (MTP) head on turboexl3 (MTP, S13; served with the exact verify).
//! The pack's layer `num_hidden_layers` (mtp_use_kda false): `eh_proj(cat[enorm(embed(x)), hnorm(h)])` through one MLA +
//! MoE block, then `final_layernorm` (exllamav3's `BailingHybridMTPModel`; `h` the trunk's final-normed hidden, the
//! clamp lists not reaching it). The weights load with the trunk (the copied trunk's `lingHead`); its 512 routed experts
//! are resident here, stacked into turboexl3's bank shape and run by `moeLing` at top-8, beside the streamed bank, which
//! never sees them. Its MLA layer caches its latent (MTP_HEAD_LATENT, S35: src/ling_mtp_latent.zig, `latentHead`). The
//! engine projects the head's output through the trunk head (`trunk_weight`): the EXL3 head as MLX affine 4-bit,
//! quantized once at load from a dense bf16 copy (MTP_HEAD_Q4, S33: `denseHead`, `quantizeHead`), so drafts select on
//! it while the trunk keeps its own EXL3 head. Adapted from mlx-serve's GLM-5.3-Flash head (plugins/glm5_next/head.zig).

const std = @import("std");
const ran = @import("ling_ran.zig");
const kit = @import("mlx_serve");
const abi = kit.abi.plugin;
const quant_abi = kit.abi.quant;
const mlx = kit.mlx;
const t3 = @import("turboexl3");
const host = @import("mlx_serve/plugins/glm5_next/host.zig");
const trunk = @import("mlx_serve/plugins/glm5_next/trunk.zig");
const glm_moe = @import("mlx_serve/plugins/glm5_next/moe.zig");
const mla = @import("mlx_serve/plugins/bailing_hybrid/mla.zig");
const Slots = @import("mlx_serve/plugins/glm5_next/slots.zig").Slots;
const routed = @import("mlx_serve/sdk/weights/routed.zig");
const log = @import("mlx_serve/sdk/util/log.zig");
const ling_exl3 = @import("ling_exl3.zig");
const ling_mtp_latent = @import("ling_mtp_latent.zig");
const ling_mla_fused = @import("ling_mla_fused.zig");
const ling_bill = @import("bill.zig");
/// C4: the history of a prompt over 32,768 tokens in forwards of at most 4,096 rows (src/ling_segments.zig).
const ling_segments = @import("ling_segments.zig");
const Layout = @import("mlx_serve/sdk/quant/native.zig").Layout;

const Array = mlx.mlx_array;
const Verdict = abi.Verdict;
const gpa = std.heap.c_allocator;

extern "c" fn mlx_eye(res: *Array, n: c_int, m: c_int, k: c_int, dtype: mlx.mlx_dtype, s: mlx.mlx_stream) c_int;

/// The draft readout's MLX affine width and group: the dense head as 4-bit, group 64, in place of its bf16 copy. Drafts
/// only: a verify reads the trunk's own EXL3 head, so the output cannot change. The bank bills what MLX holds when it
/// opens, so the bytes the bf16 copy would hold go to its rows.
pub const q4_bits: u8 = 4;
pub const q4_group: u32 = 64;

/// MTP_HEAD_Q4's run-path mark. The engine reads the draft readout it borrowed (`trunk_weight(.head)`) on each draft
/// step's output, the head forward's one row. `q4_borrowed`: the engine borrowed the 4-bit readout. `q4_steps`: the
/// head's one-row forwards (draft steps) since the request's prompt pass. A verify (plugin.zig `forward`) after such a
/// draft step marks the bit.
var q4_borrowed = false;
var q4_steps: u32 = 0;

/// A prompt pass starts a request (plugin.zig `forward`, the phase turning to prompt): its draft steps count from 0.
pub fn q4PromptStart() void {
    q4_steps = 0;
}

/// A verify forward (plugin.zig `forward`): after draft steps on the borrowed 4-bit readout, MTP_HEAD_Q4's run-path
/// mark.
pub fn q4AtVerify() void {
    if (!q4_borrowed or q4_steps == 0) return;
    ran.mark(.MTP_HEAD_Q4);
}

/// The draft readout as `trunk_weight(.head)` lends it: the head as MLX affine `q4_bits`/gs`q4_group` (`quantizeHead`).
const Readout = struct { w: Array, scales: Array, biases: Array };

/// The head's draft readout and experts, one model a process (the plugin serves one pack).
var served: ?struct { experts: *Experts, readout: Readout } = null;

/// The NextN layer's routed experts, resident: each projection's experts stacked as turboexl3's bank reads them
/// (trellis u16 [E, in/16, out/16, n], suh [E, in], svh [E, out]).
pub const Experts = struct {
    bank: t3.Bank,
    owned: [9]Array,
    s: mlx.mlx_stream,

    /// `<base>.mlp.experts.<e>.<gate|up|down>_proj.{trellis,suh,svh,mul1}` for e in 0..n_experts, every mul1 checked.
    pub fn load(src: *const quant_abi.Source, base: []const u8, n_experts: u32, s: mlx.mlx_stream) !*Experts {
        const self = try gpa.create(Experts);
        errdefer gpa.destroy(self);
        self.* = .{ .bank = undefined, .owned = @splat(.{ .ctx = null }), .s = s };
        errdefer for (self.owned) |a| if (a.ctx != null) {
            _ = mlx.mlx_array_free(a);
        };
        var nb: [256]u8 = undefined;
        inline for (.{ "gate", "up", "down" }, 0..) |p, pi| {
            var parts: [3]Array = undefined;
            inline for (.{ "trellis", "suh", "svh" }, 0..) |leaf, li| {
                const vec = mlx.mlx_vector_array_new();
                defer _ = mlx.mlx_vector_array_free(vec);
                for (0..n_experts) |e| {
                    const name = try std.fmt.bufPrint(&nb, "{s}.mlp.experts.{d}.{s}_proj.{s}", .{ base, e, p, leaf });
                    const a = src.tensor(name);
                    if (a.ctx == null) {
                        log.err("[turboexl3_ling] MISSING WEIGHT: {s}\n", .{name});
                        return error.MissingMtpWeight;
                    }
                    if (li == 0) try checkMul1(src, base, e, p);
                    _ = mlx.mlx_vector_array_append_value(vec, .{ .ctx = a.ctx });
                }
                var st = mlx.mlx_array_new();
                errdefer _ = mlx.mlx_array_free(st);
                try mlx.check(mlx.mlx_stack_axis(&st, vec, 0, s));
                if (li == 0) {
                    var v = mlx.mlx_array_new();
                    errdefer _ = mlx.mlx_array_free(v);
                    try mlx.check(mlx.mlx_view(&v, st, .uint16, s));
                    _ = mlx.mlx_array_free(st);
                    st = v;
                }
                parts[li] = st;
                self.owned[pi * 3 + li] = st;
            }
            @field(self.bank, p) = .{ .trellis = parts[0], .suh = parts[1], .svh = parts[2] };
        }
        const vec = mlx.mlx_vector_array_new_data(&self.owned, self.owned.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_eval(vec));
        return self;
    }

    fn checkMul1(src: *const quant_abi.Source, base: []const u8, e: usize, comptime p: []const u8) !void {
        var nb: [256]u8 = undefined;
        const name = try std.fmt.bufPrint(&nb, "{s}.mlp.experts.{d}.{s}_proj.mul1", .{ base, e, p });
        const m1 = src.tensor(name);
        if (m1.ctx == null) return error.MissingMtpWeight;
        var v: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&v, .{ .ctx = m1.ctx }));
        if (@as(u32, @bitCast(v)) != ling_exl3.mul1_multiplier) {
            log.err("[turboexl3_ling] {s} is 0x{x}, not mul1's 0x{x}\n", .{ name, @as(u32, @bitCast(v)), ling_exl3.mul1_multiplier });
            return error.Exl3CodebookUnsupported;
        }
    }

    pub fn deinit(self: *Experts) void {
        for (self.owned) |a| if (a.ctx != null) {
            _ = mlx.mlx_array_free(a);
        };
        gpa.destroy(self);
    }

    /// The routed sum `[n, hidden]` f32 of `x` `[n, hidden]` over the routes `idx` `[n, k]` at weights `wts` `[n, k]`:
    /// one `moeLing` call at top-k (no clamp: the lists do not reach the head), x read in f32.
    fn sum(self: *const Experts, x: Array, idx: Array, wts: Array) !Array {
        const s = self.s;
        const xs = mlx.getShape(x);
        const n = xs[0];
        const h = xs[1];
        const k = mlx.getShape(idx)[1];
        var tmp: [6]Array = @splat(mlx.mlx_array_new());
        defer for (tmp) |a| {
            _ = mlx.mlx_array_free(a);
        };
        try mlx.check(mlx.mlx_astype(&tmp[0], x, .float32, s));
        try mlx.check(mlx.mlx_reshape(&tmp[1], tmp[0], &[_]c_int{ 1, n, h }, 3, s));
        try mlx.check(mlx.mlx_astype(&tmp[2], idx, .uint32, s));
        try mlx.check(mlx.mlx_reshape(&tmp[3], tmp[2], &[_]c_int{ 1, n, k }, 3, s));
        try mlx.check(mlx.mlx_astype(&tmp[4], wts, .float32, s));
        try mlx.check(mlx.mlx_reshape(&tmp[5], tmp[4], &[_]c_int{ 1, n, k }, 3, s));
        const y = try t3.moeLing(s, tmp[1], self.bank, tmp[3], tmp[5], .mul1, 0);
        defer _ = mlx.mlx_array_free(y);
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_reshape(&out, y, &[_]c_int{ n, h }, 2, s));
        return out;
    }

    /// The experts as the MoE block's `Routed` source (no bank: no phases, no read-ahead, nothing to settle).
    pub fn routedHook(self: *Experts) routed.Routed {
        const C = struct {
            fn rows(ctx: *anyopaque, layer: u32, x: Array, idx: Array, wts: Array) routed.Ret(Array) {
                _ = layer;
                const e: *Experts = @ptrCast(@alignCast(ctx));
                return .of(e.sum(x, idx, wts));
            }
            fn settle(ctx: *anyopaque) routed.Ret(void) {
                _ = ctx;
                return .of({});
            }
        };
        return .{ .ctx = self, .math = .{ .source = C.rows }, .settle_fn = C.settle };
    }
};

/// At load (before the bank opens, so the bank's room leaves these out): the head's weights from `src`, its experts
/// resident and bound, the draft readout (the head as MLX affine 4-bit), and its layer on the latent cache
/// (`latentHead`). A pack without the layer serves without MTP.
pub fn load(m: *trunk.Model, src: *const quant_abi.Source) !void {
    const c = m.c;
    var head = (try trunk.lingHead(m, src)) orelse {
        log.info("[turboexl3_ling] MTP: the pack has no NextN layer (layer {d}): served without MTP\n", .{c.num_hidden_layers});
        return;
    };
    var bb: [64]u8 = undefined;
    const base = try std.fmt.bufPrint(&bb, "model.layers.{d}", .{c.num_hidden_layers});
    const experts = try Experts.load(src, base, c.num_experts, m.s);
    errdefer experts.deinit();
    head.moe.streamed = .{ .routed = experts.routedHook(), .layer = 0 };
    // The dense bf16 copy goes as soon as its quantization is built.
    const readout: Readout = blk: {
        const dense = try denseHead(m);
        defer _ = mlx.mlx_array_free(dense);
        break :blk try quantizeHead(m.s, dense);
    };
    m.ling_mtp = head;
    served = .{ .experts = experts, .readout = readout };
    log.info("[turboexl3_ling] MTP head loaded: NextN layer {d} (MLA + MoE, {d} experts resident on turboexl3) and a 4-bit draft readout of the EXL3 head\n", .{ c.num_hidden_layers, c.num_experts });
    {
        const sh = mlx.getShape(readout.w);
        const vocab: u64 = @intCast(sh[0]);
        const hidden: u64 = @as(u64, @intCast(sh[1])) * 32 / q4_bits;
        const bf16 = ling_bill.draftReadout(vocab, hidden, 16, 0);
        const q4 = ling_bill.draftReadout(vocab, hidden, q4_bits, q4_group);
        log.info("[turboexl3_ling] MTP_HEAD_Q4 selected: the draft readout is the head at MLX affine {d}-bit/gs{d}, {d} MiB in place of the dense bf16 {d} MiB; the bank's rows take the {d} MiB freed\n", .{ q4_bits, q4_group, q4 >> 20, bf16 >> 20, (bf16 - q4) >> 20 });
    }
    try latentHead(m);
}

/// The head's layer on the latent cache (src/ling_mtp_latent.zig), which the bill already holds (plugin.zig
/// `kvPerToken`). The latent attention folds kv_b's halves, so a quantized kv_b is refused by name, and so is a kv_b
/// that holds its stored rows in neither orientation (`kvbOrient`). The load line names the path the head's decode
/// calls run: turboexl3's plan over the operands those calls hand it (`ling_mla_fused.declines`).
fn latentHead(m: *trunk.Model) !void {
    const fa = &m.ling_mtp.?.mla;
    if (fa.kv_b.exl3 or fa.kv_b.bits != 0 or fa.kv_b.w.ctx == null) {
        log.err("[turboexl3_ling] MTP head: its kv_b is quantized; the head's latent cache needs a plain kv_b\n", .{});
        return error.LingLatentCacheNeedsPlainKvB;
    }
    const c = m.c;
    const rows: c_int = @intCast(c.num_attention_heads * (c.mla_qk_nope_head_dim + c.mla_v_head_dim));
    const kv_lora: c_int = @intCast(c.mla_kv_lora_rank);
    const held = mlx.getShape(fa.kv_b.w);
    const orient = ling_mtp_latent.kvbOrient(held, rows, kv_lora) orelse {
        log.err("[turboexl3_ling] MTP head: its kv_b is held as {any}: neither its stored rows [{d}, {d}] nor the loader's [{d}, {d}] view\n", .{ held, rows, kv_lora, kv_lora, rows });
        return error.LingLatentKvBShape;
    };
    ling_mtp_latent.head = fa;
    // Ling's activations are bf16: the query, and the latent and rope key the cache holds (the loader casts a plain f16
    // weight to them, trunk.zig `lin`).
    const why = ling_mla_fused.declines(@intCast(c.num_attention_heads), .{ rows, kv_lora }, .bfloat16, mlx.mlx_array_dtype(fa.kv_b.w));
    log.info("[turboexl3_ling] MTP head: latent cache: {d} bytes a token instead of {d}; its decode attends in the latent: {s}{s}{s}, over kv_b's stored rows [{d}, {d}] ({s})\n", .{
        (c.mla_kv_lora_rank + c.mla_qk_rope_head_dim) * 2,
        c.num_attention_heads * (c.mla_qk_nope_head_dim + c.mla_qk_rope_head_dim + c.mla_v_head_dim) * 2,
        if (why == null) "turboexl3's fused launches" else "the composed latent chain (turboexl3 declines: ",
        why orelse "",
        if (why == null) "" else ")",
        rows,
        kv_lora,
        if (orient == .loader_view) "the transpose of the loader's [kv_lora, rows] view" else "as the loader holds them",
    });
}

/// `dense` ([vocab, hidden] bf16) as MLX affine `q4_bits`/gs`q4_group`, a block of rows at a time so a block's
/// transients are all it holds beside the result (a group runs along a row, so the blocks give the whole matrix's words).
fn quantizeHead(s: mlx.mlx_stream, dense: Array) !Readout {
    const sh = mlx.getShape(dense);
    const block: c_int = 32768;
    const parts = [3]mlx.mlx_vector_array{ mlx.mlx_vector_array_new(), mlx.mlx_vector_array_new(), mlx.mlx_vector_array_new() };
    defer for (parts) |p| {
        _ = mlx.mlx_vector_array_free(p);
    };
    var r0: c_int = 0;
    while (r0 < sh[0]) : (r0 += block) {
        var rows = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(rows);
        try mlx.check(mlx.mlx_slice(&rows, dense, &[_]c_int{ r0, 0 }, 2, &[_]c_int{ @min(r0 + block, sh[0]), sh[1] }, 2, &[_]c_int{ 1, 1 }, 2, s));
        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(&triple, rows, mlx.mlx_optional_int.some(q4_group), mlx.mlx_optional_int.some(q4_bits), "affine", .{ .ctx = null }, s));
        if (mlx.mlx_vector_array_size(triple) != 3) return error.LingHeadQuantize;
        try mlx.check(mlx.mlx_eval(triple));
        for (parts, 0..) |p, i| {
            var a = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(a);
            try mlx.check(mlx.mlx_vector_array_get(&a, triple, i));
            _ = mlx.mlx_vector_array_append_value(p, a);
        }
    }
    var out: [3]Array = .{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
    errdefer for (out) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (&out, parts) |*o, p| try mlx.check(mlx.mlx_concatenate_axis(o, p, 0, s));
    const ev = mlx.mlx_vector_array_new_data(&out, out.len);
    defer _ = mlx.mlx_vector_array_free(ev);
    try mlx.check(mlx.mlx_eval(ev));
    _ = mlx.mlx_clear_cache();
    return .{ .w = out[0], .scales = out[1], .biases = out[2] };
}

/// The EXL3 head `[vocab, hidden]` as a dense bf16 matrix: turboexl3's linear over the identity's rows, a block at a
/// time (each row of `I @ W^T` is a column of W), then transposed. Built once; the engine's draft readout reads it.
fn denseHead(m: *trunk.Model) !Array {
    const s = m.s;
    const l = &m.head;
    if (!l.exl3) return error.LingHeadNotExl3;
    const tsh = mlx.getShape(l.w);
    const hidden: c_int = tsh[0] * 16;
    const block: c_int = 512;
    var eye = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eye);
    try mlx.check(mlx_eye(&eye, hidden, hidden, 0, .bfloat16, s));
    const parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts);
    var r0: c_int = 0;
    while (r0 < hidden) : (r0 += block) {
        const r1 = @min(r0 + block, hidden);
        var rows = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(rows);
        try mlx.check(mlx.mlx_slice(&rows, eye, &[_]c_int{ r0, 0 }, 2, &[_]c_int{ r1, hidden }, 2, &[_]c_int{ 1, 1 }, 2, s));
        const y = try ling_exl3.linear(s, rows, l.w, l.s, l.b);
        defer _ = mlx.mlx_array_free(y);
        try mlx.check(mlx.mlx_array_eval(y));
        _ = mlx.mlx_vector_array_append_value(parts, y);
    }
    var wt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wt);
    try mlx.check(mlx.mlx_concatenate_axis(&wt, parts, 0, s));
    var w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w);
    try mlx.check(mlx.mlx_transpose(&w, wt, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_contiguous(&out, w, false, s));
    try mlx.check(mlx.mlx_array_eval(out));
    _ = mlx.mlx_clear_cache();
    return out;
}

/// The trunk weights a drafter reads: the embedding table as loaded, the head as its draft readout (MLX affine 4-bit;
/// only while the MTP head is served; otherwise refused, and the model serves without drafters).
pub fn trunkWeight(mp: *anyopaque, which: abi.TrunkWeight, out: *quant_abi.Weight) callconv(.c) Verdict {
    const sv = served orelse return Verdict.refuse(.unsupported, "turboexl3_ling: no MTP head is served (the pack has no NextN layer)");
    switch (which) {
        .embed => return host.trunkWeight(mp, which, out),
        .head => {
            const r = sv.readout;
            out.* = .{ .w = .{ .ctx = r.w.ctx }, .scales = .{ .ctx = r.scales.ctx }, .biases = .{ .ctx = r.biases.ctx } };
            q4_borrowed = true;
            out.bits = q4_bits;
            out.group_size = q4_group;
            out.layout = @backingInt(Layout.affine);
            return .ok;
        },
    }
}

fn headLoad(mp: *anyopaque, _: *const abi.LoadCtx) callconv(.c) Verdict {
    const m = host.of(mp);
    // A pack (or a load) without the head: the engine's quiet "no MTP head" (`MissingMtpWeight`).
    const h = m.ling_mtp orelse return Verdict.refuse(.unsupported, "MissingMtpWeight");
    mlx.evalFields(h) catch |e| return Verdict.named(.failed, @errorName(e));
    m.mtp_ready = true;
    return .ok;
}

/// RQ (src/ling_rq.zig): plugin.zig's hook at each head forward, before its append, with the head cache's length and
/// the forward's rows (`rqHeadAppend`: the head's capacity replayed, and a trim first where the append would pass the
/// plan's). Set when the bank opens (plugin.zig `openBank`); null before, when no head forward runs.
pub var rq_append: ?*const fn (mp: *anyopaque, len: u64, n_rows: u64) anyerror!void = null;

fn headForward(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) Verdict {
    const m = host.of(mp);
    if (!m.mtp_ready) return Verdict.refuse(.failed, "MTP head not loaded");
    const h = &m.ling_mtp.?;
    const ops = f.cache_ops orelse return Verdict.refuse(.unsupported, "a head forward without cache_ops");
    if (f.caches.len != 1 or f.hidden_in.ctx == null or f.head != 0) return Verdict.refuse(.unsupported, "one head, one cache, the trunk's hidden");
    // RQ: the head cache's append, before each forward of the block (its history after a prompt, a draft step, a verify's
    // accepted rows).
    const append = rq_append orelse return Verdict.refuse(.failed, "a head forward before the bank opened (no RQ hook)");
    // C4: a prompt's history over 32,767 rows (a prompt over 32,768 tokens) in forwards of at most 4,096 rows; every
    // other forward at once, as before.
    const normed = history(mp, m, h, ops, f, append, ling_segments.historyRows(f.rows)) catch |e| return Verdict.named(.failed, @errorName(e));
    // A draft step: one row, whose output the engine reads through the readout it borrowed (MTP_HEAD_Q4's count).
    if (q4_borrowed and f.rows == 1) q4_steps +|= 1;
    if (f.want_hidden) out.hidden = .{ .ctx = normed.ctx } else _ = mlx.mlx_array_free(normed);
    return .ok;
}

/// The head's forward of `f.rows` rows in forwards of at most `chunk` rows (C4: src/ling_segments.zig `historyRows`):
/// each its RQ hook (`append`, the head cache's length and the forward's rows) and the block (`step`) at the head
/// cache's length, the rows `f.ids` (or `f.tokens`) and `f.hidden_in` hold at its place; a forward before the last has
/// its cache append evaluated (the head cache's view), so the next one's graph does not hold it, and its output is
/// dropped unevaluated, as a history forward's is. Returns the last forward's normed output, unevaluated. A `chunk` that
/// holds every row is one forward, as before C4.
pub fn history(mp: *anyopaque, m: *trunk.Model, h: *const trunk.LingHead, ops: *const abi.CacheOps, f: *const abi.Forward, append: *const fn (mp: *anyopaque, len: u64, n_rows: u64) anyerror!void, chunk: u64) !Array {
    const cache = f.caches.slice()[0];
    if (chunk >= f.rows) {
        try append(mp, ops.len(cache, 0), f.rows);
        return step(m, h, ops, f);
    }
    const s = m.s;
    const ids = try host.tokenIds(f);
    defer _ = mlx.mlx_array_free(ids);
    const hid: Array = .{ .ctx = f.hidden_in.ctx };
    const hs = mlx.getShape(hid);
    if (hs.len != 3 or hs[1] != @as(c_int, @intCast(f.rows))) return error.LingMtpHistoryRows;
    var at: u32 = 0;
    var n_fwd: usize = 0;
    while (true) {
        const n: u32 = @intCast(@min(chunk, f.rows - at));
        try append(mp, ops.len(cache, 0), n);
        var idv = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(idv);
        try mlx.check(mlx.mlx_slice(&idv, ids, &[_]c_int{ 0, @intCast(at) }, 2, &[_]c_int{ 1, @intCast(at + n) }, 2, &[_]c_int{ 1, 1 }, 2, s));
        var hv = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(hv);
        try mlx.check(mlx.mlx_slice(&hv, hid, &[_]c_int{ 0, @intCast(at), 0 }, 3, &[_]c_int{ hs[0], @intCast(at + n), hs[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        var sub = f.*;
        sub.rows = n;
        sub.ids = .{ .ctx = idv.ctx };
        sub.tokens = .{};
        sub.hidden_in = .{ .ctx = hv.ctx };
        const normed = try step(m, h, ops, &sub);
        at += n;
        n_fwd += 1;
        if (at == f.rows) {
            if (n_fwd > 1) historyEngaged(n_fwd, f.rows, chunk);
            return normed;
        }
        _ = mlx.mlx_array_free(normed);
        // this forward's append evaluated (the head cache's view of every row so far)
        var v: abi.KvView = .{};
        const vv = ops.view(cache, 0, .{ .ctx = s.ctx }, &v);
        if (!vv.isOk()) return error.LingMtpHistoryView;
        defer ops.release(&v);
        const ev = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(ev);
        inline for (.{ v.k, v.v }) |x| if (x.ctx != null) {
            _ = mlx.mlx_vector_array_append_value(ev, .{ .ctx = x.ctx });
        };
        try mlx.check(mlx.mlx_eval(ev));
    }
}

/// MTP_HISTORY_CHUNKS' run-path mark, at each history in two or more forwards, and its engaged line, once a process.
var history_said = false;

fn historyEngaged(n_fwd: usize, rows: u64, chunk: u64) void {
    ran.mark(.MTP_HISTORY_CHUNKS);
    if (history_said) return;
    history_said = true;
    log.info("[turboexl3_ling] MTP_HISTORY_CHUNKS engaged: the MTP head's history of {d} rows (a prompt over {d} tokens) ran in {d} forwards of at most {d} rows, each forward's cache append evaluated before the next (draft-only: the exact verify decides the text)\n", .{ rows, ling_segments.segment_rows, n_fwd, chunk });
}

/// The block over the forward's rows, appended to the head's cache at its own length; the last row's normed output.
fn step(m: *trunk.Model, h: *const trunk.LingHead, ops: *const abi.CacheOps, f: *const abi.Forward) !Array {
    const s = m.s;
    const ids = try host.tokenIds(f);
    defer _ = mlx.mlx_array_free(ids);
    const n: c_int = mlx.getShape(ids)[1];
    // The head attends (and rotates) at its cache's own length, as the GLM head: row p pairs the trunk's hidden at p
    // with token p + 1, at position p.
    const len: u32 = @intCast(ops.len(f.caches.slice()[0], 0));
    const pos = [_]u32{len};
    const sl: Slots = .{ .ops = ops, .caches = f.caches.slice(), .positions = &pos };
    const emb = try m.embedding(ids);
    defer _ = mlx.mlx_array_free(emb);
    const en = try m.rmsNorm(emb, h.enorm);
    defer _ = mlx.mlx_array_free(en);
    var hid = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hid);
    try mlx.check(mlx.mlx_astype(&hid, .{ .ctx = f.hidden_in.ctx }, mlx.mlx_array_dtype(emb), s));
    const hn = try m.rmsNorm(hid, h.hnorm);
    defer _ = mlx.mlx_array_free(hn);
    var cat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cat);
    {
        const vec = mlx.mlx_vector_array_new_data(&[_]Array{ en, hn }, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, 2, s));
    }
    const x = try m.mm(&h.eh_proj, cat);
    defer _ = mlx.mlx_array_free(x);
    const a_in = try m.rmsNorm(x, h.input_norm);
    defer _ = mlx.mlx_array_free(a_in);
    const attn = try mla.mlaAttn(m, sl, a_in, &h.mla, 0, @intCast(len), 1, n, n > 1);
    defer _ = mlx.mlx_array_free(attn);
    var x1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x1);
    try mlx.check(mlx.mlx_add(&x1, x, attn, s));
    const m_in = try m.rmsNorm(x1, h.post_norm);
    defer _ = mlx.mlx_array_free(m_in);
    const mlp = try glm_moe.forward(&h.moe, s, m_in);
    defer _ = mlx.mlx_array_free(mlp);
    var y = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_add(&y, x1, mlp, s));
    var last = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(last);
    const sh = mlx.getShape(y);
    try mlx.check(mlx.mlx_slice(&last, y, &[_]c_int{ 0, sh[1] - 1, 0 }, 3, &[_]c_int{ 1, sh[1], sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    return m.rmsNorm(last, h.final_norm);
}

fn noop(_: *anyopaque, _: u32) callconv(.c) Verdict {
    return .ok;
}
fn release(_: *anyopaque, _: u32) callconv(.c) void {}
fn truncate(_: *anyopaque, _: u32, _: u64) callconv(.c) void {}
fn adopt(_: *anyopaque, _: u32, _: abi.Handle) callconv(.c) Verdict {
    return .ok;
}
fn select(_: *anyopaque, _: abi.Array, _: *abi.Array) callconv(.c) Verdict {
    return Verdict.refuse(.unsupported, "the engine selects drafts");
}
/// The head's weights load with the trunk, before the bank opens: the bank's room already leaves them out.
fn bill(_: *const abi.Parsed) callconv(.c) u64 {
    return 0;
}

pub const hooks: abi.HeadHooks = .{ .load = &headLoad, .activate = &noop, .release = &release, .forward = &headForward, .truncate = &truncate, .adopt = &adopt, .mark = &release, .draft_select = &select, .bill = &bill };
