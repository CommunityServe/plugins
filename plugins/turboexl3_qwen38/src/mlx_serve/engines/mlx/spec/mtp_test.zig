//! copied_from: src/engines/mlx/spec/mtp_test.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
const std = @import("std");
const mtp = @import("../../../sdk/spec/mtp/mtp.zig");
const QuantMode = @import("../../../sdk/quant/params.zig").QuantMode;
const mlx = @import("../../../mlx/ffi.zig");
const model_mod = @import("../../../sdk/config/model.zig");
const params_mod = @import("../../../sdk/quant/params.zig");
const Transformer = @import("../transformer.zig").Transformer;
const KVCache = @import("../kv/cache.zig").KVCache;
const testing = std.testing;


/// Random dense bf16 sidecar (hidden 8, head_dim 4, 2 q / 2 kv heads, mlp 16, vocab 16)
/// plus the target fields `forward` reads.
const TinyHead = struct {
    tmp: std.testing.TmpDir,
    m: mtp.MtpModel,
    xfm: Transformer,

    fn init(self: *TinyHead) !void {
        const io = testing.io;
        const allocator = testing.allocator;
        const s = mlx.gpuStream();
        self.tmp = std.testing.tmpDir(.{ .iterate = true });
        errdefer self.tmp.cleanup();

        // ── synthetic DENSE sidecar (random bf16; zeros would make every rms-norm
        // output zero and the equivalence trivially true) ──
        var prng = std.Random.DefaultPrng.init(7);
        const save_map = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(save_map);
        var owned: std.ArrayList(mlx.mlx_array) = .empty;
        defer {
            for (owned.items) |a| _ = mlx.mlx_array_free(a);
            owned.deinit(allocator);
        }
        const putRand = struct {
            fn f(map: mlx.mlx_map_string_to_array, list: *std.ArrayList(mlx.mlx_array), alloc: std.mem.Allocator, rng: *std.Random.DefaultPrng, key: [*:0]const u8, shape: []const c_int, st: mlx.mlx_stream) !mlx.mlx_array {
                var n: usize = 1;
                for (shape) |d| n *= @intCast(d);
                const buf = try alloc.alloc(f32, n);
                defer alloc.free(buf);
                for (buf) |*x| x.* = rng.random().floatNorm(f32) * 0.5;
                const f32_arr = mlx.mlx_array_new_data(buf.ptr, shape.ptr, @intCast(shape.len), .float32);
                defer _ = mlx.mlx_array_free(f32_arr);
                var a = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_astype(&a, f32_arr, .bfloat16, st));
                try mlx.check(mlx.mlx_array_eval(a));
                _ = mlx.mlx_map_string_to_array_insert(map, key, a);
                try list.append(alloc, a);
                return a;
            }
        }.f;

        // hidden 8, head_dim 4, 2 q heads (x2 for the q/gate split), 2 kv heads,
        // mlp inter 16, vocab 16.
        // Disk orientation is torch [out, in]: fc maps concat(2H) -> H.
        _ = try putRand(save_map, &owned, allocator, &prng, "mtp.fc.weight", &.{ 8, 16 }, s);
        _ = try putRand(save_map, &owned, allocator, &prng, "mtp.pre_fc_norm_embedding.weight", &.{8}, s);
        _ = try putRand(save_map, &owned, allocator, &prng, "mtp.pre_fc_norm_hidden.weight", &.{8}, s);
        _ = try putRand(save_map, &owned, allocator, &prng, "mtp.norm.weight", &.{8}, s);
        _ = try putRand(save_map, &owned, allocator, &prng, "mtp.layers.0.input_layernorm.weight", &.{8}, s);
        _ = try putRand(save_map, &owned, allocator, &prng, "mtp.layers.0.post_attention_layernorm.weight", &.{8}, s);
        _ = try putRand(save_map, &owned, allocator, &prng, "mtp.layers.0.self_attn.q_norm.weight", &.{4}, s);
        _ = try putRand(save_map, &owned, allocator, &prng, "mtp.layers.0.self_attn.k_norm.weight", &.{4}, s);
        _ = try putRand(save_map, &owned, allocator, &prng, "mtp.layers.0.self_attn.q_proj.weight", &.{ 16, 8 }, s);
        _ = try putRand(save_map, &owned, allocator, &prng, "mtp.layers.0.self_attn.k_proj.weight", &.{ 8, 8 }, s);
        _ = try putRand(save_map, &owned, allocator, &prng, "mtp.layers.0.self_attn.v_proj.weight", &.{ 8, 8 }, s);
        _ = try putRand(save_map, &owned, allocator, &prng, "mtp.layers.0.self_attn.o_proj.weight", &.{ 8, 8 }, s);
        _ = try putRand(save_map, &owned, allocator, &prng, "mtp.layers.0.mlp.gate_proj.weight", &.{ 16, 8 }, s);
        _ = try putRand(save_map, &owned, allocator, &prng, "mtp.layers.0.mlp.up_proj.weight", &.{ 16, 8 }, s);
        _ = try putRand(save_map, &owned, allocator, &prng, "mtp.layers.0.mlp.down_proj.weight", &.{ 8, 16 }, s);

        var dir_buf: [512]u8 = undefined;
        const dir_n = try self.tmp.dir.realPath(io, &dir_buf);
        const dir_abs = dir_buf[0..dir_n];
        const file_path = try std.fs.path.joinZ(allocator, &.{ dir_abs, "model-mtp.safetensors" });
        defer allocator.free(file_path);
        const meta = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta);
        try mlx.check(mlx.mlx_save_safetensors(file_path.ptr, save_map, meta));

        self.m = try mtp.loadMtp(io, allocator, s, dir_abs);
        errdefer self.m.deinit();

        // ── toy target: only the fields forward() reads (config scalars, dense
        // bf16 embed table, dense bf16 lm_head) ──
        var emb_prng = std.Random.DefaultPrng.init(11);
        const mk2d = struct {
            fn f(alloc: std.mem.Allocator, rng: *std.Random.DefaultPrng, rows: usize, cols: usize, st: mlx.mlx_stream) !mlx.mlx_array {
                const buf = try alloc.alloc(f32, rows * cols);
                defer alloc.free(buf);
                for (buf) |*x| x.* = rng.random().floatNorm(f32) * 0.5;
                const shape = [_]c_int{ @intCast(rows), @intCast(cols) };
                const f32_arr = mlx.mlx_array_new_data(buf.ptr, &shape, 2, .float32);
                defer _ = mlx.mlx_array_free(f32_arr);
                var a = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_astype(&a, f32_arr, .bfloat16, st));
                try mlx.check(mlx.mlx_array_eval(a));
                return a;
            }
        }.f;
        const emb_w = try mk2d(allocator, &emb_prng, 16, 8, s);
        errdefer _ = mlx.mlx_array_free(emb_w);
        const lm_w = try mk2d(allocator, &emb_prng, 16, 8, s);

        const xfm = &self.xfm;
        xfm.rht = null;
        xfm.allocator = allocator;
        xfm.s = s;
        xfm.config = .{};
        xfm.config.hidden_size = 8;
        xfm.config.num_attention_heads = 2;
        xfm.config.num_key_value_heads = 2;
        xfm.config.head_dim = 4;
        xfm.config.query_pre_attn_scalar = 4;
        xfm.config.partial_rotary_factor = 0.5;
        xfm.config.attn_output_gate = true;
        xfm.emb_w = emb_w;
        xfm.emb_s = .{ .ctx = null };
        xfm.emb_b = .{ .ctx = null };
        xfm.lm_head_w = lm_w;
        xfm.lm_head_s = .{ .ctx = null };
        xfm.lm_head_b = .{ .ctx = null };
    }

    fn deinit(self: *TinyHead) void {
        self.m.deinit();
        _ = mlx.mlx_array_free(self.xfm.emb_w);
        _ = mlx.mlx_array_free(self.xfm.lm_head_w);
        self.tmp.cleanup();
    }
};

/// A bare Transformer carrying nothing but a quantized lm_head + the config
/// fields `headQuantParams` reads. Every function under test touches exactly
/// those, which is the property that let the qwen4_exp head reuse the scheme.
/// A minimal target for the rerank scheme: a Transformer carrying nothing
/// but a quantized `lm_head`, which is all the scheme ever reads. `pub` so
/// the qwen4 arm's tests in generate.zig build the same target.
pub const RerankFixture = struct {
    xfm: Transformer,
    dense: mlx.mlx_array,
    vocab: c_int,
    hidden: c_int,

    pub fn init(s: mlx.mlx_stream, vocab: c_int, hidden: c_int, bits: u32, gs: u32, seed: u64) !RerankFixture {
        var prng = std.Random.DefaultPrng.init(seed);
        const rand = prng.random();
        const n: usize = @intCast(vocab * hidden);
        const buf = try testing.allocator.alloc(f32, n);
        defer testing.allocator.free(buf);
        // bf16-exact levels: the dense reference and the requantized source
        // must not differ by a rounding the test would read as a rerank miss.
        for (buf) |*v| v.* = @as(f32, @floatFromInt(rand.intRangeAtMost(i32, -128, 127))) * 0.0078125;
        const shape = [_]c_int{ vocab, hidden };
        const f32_arr = mlx.mlx_array_new_data(buf.ptr, &shape, 2, .float32);
        defer _ = mlx.mlx_array_free(f32_arr);
        var dense = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&dense, f32_arr, .bfloat16, s));

        var qv = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(qv);
        try mlx.check(mlx.mlx_quantize(
            &qv,
            dense,
            mlx.mlx_optional_int.some(@intCast(gs)),
            mlx.mlx_optional_int.some(@intCast(bits)),
            "affine",
            .{ .ctx = null },
            s,
        ));
        var xfm: Transformer = undefined;
        xfm.rht = null;
        xfm.config = .{};
        xfm.config.hidden_size = @intCast(hidden);
        xfm.config.quant_mode = .affine;
        xfm.config.quant_bits = bits;
        xfm.config.quant_group_size = gs;
        xfm.s = s;
        xfm.lm_head_w = mlx.mlx_array_new();
        xfm.lm_head_s = mlx.mlx_array_new();
        xfm.lm_head_b = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&xfm.lm_head_w, qv, 0));
        try mlx.check(mlx.mlx_vector_array_get(&xfm.lm_head_s, qv, 1));
        try mlx.check(mlx.mlx_vector_array_get(&xfm.lm_head_b, qv, 2));
        return .{ .xfm = xfm, .dense = dense, .vocab = vocab, .hidden = hidden };
    }

    /// The same head left dense: `lm_head_w` the bf16 [V, H] rows, no scales.
    pub fn initDense(s: mlx.mlx_stream, vocab: c_int, hidden: c_int, seed: u64) !RerankFixture {
        var fx = try init(s, vocab, hidden, 8, 64, seed);
        _ = mlx.mlx_array_free(fx.xfm.lm_head_s);
        _ = mlx.mlx_array_free(fx.xfm.lm_head_b);
        try mlx.check(mlx.mlx_array_set(&fx.xfm.lm_head_w, fx.dense));
        fx.xfm.lm_head_s = mlx.mlx_array_new();
        fx.xfm.lm_head_b = mlx.mlx_array_new();
        return fx;
    }

    pub fn deinit(self: *RerankFixture) void {
        _ = mlx.mlx_array_free(self.dense);
        _ = mlx.mlx_array_free(self.xfm.lm_head_w);
        _ = mlx.mlx_array_free(self.xfm.lm_head_s);
        _ = mlx.mlx_array_free(self.xfm.lm_head_b);
    }

    /// `[1, 1, hidden]` bf16 activation.
    fn randomX(self: *const RerankFixture, s: mlx.mlx_stream, seed: u64) !mlx.mlx_array {
        var prng = std.Random.DefaultPrng.init(seed);
        const rand = prng.random();
        const buf = try testing.allocator.alloc(f32, @intCast(self.hidden));
        defer testing.allocator.free(buf);
        for (buf) |*v| v.* = @as(f32, @floatFromInt(rand.intRangeAtMost(i32, -64, 63))) * 0.015625;
        const shape = [_]c_int{ 1, 1, self.hidden };
        const f32_arr = mlx.mlx_array_new_data(buf.ptr, &shape, 3, .float32);
        defer _ = mlx.mlx_array_free(f32_arr);
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&out, f32_arr, .bfloat16, s));
        return out;
    }

    /// The full-vocab logits the real head would produce — the exact answer
    /// the rerank is judged against.
    fn exactLogits(self: *const RerankFixture, s: mlx.mlx_stream, x: mlx.mlx_array) !mlx.mlx_array {
        const qp = mtp.headQuantParams(&self.xfm.config, self.xfm.lm_head_w, self.xfm.lm_head_s);
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_quantized_matmul(
            &out,
            x,
            self.xfm.lm_head_w,
            self.xfm.lm_head_s,
            self.xfm.lm_head_b,
            true,
            mlx.mlx_optional_int.some(@intCast(qp.group_size)),
            mlx.mlx_optional_int.some(@intCast(qp.bits)),
            qp.mode.cstr(),
            s,
        ));
        return out;
    }
};

fn readIdScalar(arr: mlx.mlx_array) !u32 {
    try mlx.check(mlx.mlx_array_eval(arr));
    var v: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&v, arr));
    return @intCast(v);
}

fn readRow(allocator: std.mem.Allocator, s: mlx.mlx_stream, arr: mlx.mlx_array, n: usize) ![]f32 {
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, arr, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f));
    const ptr = mlx.mlx_array_data_float32(f) orelse return error.NoData;
    const out = try allocator.alloc(f32, n);
    @memcpy(out, ptr[0..n]);
    return out;
}

test "mtp: a head built from the rows it computes re-scores through its 8-bit copy, never the head's own tables" {
    // A trunk format's head (the EXL3 pack's trellis) has no affine rows to gather: the coarse head and its re-score
    // copy come from the rows it computes (`buildRerankCoarseRows`), and the shortlist re-scores through the copy.
    // Proposal side only: the verify reads the head itself.
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return;
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var fx = try RerankFixture.init(s, mtp.TOP32_MIN_ROWS + 96, 256, 8, 64, 0x7E11);
    defer fx.deinit();
    var slot: ?mtp.RerankCoarse = mtp.buildRerankCoarseRows(s, fx.dense, 3) orelse return error.CoarseBuildDeclined;
    defer if (slot) |*q| q.deinit();
    const copy = slot.?.exact orelse return error.NoRescoreCopy;
    try testing.expectEqual(fx.vocab, slot.?.rows);
    // under DRAFT_VOCAB_MIN ids every draft reads the whole head
    try testing.expect(slot.?.sub == null);
    // the head's own tables become a trellis-shaped stand-in and two side vectors: nothing in them is a row
    for ([_]*mlx.mlx_array{ &fx.xfm.lm_head_w, &fx.xfm.lm_head_s, &fx.xfm.lm_head_b }) |a| {
        _ = mlx.mlx_array_free(a.*);
        a.* = mlx.mlx_array_new();
    }
    try mlx.check(mlx.mlx_zeros(&fx.xfm.lm_head_w, &[_]c_int{ @divTrunc(fx.vocab, 16), @divTrunc(fx.hidden, 16), 96 }, 3, .int16, s));
    try mlx.check(mlx.mlx_zeros(&fx.xfm.lm_head_s, &[_]c_int{fx.hidden}, 1, .float16, s));
    try mlx.check(mlx.mlx_zeros(&fx.xfm.lm_head_b, &[_]c_int{fx.vocab}, 1, .float16, s));
    var logged = false;
    const eps: f32 = 1e-3;
    var decisive: u32 = 0;
    var seed: u64 = 1;
    while (seed <= 8) : (seed += 1) {
        const x = try fx.randomX(s, seed);
        defer _ = mlx.mlx_array_free(x);
        // the copy's full readout: what the re-score reads, row for row
        var full = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(full);
        try mlx.check(mlx.mlx_quantized_matmul(&full, x, copy.w, copy.s, copy.b, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(8), "affine", s));
        const ex = try readRow(allocator, s, full, @intCast(fx.vocab));
        defer allocator.free(ex);
        var want: u32 = 0;
        for (ex, 0..) |v, i| if (v > ex[want]) {
            want = @intCast(i);
        };
        var sl = (try mtp.rerankShortlist(s, &fx.xfm, &slot, &logged, x, null)) orelse return error.RerankDeclined;
        defer sl.deinit();
        try testing.expectEqual(fx.vocab, sl.rows);
        try mlx.check(mlx.mlx_array_eval(sl.cands));
        const cands = (mlx.mlx_array_data_uint32(sl.cands) orelse return error.NoData)[0..32];
        const picked_arr = try mtp.shortlistArgmax(s, sl);
        defer _ = mlx.mlx_array_free(picked_arr);
        const picked = try readIdScalar(picked_arr);
        if (std.mem.indexOfScalar(u32, cands, want) != null) {
            decisive += 1;
            try testing.expect(ex[picked] >= ex[want] - eps);
        }
    }
    try testing.expect(decisive >= 4);
}

test "mtp: draft rerank over a vocab subset: the subset's exact argmax, the mask honoured, vocab-wide shortlists; a head without it drafts over the whole vocab" {
    // `withVocabSubset`: single-row drafts read the coarse head's leading rows. A draft is the exact argmax over those
    // ids whenever the coarse shortlist holds it, never a later id, never a suppressed one (the vocab-wide mask's
    // leading ids go on the narrower readout), and a shortlist's `rows` stays the vocab (a sampled draft's dense
    // proposal row is vocab-wide). A head without the subset drafts the whole head's argmax, whatever its id.
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return;
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    const keep: c_int = mtp.TOP32_MIN_ROWS + 64;
    var fx = try RerankFixture.init(s, 2 * mtp.TOP32_MIN_ROWS + 96, 256, 8, 64, 0x5B5E7);
    defer fx.deinit();
    var subset: ?mtp.RerankCoarse = mtp.withVocabSubset(s, mtp.buildRerankCoarse(s, &fx.xfm, 3) orelse return error.CoarseBuildDeclined, keep);
    defer if (subset) |*q| q.deinit();
    try testing.expectEqual(fx.vocab, subset.?.rows);
    try testing.expectEqual(keep, subset.?.sub_rows);
    try testing.expect(subset.?.sub != null);
    var whole: ?mtp.RerankCoarse = mtp.buildRerankCoarse(s, &fx.xfm, 3) orelse return error.CoarseBuildDeclined;
    defer if (whole) |*q| q.deinit();
    try testing.expect(whole.?.sub == null);
    // no subset at or past the vocab, nor under the shortlist's least width
    try testing.expect(mtp.withVocabSubset(s, whole.?, fx.vocab).sub == null);
    try testing.expect(mtp.withVocabSubset(s, whole.?, mtp.TOP32_MIN_ROWS - 1).sub == null);
    var logged = false;

    const mask_buf = try allocator.alloc(bool, @intCast(fx.vocab));
    defer allocator.free(mask_buf);
    const eps: f32 = 1e-3;
    var decisive: [2]u32 = .{ 0, 0 };
    var beyond: u32 = 0;
    var seed: u64 = 1;
    while (seed <= 8) : (seed += 1) {
        const x = try fx.randomX(s, seed);
        defer _ = mlx.mlx_array_free(x);
        const exact = try fx.exactLogits(s, x);
        defer _ = mlx.mlx_array_free(exact);
        const ex = try readRow(allocator, s, exact, @intCast(fx.vocab));
        defer allocator.free(ex);
        for ([_]bool{ true, false }, 0..) |active, arm| {
            const slot: *?mtp.RerankCoarse = if (active) &subset else &whole;
            const lim: usize = if (active) @intCast(keep) else @intCast(fx.vocab);
            var want: u32 = 0;
            for (ex[0..lim], 0..) |v, i| if (v > ex[want]) {
                want = @intCast(i);
            };
            if (!active and want >= @as(u32, @intCast(keep))) beyond += 1;

            // unmasked: the readout's argmax when the coarse shortlist holds it, never an id past the subset while
            // active
            var sl = (try mtp.rerankShortlist(s, &fx.xfm, slot, &logged, x, null)) orelse return error.RerankDeclined;
            defer sl.deinit();
            try testing.expectEqual(fx.vocab, sl.rows);
            try mlx.check(mlx.mlx_array_eval(sl.cands));
            const cands = (mlx.mlx_array_data_uint32(sl.cands) orelse return error.NoData)[0..32];
            var listed = false;
            for (cands) |c| {
                if (active) try testing.expect(c < @as(u32, @intCast(keep)));
                if (c == want) listed = true;
            }
            const picked_arr = (try mtp.rerankSelect(s, &fx.xfm, slot, &logged, x, null)) orelse return error.RerankDeclined;
            defer _ = mlx.mlx_array_free(picked_arr);
            const picked = try readIdScalar(picked_arr);
            if (active) try testing.expect(picked < @as(u32, @intCast(keep)));
            if (listed) {
                decisive[arm] += 1;
                try testing.expect(ex[picked] >= ex[want] - eps);
            }

            // masked: the readout's winner suppressed through the vocab-wide mask
            @memset(mask_buf, false);
            mask_buf[want] = true;
            const mask = mlx.mlx_array_new_data(mask_buf.ptr, &[_]c_int{fx.vocab}, 1, .bool_);
            defer _ = mlx.mlx_array_free(mask);
            const masked_arr = (try mtp.rerankSelect(s, &fx.xfm, slot, &logged, x, mask)) orelse return error.RerankDeclined;
            defer _ = mlx.mlx_array_free(masked_arr);
            const masked = try readIdScalar(masked_arr);
            try testing.expect(masked != want);
            if (active) try testing.expect(masked < @as(u32, @intCast(keep)));
        }
    }
    try testing.expect(decisive[0] >= 4 and decisive[1] >= 4);
    // the whole head's winner lies past the subset on some draws, so the arms really differ
    try testing.expect(beyond >= 1);
}
