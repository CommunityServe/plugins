//! turboexl3_ling's own (not a copy): the exact verify's MLA attention rows (mla.zig `rowAttn`: each query row r of a
//! verify of S rows attends the first N_r = T - S + r + 1 keys in its own one-row MLX SDPA call) in one launch per kind
//! of row, each row's words those of its own one-row call. MLX 0.32.5 (scaled_dot_product_attention.cpp) runs a
//! one-row call on its vector kernels: the one-pass `sdpa_vector` below `planRow`'s threshold, else the two-pass
//! `sdpa_vector_2pass_1` + `_2` at a `blocks` count chosen from the keys, the device and the simdgroups (gqa x q_len);
//! a causal call of S rows would choose its blocks at gqa x S and mask by position, so its rows are not their one-row
//! calls' words. Here the rows are grouped by their one-row plan (`planRow` at q_len 1: the pass and the blocks, consecutive
//! rows with one plan a group), and each group runs MLX's kernel statements (kernels/mla/ling_verify_mla_*.metal,
//! sdpa_vector.h's, MIT) for each of its rows over that row's own N: the same key striding, the same TK-key main loop and
//! tail, the same combine. Groups are joined on the row axis. No row falls back: a verify whose rows straddle a
//! threshold runs two groups (two kinds of launch), each row still its one-row call's plan.
//! Served: one sequence (B 1), the expanded MLA cache's heads (as many key heads as query heads: gqa 1), the head dims
//! (192, 128) (Ling) or (128, 128), bf16 or f16, 1 to 16 rows; null elsewhere (the caller runs rowAttn), checked before
//! any launch. The keys and values are read in place at their strides (the cache's views), as MLX's own call reads them.
//! The rope-once cache (src/ling_mla_rope1.zig, every trunk MLA layer's): `attendRope1`, the same rows, plans and
//! groups on the rope-once texts (kernels/mla/ling_verify_mla_1p_r1.metal, _2p1_r1; pass 2 the base's), a key element
//! read from the head's nope keys below KN and from the token's one roped key past it: the expanded key's words.
const std = @import("std");
const kernelSource = @import("../../mlx/kernels.zig").source;
const mlx = @import("mlx_serve").mlx;

/// A one-row call's kernel: the one-pass vector kernel, or the two-pass kernels at `blocks` partials a row
pub const Kind = enum { one_pass, two_pass };
pub const Plan = struct {
    kind: Kind,
    blocks: u32 = 0,
    pub fn eql(a: Plan, b: Plan) bool {
        return a.kind == b.kind and a.blocks == b.blocks;
    }
};

/// The two-pass kernel's keys per main-loop step (sdpa_vector_2pass_1's TK at these head dims: no _tk1 variant runs at
/// one simdgroup a threadgroup)
pub const tk: u32 = 4;
/// The most rows a call takes (the vector kernels' q_len cap of 8, twice: a verify of up to 16 rows)
pub const max_rows: c_int = 16;

/// MLX 0.32.5's plan for one query row over `n_keys` keys at `gqa` query heads a key head (eval_gpu's vector mode at
/// q_len 1, where the causal flag is off, and sdpa_vector_2pass's blocks at n_simds = gqa): the two-pass kernels on a
/// large device ('d' or 's', the architecture string's last letter) from 1024 keys or with gqa from 4096, else the
/// one-pass kernel (MLX's `sdpa_vector`, which fits its 1024 threads at these head dims). `blocks_env`:
/// MLX_SDPA_BLOCKS's override (0 none), rounded up to 32 as MLX rounds it.
pub fn planRow(devc: u8, gqa: u32, n_keys: u32, blocks_env: u32) Plan {
    const two = ((devc == 'd' or devc == 's') and n_keys >= 1024) or (gqa > 1 and n_keys >= 4096);
    if (!two) return .{ .kind = .one_pass };
    const n_simds = gqa;
    var blocks: u32 = undefined;
    if (devc == 's') {
        blocks = 64;
        if (n_keys > 1024 and n_simds > 4) {
            if (n_keys <= 8192) {
                blocks = 128;
            } else if (n_keys <= 32768) {
                blocks = 256;
            } else if (n_keys <= 65536) {
                blocks = 512;
            } else {
                blocks = 1024;
            }
        }
    } else if (devc == 'd') {
        blocks = 128;
        if (n_simds <= 2 and n_keys > 8192) {
            blocks = 256;
        } else if (n_simds >= 6) {
            if (n_keys >= 16384 and n_keys < 65536) {
                blocks = 512;
            } else if (n_keys >= 65536) {
                blocks = 1024;
            }
        }
    } else {
        blocks = if (n_simds >= 4) 64 else 32;
    }
    if (blocks_env > 0) blocks = ((blocks_env + 31) / 32) * 32;
    return .{ .kind = .two_pass, .blocks = blocks };
}

var devc_cache: ?u8 = null;

/// The device's architecture letter, as MLX reads it for these choices (`get_architecture().back()`: mlx-c's device
/// info "architecture", e.g. applegpu_g17s), read once; 0 where it cannot be read
pub fn deviceClass() u8 {
    if (devc_cache) |c| return c;
    var c: u8 = 0;
    var dev = mlx.mlx_device{ .ctx = null };
    if (mlx.mlx_get_default_device(&dev) == 0) {
        var info = mlx.mlx_device_info_new();
        defer _ = mlx.mlx_device_info_free(info);
        if (mlx.mlx_device_info_get(&info, dev) == 0) {
            var cstr: [*:0]const u8 = undefined;
            if (mlx.mlx_device_info_get_string(&cstr, info, "architecture") == 0) {
                const arch = std.mem.span(cstr);
                if (arch.len > 0) c = arch[arch.len - 1];
            }
        }
    }
    devc_cache = c;
    return c;
}

/// MLX_SDPA_BLOCKS as MLX's config reads it (a positive integer, else 0)
pub fn blocksEnv() u32 {
    const v = std.c.getenv("MLX_SDPA_BLOCKS") orelse return 0;
    return std.fmt.parseInt(u32, std.mem.span(v), 10) catch 0;
}

/// The head dims served: Ling's (192, 128), and (128, 128)
pub fn admits(qk: c_int, v: c_int) bool {
    return (qk == 192 and v == 128) or (qk == 128 and v == 128);
}

fn free(a: mlx.mlx_array) void {
    if (a.ctx != null) _ = mlx.mlx_array_free(a);
}

fn newKernel(comptime name: [:0]const u8, ins: anytype, outs: anytype, src: [:0]const u8) !mlx.mlx_fast_metal_kernel {
    const iv = mlx.mlx_vector_string_new_data(ins, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    // the operands at their own strides: the cache's views read in place, as MLX's own call reads them
    const k = mlx.mlx_fast_metal_kernel_new(name.ptr, iv, ov, src.ptr, "", false, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    return k;
}

fn apply(k: mlx.mlx_fast_metal_kernel, cfg: mlx.mlx_fast_metal_kernel_config, s: mlx.mlx_stream, inputs: []const mlx.mlx_array, outs: []mlx.mlx_array) !void {
    const iv = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, k, iv, cfg, s));
    if (mlx.mlx_vector_array_size(ov) != outs.len) return error.MetalKernelBadOutputCount;
    for (outs) |*o| o.* = .{ .ctx = null };
    errdefer for (outs) |o| free(o);
    for (outs, 0..) |*o, i| {
        o.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(o, ov, i));
    }
}

const in_rows = [_][*:0]const u8{ "qin", "kin", "vin", "p", "sc" };
/// The rope-once texts' operands (src/ling_mla_rope1.zig): kin each head's nope keys, krin the token's one roped key
const in_rows_r1 = [_][*:0]const u8{ "qin", "kin", "vin", "krin", "p", "sc" };
const in_pass2 = [_][*:0]const u8{ "partials", "sums", "maxs", "p" };
const out_rows = [_][*:0]const u8{"oout"};
const out_pass1 = [_][*:0]const u8{ "partials", "sums", "maxs" };

var k1p: ?mlx.mlx_fast_metal_kernel = null;
var k2p1: ?mlx.mlx_fast_metal_kernel = null;
var k2p2: ?mlx.mlx_fast_metal_kernel = null;
var k1p_r1: ?mlx.mlx_fast_metal_kernel = null;
var k2p1_r1: ?mlx.mlx_fast_metal_kernel = null;

fn kernel1p() !mlx.mlx_fast_metal_kernel {
    if (k1p) |k| return k;
    const k = try newKernel("ling_verify_mla_1p", &in_rows, &out_rows, kernelSource("kernels/mla/ling_verify_mla_1p.metal"));
    k1p = k;
    return k;
}

fn kernel2p1() !mlx.mlx_fast_metal_kernel {
    if (k2p1) |k| return k;
    const k = try newKernel("ling_verify_mla_2p1", &in_rows, &out_pass1, kernelSource("kernels/mla/ling_verify_mla_2p1.metal"));
    k2p1 = k;
    return k;
}

fn kernel2p2() !mlx.mlx_fast_metal_kernel {
    if (k2p2) |k| return k;
    const k = try newKernel("ling_verify_mla_2p2", &in_pass2, &out_rows, kernelSource("kernels/mla/ling_verify_mla_2p2.metal"));
    k2p2 = k;
    return k;
}

/// The rope-once cache's one-pass and pass-1 texts (the keys from kin below KN, from krin past it); pass 2 is the base's
fn kernel1pR1() !mlx.mlx_fast_metal_kernel {
    if (k1p_r1) |k| return k;
    const k = try newKernel("ling_verify_mla_1p_r1", &in_rows_r1, &out_rows, kernelSource("kernels/mla/ling_verify_mla_1p_r1.metal"));
    k1p_r1 = k;
    return k;
}

fn kernel2p1R1() !mlx.mlx_fast_metal_kernel {
    if (k2p1_r1) |k| return k;
    const k = try newKernel("ling_verify_mla_2p1_r1", &in_rows_r1, &out_pass1, kernelSource("kernels/mla/ling_verify_mla_2p1_r1.metal"));
    k2p1_r1 = k;
    return k;
}

fn intArray(vals: []const i32) mlx.mlx_array {
    return mlx.mlx_array_new_data(vals.ptr, &[_]c_int{@intCast(vals.len)}, 1, .int32);
}

/// One group's rows [r0, r0 + rows) of `plan`, n0 the keys of its first row: [1, H, rows, V] in q's dtype. `kr`: the
/// rope-once cache's roped keys [1, 1, T, qk - kn] (k then each head's nope keys [1, H, T, kn]), on the rope-once texts;
/// null: k the expanded keys, on the base texts.
fn group(s: mlx.mlx_stream, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, sc: mlx.mlx_array, h: c_int, qk: c_int, vd: c_int, r0: c_int, rows: c_int, n0: c_int, plan: Plan, kr: ?mlx.mlx_array, kn: c_int) !mlx.mlx_array {
    const dt = mlx.mlx_array_dtype(q);
    const out_shape = [_]c_int{ 1, h, rows, vd };
    const p = intArray(&.{ r0, n0, @intCast(plan.blocks) });
    defer free(p);
    switch (plan.kind) {
        .one_pass => {
            const kern = if (kr != null) try kernel1pR1() else try kernel1p();
            const cfg = mlx.mlx_fast_metal_kernel_config_new();
            defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &out_shape, out_shape.len, dt));
            // MLX's grid: threadgroups (B H, q_len) of 1024 threads, a launch row a query row here
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, h * 1024, rows, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 1024, 1, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", qk));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "V", vd));
            var o: [1]mlx.mlx_array = undefined;
            if (kr) |rope| {
                try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "KN", kn));
                try apply(kern, cfg, s, &.{ q, k, v, rope, p, sc }, &o);
            } else try apply(kern, cfg, s, &.{ q, k, v, p, sc }, &o);
            return o[0];
        },
        .two_pass => {
            const blocks: c_int = @intCast(plan.blocks);
            const k1 = if (kr != null) try kernel2p1R1() else try kernel2p1();
            const cfg1 = mlx.mlx_fast_metal_kernel_config_new();
            defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg1);
            const part_shape = [_]c_int{ 1, h, rows, blocks, vd };
            const stat_shape = [_]c_int{ 1, h, rows, blocks };
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg1, &part_shape, part_shape.len, dt));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg1, &stat_shape, stat_shape.len, .float32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg1, &stat_shape, stat_shape.len, .float32));
            // MLX's grid: threadgroups (key heads, B, blocks) of (32, gqa, q_len), a simdgroup a query row
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg1, h * 32, 1, blocks * rows));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg1, 32, 1, rows));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg1, "T", dt));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg1, "D", qk));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg1, "V", vd));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg1, "BLOCKS", blocks));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg1, "TK", @intCast(tk)));
            var po: [3]mlx.mlx_array = undefined;
            if (kr) |rope| {
                try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg1, "KN", kn));
                try apply(k1, cfg1, s, &.{ q, k, v, rope, p, sc }, &po);
            } else try apply(k1, cfg1, s, &.{ q, k, v, p, sc }, &po);
            defer for (po) |a| free(a);
            const k2 = try kernel2p2();
            const cfg2 = mlx.mlx_fast_metal_kernel_config_new();
            defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg2);
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg2, &out_shape, out_shape.len, dt));
            // MLX's grid: threadgroups (B H, q_len) of 1024 threads
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg2, h * 1024, rows, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg2, 1024, 1, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg2, "T", dt));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg2, "V", vd));
            var o: [1]mlx.mlx_array = undefined;
            try apply(k2, cfg2, s, &.{ po[0], po[1], po[2], p }, &o);
            return o[0];
        },
    }
}

/// What `attend` ran: the row groups (1 or 2 kinds of launch) and whether any is two-pass
pub const Ran = struct { groups: u32, two_pass: bool };
pub var last: Ran = .{ .groups = 0, .two_pass = false };

/// `out` (the caller's handle) = rowAttn's result for queries [1, H, S, D] over keys [1, H, T, D] and values
/// [1, H, T, V] (the query rows the last S of the T): row r over the first T - S + r + 1 keys, each row's words its own
/// one-row MLX SDPA call's. False where it does not serve (nothing launched; the caller runs rowAttn).
pub fn attend(out: *mlx.mlx_array, queries: mlx.mlx_array, keys: mlx.mlx_array, values: mlx.mlx_array, scale: f32, seq_len: c_int, s: mlx.mlx_stream) !bool {
    const qs = mlx.getShape(queries);
    const ks = mlx.getShape(keys);
    const vs = mlx.getShape(values);
    if (qs.len != 4 or ks.len != 4 or vs.len != 4) return false;
    if (qs[0] != 1 or ks[0] != 1 or vs[0] != 1) return false;
    if (qs[1] != ks[1] or vs[1] != ks[1] or vs[2] != ks[2] or qs[3] != ks[3]) return false;
    if (!admits(qs[3], vs[3])) return false;
    const dt = mlx.mlx_array_dtype(queries);
    if ((dt != .bfloat16 and dt != .float16) or mlx.mlx_array_dtype(keys) != dt or mlx.mlx_array_dtype(values) != dt) return false;
    if (seq_len < 1 or seq_len > max_rows or qs[2] != seq_len or ks[2] < seq_len) return false;
    if (!mlx.streamIsGpu(s)) return false;
    return run(out, queries, keys, values, null, 0, scale, seq_len, s);
}

/// `attend` over the rope-once cache (src/ling_mla_rope1.zig): `nope_keys` [1, H, T, KN] each head's nope keys and
/// `kr` [1, 1, T, D - KN] the token's one roped key (views of the cache, read in place), on the rope-once texts: each
/// row's words `attend`'s over the expanded keys [1, H, T, D] (each head's nope keys ++ the roped key), so its one-row
/// MLX SDPA call's. The rows, plans and groups are `attend`'s. False where it does not serve (nothing launched).
pub fn attendRope1(out: *mlx.mlx_array, queries: mlx.mlx_array, nope_keys: mlx.mlx_array, values: mlx.mlx_array, kr: mlx.mlx_array, scale: f32, seq_len: c_int, s: mlx.mlx_stream) !bool {
    const qs = mlx.getShape(queries);
    const ks = mlx.getShape(nope_keys);
    const rs = mlx.getShape(kr);
    const vs = mlx.getShape(values);
    if (qs.len != 4 or ks.len != 4 or rs.len != 4 or vs.len != 4) return false;
    if (qs[0] != 1 or ks[0] != 1 or rs[0] != 1 or vs[0] != 1) return false;
    if (qs[1] != ks[1] or vs[1] != ks[1] or rs[1] != 1 or vs[2] != ks[2] or rs[2] != ks[2] or qs[3] != ks[3] + rs[3]) return false;
    if (ks[3] < 1 or rs[3] < 1 or !admits(qs[3], vs[3])) return false;
    const dt = mlx.mlx_array_dtype(queries);
    if ((dt != .bfloat16 and dt != .float16) or mlx.mlx_array_dtype(nope_keys) != dt or mlx.mlx_array_dtype(kr) != dt or mlx.mlx_array_dtype(values) != dt) return false;
    if (seq_len < 1 or seq_len > max_rows or qs[2] != seq_len or ks[2] < seq_len) return false;
    if (!mlx.streamIsGpu(s)) return false;
    return run(out, queries, nope_keys, values, kr, ks[3], scale, seq_len, s);
}

/// The rows' plans, groups and launches for `attend` and `attendRope1`, their checks passed (`kr` null: the expanded
/// keys on the base texts)
fn run(out: *mlx.mlx_array, queries: mlx.mlx_array, keys: mlx.mlx_array, values: mlx.mlx_array, kr: ?mlx.mlx_array, kn: c_int, scale: f32, seq_len: c_int, s: mlx.mlx_stream) !bool {
    const qs = mlx.getShape(queries);
    const vs = mlx.getShape(values);
    const h = qs[1];
    const t = mlx.getShape(keys)[2];
    const devc = deviceClass();
    const benv = blocksEnv();
    const n_rows: usize = @intCast(seq_len);
    var plans: [max_rows]Plan = undefined;
    for (plans[0..n_rows], 0..) |*pl, r| pl.* = planRow(devc, 1, @intCast(t - seq_len + @as(c_int, @intCast(r)) + 1), benv);
    const sv = [_]f32{scale};
    const sc = mlx.mlx_array_new_data(&sv, &[_]c_int{1}, 1, .float32);
    defer free(sc);
    var parts: [max_rows]mlx.mlx_array = undefined;
    var n_parts: usize = 0;
    defer for (parts[0..n_parts]) |a| free(a);
    var two = false;
    var r0: usize = 0;
    while (r0 < n_rows) {
        var r1 = r0 + 1;
        while (r1 < n_rows and plans[r1].eql(plans[r0])) r1 += 1;
        const n0: c_int = t - seq_len + @as(c_int, @intCast(r0)) + 1;
        parts[n_parts] = try group(s, queries, keys, values, sc, h, qs[3], vs[3], @intCast(r0), @intCast(r1 - r0), n0, plans[r0], kr, kn);
        n_parts += 1;
        two = two or plans[r0].kind == .two_pass;
        r0 = r1;
    }
    if (n_parts == 1) {
        try mlx.check(mlx.mlx_array_set(out, parts[0]));
    } else {
        const vec = mlx.mlx_vector_array_new_data(&parts, n_parts);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(out, vec, 2, s));
    }
    last = .{ .groups = @intCast(n_parts), .two_pass = two };
    return true;
}

test "turboexl3_ling verify rows: MLX 0.32.5's one-row plan (pass, blocks) on each device class, and its override" {
    const t = std.testing;
    // a large device ('s', 'd'): one-pass below 1024 keys, two-pass from 1024
    try t.expect(planRow('s', 1, 1023, 0).eql(.{ .kind = .one_pass }));
    try t.expect(planRow('s', 1, 1024, 0).eql(.{ .kind = .two_pass, .blocks = 64 }));
    try t.expect(planRow('s', 1, 100000, 0).eql(.{ .kind = .two_pass, .blocks = 64 }));
    try t.expect(planRow('d', 1, 8192, 0).eql(.{ .kind = .two_pass, .blocks = 128 }));
    try t.expect(planRow('d', 1, 8193, 0).eql(.{ .kind = .two_pass, .blocks = 256 }));
    // another device: one-pass at gqa 1 whatever the keys
    try t.expect(planRow('g', 1, 100000, 0).eql(.{ .kind = .one_pass }));
    try t.expect(planRow('g', 8, 4096, 0).eql(.{ .kind = .two_pass, .blocks = 64 }));
    try t.expect(planRow('s', 8, 2048, 0).eql(.{ .kind = .two_pass, .blocks = 128 }));
    // MLX_SDPA_BLOCKS, rounded up to 32
    try t.expect(planRow('s', 1, 4096, 40).eql(.{ .kind = .two_pass, .blocks = 64 }));
    try t.expect(planRow('s', 1, 100, 40).eql(.{ .kind = .one_pass }));
    try t.expect(admits(192, 128) and admits(128, 128) and !admits(192, 192) and !admits(64, 64));
}
