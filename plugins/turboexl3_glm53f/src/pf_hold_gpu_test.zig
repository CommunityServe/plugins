//! The device check of `TURBOEXL3_GLM53F_PF_MOE_HOLD` (src/pf_reuse.zig), run by the build step `moe-combine-gpu-test`
//! (linked to the server's MLX) under TURBOEXL3_GLM53F_MOE_COMBINE_GPU=1 (its test name holds that step's filter);
//! `zig build test` compiles it and its filter leaves it out.
//!
//! The hold saves a DSA layer's first chunks their fresh arrays only if its sizes are the ones the routed chain
//! allocates. This runs the served prompt path, turboexl3's `moeClampedRows` (PF_ROWS) with the size classes (PF_REUSE's
//! `Config.pf_size_class`), over two chunks of 4019 and 4087 rows (both of the 4096-row class, slots in order as
//! `prefillParts` sorts them, unit weights) at GLM-5.3-Flash's routed geometry (hidden 4096, inter 2048, K4, mul1,
//! swiglu_limit 10) over 8 experts of seeded random codewords, once on each chain the hold sizes: the exact chain
//! (`Config.pf_int8` 0: `chunkBytes`, 26 arrays) and PF_INT8's int8 chain under both schemes (`Config.pf_int8` 1 and 2:
//! `chunkBytesI8`, 32 arrays) and under PF_INT8=3 and 4 (scheme 1 with `Config.pf_trunk_int8`: the same routed chain), each
//! with its own engine (every engine released between them), the hold installed under that configuration
//! (`holdInstall` reads the chain from it). For each:
//! 1. from an empty cache, the two chunks committed together allocate at least the hold's bytes (`fresh_without`);
//! 2. the hold made by `holdInstall` on the stream, taken by the schedule at a DSA layer's start (over `Mlx`) with the
//!    cache limit at 0, then handed back at the first routed call: the cache holds exactly the hold's bytes (the raise
//!    kept every buffer), and the two chunks, each committed as `prefillParts` commits it, allocate under 1 MiB past
//!    the hand-back (`fresh_with`: the schedule's own measure, the pass line's);
//! 3. each chunk's output with the hold has the bits it has without it.
//! 4. `TURBOEXL3_GLM53F_PF_MOE_HOLD_TOP`'s hand-back into a cache over its limit (600 MiB cached under a 32 MiB limit,
//!    as a DSA layer's attention leaves it), each count the schedule's own (its take line's), the hold's every buffer
//!    waited for before the call's first allocation: without TOP the hand-back caches none, and the first allocation
//!    trims at least the excess; with TOP it releases none and the cache keeps every one, the first allocation trims
//!    nothing, the commits take at least a chunk's worth from the cache, both outputs' buffers are held ones, and the
//!    outputs have step 1's bits.
//! Two lines a chain, int8=0, 1, 2, 3 and 4 in order (the hold's chain and its arrays as `holdInstall` made them):
//!
//!     TURBOEXL3_GLM53F_PF_MOE_HOLD_FIT int8=<v> chain=exact|int8 rows=4019,4087 class=4096 arrays=26|32 hold_mb=<n>
//!         back_mb=<n> fresh_without_kb=<n> fresh_with_kb=<n> differing=<n>/<total> PASS|FAIL
//!     TURBOEXL3_GLM53F_PF_MOE_HOLD_TOP_FIT int8=<v> over_mb=600 limit_mb=32 hold_mb=<n> without: cached=<n>
//!         released=<n> in_flight=<n> back_mb=<n> excess_mb=<n> first_alloc_trim_mb=<n> commit_cache_drop_mb=<n>
//!         outputs_from_hold=<n>/<n> with: ... differing=<n>/<total> PASS|FAIL
const std = @import("std");
const kit = @import("mlx_serve");
const mlx = kit.mlx;
const t3 = @import("turboexl3");
const pf_reuse = @import("pf_reuse.zig");
const routed_t3 = @import("routed_t3.zig");
const Array = mlx.mlx_array;

const hidden: u32 = 4096;
const inter: u32 = 2048;
/// The codewords' rate (64 halfwords a tile)
const k: u32 = 4;
/// A bank of this many experts (a chunk's array sizes do not depend on it)
const experts: u32 = 8;
/// GLM-5.3-Flash's swiglu_limit
const limit: c_int = 10;
/// The layer's rows the chunks read
const x_rows: c_int = 4096;
/// Two chunks of the 4096-row class, as `prefillParts` keeps in flight
const chunk_rows = [pf_reuse.hold_chunks]u32{ 4019, 4087 };

fn words(w: []const u16, shape: []const c_int, dtype: mlx.mlx_dtype) Array {
    return mlx.mlx_array_new_data(w.ptr, shape.ptr, @intCast(shape.len), dtype);
}

fn f16Bits(v: f32) u16 {
    return @bitCast(@as(f16, @floatCast(v)));
}

fn bf16Bits(v: f32) u16 {
    return @truncate(@as(u32, @bitCast(v)) >> 16);
}

/// A bank projection `in -> out` at rate `k`: random codewords [E, in/16, out/16, 16k] u16, suh f16 [E, in] in
/// [0.5, 1.5], svh f16 [E, out] in [0.5, 1.5] / sqrt(in) (src/draft_gpu_test.zig's).
fn proj(a: std.mem.Allocator, rnd: std.Random, in: u32, out: u32) !t3.Proj {
    const n = 16 * k;
    const tw = try a.alloc(u16, @as(usize, experts) * (in / 16) * (out / 16) * n);
    defer a.free(tw);
    rnd.bytes(std.mem.sliceAsBytes(tw));
    const su = try a.alloc(u16, @as(usize, experts) * in);
    defer a.free(su);
    for (su) |*v| v.* = f16Bits(0.5 + rnd.float(f32));
    const sv = try a.alloc(u16, @as(usize, experts) * out);
    defer a.free(sv);
    for (sv) |*v| v.* = f16Bits((0.5 + rnd.float(f32)) / @sqrt(@as(f32, @floatFromInt(in))));
    const p: t3.Proj = .{
        .trellis = words(tw, &.{ @intCast(experts), @intCast(in / 16), @intCast(out / 16), @intCast(n) }, .uint16),
        .suh = words(su, &.{ @intCast(experts), @intCast(in) }, .float16),
        .svh = words(sv, &.{ @intCast(experts), @intCast(out) }, .float16),
    };
    try evalAll(&.{ p.trellis, p.suh, p.svh });
    return p;
}

fn freeProj(p: t3.Proj) void {
    for ([_]Array{ p.trellis, p.suh, p.svh }) |x| _ = mlx.mlx_array_free(x);
}

fn evalAll(arrs: []const Array) !void {
    const vec = mlx.mlx_vector_array_new_data(arrs.ptr, arrs.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_eval(vec));
}

fn asyncEval(arrs: []const Array) !void {
    const vec = mlx.mlx_vector_array_new_data(arrs.ptr, arrs.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_async_eval(vec));
}

fn cacheNow() usize {
    var c: usize = 0;
    _ = mlx.mlx_get_cache_memory(&c);
    return c;
}

fn activeNow() usize {
    var a: usize = 0;
    _ = mlx.mlx_get_active_memory(&a);
    return a;
}

/// MLX's active and cached bytes: what a run of allocations grows when no cached buffer serves it
fn memNow() usize {
    var a: usize = 0;
    _ = mlx.mlx_get_active_memory(&a);
    return a + cacheNow();
}

/// A chunk's inputs as `prefillParts` hands them to `gemmRows`: each assignment's row of x, its slot (in order: the
/// chunk's slots sorted, `experts` runs), its weight 1
const Call = struct {
    rows: Array,
    inds: Array,
    scores: Array,
    fn deinit(c: Call) void {
        for ([_]Array{ c.rows, c.inds, c.scores }) |x| _ = mlx.mlx_array_free(x);
    }
};

fn call(a: std.mem.Allocator, n: u32) !Call {
    const r = try a.alloc(u32, n);
    defer a.free(r);
    const d = try a.alloc(u32, n);
    defer a.free(d);
    const w = try a.alloc(f32, n);
    defer a.free(w);
    for (r, d, w, 0..) |*ri, *di, *wi, i| {
        ri.* = @intCast((i * 13) % @as(usize, @intCast(x_rows)));
        di.* = @intCast(i * experts / n);
        wi.* = 1.0;
    }
    const sh = [_]c_int{@intCast(n)};
    return .{
        .rows = mlx.mlx_array_new_data(r.ptr, &sh, 1, .uint32),
        .inds = mlx.mlx_array_new_data(d.ptr, &sh, 1, .uint32),
        .scores = mlx.mlx_array_new_data(w.ptr, &sh, 1, .float32),
    };
}

/// One chunk on the served prompt path (routed_t3 `gemmRows`' call)
fn chunk(s: mlx.mlx_stream, x: Array, bank: t3.Bank, c: Call) !Array {
    return t3.moeClampedRows(s, x, c.rows, bank, c.inds, c.scores, .mul1, limit, .bfloat16);
}

/// The bf16 words of `got` that differ from `want`'s (both made contiguous), of the total
fn differing(s: mlx.mlx_stream, want: Array, got: Array) !struct { n: usize, total: usize } {
    if (mlx.mlx_array_dtype(want) != .bfloat16 or mlx.mlx_array_dtype(got) != .bfloat16) return error.PfHoldGpuDtype;
    if (!std.mem.eql(c_int, mlx.getShape(want), mlx.getShape(got))) return error.PfHoldGpuShape;
    var wc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wc);
    try mlx.check(mlx.mlx_contiguous(&wc, want, false, s));
    var gc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gc);
    try mlx.check(mlx.mlx_contiguous(&gc, got, false, s));
    try evalAll(&.{ wc, gc });
    const n = mlx.mlx_array_size(wc);
    const pw = (mlx.mlx_array_data_bfloat16(wc) orelse return error.PfHoldGpuData)[0..n];
    const pg = (mlx.mlx_array_data_bfloat16(gc) orelse return error.PfHoldGpuData)[0..n];
    var d: usize = 0;
    for (pw, pg) |x, y| d += @intFromBool(x != y);
    return .{ .n = d, .total = n };
}

test "turboexl3_glm53f moe combine gpu: TURBOEXL3_GLM53F_PF_MOE_HOLD's arrays are the routed chain's own for two 4096-row-class chunks, on the exact chain and on PF_INT8's int8 chain under both schemes, PF_INT8=3 and PF_INT8=4: handed back, the chunks take every one and allocate under 1 MiB more, with the same bits; into a cache over its limit, PF_MOE_HOLD_TOP's hand-back caches every one where PF_MOE_HOLD's caches none" {
    if (std.c.getenv("TURBOEXL3_GLM53F_MOE_COMBINE_GPU") == null) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const a = std.heap.page_allocator;
    var prng = std.Random.DefaultPrng.init(20261110);
    const rnd = prng.random();
    // the served routed prefill with PF_REUSE's size classes, each chain's engine made under its configuration (every
    // engine released with the stream synchronized, before, between the chains and after)
    const saved = t3.config.get().*;
    try mlx.check(mlx.mlx_synchronize(s));
    t3.releaseEngines();
    defer {
        _ = mlx.mlx_synchronize(s);
        t3.releaseEngines();
        t3.configure(saved) catch {};
    }
    const bank: t3.Bank = .{ .gate = try proj(a, rnd, hidden, inter), .up = try proj(a, rnd, hidden, inter), .down = try proj(a, rnd, inter, hidden) };
    defer for ([_]t3.Proj{ bank.gate, bank.up, bank.down }) |p| freeProj(p);
    const xs = try a.alloc(u16, @as(usize, @intCast(x_rows)) * hidden);
    defer a.free(xs);
    for (xs) |*v| v.* = bf16Bits(rnd.floatNorm(f32));
    const x = words(xs, &.{ x_rows, @intCast(hidden) }, .bfloat16);
    defer _ = mlx.mlx_array_free(x);
    var calls: [chunk_rows.len]Call = undefined;
    var made: usize = 0;
    defer for (calls[0..made]) |c| c.deinit();
    for (chunk_rows, &calls) |n, *c| {
        c.* = try call(a, n);
        made += 1;
    }
    try evalAll(&.{ x, calls[0].rows, calls[0].inds, calls[0].scores, calls[1].rows, calls[1].inds, calls[1].scores });
    try std.testing.expectEqual(@as(u32, 4096), t3.prefill.classRows(chunk_rows[0]));
    try std.testing.expectEqual(@as(u32, 4096), t3.prefill.classRows(chunk_rows[1]));
    var passed: usize = 0;
    for ([_]u8{ 0, 1, 2, 3, 4 }) |v| {
        var cfg = saved;
        cfg.pf_size_class = true;
        // 3 and 4: TURBOEXL3_GLM53F_PF_INT8=3's and =4's configuration (routed_t3's `PfInt8.apply`): scheme 1's routed
        // chain with the trunk's int8 on
        cfg.pf_int8 = if (v >= 3) 1 else v;
        cfg.pf_trunk_int8 = v >= 3;
        try mlx.check(mlx.mlx_synchronize(s));
        t3.releaseEngines();
        try t3.configure(cfg);
        if (try fitChain(s, x, bank, &calls, v)) passed += 1;
    }
    try std.testing.expectEqual(@as(usize, 5), passed);
}

/// Steps 1-4 on the chain turboexl3's configuration names (`Config.pf_int8` `v`, configured by the caller with the size
/// classes): its two lines; true when both pass.
fn fitChain(s: mlx.mlx_stream, x: Array, bank: t3.Bank, calls: *const [chunk_rows.len]Call, v: u8) !bool {
    // the engine, its tables and kernels made, both chunks run once
    {
        const y0 = try chunk(s, x, bank, calls[0]);
        defer _ = mlx.mlx_array_free(y0);
        const y1 = try chunk(s, x, bank, calls[1]);
        defer _ = mlx.mlx_array_free(y1);
        try evalAll(&.{ y0, y1 });
    }
    try mlx.check(mlx.mlx_synchronize(s));
    // 1. without the hold, from an empty cache: both chunks committed together, so neither takes the other's arrays
    _ = mlx.mlx_clear_cache();
    const m0 = memNow();
    const w0 = try chunk(s, x, bank, calls[0]);
    defer _ = mlx.mlx_array_free(w0);
    const w1 = try chunk(s, x, bank, calls[1]);
    defer _ = mlx.mlx_array_free(w1);
    try evalAll(&.{ w0, w1 });
    try mlx.check(mlx.mlx_synchronize(s));
    const fresh_without = memNow() -| m0;
    // 2. the hold: made on the stream on the configured chain, taken at a DSA layer's start from an empty cache whose
    // limit is 0 (a buffer freed over the limit is released), handed back at the first routed call
    const total = try pf_reuse.holdInstall(s, hidden, inter, routed_t3.prompt_rows);
    defer pf_reuse.holdReset();
    const chain = pf_reuse.heldChain();
    const arrays = pf_reuse.heldArrays();
    const chain_ok = chain == pf_reuse.Chain.of(v) and arrays == chain.arrays() * pf_reuse.hold_chunks;
    _ = mlx.mlx_clear_cache();
    var lim0: usize = 0;
    try mlx.check(mlx.mlx_set_cache_limit(&lim0, 0));
    defer {
        var l: usize = 0;
        _ = mlx.mlx_set_cache_limit(&l, lim0);
    }
    var sc: pf_reuse.Schedule(pf_reuse.Mlx) = .{ .b = .{}, .hold = total };
    sc.begin(true, 16384, 1);
    sc.layer(3, false, true);
    const taken = sc.holding and sc.stats.mh_taken == 1;
    try mlx.check(mlx.mlx_synchronize(s));
    sc.routed();
    // every buffer back once its launch's command buffer lets go of it (served, the layer's attention runs between)
    const ts = std.c.timespec{ .sec = 0, .nsec = 5 * std.time.ns_per_ms };
    var spins: usize = 0;
    while (cacheNow() < total and spins < 200) : (spins += 1) _ = std.c.nanosleep(&ts, null);
    const back = cacheNow();
    // the chunks as `prefillParts` commits them: one at a time, the measure after the second
    const g0 = try chunk(s, x, bank, calls[0]);
    defer _ = mlx.mlx_array_free(g0);
    try asyncEval(&.{g0});
    sc.committed(1, 0);
    const g1 = try chunk(s, x, bank, calls[1]);
    defer _ = mlx.mlx_array_free(g1);
    try asyncEval(&.{g1});
    sc.committed(2, 0);
    try mlx.check(mlx.mlx_synchronize(s));
    const fresh_with = sc.stats.mh_fresh_max;
    const measured = sc.stats.mh_measured;
    sc.end();
    // 3. the same bits
    const d0 = try differing(s, w0, g0);
    const d1 = try differing(s, w1, g1);
    const ok = chain_ok and taken and measured == 1 and back == total and fresh_without >= total and fresh_with < (1 << 20) and d0.n == 0 and d1.n == 0 and d0.total == chunk_rows[0] * hidden and d1.total == chunk_rows[1] * hidden;
    std.debug.print("\nTURBOEXL3_GLM53F_PF_MOE_HOLD_FIT int8={d} chain={s} rows={d},{d} class=4096 arrays={d} hold_mb={d} back_mb={d} fresh_without_kb={d} fresh_with_kb={d} differing={d}/{d} {s}\n", .{ v, @tagName(chain), chunk_rows[0], chunk_rows[1], arrays, total >> 20, back >> 20, fresh_without >> 10, fresh_with >> 10, d0.n + d1.n, d0.total + d1.total, if (ok) "PASS" else "FAIL" });
    // 4. PF_MOE_HOLD_TOP: the hand-back into a cache over its limit, without TOP and with it
    const arm_no = try overLimit(s, x, bank, calls, total, false, .{ w0, w1 });
    const arm_top = try overLimit(s, x, bank, calls, total, true, .{ w0, w1 });
    const sn = arm_no.st;
    const stp = arm_top.st;
    const one_mib: usize = 1 << 20;
    const no_ok = sn.mh_used == 1 and sn.mh_hb_bufs == arrays and sn.mh_hb_cached == 0 and arm_no.back < one_mib and sn.mh_over_limit == 1 and sn.mh_excess > 0 and sn.mh_first_trims == 1 and sn.mh_first_drop >= sn.mh_excess and arm_no.differing == 0;
    const top_ok = stp.mh_used == 1 and stp.mh_hb_bufs == arrays and stp.mh_hb_released == 0 and stp.mh_hb_cached + stp.mh_hb_deferred == arrays and arm_top.back >= total and arm_top.back < total + one_mib and stp.mh_over_limit == 0 and stp.mh_first_trims == 0 and stp.mh_commits == 2 and stp.mh_commit_drop >= @as(i64, @intCast(total / 2)) and stp.mh_y_held == 2 and stp.mh_y_seen == 2 and arm_top.differing == 0 and arm_top.total == d0.total + d1.total;
    std.debug.print("\nTURBOEXL3_GLM53F_PF_MOE_HOLD_TOP_FIT int8={d} over_mb={d} limit_mb={d} hold_mb={d} without: cached={d} released={d} in_flight={d} back_mb={d} excess_mb={d} first_alloc_trim_mb={d} commit_cache_drop_mb={d} outputs_from_hold={d}/{d} with: cached={d} released={d} in_flight={d} back_mb={d} excess_mb={d} first_alloc_trim_mb={d} commit_cache_drop_mb={d} outputs_from_hold={d}/{d} differing={d}/{d} {s}\n", .{ v, over_bytes >> 20, over_limit_bytes >> 20, total >> 20, sn.mh_hb_cached, sn.mh_hb_released, sn.mh_hb_deferred, arm_no.back >> 20, sn.mh_excess >> 20, sn.mh_first_drop >> 20, @divTrunc(sn.mh_commit_drop, 1 << 20), sn.mh_y_held, sn.mh_y_seen, stp.mh_hb_cached, stp.mh_hb_released, stp.mh_hb_deferred, arm_top.back >> 20, stp.mh_excess >> 20, stp.mh_first_drop >> 20, @divTrunc(stp.mh_commit_drop, 1 << 20), stp.mh_y_held, stp.mh_y_seen, arm_no.differing + arm_top.differing, arm_no.total + arm_top.total, if (no_ok and top_ok) "PASS" else "FAIL" });
    return ok and no_ok and top_ok;
}

/// Step 4's cache: 600 MiB of buffers over a 32 MiB limit at the hand-back
const over_bytes: usize = 600 << 20;
const over_limit_bytes: usize = 32 << 20;

/// A committed chunk output's buffer address (routed_t3's `bufferAddr`)
fn addrOf(y: Array) usize {
    return if (mlx.mlx_array_data_bfloat16(y)) |p| @intFromPtr(p) else 0;
}

/// Step 4 for one arm: the schedule's counts, what the cache gained once every held buffer was freed, and the outputs'
/// words that differ from step 1's
const Over = struct { st: pf_reuse.Stats, back: usize, differing: usize, total: usize };

/// Step 4: from an empty cache under a 4 GiB limit, a first routed call, then a DSA layer's hold; 600 MiB freed into the
/// cache (an array made from host bytes: no launch, so no command buffer frees anything later) and its limit set to
/// 32 MiB (MLX trims at its next allocation); the hold handed back (`top`: PF_MOE_HOLD_TOP), every held buffer waited
/// for (a launch's command buffer lets go of the rest at its completion); the call's first allocation (a chunk's 16 KiB
/// row table) and the two chunks as `prefillParts` commits them, at the take's points
fn overLimit(s: mlx.mlx_stream, x: Array, bank: t3.Bank, calls: *const [chunk_rows.len]Call, total: usize, top: bool, want: [2]Array) !Over {
    try mlx.check(mlx.mlx_synchronize(s));
    _ = mlx.mlx_clear_cache();
    var lim0: usize = 0;
    try mlx.check(mlx.mlx_set_cache_limit(&lim0, 4 << 30));
    defer {
        var l: usize = 0;
        _ = mlx.mlx_set_cache_limit(&l, lim0);
    }
    var sc: pf_reuse.Schedule(pf_reuse.Mlx) = .{ .b = .{}, .hold = total, .top = top };
    sc.begin(true, 16384, 1);
    defer sc.end();
    sc.routed();
    sc.layer(7, false, true);
    if (!sc.holding) return error.PfHoldGpuNoHold;
    try mlx.check(mlx.mlx_synchronize(s));
    {
        const host = try std.heap.page_allocator.alloc(u8, over_bytes);
        defer std.heap.page_allocator.free(host);
        @memset(host, 0);
        const f = mlx.mlx_array_new_data(host.ptr, &[_]c_int{@intCast(over_bytes)}, 1, .uint8);
        if (f.ctx == null) return error.PfHoldGpuFiller;
        _ = mlx.mlx_array_free(f);
    }
    var l32: usize = 0;
    try mlx.check(mlx.mlx_set_cache_limit(&l32, over_limit_bytes));
    const a_in = activeNow();
    const c_in = cacheNow();
    sc.routed();
    const ts = std.c.timespec{ .sec = 0, .nsec = 5 * std.time.ns_per_ms };
    var spins: usize = 0;
    while (activeNow() + total > a_in and spins < 200) : (spins += 1) _ = std.c.nanosleep(&ts, null);
    const back = cacheNow() -| c_in;
    var tbl_rows: [4096]u32 = undefined;
    for (&tbl_rows, 0..) |*v, i| v.* = @intCast(i);
    sc.at(.first_alloc_before);
    const tbl = mlx.mlx_array_new_data(&tbl_rows, &[_]c_int{tbl_rows.len}, 1, .uint32);
    defer _ = mlx.mlx_array_free(tbl);
    sc.at(.first_alloc_after);
    const g0 = try chunk(s, x, bank, calls[0]);
    defer _ = mlx.mlx_array_free(g0);
    sc.at(.commit_before);
    try asyncEval(&.{g0});
    sc.committed(1, addrOf(g0));
    const g1 = try chunk(s, x, bank, calls[1]);
    defer _ = mlx.mlx_array_free(g1);
    sc.at(.commit_before);
    try asyncEval(&.{g1});
    sc.committed(2, addrOf(g1));
    try mlx.check(mlx.mlx_synchronize(s));
    const st = sc.stats;
    const d0 = try differing(s, want[0], g0);
    const d1 = try differing(s, want[1], g1);
    return .{ .st = st, .back = back, .differing = d0.n + d1.n, .total = d0.total + d1.total };
}
