//! DeepSeek-V4.1's wide-span attention (`attnFused` of the vendored slice) on the M5 GPU's matrix units: the same
//! one-pass kernel and tiling (16 heads a threadgroup, 8 simdgroups of D / 8 channels, tiles of 16 keys, online f32
//! softmax seeded by the sink, P split into a bf16 high and low part), with both products through Metal 4's tensor ops
//! (MetalPerformancePrimitives `matmul2d`, as MLX's `steel/attn/nax.h`). `dsv41_model.zig`'s `fusedHeads` calls it
//! first (patches/attn-nax.patch), for picked rows as values (`attnFusedNax`) or read packed from the compressed store
//! (`attnFusedNaxPacked`), and keeps `attnFused` / `attnFusedPacked` where `fits` is false.
const std = @import("std");
const mlx = @import("sushi/mlx.zig");
const sc_mod = @import("sushi/dsv41/dsv41_scope.zig");
const kernels = @import("sushi/dsv41/dsv41_kernels.zig");
const qat_mod = @import("sushi/dsv41/dsv41_qat.zig");

const A = sc_mod.A;
const Scope = sc_mod.Scope;
const dim = sc_mod.dim;
const ci = sc_mod.ci;

/// One simdgroup's 16 x N x KK multiply-accumulate on register fragments (one of N, KK must be 32: both inputs are
/// cooperative tensors). A 16 x 16 fragment's lane holds 8 elements, rows fm and fm + 8, columns fn .. fn + 3
/// (`dsv41_nax_r` / `dsv41_nax_c` of element i); operands are in their logical layout (A 16 x KK, B KK x N), and a
/// dimension of 32 is two consecutive 16-wide fragments.
const HEADER =
    \\#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
    \\inline short dsv41_nax_r(ushort lane, int i) { const short qid = short(lane) >> 2; return ((qid & 4) | ((short(lane) >> 1) & 3)) + (i >> 2) * 8; }
    \\inline short dsv41_nax_c(ushort lane, int i) { const short qid = short(lane) >> 2; return ((qid & 2) | (short(lane) & 1)) * 4 + (i & 3); }
    \\template <int N, int KK, typename AT, typename BT>
    \\inline void dsv41_nax_mma(thread float* c, thread const AT* a, thread const BT* b) {
    \\  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, N, KK, false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    \\  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    \\  auto ca = op.template get_left_input_cooperative_tensor<AT, BT, float>();
    \\  auto cb = op.template get_right_input_cooperative_tensor<AT, BT, float>();
    \\  auto cc = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ca)>, metal::remove_addrspace_t<decltype(cb)>, float>();
    \\  #pragma clang loop unroll(full)
    \\  for (int i = 0; i < KK / 2; ++i) ca[i] = a[i];
    \\  #pragma clang loop unroll(full)
    \\  for (int i = 0; i < KK * N / 32; ++i) cb[i] = b[i];
    \\  #pragma clang loop unroll(full)
    \\  for (int i = 0; i < N / 2; ++i) cc[i] = c[i];
    \\  op.run(ca, cb, cc);
    \\  #pragma clang loop unroll(full)
    \\  for (int i = 0; i < N / 2; ++i) c[i] = cc[i];
    \\}
;

/// `attnFused`'s kernel with the scores as 16 heads x 16 keys over 32-channel steps and P x V as 16 heads x 32
/// channels over the 16 keys. The keys of row l (its band of window rows from `wp`, then its K picked compressed rows,
/// `topk` < 0 or >= nc skipped) are staged in threadgroup memory 16 at a time and shared by the 16 heads; each
/// simdgroup scores its channels (partials summed in simdgroup order) and accumulates its slice of the output.
/// meta = {wp, window, K, nc}. The tile load between `HEAD` and `TAIL` is the slice's own (`attnFused` and
/// `attnFusedPacked`), so the tile holds the bytes those kernels stage.
const HEAD =
    \\const uint tid = thread_index_in_threadgroup, sg = simdgroup_index_in_threadgroup;
    \\const ushort lane = thread_index_in_simdgroup;
    \\const int l = threadgroup_position_in_grid.y, h0 = threadgroup_position_in_grid.x * HB;
    \\const int wp = meta[0], window = meta[1], K = meta[2], nc = meta[3];
    \\const int H = q_shape[1];
    \\const int c1 = wp + l, c0 = metal::max(0, c1 - window + 1), nw = c1 - c0 + 1, total = nw + K;
    \\constexpr int DS = D / NSG, QS = DS / 32, OF = DS / 16, NT = NSG * 32, TPH = NT / HB, NJ = KT / TPH;
    \\const int d0 = int(sg) * DS;
    \\threadgroup T kt[KT * D];
    \\threadgroup float sp[NSG * HB * KT];
    \\threadgroup T ph[HB * KT], pl[HB * KT];
    \\threadgroup float cr[HB], rl[HB];
    \\threadgroup int rsrc[KT];
    \\const int hh = int(tid) / TPH, sub = int(tid) % TPH;
    \\float m = sink[h0 + hh], lsum = 1.0f;
    \\const float scale = scal[0];
    \\T qa[QS][16];
    \\float o[OF * 8];
    \\#pragma clang loop unroll(full)
    \\for (int ks = 0; ks < QS; ++ks)
    \\  #pragma clang loop unroll(full)
    \\  for (int i = 0; i < 16; ++i) qa[ks][i] = q[(size_t(l) * H + h0 + dsv41_nax_r(lane, i % 8)) * D + d0 + ks * 32 + (i / 8) * 16 + dsv41_nax_c(lane, i % 8)];
    \\#pragma clang loop unroll(full)
    \\for (int i = 0; i < OF * 8; ++i) o[i] = 0.0f;
    \\for (int t0 = 0; t0 < total; t0 += KT) {
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  if (tid < KT) {
    \\    const int t = t0 + int(tid);
    \\    int src = -1;
    \\    if (t < nw) src = c0 + t;
    \\    else if (t < total) { const int c = topk[size_t(l) * K + t - nw]; if (c >= 0 && c < nc) src = -2 - c; }
    \\    rsrc[tid] = src;
    \\  }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  bool any = false;
    \\  for (int j = 0; j < KT; ++j) any = any || rsrc[j] != -1;
    \\  if (!any) continue;
;

/// The second half, from the staged tile on: the scores, the online softmax and P x V.
const TAIL =
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  float s[8];
    \\  #pragma clang loop unroll(full)
    \\  for (int i = 0; i < 8; ++i) s[i] = 0.0f;
    \\  #pragma clang loop unroll(full)
    \\  for (int ks = 0; ks < QS; ++ks) {
    \\    T b[16];
    \\    #pragma clang loop unroll(full)
    \\    for (int i = 0; i < 16; ++i) b[i] = kt[dsv41_nax_c(lane, i % 8) * D + d0 + ks * 32 + (i / 8) * 16 + dsv41_nax_r(lane, i % 8)];
    \\    dsv41_nax_mma<16, 32>(s, qa[ks], b);
    \\  }
    \\  #pragma clang loop unroll(full)
    \\  for (int i = 0; i < 8; ++i) sp[(int(sg) * HB + dsv41_nax_r(lane, i)) * KT + dsv41_nax_c(lane, i)] = s[i];
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  float sv[NJ], tmax = -INFINITY;
    \\  for (int jj = 0; jj < NJ; ++jj) {
    \\    const int j = sub + jj * TPH;
    \\    float v = -INFINITY;
    \\    if (rsrc[j] != -1) {
    \\      float a = 0.0f;
    \\      for (int g = 0; g < NSG; ++g) a += sp[(g * HB + hh) * KT + j];
    \\      v = a * scale;
    \\    }
    \\    sv[jj] = v;
    \\    tmax = metal::max(tmax, v);
    \\  }
    \\  for (int off = TPH / 2; off > 0; off >>= 1) tmax = metal::max(tmax, metal::simd_shuffle_xor(tmax, ushort(off)));
    \\  const float mn = metal::max(m, tmax), corr = metal::precise::exp(m - mn);
    \\  float psum = 0.0f;
    \\  for (int jj = 0; jj < NJ; ++jj) {
    \\    const int j = sub + jj * TPH;
    \\    const float p = metal::precise::exp(sv[jj] - mn);
    \\    const T hi = T(p);
    \\    psum += p;
    \\    ph[hh * KT + j] = hi;
    \\    pl[hh * KT + j] = T(p - float(hi));
    \\  }
    \\  for (int off = TPH / 2; off > 0; off >>= 1) psum += metal::simd_shuffle_xor(psum, ushort(off));
    \\  lsum = lsum * corr + psum;
    \\  m = mn;
    \\  if (sub == 0) cr[hh] = corr;
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  #pragma clang loop unroll(full)
    \\  for (int i = 0; i < OF * 8; ++i) o[i] *= cr[dsv41_nax_r(lane, i % 8)];
    \\  T ah[8], al[8];
    \\  #pragma clang loop unroll(full)
    \\  for (int i = 0; i < 8; ++i) { ah[i] = ph[dsv41_nax_r(lane, i) * KT + dsv41_nax_c(lane, i)]; al[i] = pl[dsv41_nax_r(lane, i) * KT + dsv41_nax_c(lane, i)]; }
    \\  #pragma clang loop unroll(full)
    \\  for (int g = 0; g < OF / 2; ++g) {
    \\    T b[16];
    \\    #pragma clang loop unroll(full)
    \\    for (int f = 0; f < 2; ++f)
    \\      #pragma clang loop unroll(full)
    \\      for (int i = 0; i < 8; ++i) b[f * 8 + i] = kt[dsv41_nax_r(lane, i) * D + d0 + g * 32 + f * 16 + dsv41_nax_c(lane, i)];
    \\    dsv41_nax_mma<32, 16>(o + g * 16, al, b);
    \\    dsv41_nax_mma<32, 16>(o + g * 16, ah, b);
    \\  }
    \\}
    \\if (sub == 0) rl[hh] = 1.0f / lsum;
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\#pragma clang loop unroll(full)
    \\for (int f = 0; f < OF; ++f)
    \\  #pragma clang loop unroll(full)
    \\  for (int i = 0; i < 8; ++i) {
    \\    const int r = dsv41_nax_r(lane, i);
    \\    out[(size_t(l) * H + h0 + r) * D + d0 + f * 16 + dsv41_nax_c(lane, i)] = OT(o[f * 8 + i] * rl[r]);
    \\  }
;

var def = kernels.Def{ .name = "sushi_ds41_attn_nax", .ins = &.{ "q", "keys", "comp", "topk", "sink", "meta", "scal" }, .outs = &.{"out"}, .source = HEAD ++ kernels.ATTN_FUSED_LOAD_ROWS ++ TAIL, .header = HEADER };
var packed_def = kernels.Def{ .name = "sushi_ds41_attn_nax_fp4", .ins = &.{ "q", "keys", "ccodes", "cscales", "topk", "sink", "meta", "scal" }, .outs = &.{"out"}, .source = HEAD ++ kernels.ATTN_FUSED_LOAD_FP4 ++ TAIL, .header = qat_mod.PACK_HEADER ++ "\n" ++ HEADER };

const hb = 16;
const nsg = 8;
const kt = 16;

/// Whether `attnFusedNax` serves `h` heads of `d` channels in `dt` on this GPU: whole head blocks, 32-channel steps
/// in each simdgroup's slice, the threadgroup within Metal's 32 KB, and the matrix units present (`naxAvailable`).
pub fn fits(h: usize, d: usize, dt: mlx.mlx_dtype) bool {
    if (dt != .bfloat16 and dt != .float16) return false;
    const tg_bytes = kt * d * 2 + nsg * hb * kt * 4 + 2 * hb * kt * 2 + 2 * hb * 4 + kt * 4;
    if (h % hb != 0 or d % (nsg * 32) != 0 or tg_bytes > 32768) return false;
    return naxAvailable();
}

/// `attnFused`'s heads `[L, H, D]` for `q` `[L, H, D]` against the window rows `keys_w` (query i sees rows
/// (wp + i - window, wp + i]) and the compressed rows `comp_rows` it picked (`topk` `[L, K]`), on the matrix units.
/// `fits(H, D, dtype(q))` must hold.
pub fn attnFusedNax(sc: *Scope, sink: A, q: A, keys_w: A, wp: usize, window: usize, comp_rows: A, topk: A, scale: f32, out_dt: mlx.mlx_dtype) !A {
    const dt = mlx.mlx_array_dtype(q);
    const K = if (comp_rows.ctx != null) dim(topk, 1) else 0;
    const comp = if (K > 0) try sc.astype(comp_rows, dt) else try sc.astype(keys_w, dt);
    const picks = if (K > 0) try sc.astype(topk, .int32) else sc_mod.none;
    return launch(sc, &def, sink, q, keys_w, &.{comp}, wp, window, K, if (K > 0) dim(comp_rows, 0) else 0, picks, scale, out_dt, null);
}

/// `attnFusedNax` with the picked rows read where the compressed store keeps them, as `attnFusedPacked` takes them:
/// `codes` `[rows, D / 2]` and `scales` `[rows, D / group]` (`qat_mod.packFp4`'s form), `topk` the ids into them.
/// The tile load is `attnFusedPacked`'s, so the heads are the bytes `attnFusedNax` gives for the unpacked rows.
pub fn attnFusedNaxPacked(sc: *Scope, sink: A, q: A, keys_w: A, wp: usize, window: usize, codes: A, scales: A, mode: qat_mod.Qat, group: usize, topk: A, scale: f32, out_dt: mlx.mlx_dtype) !A {
    std.debug.assert(group % 8 == 0 and dim(codes, 1) * 2 == dim(q, 2));
    return launch(sc, &packed_def, sink, q, keys_w, &.{ codes, scales }, wp, window, dim(topk, 1), dim(codes, 0), try sc.astype(topk, .int32), scale, out_dt, .{ .group = ci(group), .mode = qat_mod.packMode(mode) });
}

/// `kernels.launch`'s template-integer type (private to the slice).
const Tmpl = std.meta.Child(@typeInfo(@TypeOf(kernels.launch)).@"fn".param_types[6].?);

/// The packed store's template values: the scale group and `qat_mod.packMode` of its format.
const Fp4 = struct { group: c_int, mode: c_int };

/// `comp`: the picked rows' arrays, as `d` takes them; `picks`: `topk` as int32, none without picks; `fp4` for the
/// packed kernel.
fn launch(sc: *Scope, d: *kernels.Def, sink: A, q: A, keys_w: A, comp: []const A, wp: usize, window: usize, K: usize, nc: usize, picks: A, scale: f32, out_dt: mlx.mlx_dtype, fp4: ?Fp4) !A {
    const L = dim(q, 0);
    const H = dim(q, 1);
    const D = dim(q, 2);
    const dt = mlx.mlx_array_dtype(q);
    const meta_v = [_]i32{ ci(wp), ci(window), ci(K), ci(nc) };
    const meta = try sc.ints(&meta_v, &.{meta_v.len});
    var ins: [8]A = undefined;
    ins[0] = q;
    ins[1] = try sc.astype(keys_w, dt);
    @memcpy(ins[2..][0..comp.len], comp);
    var n = 2 + comp.len;
    for ([_]A{ if (picks.ctx != null) picks else meta, sink, meta, try kernels.f32Const(sc, &.{scale}) }) |a| {
        ins[n] = a;
        n += 1;
    }
    var tmpl: [6]Tmpl = undefined;
    const base = [_]Tmpl{ .{ .name = "D", .v = ci(D) }, .{ .name = "HB", .v = hb }, .{ .name = "NSG", .v = nsg }, .{ .name = "KT", .v = kt } };
    @memcpy(tmpl[0..base.len], &base);
    var nt: usize = base.len;
    if (fp4) |f| {
        tmpl[4] = .{ .name = "G", .v = f.group };
        tmpl[5] = .{ .name = "MODE", .v = f.mode };
        nt = 6;
    }
    var o: [1]A = undefined;
    try kernels.launch(sc, d, ins[0..n], &.{.{ .shape = &.{ ci(L), ci(H), ci(D) }, .dt = out_dt }}, .{ H / hb * nsg * 32, L, 1 }, .{ nsg * 32, 1, 1 }, tmpl[0..nt], &.{ .{ .name = "T", .dt = dt }, .{ .name = "OT", .dt = out_dt } }, &o);
    return o[0];
}

var nax_cache: ?bool = null;

/// Whether the default GPU has the matrix units `matmul2d` runs on, as MLX's `is_nax_available` decides: macOS 26.2
/// or later and an `applegpu_g<N>` architecture with N >= 17 (18 for a phone part). Read once.
pub fn naxAvailable() bool {
    if (nax_cache) |v| return v;
    const v = probeNax();
    nax_cache = v;
    return v;
}

fn probeNax() bool {
    var buf: [32]u8 = undefined;
    var len: usize = buf.len;
    if (std.c.sysctlbyname("kern.osproductversion", &buf, &len, null, 0) != 0) return false;
    if (!versionAtLeast(std.mem.sliceTo(buf[0..len], 0), 26, 2)) return false;
    if (mlx.noGpuBackend()) return false;
    const had = mlx.errorPending();
    const dev = mlx.mlx_device_new_type(.gpu, 0);
    defer _ = mlx.mlx_device_free(dev);
    var info = mlx.mlx_device_info_new();
    defer _ = mlx.mlx_device_info_free(info);
    var arch: [*:0]const u8 = undefined;
    if (mlx.mlx_device_info_get(&info, dev) != 0 or mlx.mlx_device_info_get_string(&arch, info, "architecture") != 0) {
        mlx.dropLatchedErrorUnless(had);
        return false;
    }
    return archHasNax(std.mem.span(arch));
}

/// `applegpu_g17s` and later (`applegpu_g18p` and later for a phone part).
fn archHasNax(arch: []const u8) bool {
    const prefix = "applegpu_g";
    if (!std.mem.startsWith(u8, arch, prefix)) return false;
    const digits = arch[prefix.len..];
    const n_end = std.mem.indexOfNone(u8, digits, "0123456789") orelse digits.len;
    const gen = std.fmt.parseInt(u32, digits[0..n_end], 10) catch return false;
    const phone = n_end < digits.len and digits[digits.len - 1] == 'p';
    return gen >= @as(u32, if (phone) 18 else 17);
}

/// "27.2", "26.2.1": at least major.minor.
fn versionAtLeast(v: []const u8, major: u32, minor: u32) bool {
    var it = std.mem.splitScalar(u8, v, '.');
    const ma = std.fmt.parseInt(u32, it.next() orelse return false, 10) catch return false;
    const mi = std.fmt.parseInt(u32, it.next() orelse "0", 10) catch return false;
    return ma > major or (ma == major and mi >= minor);
}

test "sushi_ds41 nax gate: M5 parts and macOS 26.2+ only" {
    try std.testing.expect(archHasNax("applegpu_g17s"));
    try std.testing.expect(archHasNax("applegpu_g17d"));
    try std.testing.expect(archHasNax("applegpu_g18p"));
    try std.testing.expect(!archHasNax("applegpu_g17p"));
    try std.testing.expect(!archHasNax("applegpu_g16s"));
    try std.testing.expect(!archHasNax("applegpu_g"));
    try std.testing.expect(!archHasNax("apple_m5"));
    try std.testing.expect(versionAtLeast("27.2", 26, 2));
    try std.testing.expect(versionAtLeast("26.2.1", 26, 2));
    try std.testing.expect(!versionAtLeast("26.1", 26, 2));
    try std.testing.expect(!versionAtLeast("15.7", 26, 2));
}
