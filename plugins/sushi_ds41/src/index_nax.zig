//! DeepSeek-V4.1's prefill index scores (`indexScoresTile` of the vendored slice) on the M5 GPU's matrix units: each
//! threadgroup stages one tile of KT keys and scores R query rows against it, so a key is read once for R rows
//! instead of once a row, and every head's dot product runs through Metal 4's `matmul2d` (16 heads x 32 keys over
//! 16-channel steps, f32 accumulators). The relu, the head weights and the head sum stay in registers, as in
//! `indexScoresTile`, so no `[L, IH, nb]` plane is written. A key's score depends only on its own column, so a slice
//! of the key list scores the bits the whole list's columns would (`indexStream` relies on it).
const std = @import("std");
const mlx = @import("sushi/mlx.zig");
const sc_mod = @import("sushi/dsv41/dsv41_scope.zig");
const kernels = @import("sushi/dsv41/dsv41_kernels.zig");
const attn_nax = @import("attn_nax.zig");

const A = sc_mod.A;
const Scope = sc_mod.Scope;
const dim = sc_mod.dim;
const ci = sc_mod.ci;

/// Simdgroup s takes (row, key group) pairs s, s + NSG, ...: R rows x KT / 32 groups of 32 keys. Each pair is two
/// 16 x 32 fragments (heads 0-15 and 16-31), relu'd, weighted, folded over the fragment's rows (i and i + 4 hold
/// rows r and r + 8) and summed across the lanes that hold the other rows (lane bits 1, 2 and 4); the four lanes left
/// (bits 0 and 3) write the group's 32 scores. A tile no row of the group can see is -inf without the work.
/// meta = {p0, ratio, key offset}: `keys` may be a slice of a longer key list starting at that offset.
const SOURCE =
    \\const uint tid = thread_index_in_threadgroup, sg = simdgroup_index_in_threadgroup;
    \\const ushort lane = thread_index_in_simdgroup;
    \\const int n0 = threadgroup_position_in_grid.x * KT, l0 = threadgroup_position_in_grid.y * R;
    \\const int L = q_shape[0], nb = keys_shape[0];
    \\const int end = metal::min(nb, n0 + KT);
    \\const int lmax = metal::min(L, l0 + R) - 1;
    \\const int seen_max = (meta[0] + lmax + 1) / meta[1] - meta[2];
    \\if (n0 >= seen_max) {
    \\  for (int r = 0; r < R && l0 + r < L; ++r)
    \\    for (int n = n0 + int(tid); n < end; n += NSG * 32) scores[size_t(l0 + r) * nb + n] = -INFINITY;
    \\  return;
    \\}
    \\constexpr int C4 = IHD / 4, NG = KT / 32;
    \\using T4 = metal::vec<T, 4>;
    \\threadgroup T kt[KT * IHD];
    \\threadgroup float wt[R * 32];
    \\for (uint i = tid; i < uint(KT * C4); i += NSG * 32) {
    \\  const int n = n0 + int(i / C4);
    \\  ((threadgroup T4*)kt)[i] = n < nb ? ((const device T4*)keys)[size_t(n) * C4 + i % C4] : T4(0);
    \\}
    \\for (uint i = tid; i < uint(R * 32); i += NSG * 32) {
    \\  const int l = l0 + int(i) / 32;
    \\  wt[i] = l < L ? float(w[size_t(l) * 32 + i % 32]) * wsc[0] : 0.0f;
    \\}
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\for (int pair = int(sg); pair < R * NG; pair += NSG) {
    \\  const int r = pair / NG, g = pair % NG, l = l0 + r;
    \\  if (l >= L) continue;
    \\  const int seen = (meta[0] + l + 1) / meta[1] - meta[2];
    \\  const int k0 = n0 + g * 32;
    \\  if (k0 >= seen) {
    \\    if (lane < 32 && k0 + int(lane) < nb) scores[size_t(l) * nb + k0 + lane] = -INFINITY;
    \\    continue;
    \\  }
    \\  float lo[16], hi[16];
    \\  #pragma clang loop unroll(full)
    \\  for (int i = 0; i < 16; ++i) { lo[i] = 0.0f; hi[i] = 0.0f; }
    \\  const device T* qr = q + size_t(l) * 32 * IHD;
    \\  const threadgroup T* kr = kt + g * 32 * IHD;
    \\  for (int d = 0; d < IHD; d += 16) {
    \\    T alo[8], ahi[8], b[16];
    \\    #pragma clang loop unroll(full)
    \\    for (int i = 0; i < 8; ++i) {
    \\      alo[i] = qr[dsv41_nax_r(lane, i) * IHD + d + dsv41_nax_c(lane, i)];
    \\      ahi[i] = qr[(16 + dsv41_nax_r(lane, i)) * IHD + d + dsv41_nax_c(lane, i)];
    \\    }
    \\    #pragma clang loop unroll(full)
    \\    for (int i = 0; i < 16; ++i) b[i] = kr[(dsv41_nax_c(lane, i % 8) + (i / 8) * 16) * IHD + d + dsv41_nax_r(lane, i % 8)];
    \\    dsv41_nax_mma<32, 16>(lo, alo, b);
    \\    dsv41_nax_mma<32, 16>(hi, ahi, b);
    \\  }
    \\  const threadgroup float* wr = wt + r * 32;
    \\  float s[8];
    \\  #pragma clang loop unroll(full)
    \\  for (int f = 0; f < 2; ++f)
    \\    #pragma clang loop unroll(full)
    \\    for (int j = 0; j < 4; ++j) {
    \\      const int a = f * 8 + j, b2 = f * 8 + j + 4;
    \\      const int ra = dsv41_nax_r(lane, j), rb = dsv41_nax_r(lane, j + 4);
    \\      s[f * 4 + j] = (metal::max(lo[a], 0.0f) * wr[ra] + metal::max(lo[b2], 0.0f) * wr[rb])
    \\                   + (metal::max(hi[a], 0.0f) * wr[16 + ra] + metal::max(hi[b2], 0.0f) * wr[16 + rb]);
    \\    }
    \\  #pragma clang loop unroll(full)
    \\  for (int i = 0; i < 8; ++i) {
    \\    s[i] += metal::simd_shuffle_xor(s[i], ushort(2));
    \\    s[i] += metal::simd_shuffle_xor(s[i], ushort(4));
    \\    s[i] += metal::simd_shuffle_xor(s[i], ushort(16));
    \\  }
    \\  if ((lane & 22u) == 0u)
    \\    #pragma clang loop unroll(full)
    \\    for (int i = 0; i < 8; ++i) {
    \\      const int n = k0 + dsv41_nax_c(lane, i % 4) + (i / 4) * 16;
    \\      if (n < nb) scores[size_t(l) * nb + n] = n < seen ? s[i] : -INFINITY;
    \\    }
    \\}
;

var def = kernels.Def{ .name = "sushi_ds41_index_nax", .ins = &.{ "q", "keys", "w", "meta", "wsc" }, .outs = &.{"scores"}, .source = SOURCE, .header = attn_nax.HEADER };

/// Measured on the M5 Ultra at a 256-row span over 23,680 keys: R 4 / 8 / 16 with KT 64 / 64 / 32 took 0.771 / 0.716 /
/// 0.703 ms against `indexScoresTile`'s 2.323.
const r = 16;
const kt = 32;
const nsg = 8;

/// Whether `indexScoresNax` serves `ih` heads of `ihd` channels in `dt` on this GPU (32 heads, 16-channel steps, the
/// matrix units present).
pub fn fits(ih: usize, ihd: usize, dt: mlx.mlx_dtype) bool {
    if (dt != .bfloat16 and dt != .float16) return false;
    if (ih != 32 or ihd % 16 != 0 or kt * ihd * 2 + r * 32 * 4 > 32768) return false;
    return attn_nax.naxAvailable();
}

/// `indexScoresTile`'s scores `[L, nb]` (f32, -inf past what a row sees) of `q` `[L, 32, IHD]` against `keys`
/// `[nb, IHD]` from `key0` on, with head weights `w` `[L, 32]` times `wscale`; `q` is cast to the keys' dtype.
/// `key0` must be a multiple of the key tile. `fits(32, IHD, dtype(keys))` must hold.
pub fn indexScoresNax(sc: *Scope, q: A, keys: A, w: A, wscale: f32, p0: usize, ratio: usize, key0: usize) !A {
    const L = dim(q, 0);
    const nb = dim(keys, 0);
    const dt = mlx.mlx_array_dtype(keys);
    std.debug.assert(key0 % kt == 0);
    const meta = try sc.ints(&.{ ci(p0), ci(ratio), ci(key0) }, &.{3});
    var o: [1]A = undefined;
    try kernels.launch(sc, &def, &.{ try sc.astype(q, dt), keys, w, meta, try kernels.f32Const(sc, &.{wscale}) }, &.{.{ .shape = &.{ ci(L), ci(nb) }, .dt = .float32 }}, .{ (nb + kt - 1) / kt * nsg * 32, (L + r - 1) / r, 1 }, .{ nsg * 32, 1, 1 }, &.{
        .{ .name = "IHD", .v = ci(dim(q, 2)) }, .{ .name = "KT", .v = kt }, .{ .name = "R", .v = r }, .{ .name = "NSG", .v = nsg },
    }, &.{.{ .name = "T", .dt = dt }}, &o);
    return o[0];
}
