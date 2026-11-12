// turboexl3_glm53f TURBOEXL3_GLM53F_DSA_LFUSE (src/dsa_lfuse.zig), launch 2 of 2: a verify window's precise softmax
// over its masked scores (launch 1's) and the second matmul at GLM-5.3-Flash's dims (64 heads, 512-wide latent rows,
// 2051 selected slots a row), the stock chain's softmax and matmul in one launch.
//
// The stock chain's softmax (mlx_softmax_axis, precise, over 2051 slots: MLX's block_softmax_precise kernel, one
// threadgroup a row of 544 threads, 17 simdgroups): thread t reads slots 4t .. 4t + 3 as f32 (past the end -inf), its
// running max from -FLT_MAX, simd_max, the threadgroup's max; then each thread's exp(x - max) (fast::exp) summed from 0
// in slot order, simd_sum, the 17 simdgroup sums simd_summed (the other lanes 0), 1 / that, and p = T(exp * (1 / sum)).
// Its second matmul mlx_matmul(p, keys) over the gathered keys [2051, 512]: MLX's regular NAX GEMM as
// src/kernels/dsa_vattn.metal reads it (keys from the latent at the clamped index), 8 whole 256-k stages and 3 slots
// past them under load_safe's limits. Here each simdgroup takes 8 heads' max and sum with one lane per MLX thread of
// each of the 17 simdgroups in turn (the same lane values, so the same simd_max and simd_sum), then the same GEMM loop
// with each A element computed as the softmax writes it, from the same scores and the same max and sum: the same p, so
// the same output.
//
// Template: T (bf16 or f16). Inputs sc [n, 64, 2051] (launch 1's scores), lat [1, slots, 512], idx int32 [n, 2051];
// output d [n, 64, 512]. Grid: (4, 1, n) threadgroups of 256 threads, simdgroup s at heads 32 * (s / 4) and latent
// columns 32 * (s % 4) of the threadgroup's 64 x 128 tile.
constexpr int H = 64;
constexpr int R = 512;
constexpr int W = 2051;
constexpr int NSG = ((W + 3) / 4 + 31) / 32;  // the softmax's simdgroups: 17
constexpr short BM = 64;
constexpr short BN = 128;
constexpr short BK = 256;
constexpr short WN = 4;
constexpr short SM = 32;
constexpr short SN = 32;
constexpr short SK = 32;
constexpr short TM = SM / 16;
constexpr short TN = SN / 16;
constexpr short TK = SK / 16;
constexpr int k_iterations = W / BK;
static_assert(H == 8 * 8 && H % BM == 0 && R % BN == 0 && NSG <= 32, "8 simdgroups of 8 heads; whole tiles of heads and latent columns");

const int row = int(threadgroup_position_in_grid.z);
const int c_col = int(threadgroup_position_in_grid.x) * BN;
const short sgi = short(simdgroup_index_in_threadgroup);
const uint lane = thread_index_in_simdgroup;
const short tm = SM * (sgi / WN);
const short tn = SN * (sgi % WN);

const device T* S = sc + size_t(row) * H * W;
const device int* ix = idx + size_t(row) * W;
device T* D = d + size_t(row) * H * R + size_t(tm) * R + (c_col + tn);

threadgroup float smax[H];
threadgroup float sinv[H];

// the softmax's max and 1 / sum for heads 8 sgi .. 8 sgi + 7
for (int hh = 0; hh < 8; hh++) {
  const int h = sgi * 8 + hh;
  const device T* srow = S + size_t(h) * W;
  float maxval = -FLT_MAX;
  for (int t = int(lane); t < NSG * 32; t += 32) {
    const int base = t * 4;
    float ld[4];
    if (base + 4 <= W) {
      for (int i = 0; i < 4; i++) {
        ld[i] = float(srow[base + i]);
      }
    } else {
      for (int i = 0; i < 4; i++) {
        ld[i] = (base + i < W) ? float(srow[base + i]) : -INFINITY;
      }
    }
    for (int i = 0; i < 4; i++) {
      maxval = (maxval < ld[i]) ? ld[i] : maxval;
    }
  }
  maxval = metal::simd_max(maxval);
  float mine = 0.0f;
  for (int g = 0; g < NSG; g++) {
    const int base = (g * 32 + int(lane)) * 4;
    float ld[4];
    if (base + 4 <= W) {
      for (int i = 0; i < 4; i++) {
        ld[i] = float(srow[base + i]);
      }
    } else {
      for (int i = 0; i < 4; i++) {
        ld[i] = (base + i < W) ? float(srow[base + i]) : -INFINITY;
      }
    }
    float normalizer = 0;
    for (int i = 0; i < 4; i++) {
      normalizer += metal::fast::exp(ld[i] - maxval);
    }
    normalizer = metal::simd_sum(normalizer);
    if (int(lane) == g) {
      mine = normalizer;
    }
  }
  const float total = metal::simd_sum(mine);
  if (lane == 0) {
    smax[h] = maxval;
    sinv[h] = 1 / total;
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// The lane's place in a 16 x 16 fragment (BaseNAXFrag's): rows sc.y and sc.y + 8, columns sc.x .. sc.x + 3.
const short2 fc = BaseNAXFrag::get_coord();

NAXTile<float, TM, TN> Dtile;
Dtile.clear();

// One 32-k step at k0: A the softmax's p at the simdgroup's heads, B gathered; `kmax` the step's slots before W.
auto step = [&](const int k0, const short kmax, const bool safe) {
  NAXTile<T, TM, TK> Atile;
  // A [64, W] = p: element (row r, column c) = T(exp(scores[tm + r][k0 + c] - max) * (1 / sum)), 0 past W
  STEEL_PRAGMA_UNROLL
  for (short im = 0; im < TM; im++) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 2; i++) {
      const int h = tm + im * 16 + fc.y + i * 8;
      const device T* prow = S + size_t(h) * W + k0;
      const float m = smax[h];
      const float inv = sinv[h];
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        thread auto& f = Atile.frag_at(im, ik);
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < 4; j++) {
          const short c = ik * 16 + fc.x + j;
          f[i * 4 + j] = (!safe || c < kmax) ? static_cast<T>(metal::fast::exp(float(prow[c]) - m) * inv) : T(0);
        }
      }
    }
  }
  NAXTile<T, TK, TN> Btile;
  // B [W, R] = keys: element (row r, column c) = keys[k0 + r][c_col + tn + c]
  STEEL_PRAGMA_UNROLL
  for (short ir = 0; ir < TK; ir++) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 2; i++) {
      const short r = ir * 16 + fc.y + i * 8;
      const bool rok = !safe || r < kmax;
      const device T* src = lat + size_t(rok ? max(ix[k0 + r], 0) : 0) * R + (c_col + tn);
      STEEL_PRAGMA_UNROLL
      for (short ic = 0; ic < TN; ic++) {
        thread auto& f = Btile.frag_at(ir, ic);
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < 4; j++) {
          const short c = ic * 16 + fc.x + j;
          f[i * 4 + j] = rok ? static_cast<T>(src[c]) : T(0);
        }
      }
    }
  }
  tile_matmad_nax(Dtile, Atile, metal::bool_constant<false>{}, Btile, metal::bool_constant<false>{});
};

// gemm_loop's k iterations: 8 whole 256-k stages (every simdgroup whole), then the 3 slots past them under load_safe's
// limits
STEEL_PRAGMA_NO_UNROLL
for (int kk0 = 0; kk0 < k_iterations; kk0++) {
  threadgroup_barrier(mem_flags::mem_none);
  STEEL_PRAGMA_NO_UNROLL
  for (int kk1 = 0; kk1 < BK; kk1 += SK) {
    volatile int compiler_barrier;
    step(kk0 * BK + kk1, SK, false);
    (void)compiler_barrier;
  }
}
simdgroup_barrier(mem_flags::mem_none);
{
  const short rem_bk = short(W - k_iterations * BK);
  STEEL_PRAGMA_NO_UNROLL
  for (int kk1 = 0; kk1 < rem_bk; kk1 += SK) {
    step(k_iterations * BK + kk1, short(max(0, rem_bk - kk1)), true);
  }
}

Dtile.store(D, R);
