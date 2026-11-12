// turboexl3_glm53f TURBOEXL3_GLM53F_DSA_LFUSE (src/dsa_lfuse.zig), launch 1 of 2: a verify window's sparse latent
// attention scores at GLM-5.3-Flash's dims (64 heads, 512-wide latent rows, 2051 selected slots a row), the stock
// chain's q4 copy, scale multiply, first matmul, mask and where in one launch.
//
// The stock chain (glm5_next `sparseLatentAttention` then `latentAttendFolded`, B 1, one chunk): q4 [n, 64, 1, 512] =
// q_abs [1, 64, n, 512] with its rows and heads swapped (a copy), qs = q4 * T(scale) (MLX's multiply in T), scores =
// mlx_matmul(qs, keys^T) over the gathered keys [n, 2051, 512] (MLX's regular NAX GEMM: bm 64, bn 128, bk 256, wm 2,
// wn 4; each simdgroup's tile_matmad_nax over 32-k steps in ascending k, the past-the-end columns zero), then
// where(idx >= 0, scores, T(-inf)). Here the same GEMM loop as src/kernels/dsa_vattn.metal's (keys read from the latent
// at the clamped index), with A read from q_abs at the window row (head stride VS * 512) and each element multiplied by
// T(scale) in T as MLX's multiply does, and the store writing T(-inf) where the slot is unused: the same values.
//
// Template: T (bf16 or f16), VS (q_abs's rows: the window's), VSC (the scale's f32 bits). Inputs q [1, 64, VS, 512],
// lat [1, slots, 512], idx int32 [VS, 2051]; output d [VS, 64, 2051]. Grid: (17, 1, VS) threadgroups of 256 threads,
// simdgroup s at heads 32 * (s / 4) and slots 32 * (s % 4) of the threadgroup's 64 x 128 tile.
constexpr int H = 64;
constexpr int R = 512;
constexpr int W = 2051;
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
constexpr int k_iterations = R / BK;
static_assert(H % BM == 0 && R % BK == 0, "the heads fill whole 64-row tiles and the latent whole 256-k stages");

const int row = int(threadgroup_position_in_grid.z);
const int c_col = int(threadgroup_position_in_grid.x) * BN;
const short sgi = short(simdgroup_index_in_threadgroup);
const short tm = SM * (sgi / WN);
const short tn = SN * (sgi % WN);

const short sgp_sn = short(min(int(SN), W - (c_col + tn)));
const bool full_n = sgp_sn == SN;
const bool has_output = sgp_sn > 0;

// A: the window row's absorbed queries, head h at q[(h * VS + row) * R]
const device T* A = q + (size_t(tm) * VS + row) * R;
const device int* ix = idx + size_t(row) * W;
device T* D = d + size_t(row) * H * W + size_t(tm) * W + (c_col + tn);
const T scale_t = static_cast<T>(as_type<float>(uint(VSC)));

// The lane's place in a 16 x 16 fragment (BaseNAXFrag's): rows sc.y and sc.y + 8, columns sc.x .. sc.x + 3.
const short2 sc = BaseNAXFrag::get_coord();

// The lane's 4 key rows (the simdgroup's slots c_col + tn + r), fixed over k; a slot past W reads nothing.
const device T* krow[TN][2];
bool kok[TN][2];
STEEL_PRAGMA_UNROLL
for (short ir = 0; ir < TN; ir++) {
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < 2; i++) {
    const short r = ir * 16 + sc.y + i * 8;
    kok[ir][i] = r < sgp_sn;
    krow[ir][i] = lat + size_t(kok[ir][i] ? max(ix[c_col + tn + r], 0) : 0) * R;
  }
}

NAXTile<float, TM, TN> Dtile;
Dtile.clear();

// One 32-k step at k0: A as MLX loads qs (q times the scale, in T), B gathered; `bsafe` a partial simdgroup's limits.
auto step = [&](const int k0, const bool bsafe) {
  NAXTile<T, TM, TK> Atile;
  Atile.load(A + k0, VS * R);
  STEEL_PRAGMA_UNROLL
  for (short im = 0; im < TM; im++) {
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < TK; ik++) {
      thread auto& f = Atile.frag_at(im, ik);
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < 8; e++) {
        f[e] = f[e] * scale_t;
      }
    }
  }
  NAXTile<T, TN, TK> Btile;
  // B [W, R] stored as keys rows: element (row r, column c) = keys[c_col + tn + r][k0 + c]
  STEEL_PRAGMA_UNROLL
  for (short ir = 0; ir < TN; ir++) {
    STEEL_PRAGMA_UNROLL
    for (short ic = 0; ic < TK; ic++) {
      thread auto& f = Btile.frag_at(ir, ic);
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < 2; i++) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < 4; j++) {
          const short c = ic * 16 + sc.x + j;
          f[i * 4 + j] = (!bsafe || kok[ir][i]) ? static_cast<T>(krow[ir][i][k0 + c]) : T(0);
        }
      }
    }
  }
  tile_matmad_nax(Dtile, Atile, metal::bool_constant<false>{}, Btile, metal::bool_constant<true>{});
};

// gemm_loop's k iterations (the latent is whole 256-k stages): a partial simdgroup takes load_safe's limits on B's rows
// past W, where MLX's does; a simdgroup with no output skips its steps
STEEL_PRAGMA_NO_UNROLL
for (int kk0 = 0; kk0 < k_iterations; kk0++) {
  threadgroup_barrier(mem_flags::mem_none);
  if (!full_n && !has_output) {
    continue;
  }
  STEEL_PRAGMA_NO_UNROLL
  for (int kk1 = 0; kk1 < BK; kk1 += SK) {
    volatile int compiler_barrier;
    step(kk0 * BK + kk1, !full_n);
    (void)compiler_barrier;
  }
}

if (has_output) {
  // the where: T(-inf) at the unused slots (index -1), the score elsewhere
  STEEL_PRAGMA_UNROLL
  for (short im = 0; im < TM; im++) {
    STEEL_PRAGMA_UNROLL
    for (short jn = 0; jn < TN; jn++) {
      thread auto& f = Dtile.frag_at(im, jn);
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < 4; j++) {
        const short c = jn * 16 + sc.x + j;
        if (c < sgp_sn && ix[c_col + tn + c] < 0) {
          f[j] = -INFINITY;
          f[4 + j] = -INFINITY;
        }
      }
    }
  }
  if (full_n) {
    Dtile.store(D, W);
  } else {
    Dtile.store_safe(D, W, short2(sgp_sn, SM));
  }
}
