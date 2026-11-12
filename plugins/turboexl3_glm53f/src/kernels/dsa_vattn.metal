// turboexl3_glm53f TURBOEXL3_GLM53F_DSA_VATTN (src/dsa_vattn.zig): one of the two matmuls of a verify window's sparse
// latent attention (glm5_next `latentAttendFolded` after `sparseLatentAttention`'s gather), with the gathered keys read
// through the selection's indices where MLX reads the gathered copy.
//
// The stock chain gathers keys [n, W, R] = latent[max(idx, 0)] (take), then runs mlx_matmul(qs, keys^T) and
// mlx_matmul(p, keys) over a batch of n >= 2 verify rows: MLX's regular NAX GEMM, steel_gemm_fused_nax (bm 64, bn 128,
// bk 256, wm 2, wn 4 on this GPU; every NAX tiling of it gives a simdgroup the same 32 x 32 tile), each simdgroup's
// tile_matmad_nax over 32-k steps in ascending k (gemm_nax.h `gemm_loop`), the past-the-end rows and columns zero
// (load_safe), the out-of-range simdgroups idle. Here the same loop with the same loads, the B tile's rows (the keys)
// taken from the latent at the clamped index of each row, the value the gathered copy holds there: the same op on the
// same operands in the same order, so the same output.
//
// Template: T (bf16 or f16), VM, VN, VK (the GEMM: A [n, VM, VK] row-major, D [n, VM, VN]), VR (the latent's row width),
// VW (the indices a row: idx [n, VW]), VTB (1: D = A . keys^T, keys [VN = VW, VK = VR]; 0: D = A . keys, keys [VK = VW,
// VN = VR]). Grid: (ceil(VN / 128), ceil(VM / 64), n) threadgroups of 256 threads, simdgroup s at rows 32 * (s / 4) and
// columns 32 * (s % 4) of the threadgroup's 64 x 128 tile, as MLX's (32, 4, 2) threadgroup places it.
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
constexpr bool TB = VTB != 0;
constexpr short RB = TB ? TN : TK;
constexpr short CB = TB ? TK : TN;
constexpr bool kAlignedM = (VM % BM) == 0;
constexpr bool kAlignedK = (VK % BK) == 0;
constexpr int k_iterations = VK / BK;

const int bz = int(threadgroup_position_in_grid.z);
const int c_row = int(threadgroup_position_in_grid.y) * BM;
const int c_col = int(threadgroup_position_in_grid.x) * BN;
const short sgi = short(simdgroup_index_in_threadgroup);
const short tm = SM * (sgi / WN);
const short tn = SN * (sgi % WN);

const short sgp_sm = kAlignedM ? SM : short(min(int(SM), VM - (c_row + tm)));
const short sgp_sn = short(min(int(SN), VN - (c_col + tn)));
const bool full_m = sgp_sm == SM;
const bool full_n = sgp_sn == SN;
const bool has_output = sgp_sm > 0 && sgp_sn > 0;

const device T* A = a + size_t(bz) * VM * VK + size_t(c_row + tm) * VK;
const device int* ix = idx + size_t(bz) * VW;
device T* D = d + size_t(bz) * VM * VN + size_t(c_row + tm) * VN + (c_col + tn);

// The lane's place in a 16 x 16 fragment (BaseNAXFrag's): rows sc.y and sc.y + 8, columns sc.x .. sc.x + 3.
const short2 sc = BaseNAXFrag::get_coord();

// TB: the lane's 4 key rows (the simdgroup's columns c_col + tn + r), fixed over k; a row past VN reads nothing.
const device T* krow[TN][2];
bool kok[TN][2];
if constexpr (TB) {
  STEEL_PRAGMA_UNROLL
  for (short ir = 0; ir < TN; ir++) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 2; i++) {
      const short r = ir * 16 + sc.y + i * 8;
      kok[ir][i] = r < sgp_sn;
      krow[ir][i] = lat + size_t(kok[ir][i] ? max(ix[c_col + tn + r], 0) : 0) * VR;
    }
  }
}

NAXTile<float, TM, TN> Dtile;
Dtile.clear();

// One 32-k step at k0: A as MLX loads it, B gathered. `kmax` the step's k past the end (VK - k0 for the remainder),
// `bsafe` whether B takes load_safe's limits (a partial simdgroup, or the remainder).
auto step = [&](const int k0, const short kmax, const bool asafe, const bool bsafe) {
  NAXTile<T, TM, TK> Atile;
  NAXTile<T, RB, CB> Btile;
  if (!asafe) {
    Atile.load(A + k0, VK);
  } else {
    Atile.load_safe(A + k0, VK, short2(kmax, sgp_sm));
  }
  if constexpr (TB) {
    // B [VN, VK] stored as keys rows: element (row r, column c) = keys[c_col + tn + r][k0 + c]
    STEEL_PRAGMA_UNROLL
    for (short ir = 0; ir < RB; ir++) {
      STEEL_PRAGMA_UNROLL
      for (short ic = 0; ic < CB; ic++) {
        thread auto& f = Btile.frag_at(ir, ic);
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < 4; j++) {
            const short c = ic * 16 + sc.x + j;
            f[i * 4 + j] = (!bsafe || (kok[ir][i] && c < kmax)) ? static_cast<T>(krow[ir][i][k0 + c]) : T(0);
          }
        }
      }
    }
  } else {
    // B [VK, VN] = keys: element (row r, column c) = keys[k0 + r][c_col + tn + c]
    STEEL_PRAGMA_UNROLL
    for (short ir = 0; ir < RB; ir++) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < 2; i++) {
        const short r = ir * 16 + sc.y + i * 8;
        const bool rok = !bsafe || r < kmax;
        const device T* src = lat + size_t(rok ? max(ix[k0 + r], 0) : 0) * VR + (c_col + tn);
        STEEL_PRAGMA_UNROLL
        for (short ic = 0; ic < CB; ic++) {
          thread auto& f = Btile.frag_at(ir, ic);
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < 4; j++) {
            const short c = ic * 16 + sc.x + j;
            f[i * 4 + j] = (rok && (!bsafe || c < sgp_sn)) ? static_cast<T>(src[c]) : T(0);
          }
        }
      }
    }
  }
  tile_matmad_nax(Dtile, Atile, metal::bool_constant<false>{}, Btile, metal::bool_constant<TB>{});
};

// gemm_loop's k iterations: a partial simdgroup takes load_safe's limits (A's rows past VM, B's rows or columns past
// VN, each where MLX's does), a simdgroup with no output skips its steps
const bool part = !full_m || !full_n;
STEEL_PRAGMA_NO_UNROLL
for (int kk0 = 0; kk0 < k_iterations; kk0++) {
  threadgroup_barrier(mem_flags::mem_none);
  if (part && !has_output) {
    continue;
  }
  STEEL_PRAGMA_NO_UNROLL
  for (int kk1 = 0; kk1 < BK; kk1 += SK) {
    volatile int compiler_barrier;
    step(kk0 * BK + kk1, SK, !full_m, !full_n);
    (void)compiler_barrier;
  }
}
// the k past the last whole BK: load_safe's limits for every simdgroup
if (!kAlignedK) {
  simdgroup_barrier(mem_flags::mem_none);
  if (has_output) {
    const short rem_bk = short(VK - k_iterations * BK);
    STEEL_PRAGMA_NO_UNROLL
    for (int kk1 = 0; kk1 < rem_bk; kk1 += SK) {
      step(k_iterations * BK + kk1, short(max(0, rem_bk - kk1)), true, true);
    }
  }
}

if (has_output) {
  if (!part) {
    Dtile.store(D, VN);
  } else {
    Dtile.store_safe(D, VN, short2(sgp_sn, sgp_sm));
  }
}
