// turboexl3_glm53f TURBOEXL3_GLM53F_ROUTER, phase 1 (src/fused_router.zig): a decode call's router logits as MLX's
// split-K NAX GEMM leaves them, its partials P[part][row][expert] (f32), for M rows of x (bf16 [M, K]) against the
// router as stored (WT [N, K]: the pack's f16, or bf16).
//
// The stock route casts x to f32 and runs mlx_matmul(x, W) over the router's f32 copy, a transposed view of the stored
// weight widened, so MLX takes steel_gemm_splitk_nax_nt (bm 64, bn 64, bk 256, wm 2, wn 2; split K at 2048): one chain a
// 32-expert block and a K partition, each a simdgroup's tile_matmad_nax over 32-k steps (two 16x32x16 matmul2d ops,
// relaxed precision, transpose_b) in ascending k, the rows past M zero (load_safe). Here a threadgroup runs one such
// chain: the same op on the same operands in the same order. The operands are the same numbers: x is bf16 and the weight
// f16 or bf16, so their f32 copies are their exact widenings, read here as stored and widened in the fragment. The
// chain's simdgroup runs no load of its own in the loop: all 256 threads stream the weight through a register ring D
// stages deep into a threadgroup ring of R slots (a stage: the block's 32 experts by SK k, held as uint4 words so every
// 16-byte store is aligned), and x's partition slice sits in threadgroup memory. Phase 2 sums the partials as MLX's
// split-K accumulate does.
//
// Template: WT (the weight's type), M (rows, 2..4), N (experts, a multiple of 32), K (the hidden), PS (the partition's
// k, MLX's 2048). Grid: (N / 32) * (K / PS) threadgroups of 256 threads, the chain (block, partition) =
// (tg % NB, tg / NB).
using namespace mlx::steel;

constexpr int NB = N / 32;
constexpr int SK = 64;
constexpr int NST = PS / SK;
constexpr int D = 8;
constexpr int R = 3;
static_assert(N % 32 == 0 && K % PS == 0 && PS % SK == 0 && NST % D == 0, "router_p1: the shapes do not tile");
static_assert(M >= 1 && M <= 4, "router_p1: 1 to 4 rows");
static_assert(sizeof(WT) == 2, "router_p1: a 2-byte weight");

const int tgi = int(threadgroup_position_in_grid.x);
const int tid = int(thread_position_in_threadgroup.x);
const bool chain = simdgroup_index_in_threadgroup == 0;
const int n0 = (tgi % NB) * 32;
const int part = tgi / NB;
const int k0 = part * PS;

threadgroup bfloat16_t xs[M * PS];
// R slots of a stage: 32 experts by SK k of WT, as uint4 words (8 values a word)
threadgroup uint4 ring4[R * 32 * SK / 8];

// x's partition slice
for (int i = tid; i < M * PS; i += 256) {
  const int r = i / PS;
  xs[i] = x[r * K + k0 + (i - r * PS)];
}

// A stage is 32 experts by SK k, one uint4 word (8 values) a thread: expert tid / 8, k (tid % 8) * 8.
const int bn = tid >> 3;
const int bq = tid & 7;
const device uint4* src = (const device uint4*)(w + (size_t)(n0 + bn) * K + k0 + bq * 8);
threadgroup uint4* dst = ring4 + bn * (SK / 8) + bq;
constexpr int SRC_STAGE = SK / 8;
constexpr int DST_SLOT = 32 * SK / 8;

uint4 rb[D];
STEEL_PRAGMA_UNROLL
for (int d = 0; d < D; ++d) {
  rb[d] = src[d * SRC_STAGE];
}
dst[0] = rb[0];
rb[0] = src[D * SRC_STAGE];

NAXTile<float, 1, 2> Dtile;
Dtile.clear();
const short2 sc = BaseNAXFrag::get_coord();

for (int g = 0; g < NST; g += D) {
  STEEL_PRAGMA_UNROLL
  for (int d = 0; d < D; ++d) {
    const int i = g + d;
    // stage i + 1 into its slot, then stage i + 1 + D into the registers it leaves
    if (i + 1 < NST) {
      dst[((i + 1) % R) * DST_SLOT] = rb[(d + 1) % D];
      if (i + 1 + D < NST) {
        rb[(d + 1) % D] = src[(i + 1 + D) * SRC_STAGE];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (chain) {
      const threadgroup WT* bs = (const threadgroup WT*)(ring4 + (i % R) * DST_SLOT);
      STEEL_PRAGMA_UNROLL
      for (int s2 = 0; s2 < SK; s2 += 32) {
        NAXTile<float, 1, 2> Atile;
        NAXTile<float, 2, 2> Btile;
        STEEL_PRAGMA_UNROLL
        for (short kk = 0; kk < 2; ++kk) {
          const int kc = i * SK + s2 + kk * 16 + sc.x;
          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < 2; ++ii) {
            const int row = sc.y + ii * 8;
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < 4; ++j) {
              Atile.frag_at(0, kk)[ii * 4 + j] = row < M ? static_cast<float>(xs[row * PS + kc + j]) : float(0);
            }
          }
          STEEL_PRAGMA_UNROLL
          for (short nn = 0; nn < 2; ++nn) {
            STEEL_PRAGMA_UNROLL
            for (short ii = 0; ii < 2; ++ii) {
              const threadgroup WT* bp = bs + (nn * 16 + sc.y + ii * 8) * SK + s2 + kk * 16 + sc.x;
              STEEL_PRAGMA_UNROLL
              for (short j = 0; j < 4; ++j) {
                Btile.frag_at(nn, kk)[ii * 4 + j] = static_cast<float>(bp[j]);
              }
            }
          }
        }
        tile_matmad_nax(Dtile, Atile, metal::bool_constant<false>{}, Btile, metal::bool_constant<true>{});
      }
    }
  }
}

// the chain's partial, rows < M (MLX's store_safe)
if (chain) {
  device float* out = p + (size_t)part * M * N + n0;
  STEEL_PRAGMA_UNROLL
  for (short nn = 0; nn < 2; ++nn) {
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < 2; ++ii) {
      const int row = sc.y + ii * 8;
      if (row < M) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < 4; ++j) {
          out[row * N + nn * 16 + sc.x + j] = Dtile.frag_at(0, nn)[ii * 4 + j];
        }
      }
    }
  }
}
