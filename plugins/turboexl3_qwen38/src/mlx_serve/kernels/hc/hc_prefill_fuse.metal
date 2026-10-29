// The fused read's helpers (hc_prefill_nax.zig), composed after hc_prefill_nax.metal. The inject product is MLX's NAX
// "nn" GEMM at N = HC (columns past HC load as zeros); MLX splits its K exactly where it splits the down product's.

/// `hcpf_nax_steps` (the down chain, MLX's "nt" kernel) with the inject chain (MLX's "nn" kernel of the same rows) on the
/// SAME A tile: the normalized rows are read once for both products. `BI` points at row k0 of the [K, HC] inject weight.
template <short TM, bool kAlignedM, int BAR, int HC>
METAL_FUNC void hcpf_fuse_steps(
    thread NAXTile<float, TM, 2>& Dtile,
    thread NAXTile<float, TM, 2>& Itile,
    const device bfloat16_t* A,
    const int lda,
    const device bfloat16_t* B,
    const int ldb,
    const device bfloat16_t* BI,
    const int steps,
    const short sgp_sm) {
  // Rows past M load zeros (MLX's load_safe limits); a fragment with no row inside M skips the math, as MLX's gemm_loop
  // does, but still meets every threadgroup barrier.
  const bool has_output = kAlignedM || sgp_sm > 0;
  STEEL_PRAGMA_NO_UNROLL
  for (int s = 0; s < steps; s++) {
    if constexpr (BAR > 0) {
      if (s % BAR == 0) {
        threadgroup_barrier(mem_flags::mem_none);
      }
    }
    if (has_output) {
      NAXTile<bfloat16_t, TM, 2> Atile;
      NAXTile<bfloat16_t, 2, 2> Btile;
      NAXTile<bfloat16_t, 2, 2> Ibtile;

      volatile int compiler_barrier;

      if constexpr (kAlignedM) {
        Atile.load(A, lda);
      } else {
        Atile.load_safe(A, lda, short2(32, sgp_sm));
      }
      Btile.load(B, ldb);
      // MLX's nn B tile at N = HC: rows k .. k + 31 of the [K, HC] weight, columns past HC zero.
      Ibtile.load_safe(BI, HC, short2(HC, 32));

      tile_matmad_nax(
          Dtile,
          Atile,
          metal::bool_constant<false>{},
          Btile,
          metal::bool_constant<true>{});
      tile_matmad_nax(
          Itile,
          Atile,
          metal::bool_constant<false>{},
          Ibtile,
          metal::bool_constant<false>{});

      (void)compiler_barrier;
    }
    A += 32;
    B += 32;
    BI += 32 * HC;
  }
}

/// Split mode: park this lane's live inject partials (columns < HC sit in fragment column 0; lane coordinates `sc` =
/// BaseNAXFrag::get_coord(): columns sc.x .. sc.x + 3, rows sc.y and sc.y + 8 of a 16 x 16 fragment) as [TR rows][HC] f32.
template <short TM, int HC>
METAL_FUNC void hcpf_inj_park(thread NAXTile<float, TM, 2>& I, threadgroup float* dst, const short2 sc) {
  STEEL_PRAGMA_UNROLL
  for (short mm = 0; mm < TM; mm++) {
    thread auto& f = I.frag_at(mm, 0);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 2; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < 4; j++) {
        const short c = sc.x + j;
        if (c < HC) {
          dst[(mm * 16 + sc.y + i * 8) * HC + c] = f[i * 4 + j];
        }
      }
    }
  }
}

/// Split mode, partition 0: MLX's accum order over the parked partials (`AccT out = 0; out += C_split[p]` for p in order):
/// 0 + P0, then + P1, + P2, ... `src` is partition 1's [TR][HC] block for this row block; partition q's sits (q - 1) *
/// `stride` floats further.
template <short TM, int HC, int NP>
METAL_FUNC void hcpf_inj_sum(thread NAXTile<float, TM, 2>& I, const threadgroup float* src, const int stride, const short2 sc) {
  STEEL_PRAGMA_UNROLL
  for (short mm = 0; mm < TM; mm++) {
    thread auto& f = I.frag_at(mm, 0);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 2; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < 4; j++) {
        const short c = sc.x + j;
        if (c < HC) {
          const int at = (mm * 16 + sc.y + i * 8) * HC + c;
          float a = 0.0f + f[i * 4 + j];
          for (int q = 1; q < NP; q++) {
            a = a + src[(q - 1) * stride + at];
          }
          f[i * 4 + j] = a;
        }
      }
    }
  }
}

/// hc_prefill_norm.metal's normalized value, its two roundings: bf16(f32(bf16(f32(x) * inv)) * f32(w)), x the (written)
/// stream value, inv the row's f32 scale exactly as that kernel computes it (metal::precise::rsqrt(sum / H + eps), the
/// pre-pass hc_prefill_norm_inv.metal stores it), w the norm weight.
METAL_FUNC bfloat16_t hcpf_norm(const bfloat16_t x, const float inv, const bfloat16_t w) {
  return static_cast<bfloat16_t>(float(static_cast<bfloat16_t>(float(x) * inv)) * float(w));
}
