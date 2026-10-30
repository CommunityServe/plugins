// The rearranged read's helpers (hc_prefill_nax.zig), composed after hc_prefill_fuse.metal: chains that already exist,
// element for element (hcpf_fuse_steps' inject half; hcpf_nax_steps for HC B tiles sharing one A tile).

/// hcpf_fuse_steps' inject chain alone (MLX's nn kernel at N = HC: rows k .. k + 31 of the [K, HC] weight, columns past
/// HC zero) on A tiles this simdgroup loads itself, the same rows; rows past M load zeros and still meet every barrier.
template <short TM, bool kAlignedM, int BAR, int HC>
METAL_FUNC void hcpf_inj_steps(
    thread NAXTile<float, TM, 2>& Itile,
    const device bfloat16_t* A,
    const int lda,
    const device bfloat16_t* BI,
    const int steps,
    const short sgp_sm) {
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
      NAXTile<bfloat16_t, 2, 2> Ibtile;

      volatile int compiler_barrier;

      if constexpr (kAlignedM) {
        Atile.load(A, lda);
      } else {
        Atile.load_safe(A, lda, short2(32, sgp_sm));
      }
      // MLX's nn B tile at N = HC: rows k .. k + 31 of the [K, HC] weight, columns past HC zero.
      Ibtile.load_safe(BI, HC, short2(HC, 32));

      tile_matmad_nax(
          Itile,
          Atile,
          metal::bool_constant<false>{},
          Ibtile,
          metal::bool_constant<false>{});

      (void)compiler_barrier;
    }
    A += 32;
    BI += 32 * HC;
  }
}

/// hcpf_nax_steps (MLX's "nt" chain) for HC accumulators at once: per 32-deep step one A tile, then stream h's B tile
/// (`B + h bstride`) into D[h]. Each D[h] element's chain is naxUpMix's simdgroup (wm, h)'s; the chains only share A.
template <short TM, bool kAlignedM, int BAR, int HC>
METAL_FUNC void hcpf_upmix_steps(
    thread NAXTile<float, TM, 2>* D,
    const device bfloat16_t* A,
    const int lda,
    const device bfloat16_t* B,
    const int ldb,
    const size_t bstride,
    const int steps,
    const short sgp_sm) {
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

      volatile int compiler_barrier;

      if constexpr (kAlignedM) {
        Atile.load(A, lda);
      } else {
        Atile.load_safe(A, lda, short2(32, sgp_sm));
      }
      STEEL_PRAGMA_UNROLL
      for (short h = 0; h < HC; h++) {
        NAXTile<bfloat16_t, 2, 2> Btile;
        Btile.load(B + h * bstride, ldb);
        tile_matmad_nax(
            D[h],
            Atile,
            metal::bool_constant<false>{},
            Btile,
            metal::bool_constant<true>{});
      }

      (void)compiler_barrier;
    }
    A += 32;
    B += 32;
  }
}
