// The hc read's NAX GEMM chain (hc_prefill_nax.zig), composed after MLX's steel NAX headers: per 32-deep K step one
// NAXTile of A and of B and tile_matmad_nax's two 16x32x16 multiply-accumulates in ascending K from a zero f32
// accumulator, MLX's "nt" chain element for element; TM 16-row fragments a simdgroup only change how B tiles are shared.
template <short TM, bool kAlignedM, int BAR>
METAL_FUNC void hcpf_nax_steps(
    thread NAXTile<float, TM, 2>& Dtile,
    const device bfloat16_t* A,
    const int lda,
    const device bfloat16_t* B,
    const int ldb,
    const int steps,
    const short sgp_sm) {
  // Rows past M load zeros (MLX's load_safe limits); a fragment with no row inside M skips the math, as MLX's
  // gemm_loop does, but still meets every threadgroup barrier.
  const bool has_output = kAlignedM || sgp_sm > 0;
  STEEL_PRAGMA_NO_UNROLL
  for (int s = 0; s < steps; s++) {
    if constexpr (BAR > 0) {
      // MLX's per-BK `threadgroup_barrier(mem_flags::mem_none)` (keeps the simdgroups sharing A/B rows in step).
      if (s % BAR == 0) {
        threadgroup_barrier(mem_flags::mem_none);
      }
    }
    if (has_output) {
      NAXTile<bfloat16_t, TM, 2> Atile;
      NAXTile<bfloat16_t, 2, 2> Btile;

      volatile int compiler_barrier;

      if constexpr (kAlignedM) {
        Atile.load(A, lda);
      } else {
        Atile.load_safe(A, lda, short2(32, sgp_sm));
      }
      Btile.load(B, ldb);

      tile_matmad_nax(
          Dtile,
          Atile,
          metal::bool_constant<false>{},
          Btile,
          metal::bool_constant<true>{});

      (void)compiler_barrier;
    }
    A += 32;
    B += 32;
  }
}
