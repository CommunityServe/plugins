// naxDown: y [M, N] = x [M, K] . w^T (w the [N, K] bf16 rows as u32 words), bit for bit MLX's NAX matmul: one chain
// over K (NP 1), or MLX's split-K partitions (NP > 1) parked as f32 and summed in partition order, rounded once.
// Template: K, N, TM, WM, WN, NP, PART, SWZ (MLX's threadblock swizzle), BAR (K steps per barrier, 0 none). `rows`: M.
  constexpr int TR = 16 * TM;
  constexpr int TPG = WM * WN;
  constexpr int BM = TR * WM;
  constexpr int BN = 32 * WN;
  constexpr int TILES_N = N / BN;
  constexpr int LDP = 36;
  threadgroup float partials[NP > 1 ? (NP - 1) * TPG * TR * LDP : 1];
  const int M = rows;
  const int tiles_m = (M + BM - 1) / BM;
  const int gx = int(threadgroup_position_in_grid.x);
  const int gy = int(threadgroup_position_in_grid.y);
  const int tile_y = (gy << SWZ) + (gx & ((1 << SWZ) - 1));
  const int tile_x = gx >> SWZ;
  if (tile_x >= TILES_N || tile_y >= tiles_m) {
    return;
  }
  const int sg = int(simdgroup_index_in_threadgroup);
  const int p = sg / TPG;
  const int t = sg % TPG;
  const int row0 = tile_y * BM + TR * (t / WN);
  const int col0 = tile_x * BN + 32 * (t % WN);
  const short sgp_sm = short(clamp(M - row0, 0, TR));
  const int k0 = p * PART;
  const int steps = (min(k0 + PART, K) - k0) / 32;
  const device bfloat16_t* A = x + (size_t(row0) * K + k0);
  const device bfloat16_t* B = (const device bfloat16_t*)w + (size_t(col0) * K + k0);

  NAXTile<float, TM, 2> Dtile;
  Dtile.clear();
  if (sgp_sm == TR) {
    hcpf_nax_steps<TM, true, BAR>(Dtile, A, K, B, K, steps, sgp_sm);
  } else {
    hcpf_nax_steps<TM, false, BAR>(Dtile, A, K, B, K, steps, sgp_sm);
  }

  if constexpr (NP > 1) {
    // Partials p >= 1 park as f32 (exact, like MLX's float32 C_split); partition 0 adds them in order.
    if (p > 0) {
      Dtile.template store<float, LDP, 1>(&partials[((p - 1) * TPG + t) * TR * LDP]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (p > 0) {
      return;
    }
    thread float* acc = Dtile.elems();
    STEEL_PRAGMA_UNROLL
    for (short e = 0; e < 16 * TM; e++) {
      acc[e] = 0.0f + acc[e];
    }
    for (int q = 1; q < NP; q++) {
      NAXTile<float, TM, 2> Ptile;
      Ptile.template load<float, LDP, 1>(&partials[((q - 1) * TPG + t) * TR * LDP]);
      const thread float* part = Ptile.elems();
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < 16 * TM; e++) {
        acc[e] = acc[e] + part[e];
      }
    }
  }

  if (sgp_sm <= 0) {
    return;
  }
  device bfloat16_t* Y = y + (size_t(row0) * N + col0);
  if (sgp_sm == TR) {
    Dtile.store(Y, N);
  } else {
    Dtile.store_safe(Y, N, short2(32, sgp_sm));
  }
