// fuseDown: act = stab[bits(bf16(x . w^T))] and, with INJ, inj = gtab[bits(bf16(x . iw))] in one launch; the tables
// are the chain's own silu and gate of every bf16 pattern. The down chain is naxDown's; the inject chain runs in the wn 0
// simdgroups of the first column tile. Template: K, N, HC, INJ, TM, WM, WN, NP, PART, SWZ, BAR. `rows`: M.
  constexpr int TR = 16 * TM;
  constexpr int TPG = WM * WN;
  constexpr int BM = TR * WM;
  constexpr int BN = 32 * WN;
  constexpr int TILES_N = N / BN;
  constexpr int LDP = 36;
  threadgroup float partials[NP > 1 ? (NP - 1) * TPG * TR * LDP : 1];
  threadgroup float ipartials[(NP > 1 && INJ != 0) ? (NP - 1) * WM * TR * HC : 1];
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
  const int wm = t / WN;
  const int row0 = tile_y * BM + TR * wm;
  const int col0 = tile_x * BN + 32 * (t % WN);
  const short sgp_sm = short(clamp(M - row0, 0, TR));
  const int k0 = p * PART;
  const int steps = (min(k0 + PART, K) - k0) / 32;
  // One inject simdgroup per wm row block, in the first column tile.
  const bool inj_sg = INJ != 0 && tile_x == 0 && (t % WN) == 0;
  const device bfloat16_t* A = x + (size_t(row0) * K + k0);
  const device bfloat16_t* B = (const device bfloat16_t*)w + (size_t(col0) * K + k0);

  NAXTile<float, TM, 2> Dtile;
  Dtile.clear();
  NAXTile<float, TM, 2> Itile;
  Itile.clear();
  if (inj_sg) {
    const device bfloat16_t* BI = iw + size_t(k0) * HC;
    if (sgp_sm == TR) {
      hcpf_fuse_steps<TM, true, BAR, HC>(Dtile, Itile, A, K, B, K, BI, steps, sgp_sm);
    } else {
      hcpf_fuse_steps<TM, false, BAR, HC>(Dtile, Itile, A, K, B, K, BI, steps, sgp_sm);
    }
  } else if (sgp_sm == TR) {
    hcpf_nax_steps<TM, true, BAR>(Dtile, A, K, B, K, steps, sgp_sm);
  } else {
    hcpf_nax_steps<TM, false, BAR>(Dtile, A, K, B, K, steps, sgp_sm);
  }

  if constexpr (NP > 1) {
    // Partials p >= 1 park as f32 (exact, like MLX's float32 C_split); partition 0 adds them in order.
    const short2 sc = BaseNAXFrag::get_coord();
    if (p > 0) {
      Dtile.template store<float, LDP, 1>(&partials[((p - 1) * TPG + t) * TR * LDP]);
      if (inj_sg) {
        hcpf_inj_park<TM, HC>(Itile, &ipartials[((p - 1) * WM + wm) * TR * HC], sc);
      }
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
    if (inj_sg) {
      hcpf_inj_sum<TM, HC, NP>(Itile, &ipartials[wm * TR * HC], WM * TR * HC, sc);
    }
  }

  if (sgp_sm <= 0) {
    return;
  }
  {
    // act: the product rounded to bf16 as MLX's store rounds it, then the chain's silu of that bf16 value.
    NAXTile<bfloat16_t, TM, 2> Otile;
    const thread float* acc = Dtile.elems();
    thread bfloat16_t* o = Otile.elems();
    STEEL_PRAGMA_UNROLL
    for (short e = 0; e < 16 * TM; e++) {
      o[e] = stab[as_type<ushort>(static_cast<bfloat16_t>(acc[e]))];
    }
    device bfloat16_t* Y = act + (size_t(row0) * N + col0);
    if (sgp_sm == TR) {
      Otile.store(Y, N);
    } else {
      Otile.store_safe(Y, N, short2(32, sgp_sm));
    }
  }
  if (inj_sg) {
    // inj: the inject product rounded to bf16 (MLX's store), then the chain's gate; only columns < HC are stored.
    NAXTile<bfloat16_t, TM, 2> Gtile;
    const thread float* acc = Itile.elems();
    thread bfloat16_t* o = Gtile.elems();
    STEEL_PRAGMA_UNROLL
    for (short e = 0; e < 16 * TM; e++) {
      o[e] = gtab[as_type<ushort>(static_cast<bfloat16_t>(acc[e]))];
    }
    Gtile.store_safe(inj + size_t(row0) * HC, HC, short2(HC, sgp_sm));
  }
