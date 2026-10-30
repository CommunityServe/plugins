// readDown: fuseDown with the inject chain in simdgroups of its own (IPT a threadgroup; inject simdgroup j of column tile
// tile_x runs row block j TILES_N + tile_x, idle past WM), so every simdgroup runs one chain; act and inj are fuseDown's.
// One chain over K. Template: K, N, HC, TM, WM, WN, SWZ, BAR. `rows`: M.
  constexpr int TR = 16 * TM;
  constexpr int TPG = WM * WN;
  constexpr int BM = TR * WM;
  constexpr int BN = 32 * WN;
  constexpr int TILES_N = N / BN;
  constexpr int STEPS = K / 32;
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

  if (sg >= TPG) {
    // An inject simdgroup: row block wm of this row tile, or none past WM.
    const int wm = (sg - TPG) * TILES_N + tile_x;
    const int row0 = tile_y * BM + TR * wm;
    const short sgp_sm = wm < WM ? short(clamp(M - row0, 0, TR)) : short(0);
    const device bfloat16_t* A = x + size_t(row0) * K;
    NAXTile<float, TM, 2> Itile;
    Itile.clear();
    if (sgp_sm == TR) {
      hcpf_inj_steps<TM, true, BAR, HC>(Itile, A, K, iw, STEPS, sgp_sm);
    } else {
      hcpf_inj_steps<TM, false, BAR, HC>(Itile, A, K, iw, STEPS, sgp_sm);
    }
    if (sgp_sm <= 0) {
      return;
    }
    // inj: the inject product rounded to bf16 (MLX's store), then the chain's gate; only columns < HC are stored.
    NAXTile<bfloat16_t, TM, 2> Gtile;
    const thread float* acc = Itile.elems();
    thread bfloat16_t* o = Gtile.elems();
    STEEL_PRAGMA_UNROLL
    for (short e = 0; e < 16 * TM; e++) {
      o[e] = gtab[as_type<ushort>(static_cast<bfloat16_t>(acc[e]))];
    }
    Gtile.store_safe(inj + size_t(row0) * HC, HC, short2(HC, sgp_sm));
    return;
  }

  const int wm = sg / WN;
  const int row0 = tile_y * BM + TR * wm;
  const int col0 = tile_x * BN + 32 * (sg % WN);
  const short sgp_sm = short(clamp(M - row0, 0, TR));
  const device bfloat16_t* A = x + size_t(row0) * K;
  const device bfloat16_t* B = (const device bfloat16_t*)w + size_t(col0) * K;

  NAXTile<float, TM, 2> Dtile;
  Dtile.clear();
  if (sgp_sm == TR) {
    hcpf_nax_steps<TM, true, BAR>(Dtile, A, K, B, K, STEPS, sgp_sm);
  } else {
    hcpf_nax_steps<TM, false, BAR>(Dtile, A, K, B, K, STEPS, sgp_sm);
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
