// fuseDownNorm: fuseDown with the norm in its A loader (hcpf_norm of the written stream `xs`, the row scales `invs` and
// the weight `nw`, staged in threadgroup memory a 32-deep step at a time), so the normalized rows are never written.
// One chain over K only. Template: K, N, H, HC, INJ, TM, WM, WN, SWZ. `rows`: M.
  constexpr int TR = 16 * TM;
  constexpr int TPG = WM * WN;
  constexpr int BM = TR * WM;
  constexpr int BN = 32 * WN;
  constexpr int TILES_N = N / BN;
  constexpr int TPT = 32 * TPG;
  constexpr int LDA = 40;
  constexpr int RUNS = BM * 4;
  constexpr int RPT = (RUNS + TPT - 1) / TPT;
  constexpr int STEPS = K / 32;
  threadgroup bfloat16_t abuf[2 * BM * LDA];
  const int M = rows;
  const int tiles_m = (M + BM - 1) / BM;
  const int gx = int(threadgroup_position_in_grid.x);
  const int gy = int(threadgroup_position_in_grid.y);
  const int tile_y = (gy << SWZ) + (gx & ((1 << SWZ) - 1));
  const int tile_x = gx >> SWZ;
  if (tile_x >= TILES_N || tile_y >= tiles_m) {
    return;
  }
  const int tid = int(thread_index_in_threadgroup);
  const int sg = int(simdgroup_index_in_threadgroup);
  const int wm = sg / WN;
  const int brow = tile_y * BM;
  const int row0 = brow + TR * wm;
  const int col0 = tile_x * BN + 32 * (sg % WN);
  const short sgp_sm = short(clamp(M - row0, 0, TR));
  const bool inj_sg = INJ != 0 && tile_x == 0 && (sg % WN) == 0;
  const device bfloat16_t* B = (const device bfloat16_t*)w + size_t(col0) * K;

  // Step 0's runs.
  bfloat16_t xr[RPT][8];
  float ir[RPT];
  for (int q = 0; q < RPT; q++) {
    const int run = tid + q * TPT;
    const int grow = brow + (run >> 2);
    ir[q] = 0.0f;
    if (run < RUNS && grow < M) {
      const device bfloat16_t* src = xs + (size_t(grow) * K + (run & 3) * 8);
      for (int e = 0; e < 8; e++) {
        xr[q][e] = src[e];
      }
      ir[q] = invs[grow * HC];
    }
  }

  NAXTile<float, TM, 2> Dtile;
  Dtile.clear();
  NAXTile<float, TM, 2> Itile;
  Itile.clear();
  for (int s = 0; s < STEPS; s++) {
    const int kb = s * 32;
    threadgroup bfloat16_t* buf = abuf + (s & 1) * (BM * LDA);
    for (int q = 0; q < RPT; q++) {
      const int run = tid + q * TPT;
      if (run < RUNS) {
        const int r = run >> 2;
        const int c8 = (run & 3) * 8;
        threadgroup bfloat16_t* dst = buf + r * LDA + c8;
        if (brow + r < M) {
          for (int e = 0; e < 8; e++) {
            dst[e] = hcpf_norm(xr[q][e], ir[q], nw[kb + c8 + e]);
          }
        } else {
          for (int e = 0; e < 8; e++) {
            dst[e] = bfloat16_t(0.0f);
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (s + 1 < STEPS) {
      // The next step's runs go out before this step's math (a 32-deep step never crosses a stream: H % 32 == 0).
      const int kn = kb + 32;
      const int hn = kn / H;
      for (int q = 0; q < RPT; q++) {
        const int run = tid + q * TPT;
        const int grow = brow + (run >> 2);
        if (run < RUNS && grow < M) {
          const device bfloat16_t* src = xs + (size_t(grow) * K + kn + (run & 3) * 8);
          for (int e = 0; e < 8; e++) {
            xr[q][e] = src[e];
          }
          ir[q] = invs[grow * HC + hn];
        }
      }
    }
    if (sgp_sm > 0) {
      NAXTile<bfloat16_t, TM, 2> Atile;
      NAXTile<bfloat16_t, 2, 2> Btile;

      volatile int compiler_barrier;

      Atile.template load<bfloat16_t, LDA, 1>(buf + TR * wm * LDA);
      Btile.load(B + kb, K);
      tile_matmad_nax(
          Dtile,
          Atile,
          metal::bool_constant<false>{},
          Btile,
          metal::bool_constant<true>{});
      if (inj_sg) {
        NAXTile<bfloat16_t, 2, 2> Ibtile;
        Ibtile.load_safe(iw + size_t(kb) * HC, HC, short2(HC, 32));
        tile_matmad_nax(
            Itile,
            Atile,
            metal::bool_constant<false>{},
            Ibtile,
            metal::bool_constant<false>{});
      }

      (void)compiler_barrier;
    }
  }

  if (sgp_sm <= 0) {
    return;
  }
  {
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
    NAXTile<bfloat16_t, TM, 2> Gtile;
    const thread float* acc = Itile.elems();
    thread bfloat16_t* o = Gtile.elems();
    STEEL_PRAGMA_UNROLL
    for (short e = 0; e < 16 * TM; e++) {
      o[e] = gtab[as_type<ushort>(static_cast<bfloat16_t>(acc[e]))];
    }
    Gtile.store_safe(inj + size_t(row0) * HC, HC, short2(HC, sgp_sm));
  }
