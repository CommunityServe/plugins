// rwDown: fuseDownNorm with the inject chain in simdgroups of its own (readDown's assignment) and SD 32-deep steps a
// stage (one barrier a stage; a stage never crosses a stream), so act and inj are fuseDownNorm's.
// One chain over K. Template: K, N, H, HC, TM, WM, WN, SWZ, SD. `rows`: M.
  constexpr int TR = 16 * TM;
  constexpr int TPG = WM * WN;
  constexpr int BM = TR * WM;
  constexpr int BN = 32 * WN;
  constexpr int TILES_N = N / BN;
  constexpr int IPT = (WM + TILES_N - 1) / TILES_N;
  constexpr int TPT = 32 * (TPG + IPT);
  constexpr int SK = 32 * SD;
  constexpr int LDA = SK + 8;
  constexpr int RPR = SK / 8;
  constexpr int RUNS = BM * RPR;
  constexpr int RPT = (RUNS + TPT - 1) / TPT;
  constexpr int STAGES = K / SK;
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
  // A down simdgroup (sg < TPG): row block sg / WN, columns 32 (sg % WN) of this column tile. An inject simdgroup: row
  // block (sg - TPG) TILES_N + tile_x, none past WM.
  const bool inj_sg = sg >= TPG;
  const int wm = inj_sg ? (sg - TPG) * TILES_N + tile_x : sg / WN;
  const int brow = tile_y * BM;
  const int row0 = brow + TR * wm;
  const int col0 = inj_sg ? 0 : tile_x * BN + 32 * (sg % WN);
  const short sgp_sm = wm < WM ? short(clamp(M - row0, 0, TR)) : short(0);
  const device bfloat16_t* B = (const device bfloat16_t*)w + size_t(col0) * K;

  // Stage 0's runs.
  bfloat16_t xr[RPT][8];
  float ir[RPT];
  for (int q = 0; q < RPT; q++) {
    const int run = tid + q * TPT;
    const int grow = brow + run / RPR;
    ir[q] = 0.0f;
    if (run < RUNS && grow < M) {
      const device bfloat16_t* src = xs + (size_t(grow) * K + (run % RPR) * 8);
      for (int e = 0; e < 8; e++) {
        xr[q][e] = src[e];
      }
      ir[q] = invs[grow * HC];
    }
  }

  NAXTile<float, TM, 2> Dtile;
  Dtile.clear();
  for (int st = 0; st < STAGES; st++) {
    const int ks = st * SK;
    threadgroup bfloat16_t* buf = abuf + (st & 1) * (BM * LDA);
    for (int q = 0; q < RPT; q++) {
      const int run = tid + q * TPT;
      if (run < RUNS) {
        const int r = run / RPR;
        const int c8 = (run % RPR) * 8;
        threadgroup bfloat16_t* dst = buf + r * LDA + c8;
        if (brow + r < M) {
          for (int e = 0; e < 8; e++) {
            dst[e] = hcpf_norm(xr[q][e], ir[q], nw[ks + c8 + e]);
          }
        } else {
          for (int e = 0; e < 8; e++) {
            dst[e] = bfloat16_t(0.0f);
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (st + 1 < STAGES) {
      // The next stage's runs go out before this stage's math.
      const int kn = ks + SK;
      const int hn = kn / H;
      for (int q = 0; q < RPT; q++) {
        const int run = tid + q * TPT;
        const int grow = brow + run / RPR;
        if (run < RUNS && grow < M) {
          const device bfloat16_t* src = xs + (size_t(grow) * K + kn + (run % RPR) * 8);
          for (int e = 0; e < 8; e++) {
            xr[q][e] = src[e];
          }
          ir[q] = invs[grow * HC + hn];
        }
      }
    }
    if (sgp_sm > 0) {
      const threadgroup bfloat16_t* ablk = buf + TR * wm * LDA;
      for (int t = 0; t < SD; t++) {
        const int kb = ks + 32 * t;
        NAXTile<bfloat16_t, TM, 2> Atile;

        volatile int compiler_barrier;

        Atile.template load<bfloat16_t, LDA, 1>(ablk + 32 * t);
        if (inj_sg) {
          // MLX's nn B tile at N = HC: rows kb .. kb + 31 of the [K, HC] weight, columns past HC zero.
          NAXTile<bfloat16_t, 2, 2> Ibtile;
          Ibtile.load_safe(iw + size_t(kb) * HC, HC, short2(HC, 32));
          tile_matmad_nax(
              Dtile,
              Atile,
              metal::bool_constant<false>{},
              Ibtile,
              metal::bool_constant<false>{});
        } else {
          NAXTile<bfloat16_t, 2, 2> Btile;
          Btile.load(B + kb, K);
          tile_matmad_nax(
              Dtile,
              Atile,
              metal::bool_constant<false>{},
              Btile,
              metal::bool_constant<true>{});
        }

        (void)compiler_barrier;
      }
    }
  }

  if (sgp_sm <= 0) {
    return;
  }
  if (inj_sg) {
    // inj: the inject product rounded to bf16 (MLX's store), then the chain's gate; only columns < HC are stored.
    NAXTile<bfloat16_t, TM, 2> Gtile;
    const thread float* acc = Dtile.elems();
    thread bfloat16_t* o = Gtile.elems();
    STEEL_PRAGMA_UNROLL
    for (short e = 0; e < 16 * TM; e++) {
      o[e] = gtab[as_type<ushort>(static_cast<bfloat16_t>(acc[e]))];
    }
    Gtile.store_safe(inj + size_t(row0) * HC, HC, short2(HC, sgp_sm));
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
