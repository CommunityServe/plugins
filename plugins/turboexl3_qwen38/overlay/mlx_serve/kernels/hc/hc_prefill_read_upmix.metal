// readUpMix: naxUpMix with one simdgroup holding every stream's up tile for its rows in registers, mixed on the
// accumulators (fragment (mm, nn) holds row 16 mm + sc.y + 8 i, column 16 nn + sc.x + j at elems()[8 (2 mm + nn) + 4 i + j]).
// Template: T, K (the rank R), H, HC, TM, WM, WN, SWZ, BAR. `rows`: M.
  constexpr int TR = 16 * TM;
  constexpr int BM = TR * WM;
  constexpr int BN = 32 * WN;
  constexpr int TILES_N = H / BN;
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
  const int row0 = tile_y * BM + TR * (sg / WN);
  const int c0 = tile_x * BN + 32 * (sg % WN);
  const short sgp_sm = short(clamp(M - row0, 0, TR));

  NAXTile<float, TM, 2> D[HC];
  STEEL_PRAGMA_UNROLL
  for (short s = 0; s < HC; s++) {
    D[s].clear();
  }
  const device bfloat16_t* A = act + size_t(row0) * K;
  const device bfloat16_t* B = (const device bfloat16_t*)w + size_t(c0) * K;
  if (sgp_sm == TR) {
    hcpf_upmix_steps<TM, true, BAR, HC>(D, A, K, B, K, size_t(H) * K, K / 32, sgp_sm);
  } else {
    hcpf_upmix_steps<TM, false, BAR, HC>(D, A, K, B, K, size_t(H) * K, K / 32, sgp_sm);
  }
  if (sgp_sm <= 0) {
    return;
  }

  // hc_prefill_mix.metal, per (row, column): streams in order, bf16 after every product and every sum.
  const T inv = T(1.0f / float(HC));
  const short2 sc = BaseNAXFrag::get_coord();
  STEEL_PRAGMA_UNROLL
  for (short mm = 0; mm < TM; mm++) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 2; i++) {
      const int r = 16 * mm + sc.y + 8 * i;
      if (r >= sgp_sm) {
        continue;
      }
      const size_t grow = size_t(row0 + r);
      STEEL_PRAGMA_UNROLL
      for (short nn = 0; nn < 2; nn++) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < 4; j++) {
          const int col = c0 + 16 * nn + sc.x + j;
          const int e = 8 * (2 * mm + nn) + 4 * i + j;
          T value = T(0);
          STEEL_PRAGMA_UNROLL
          for (short s = 0; s < HC; s++) {
            const T u = static_cast<T>(D[s].elems()[e]);
            const T sig = sigtab[as_type<ushort>(u)];
            const T nrm = normed[(grow * HC + s) * H + col];
            const T product = T(float(sig) * float(nrm));
            value = s == 0 ? product : T(float(product) + float(value));
          }
          out[grow * H + col] = T(float(value) * float(inv));
        }
      }
    }
  }
