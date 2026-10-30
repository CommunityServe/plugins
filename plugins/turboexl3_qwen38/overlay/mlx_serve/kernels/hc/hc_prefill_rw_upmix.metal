// rwUpMix: readUpMix's register form with upMixNorm's normalized values (hcpf_norm of `xs`, `invs`, `nw`); a lane computes
// all its outputs before its first store. PF 1 reads its stream values and row scales before the K loop.
// Template: T, K (the rank R), H, HC, TM, WM, WN, SWZ, BAR, PF. `rows`: M.
  constexpr int TR = 16 * TM;
  constexpr int BM = TR * WM;
  constexpr int BN = 32 * WN;
  constexpr int TILES_N = H / BN;
  constexpr int NXV = PF != 0 ? TM * 2 * HC * 8 : 1;
  constexpr int NIV = PF != 0 ? TM * 2 * HC : 1;
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
  const short2 sc = BaseNAXFrag::get_coord();

  // PF: this lane's stream values xv[((2 mm + i) HC + s) 8 + 4 nn + j] and row scales iv[(2 mm + i) HC + s]; rows past
  // M read nothing.
  T xv[NXV];
  float iv[NIV];
  if constexpr (PF != 0) {
    STEEL_PRAGMA_UNROLL
    for (short mm = 0; mm < TM; mm++) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < 2; i++) {
        const int r = 16 * mm + sc.y + 8 * i;
        const size_t grow = size_t(row0 + r);
        STEEL_PRAGMA_UNROLL
        for (short s = 0; s < HC; s++) {
          const int ri = (2 * mm + i) * HC + s;
          iv[ri] = r < sgp_sm ? invs[grow * HC + s] : 0.0f;
          STEEL_PRAGMA_UNROLL
          for (short nn = 0; nn < 2; nn++) {
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < 4; j++) {
              xv[ri * 8 + 4 * nn + j] = r < sgp_sm ? xs[(grow * HC + s) * H + c0 + 16 * nn + sc.x + j] : T(0);
            }
          }
        }
      }
    }
  }

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

  // hc_prefill_mix.metal, per (row, column): streams in order, bf16 after every product and every sum; each stream's
  // normalized value hcpf_norm's.
  const T inv = T(1.0f / float(HC));
  T res[16 * TM];
  STEEL_PRAGMA_UNROLL
  for (short mm = 0; mm < TM; mm++) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 2; i++) {
      const int r = 16 * mm + sc.y + 8 * i;
      if (r >= sgp_sm) {
        continue;
      }
      const size_t grow = size_t(row0 + r);
      float ivr[HC];
      STEEL_PRAGMA_UNROLL
      for (short s = 0; s < HC; s++) {
        if constexpr (PF != 0) {
          ivr[s] = iv[(2 * mm + i) * HC + s];
        } else {
          ivr[s] = invs[grow * HC + s];
        }
      }
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
            T xval;
            if constexpr (PF != 0) {
              xval = xv[((2 * mm + i) * HC + s) * 8 + 4 * nn + j];
            } else {
              xval = xs[(grow * HC + s) * H + col];
            }
            const T nrm = hcpf_norm(xval, ivr[s], nw[s * H + col]);
            const T product = T(float(sig) * float(nrm));
            value = s == 0 ? product : T(float(product) + float(value));
          }
          res[e] = T(float(value) * float(inv));
        }
      }
    }
  }
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
          out[grow * H + c0 + 16 * nn + sc.x + j] = res[8 * (2 * mm + nn) + 4 * i + j];
        }
      }
    }
  }
