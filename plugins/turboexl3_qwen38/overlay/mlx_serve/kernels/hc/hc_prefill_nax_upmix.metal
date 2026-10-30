// naxUpMix: the up product act [M, K] . w^T (w the [HC H, K] bf16 rows) with hc_prefill_mix.metal as its epilogue; each
// stream's up tile is rounded to bf16 into threadgroup memory as MLX's store rounds it, then mixed as the mix kernel.
// Template: T, K (the rank R), H, HC (divides 16 TM), TM, WM, SWZ, BAR, PF (normed values read before the K loop).
  constexpr int TR = 16 * TM;
  constexpr int BM = TR * WM;
  constexpr int TILES_N = H / 32;
  constexpr int LDT = 40;
  constexpr int RPS = TR / HC;
  threadgroup bfloat16_t upt[WM * HC * TR * LDT];
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
  const int lane = int(thread_index_in_simdgroup);
  const int wm = sg / HC;
  const int h = sg % HC;
  const int row0 = tile_y * BM + TR * wm;
  const int c0 = tile_x * 32;
  const short sgp_sm = short(clamp(M - row0, 0, TR));
  const int er0 = RPS * h;

  T nv[PF != 0 ? RPS * HC : 1];
  if constexpr (PF != 0) {
    for (int i = 0; i < RPS; i++) {
      const int grow = row0 + er0 + i;
      for (int s = 0; s < HC; s++) {
        nv[i * HC + s] = grow < M ? normed[(size_t(grow) * HC + s) * H + c0 + lane] : T(0);
      }
    }
  }

  NAXTile<float, TM, 2> Dtile;
  Dtile.clear();
  const device bfloat16_t* A = act + size_t(row0) * K;
  const device bfloat16_t* B = (const device bfloat16_t*)w + (size_t(h) * H + c0) * K;
  if (sgp_sm == TR) {
    hcpf_nax_steps<TM, true, BAR>(Dtile, A, K, B, K, K / 32, sgp_sm);
  } else {
    hcpf_nax_steps<TM, false, BAR>(Dtile, A, K, B, K, K / 32, sgp_sm);
  }
  Dtile.template store<bfloat16_t, LDT, 1>(&upt[(wm * HC + h) * TR * LDT]);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // hc_prefill_mix.metal, per (row, column): streams in order, bf16 after every product and every sum.
  const T inv = T(1.0f / float(HC));
  const threadgroup bfloat16_t* blk = &upt[wm * HC * TR * LDT];
  for (int i = 0; i < RPS; i++) {
    const int r = er0 + i;
    const int grow = row0 + r;
    if (grow >= M) {
      break;
    }
    T value = T(0);
    for (int s = 0; s < HC; s++) {
      const T u = blk[(s * TR + r) * LDT + lane];
      const T sig = sigtab[as_type<ushort>(u)];
      T nrm;
      if constexpr (PF != 0) {
        nrm = nv[i * HC + s];
      } else {
        nrm = normed[(size_t(grow) * HC + s) * H + c0 + lane];
      }
      const T product = T(float(sig) * float(nrm));
      value = s == 0 ? product : T(float(product) + float(value));
    }
    out[size_t(grow) * H + c0 + lane] = T(float(value) * float(inv));
  }
