const uint t = thread_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const int s = int(threadgroup_position_in_grid.y) % HC;
const int g = int(threadgroup_position_in_grid.y) / HC;
const int r0 = int(threadgroup_position_in_grid.z) * RBD;
const int nr = min(RBD, ROWS - r0);
constexpr int SGD = 8;
constexpr int PER = H / 256;
constexpr int K = HC * H;
constexpr int KW = K / 2;
constexpr int NSLOT = H / 16;
constexpr int SITER = NSLOT / 32;
constexpr int PR = R + HC;
constexpr int XQ = H / 8;
constexpr int SQ = NSLOT / 4;
constexpr int NDG = R / (SGD * DRP);
threadgroup uint4 xs4[RBD * XQ];
threadgroup float4 xsm4[RBD * SQ];
threadgroup float npart[RBD * 8];
threadgroup float tgi[8 * HC];
threadgroup T* xs = (threadgroup T*)xs4;
const int base = s * H;
const bool down = g < NDG;
const bool writer = g == (INJ != 0 ? NDG : 0);
const int n0 = g * (SGD * DRP) + int(sg);

// LF 1: the norm's operands for every row of the block, then the norm weight, then this simdgroup's first down row;
// LF 0: the weights first (the two-launch read's order).
uint4 wq[SITER][2];
if (LF == 0 && down) {
  for (int i = 0; i < SITER; ++i) {
    const int slot = i * 32 + int(lane);
    const device uint4* wp = (const device uint4*)(dw_q + (size_t)n0 * KW + (size_t)(base / 2 + slot * 8));
    wq[i][0] = wp[0];
    wq[i][1] = wp[1];
  }
}
T xin[RBD][PER];
T woin[RBD][PER];
float gate[RBD];
for (int rr = 0; rr < RBD; ++rr) {
  gate[rr] = 0.0f;
  if (rr < nr) {
    const size_t xrow = (size_t)(r0 + rr) * K;
    for (int i = 0; i < PER; ++i) xin[rr][i] = x_in[xrow + base + int(t) + 256 * i];
    if (WR) {
      gate[rr] = float(wi_in[(r0 + rr) * HC + s]);
      for (int i = 0; i < PER; ++i) woin[rr][i] = wo_in[(size_t)(r0 + rr) * H + int(t) + 256 * i];
    }
  }
}
float wv[PER];
for (int i = 0; i < PER; ++i) wv[i] = float(nw[base + int(t) + 256 * i]);
if (LF != 0 && down) {
  for (int i = 0; i < SITER; ++i) {
    const int slot = i * 32 + int(lane);
    const device uint4* wp = (const device uint4*)(dw_q + (size_t)n0 * KW + (size_t)(base / 2 + slot * 8));
    wq[i][0] = wp[0];
    wq[i][1] = wp[1];
  }
}

// Pending hcWrite (stream' = T(stream + T(out * inj))) and the sums of squares, ND_SOURCE's per row, behind one barrier.
T xw[RBD][PER];
for (int rr = 0; rr < RBD; ++rr) {
  if (rr < nr) {
    float a = 0.0f;
    for (int i = 0; i < PER; ++i) {
      const T v = WR ? T(float(xin[rr][i]) + float(T(float(woin[rr][i]) * gate[rr]))) : xin[rr][i];
      xw[rr][i] = v;
      const float xv = float(v);
      a += xv * xv;
    }
    a = simd_sum(a);
    if (lane == 0) npart[rr * 8 + int(sg)] = a;
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (int rr = 0; rr < RBD; ++rr) {
  if (rr < nr) {
    float tot = 0.0f;
    for (int q = 0; q < 8; ++q) tot += npart[rr * 8 + q];
    const float rsh = rsqrt(tot / float(H) + eps[0]);
    for (int i = 0; i < PER; ++i) xs[rr * H + int(t) + 256 * i] = T(float(T(float(xw[rr][i]) * rsh)) * wv[i]);
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (INJ == 0 && writer) {
  for (int rr = 0; rr < RBD; ++rr) {
    if (rr < nr) {
      const size_t xrow = (size_t)(r0 + rr) * K;
      for (int i = 0; i < PER; ++i) {
        xn_out[xrow + base + int(t) + 256 * i] = xs[rr * H + int(t) + 256 * i];
        if (WR) xs_out[xrow + base + int(t) + 256 * i] = xw[rr][i];
      }
    }
  }
}
// The slot sums, once per threadgroup: slot n's 16 normalized values in element order (ND_SOURCE's xsum).
if (down) {
  for (int e = int(t); e < nr * NSLOT; e += 256) {
    const int rr = e / NSLOT;
    const int slot = e - rr * NSLOT;
    const uint4 v0 = xs4[rr * XQ + slot * 2];
    const uint4 v1 = xs4[rr * XQ + slot * 2 + 1];
    float xsum = 0.0f;
    for (int w = 0; w < 4; ++w) {
      xsum += hc_val<T, true>(v0[w] & 0xFFFFu);
      xsum += hc_val<T, true>((v0[w] >> 16) & 0xFFFFu);
    }
    for (int w = 0; w < 4; ++w) {
      xsum += hc_val<T, true>(v1[w] & 0xFFFFu);
      xsum += hc_val<T, true>((v1[w] >> 16) & 0xFFFFu);
    }
    ((threadgroup float*)xsm4)[e] = xsum;
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (down) {
const float ps = 1.0f;
const float pb = 0.0f;
const threadgroup float* xsmf = (const threadgroup float*)xsm4;
float res[DRP][RBD];
for (int p = 0; p < DRP; ++p) {
  float acc[RBD];
  for (int rr = 0; rr < RBD; ++rr) acc[rr] = 0.0f;
  for (int i = 0; i < SITER; ++i) {
    const int slot = i * 32 + int(lane);
    // Element e of the slot is half e % 2 of word e / 2 (little-endian bf16 pairs), as ND_SOURCE unpacks it.
    float wf[16];
    for (int w = 0; w < 4; ++w) {
      wf[2 * w] = hc_val<T, true>(wq[i][0][w] & 0xFFFFu);
      wf[2 * w + 1] = hc_val<T, true>((wq[i][0][w] >> 16) & 0xFFFFu);
      wf[8 + 2 * w] = hc_val<T, true>(wq[i][1][w] & 0xFFFFu);
      wf[8 + 2 * w + 1] = hc_val<T, true>((wq[i][1][w] >> 16) & 0xFFFFu);
    }
    if (p + 1 < DRP) {
      const device uint4* wp = (const device uint4*)(dw_q + (size_t)(n0 + (p + 1) * SGD) * KW + (size_t)(base / 2 + slot * 8));
      wq[i][0] = wp[0];
      wq[i][1] = wp[1];
    }
    for (int rr = 0; rr < RBD; ++rr) {
      if (rr < nr) {
        const uint4 v0 = xs4[rr * XQ + slot * 2];
        const uint4 v1 = xs4[rr * XQ + slot * 2 + 1];
        float xr[16];
        for (int w = 0; w < 4; ++w) {
          xr[2 * w] = hc_val<T, true>(v0[w] & 0xFFFFu);
          xr[2 * w + 1] = hc_val<T, true>((v0[w] >> 16) & 0xFFFFu);
          xr[8 + 2 * w] = hc_val<T, true>(v1[w] & 0xFFFFu);
          xr[8 + 2 * w + 1] = hc_val<T, true>((v1[w] >> 16) & 0xFFFFu);
        }
        const float xsum = xsmf[rr * NSLOT + slot];
        float d = 0.0f;
        for (int e = 0; e < 16; ++e) d += xr[e] * wf[e];
        acc[rr] += ps * d + pb * xsum;
      }
    }
  }
  for (int rr = 0; rr < RBD; ++rr) res[p][rr] = simd_sum(acc[rr]);
}
  if (lane == 0) {
    for (int p = 0; p < DRP; ++p) {
      for (int rr = 0; rr < RBD; ++rr) {
        if (rr < nr) parts_out[(size_t)((r0 + rr) * HC + s) * PR + n0 + p * SGD] = res[p][rr];
      }
    }
  }
} else if (INJ != 0) {
  // The inject threadgroup: its inject weights load before its stores (a later load would wait on them).
  T iwv[PER][HC];
  for (int i = 0; i < PER; ++i) {
    const size_t k = (size_t)(base + int(t) + 256 * i);
    for (int c = 0; c < HC; ++c) iwv[i][c] = iw[k * HC + c];
  }
  for (int rr = 0; rr < RBD; ++rr) {
    if (rr < nr) {
      const size_t xrow = (size_t)(r0 + rr) * K;
      for (int i = 0; i < PER; ++i) {
        xn_out[xrow + base + int(t) + 256 * i] = xs[rr * H + int(t) + 256 * i];
        if (WR) xs_out[xrow + base + int(t) + 256 * i] = xw[rr][i];
      }
    }
  }
  // This stream's partial of the inject matvec per row (NDM_SOURCE's); the up launch adds the streams.
  for (int rr = 0; rr < RBD; ++rr) {
    if (rr < nr) {
      float ip[HC];
      for (int c = 0; c < HC; ++c) ip[c] = 0.0f;
      for (int i = 0; i < PER; ++i) {
        for (int c = 0; c < HC; ++c) ip[c] += float(xs[rr * H + int(t) + 256 * i]) * float(iwv[i][c]);
      }
      for (int c = 0; c < HC; ++c) {
        const float pa = simd_sum(ip[c]);
        if (lane == 0) tgi[sg * HC + c] = pa;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (t < uint(HC)) {
        float tt = 0.0f;
        for (int q = 0; q < 8; ++q) tt += tgi[q * HC + t];
        parts_out[(size_t)((r0 + rr) * HC + s) * PR + R + t] = tt;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
}