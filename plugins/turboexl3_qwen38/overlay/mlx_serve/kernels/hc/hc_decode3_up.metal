const uint t = thread_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const int r0 = int(threadgroup_position_in_grid.z) * RBU;
const int nr = min(RBU, ROWS - r0);
const int j = int(threadgroup_position_in_grid.y) * SGU + int(sg);
constexpr int R_by_p = R / 2;
constexpr int RIT = R_by_p / 32;
constexpr int PR = R + HC;
constexpr int TPT = 32 * SGU;
constexpr int AIT = (RBU * R + TPT - 1) / TPT;
// at most four of a thread's activation steps keep their stream partials in registers ahead of the weights
constexpr int AIT0 = AIT < 4 ? AIT : 4;
threadgroup float2 act2[RBU * R / 2];
threadgroup float* actf = (threadgroup float*)act2;

// LF 1: the stream partials (and this threadgroup's inject partials) gate the barrier, their loads go out first;
// LF 0: the weights first (U2M's order). Lane L holds U2M's words L + 32 i of each stream's row h * H + j.
uint pw[HC][RIT];
if (LF == 0) {
  for (int h = 0; h < HC; ++h) {
    const size_t hrow = (size_t)h * H + (size_t)j;
    for (int i = 0; i < RIT; ++i) pw[h][i] = uw_q[hrow * R_by_p + (size_t)(int(lane) + 32 * i)];
  }
}
float pv[AIT0][HC];
for (int q = 0; q < AIT0; ++q) {
  const int e = int(t) + q * TPT;
  const int rr = e / R;
  if (rr < nr) {
    const int i = e - rr * R;
    for (int s = 0; s < HC; ++s) pv[q][s] = parts_in[(size_t)((r0 + rr) * HC + s) * PR + i];
  }
}
const bool injt = INJ != 0 && threadgroup_position_in_grid.y == 0 && t < uint(nr * HC);
float ipv[HC];
if (injt) {
  const int rr = int(t) / HC;
  const int c = int(t) - rr * HC;
  for (int s = 0; s < HC; ++s) ipv[s] = parts_in[(size_t)((r0 + rr) * HC + s) * PR + R + c];
}
if (LF != 0) {
  for (int h = 0; h < HC; ++h) {
    const size_t hrow = (size_t)h * H + (size_t)j;
    for (int i = 0; i < RIT; ++i) pw[h][i] = uw_q[hrow * R_by_p + (size_t)(int(lane) + 32 * i)];
  }
}
// This lane's (row, stream) = (lane / HC, lane % HC) normalized value.
const int myr = int(lane) / HC;
const int myh = int(lane) - myr * HC;
float xnv = 0.0f;
if (myr < nr) xnv = float(xn_in[(size_t)(r0 + myr) * HC * H + (size_t)myh * H + (size_t)j]);
// act = silu(T(sum of the stream partials)), as U2M computes it; staged as the f32 value of the T result.
for (int q = 0; q < AIT0; ++q) {
  const int e = int(t) + q * TPT;
  const int rr = e / R;
  if (rr < nr) {
    float tsum = 0.0f;
    for (int s = 0; s < HC; ++s) tsum += pv[q][s];
    const T v = T(tsum);
    const T sig = T(1.0f / (1.0f + metal::exp(-float(v))));
    actf[e] = float(v * sig);
  }
}
for (int q = AIT0; q < AIT; ++q) {
  const int e = int(t) + q * TPT;
  const int rr = e / R;
  if (rr < nr) {
    const int i = e - rr * R;
    float tsum = 0.0f;
    for (int s = 0; s < HC; ++s) tsum += parts_in[(size_t)((r0 + rr) * HC + s) * PR + i];
    const T v = T(tsum);
    const T sig = T(1.0f / (1.0f + metal::exp(-float(v))));
    actf[e] = float(v * sig);
  }
}
if (injt) {
  float tsum = 0.0f;
  for (int s = 0; s < HC; ++s) tsum += ipv[s];
  const T v = T(tsum);
  const T sig = T(1.0f / (1.0f + metal::exp(-float(v))));
  inj_out[(size_t)r0 * HC + t] = sig * T(2.0f);
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// U2M's weight values (two bf16 a word), unpacked once for every row (WH 1) or at each use (WH 0).
float wl[HC][RIT];
float wh[HC][RIT];
if (WH != 0) {
  for (int h = 0; h < HC; ++h) {
    for (int i = 0; i < RIT; ++i) {
      wl[h][i] = hc_val<T, true>(pw[h][i] & 0xFFFFu);
      wh[h][i] = hc_val<T, true>(pw[h][i] >> 16);
    }
  }
}
float mine = 0.0f;
for (int rr = 0; rr < RBU; ++rr) {
  if (rr < nr) {
    float al[RIT];
    float ah[RIT];
    for (int i = 0; i < RIT; ++i) {
      const float2 av = act2[rr * (R / 2) + int(lane) + 32 * i];
      al[i] = av.x;
      ah[i] = av.y;
    }
    for (int h = 0; h < HC; ++h) {
      float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
      for (int i = 0; i < RIT; ++i) {
        a0 += al[i] * (WH != 0 ? wl[h][i] : hc_val<T, true>(pw[h][i] & 0xFFFFu));
        a1 += ah[i] * (WH != 0 ? wh[h][i] : hc_val<T, true>(pw[h][i] >> 16));
      }
      const float acc = simd_sum((a0 + a1) + (a2 + a3));
      if (int(lane) == rr * HC + h) mine = acc;
    }
  }
}
// One lane per (row, stream): U2M's gate and mix term; the row's terms add in stream order.
const T u = T(mine);
const T sgv = T(1.0f / (1.0f + metal::exp(-float(u))));
const float m = float(T(float(sgv) * xnv));
const ushort lb = ushort(myr * HC);
float sum = 0.0f;
for (int h = 0; h < HC; ++h) sum += simd_shuffle(m, ushort(lb + ushort(h)));
if (myh == 0 && myr < nr) mixed_out[(size_t)(r0 + myr) * H + (size_t)j] = T(float(T(sum)) * float(T(1.0f / float(HC))));