// turboexl3_ling's own (not a copy): kernels/kda/ling_kda_blocked.metal with R value rows a simdgroup, the blocked
// prompt recurrence's served kernel (sdk/blocks/kda/ling.zig `rec_shape`: NS 8, TB 16, R 2). Each value row's arithmetic
// is ling_kda_blocked.metal's, statement for statement: its dk lanes 4 channels each in the same order, the decay then
// the k product summed per lane, simd_sum, the delta, the update then the q product, simd_sum, y rounded to OutT, the
// state in f32 between tokens and StT at the end. The R rows' chains are independent, so their latencies overlap. Grid
// (32, Dv / R, B * Hv), threadgroup (32, NS, 1): NS simdgroups and NS * R value rows a threadgroup, each block of TB
// tokens' k, q, gate, v and beta staged once for them.
auto n = thread_position_in_grid.z;
auto b_idx = n / Hv;
auto hv_idx = n % Hv;
auto hk_idx = hv_idx / (Hv / Hk);
constexpr int n_per_t = Dk / 32;
constexpr uint NT = 32 * uint(NS);
constexpr uint VR = uint(NS) * uint(R);

auto dk_idx = thread_position_in_threadgroup.x;
auto sg = thread_position_in_threadgroup.y;
auto sg_idx = thread_position_in_grid.y;
const uint tid = sg * 32 + dk_idx;
const uint dv0 = (sg_idx - sg) * uint(R);

auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
auto y_ = y + b_idx * T * Hv * Dv + hv_idx * Dv;
auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;
auto beta_ = beta + b_idx * T * Hv;

threadgroup InT k_s[TB][Dk];
threadgroup InT q_s[TB][Dk];
threadgroup float g_s[TB][Dk];
threadgroup InT v_s[TB][VR];
threadgroup InT b_s[TB];

float state[R][n_per_t];
for (int r = 0; r < R; ++r) {
  const uint dv_idx = sg_idx * uint(R) + uint(r);
  auto i_state = state_in + (n * Dv + dv_idx) * Dk;
  for (int i = 0; i < n_per_t; ++i) {
    auto s_idx = n_per_t * dk_idx + i;
    state[r][i] = static_cast<float>(i_state[s_idx]);
  }
}

for (int t0 = 0; t0 < T; t0 += TB) {
  const int tt = min(TB, T - t0);
  for (uint p = tid; p < uint(tt * Dk); p += NT) {
    const uint rr = p / uint(Dk);
    const uint d = p % uint(Dk);
    k_s[rr][d] = k_[(t0 + rr) * Hk * Dk + d];
    q_s[rr][d] = q_[(t0 + rr) * Hk * Dk + d];
    g_s[rr][d] = g_[(t0 + rr) * Hv * Dk + d];
  }
  for (uint p = tid; p < uint(tt) * VR; p += NT) {
    const uint rr = p / VR;
    const uint d = p % VR;
    v_s[rr][d] = v_[(t0 + rr) * Hv * Dv + dv0 + d];
  }
  for (uint p = tid; p < uint(tt); p += NT) {
    b_s[p] = beta_[(t0 + p) * Hv + hv_idx];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int t = 0; t < tt; ++t) {
    float kv[R];
    for (int r = 0; r < R; ++r) {
      float kv_mem = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[r][i] = state[r][i] * g_s[t][s_idx];
        kv_mem += state[r][i] * k_s[t][s_idx];
      }
      kv[r] = simd_sum(kv_mem);
    }
    for (int r = 0; r < R; ++r) {
      auto delta = (v_s[t][sg * uint(R) + uint(r)] - kv[r]) * b_s[t];

      float out = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[r][i] = state[r][i] + k_s[t][s_idx] * delta;
        out += state[r][i] * q_s[t][s_idx];
      }
      out = simd_sum(out);
      if (thread_index_in_simdgroup == 0) {
        y_[(t0 + t) * Hv * Dv + sg_idx * uint(R) + uint(r)] = static_cast<OutT>(out);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

for (int r = 0; r < R; ++r) {
  const uint dv_idx = sg_idx * uint(R) + uint(r);
  auto o_state = state_out + (n * Dv + dv_idx) * Dk;
  for (int i = 0; i < n_per_t; ++i) {
    auto s_idx = n_per_t * dk_idx + i;
    o_state[s_idx] = static_cast<StT>(state[r][i]);
  }
}
