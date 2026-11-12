// turboexl3_ling's own (not a copy): the KDA (vector-gate) recurrence of kernels/gdn/gated_delta.fmt.metal for prompt
// rows, with each block of TB tokens' k, q, gate, v and beta staged once per threadgroup of NS value rows (the stock
// kernel's threadgroup of 4 rows reads them from device memory for every token). The arithmetic is the stock kernel's,
// statement for statement: each value row's dk lanes 4 channels each, the decay then the k product summed per lane,
// simd_sum, the delta, the update then the q product, simd_sum, y rounded to OutT, the state in f32 between tokens
// and StT at the end. Grid (32, Dv, B * Hv) as the stock kernel's, threadgroup (32, NS, 1).
// Not served: kernels/kda/ling_kda_blocked_rows.metal (R value rows a simdgroup) serves. src/ling_kda_test.zig runs
// this text at NS 16, TB 16, the launch that one replaced, as the oracle of its byte test.
auto n = thread_position_in_grid.z;
auto b_idx = n / Hv;
auto hv_idx = n % Hv;
auto hk_idx = hv_idx / (Hv / Hk);
constexpr int n_per_t = Dk / 32;
constexpr uint NT = 32 * uint(NS);

auto dk_idx = thread_position_in_threadgroup.x;
auto row = thread_position_in_threadgroup.y;
auto dv_idx = thread_position_in_grid.y;
const uint tid = row * 32 + dk_idx;
const uint dv0 = dv_idx - row;

auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
auto y_ = y + b_idx * T * Hv * Dv + hv_idx * Dv;
auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;
auto beta_ = beta + b_idx * T * Hv;

threadgroup InT k_s[TB][Dk];
threadgroup InT q_s[TB][Dk];
threadgroup float g_s[TB][Dk];
threadgroup InT v_s[TB][NS];
threadgroup InT b_s[TB];

auto i_state = state_in + (n * Dv + dv_idx) * Dk;
auto o_state = state_out + (n * Dv + dv_idx) * Dk;

float state[n_per_t];
for (int i = 0; i < n_per_t; ++i) {
  auto s_idx = n_per_t * dk_idx + i;
  state[i] = static_cast<float>(i_state[s_idx]);
}

for (int t0 = 0; t0 < T; t0 += TB) {
  const int tt = min(TB, T - t0);
  for (uint p = tid; p < uint(tt * Dk); p += NT) {
    const uint r = p / uint(Dk);
    const uint d = p % uint(Dk);
    k_s[r][d] = k_[(t0 + r) * Hk * Dk + d];
    q_s[r][d] = q_[(t0 + r) * Hk * Dk + d];
    g_s[r][d] = g_[(t0 + r) * Hv * Dk + d];
  }
  for (uint p = tid; p < uint(tt * NS); p += NT) {
    const uint r = p / uint(NS);
    const uint d = p % uint(NS);
    v_s[r][d] = v_[(t0 + r) * Hv * Dv + dv0 + d];
  }
  for (uint p = tid; p < uint(tt); p += NT) {
    b_s[p] = beta_[(t0 + p) * Hv + hv_idx];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int t = 0; t < tt; ++t) {
    float kv_mem = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      state[i] = state[i] * g_s[t][s_idx];
      kv_mem += state[i] * k_s[t][s_idx];
    }
    kv_mem = simd_sum(kv_mem);

    auto delta = (v_s[t][row] - kv_mem) * b_s[t];

    float out = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      state[i] = state[i] + k_s[t][s_idx] * delta;
      out += state[i] * q_s[t][s_idx];
    }
    out = simd_sum(out);
    if (thread_index_in_simdgroup == 0) {
      y_[(t0 + t) * Hv * Dv + dv_idx] = static_cast<OutT>(out);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

for (int i = 0; i < n_per_t; ++i) {
  auto s_idx = n_per_t * dk_idx + i;
  o_state[s_idx] = static_cast<StT>(state[i]);
}
