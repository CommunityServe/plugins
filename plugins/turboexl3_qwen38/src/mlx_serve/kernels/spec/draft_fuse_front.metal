// The pick's token and the next draft step's front: row `tok` of the embedding table cast to NT, then MLX's
// rms_single_row (rms_norm.metal) statement for statement; threads past the row hold zeros, as MLX's out-of-row lanes
// do.
constexpr int N_READS = 4;
uint tid  = thread_position_in_threadgroup.x;
uint lane = thread_index_in_simdgroup;
uint sg   = simdgroup_index_in_threadgroup;
threadgroup uint df_tok[1];
threadgroup float local_inv_mean[1];
threadgroup float local_sums[32];
// The token: the candidates' entry at the chain's argmax (shortlistArgmax's take, then its astype), or the argmax
// itself.
if (tid == 0) {
    const uint s0 = uint(sel[0]);
    const uint tokv = (HAS_CANDS != 0) ? cands[s0] : s0;
    id[0] = static_cast<ID_T>(tokv);
    df_tok[0] = tokv;
}

threadgroup_barrier(mem_flags::mem_threadgroup);
{
    const uint tok = df_tok[0];
    const uint lid = tid;
    const uint simd_lane_id = lane;
    const uint simd_group_id = sg;
    const uint axis_size = uint(D);
    const uint w_stride = 1u;
    float acc = 0;
    float thread_x[N_READS];
    auto xe = emb + (ulong(tok) * ulong(axis_size) + ulong(lid * N_READS));
    if (lid * N_READS + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        thread_x[i] = static_cast<NT>(xe[i]);
        acc += thread_x[i] * thread_x[i];
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        thread_x[i] = (lid * N_READS + i < axis_size) ? (float)static_cast<NT>(xe[i]) : 0;
        acc += thread_x[i] * thread_x[i];
      }
    }
    acc = simd_sum(acc);
    //  Initialize shared memory
    if (simd_group_id == 0) {
      local_sums[simd_lane_id] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write simd accumulations into shared memory
    if (simd_lane_id == 0) {
      local_sums[simd_group_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Accumulate over simd groups
    if (simd_group_id == 0) {
      acc = simd_sum(local_sums[simd_lane_id]);
      if (simd_lane_id == 0) {
        local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write the outputs using cached x values
    const device NT* w = nw + w_stride * lid * N_READS;
    device NT* out = en + lid * N_READS;
    if (lid * N_READS + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        out[i] =
            w[w_stride * i] * static_cast<NT>(thread_x[i] * local_inv_mean[0]);
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if ((lid * N_READS + i) < axis_size) {
          out[i] =
              w[w_stride * i] * static_cast<NT>(thread_x[i] * local_inv_mean[0]);
        }
      }
    }
}
