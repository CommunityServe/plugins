// turboexl3_ling's own: the KDA step's epilogue (src/ling_kda_epi.zig, sdk/blocks/kda/ling.zig `decodeStepEpi`
// and `verifyRowsEpi`), appended to kernels/kda/ling_kda_decode.metal, its header kernels/kda/ling_kda_oprep.metal. It
// writes what the launches after the step made of the step's inputs and its y, each word theirs:
// 1. ROWS_EXACT: the window's inputs kept for a partial accept's rollback, concat(conv_state, qkv) [1, NKEEP + TOK, C]
//    (the verify's concatenate, a copy), this head's q, k and v channels;
// 2. o_proj's DEC-ONE prepare of y (`ling_kda_oprep`, prep_sc's statements): this head's DV = 128 values of a row are
//    o_proj's input block h, read back from y once the step stored every row; simdgroup t prepares row t into oxp (lane
//    order) and the block's sum into oxs;
// 3. the arrival counters of o_proj's GEMV, NCNT of them, zeroed (prep_sc's).
#if ROWS_EXACT
for (uint e = tid; e < uint((NKEEP + TOK) * 3 * DK); e += 1024) {
  const uint srow = e / uint(3 * DK);
  const uint rem = e % uint(3 * DK);
  const uint part = rem / uint(DK);
  const uint i = rem % uint(DK);
  const uint channel = (part == 0 ? h * uint(DK)
                     : (part == 1 ? uint(HK) * uint(DK) + h * uint(DK)
                                  : 2 * uint(HK) * uint(DK) + h * uint(DV))) + i;
  conv_in[srow * uint(C) + channel] = srow < uint(NKEEP) ? conv_state[srow * uint(C) + channel]
                                                          : qkv[(srow - uint(NKEEP)) * uint(C) + channel];
}
#endif
threadgroup_barrier(mem_flags::mem_device);
if (sg < uint(TOK)) {
  ling_kda_oprep<T, uint(HV * DV)>(y, osuh, oxp, oxs, sg, h, lane);
}
for (uint e = h * 1024u + tid; e < uint(NCNT); e += uint(HV) * 1024u) {
  ocnt[e] = 0u;
}
