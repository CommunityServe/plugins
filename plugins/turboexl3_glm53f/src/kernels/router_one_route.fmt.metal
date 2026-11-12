// turboexl3_glm53f TURBOEXL3_GLM53F_ROUTER_ONE: the router's phase 2 in the threadgroup that arrives last
// (src/fused_router.zig composes it; a std.fmt template: doubled braces are one). The stock router text's (mlx-serve
// kernels/moe/router.fmt.metal, sigmoid_bias) score loop runs over every row's (row, expert) pairs at once, the
// partials read first (`pload`, the header's), then each row's selection runs on its own simdgroup: simdgroup r takes
// row r, as the stock launch's simdgroup 0 takes its one row. The loop's statements and the selection's are the stock
// text's, composed from it with each edit found once: the read (the same (0 + P0) + P1) and the indexing of rk, rw and
// sel by a row's slice. Phase 1's weight ring holds the arrays: every read of it is behind the arrival's barriers.
threadgroup float* rk = reinterpret_cast<threadgroup float*>(ring4);
threadgroup float* rw = rk + M * NE;
threadgroup uint* sel = reinterpret_cast<threadgroup uint*>(rw + M * NE);
static_assert(sizeof(float) * 2 * M * NE + sizeof(uint) * M * NK <= sizeof(ring4), "router_one: phase 2's arrays do not fit phase 1's ring");
static_assert(TG == 256 && M <= TG / 32, "router_one: a row a simdgroup of the 256 threads");
const uint lane = thread_index_in_simdgroup;
constexpr uint PAIRS = uint(M) * uint(NE);
constexpr uint PER = (PAIRS + uint(TG) - 1u) / uint(TG);
float lp0[PER];
float lp1[PER];
for (uint k = 0; k < PER; ++k) {{
  const uint i = uint(tid) + k * uint(TG);
  if (i < PAIRS) {{
    lp0[k] = pload(p, i);
    lp1[k] = pload(p, PSTRIDE + i);
  }}
}}
for (uint k = 0; k < PER; ++k) {{
  const uint i = uint(tid) + k * uint(TG);
  if (i < PAIRS) {{
    const uint e = i % uint(NE);
{s}
  }}
}}
threadgroup_barrier(mem_flags::mem_threadgroup);
{{
const uint row = simdgroup_index_in_threadgroup;
const size_t rbase = (size_t)row * (size_t)NE;
{s}
}}
