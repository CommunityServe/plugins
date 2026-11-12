// turboexl3_glm53f TURBOEXL3_GLM53F_ROUTER_ONE: the predictor's phase 2 in the threadgroup that arrives last
// (src/fused_router.zig composes it; a std.fmt template: doubled braces are one). Every row's (row, expert) pair, the
// partials read first (`pload`, the header's): the stock router text's sigmoid (router_sigmoid_score.metal, MLX's
// Sigmoid with metal::precise::exp) over (0 + P0) + P1, then the correction added as MLX's add adds it: each pair's
// statements the two-launch call's second launch runs for it.
static_assert(TG == 256, "router_one: the 256 threads");
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
    const size_t rbase = (size_t)(i - e);
{s}
    scores[rbase + e] = sg + correction[e];
  }}
}}
