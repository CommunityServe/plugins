// turboexl3_ling's own (not a copy): Ling-3.1-flash's per-channel KDA gate in one launch, the ops of
// `kdaGateChain` (sdk/blocks/kda/gate.zig) in their order, each MLX's own formula (unary_ops.h, binary ops):
// Exp of the f32 A_log (metal::precise::exp), the add of a and dt_bias in f32 (MLX promotes the bf16 a to f32), the
// product, Sigmoid (1 / (1 + precise::exp(|x|)), then the branch on the sign), times the bound, Exp. One thread an
// element; g f32 [n_total], A_log and dt_bias f32 [W] (per key channel), a [n_total] in the activation dtype.
const uint i = thread_position_in_grid.x;
if (i >= uint(n_total)) {
  return;
}
const uint c = i % uint(W);
const float ea = metal::precise::exp(static_cast<float>(A_log[c]));
const float apd = static_cast<float>(a[i]) + static_cast<float>(dt_bias[c]);
const float sc = ea * apd;
const float sy = 1 / (1 + metal::precise::exp(metal::abs(sc)));
const float sg = (sc < 0) ? sy : 1 - sy;
const float gl = sg * lb;
g[i] = metal::precise::exp(gl);
