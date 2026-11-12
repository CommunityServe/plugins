// TURBOEXL3_GLM53F_KDA_GATE (src/kda_prompt.zig): kda.zig's bounded forget gate (sdk/blocks/kda/gate.zig
// `kdaGateChain`) in one dispatch, each element through the MLX kernels the chain runs, in their operand order and
// types:
//   binary_ops.h `Add` in the promoted type AddT (the gate input's bf16 and dt_bias's f32: f32), `astype` to f32,
//   binary_ops.h `Multiply` in f32 by the chain's own exp(astype(A_log, f32)) (the input `ea`), unary_ops.h `Sigmoid`
//   (kda_sigmoid), binary_ops.h `Multiply` in f32 by the chain's f32 scalar (`bound`), unary_ops.h `Exp`
//   (precise::exp).
// Grid (N, rows, 1): one thread an element.
const uint c = thread_position_in_grid.x;
const uint t = thread_position_in_grid.y;
const size_t at = size_t(t) * N + c;
const AddT added = static_cast<AddT>(a[at]) + static_cast<AddT>(dt[c]);
const float scaled = ea[c] * static_cast<float>(added);
const float sg = kda_sigmoid(scaled);
g[at] = metal::precise::exp(sg * bound);
