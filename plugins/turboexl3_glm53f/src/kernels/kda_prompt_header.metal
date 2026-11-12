// src/kda_prompt.zig's texts: unary_ops.h `Sigmoid` of the server's MLX (64ea011c), its body verbatim as a function, so
// the op chain's `mlx_sigmoid` and these texts take the same overloads (MLX's utils.h is before this header).
template <typename T>
inline T kda_sigmoid(T x) {
  auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}
