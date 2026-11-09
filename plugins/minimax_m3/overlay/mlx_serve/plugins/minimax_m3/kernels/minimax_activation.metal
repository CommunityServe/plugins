// The original pointwise operations' dtype boundaries are explicit, including
// sigmoid's output. Comparison-based clipping retains NaNs and signed zeros.
const uint i = thread_position_in_grid.x;
if (i >= N) return;
const T limit = T(as_type<float>(uint(LIMIT_BITS)));
const T negative_limit = T(-as_type<float>(uint(LIMIT_BITS)));
const T alpha = T(as_type<float>(uint(ALPHA_BITS)));
const T beta = T(as_type<float>(uint(BETA_BITS)));
const T a = gate[i];
const T b = up[i];
const T gc = isnan(a) ? a : (a < limit ? a : limit);
const T lower = isnan(b) ? b : (b > negative_limit ? b : negative_limit);
const T uc = isnan(lower) ? lower : (lower < limit ? lower : limit);
const T scaled = T(gc * alpha);
const auto y = 1 / (1 + metal::precise::exp(metal::abs(scaled)));
const T sig = T(scaled < 0 ? y : 1 - y);
const T gated = T(gc * sig);
const T shifted = T(uc + beta);
out[i] = T(gated * shifted);
