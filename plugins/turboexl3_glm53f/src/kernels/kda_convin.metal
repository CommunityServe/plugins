// TURBOEXL3_GLM53F_KDA_CONVIN (src/kda_trim.zig): after the prologue text (kernels/kda_prologue.metal, its statements
// unchanged), the capture's conv input [prev | qkv] (MLX's concatenate's values: a copy) written beside q, k and v. Each
// thread copies its 4 channels of qkv's row t to row TAPS - 1 + t, and the threads of row 0 their channels of the TAPS - 1
// prev rows to rows 0 to TAPS - 2: every element once.
for (int j = 0; j < 4; j++) {
  conv_in[size_t(TAPS - 1 + t) * CDIM + c + j] = qkv[size_t(t) * CDIM + c + j];
}
if (t == 0) {
  for (int r = 0; r < TAPS - 1; r++) {
    for (int j = 0; j < 4; j++) {
      conv_in[size_t(r) * CDIM + c + j] = prev[size_t(r) * CDIM + c + j];
    }
  }
}
