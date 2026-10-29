uint i = thread_position_in_grid.x;
if (i >= uint(N_size)) return;
uint tt = uint(SL) + uint(S_rows);
uint c = i % uint(C);
uint t = (i / uint(C)) % tt;
uint bb = i / (uint(C) * tt);
// mlx concatenate([state, gvn], 1): the state's rows copied, the new rows hcGroupNorm's weight product (mlx Multiply)
if (t < uint(SL)) {
  cat[i] = prev[(bb * uint(SL) + t) * uint(C) + c];
} else {
  cat[i] = nc[(bb * uint(S_rows) + (t - uint(SL))) * uint(C) + c] * wc[c];
}