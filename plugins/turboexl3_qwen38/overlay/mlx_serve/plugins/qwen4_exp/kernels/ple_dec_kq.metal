uint i = thread_position_in_grid.x;
if (i >= uint(N_size)) return;
uint j = i % uint(HW);
// hcGroupNorm's weight products of the keys and of the queries, then their product: three mlx Multiply, each rounded to
// T
T a = nk[i] * wk[j];
T b = nq[i] * wq[j];
kq[i] = a * b;