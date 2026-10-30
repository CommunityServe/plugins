uint i = thread_position_in_grid.x;
if (i >= uint(N_size)) return;
// the chain's `silu` (MLX's sigmoid read from its table over every 16-bit pattern, then mlx Multiply) and the conv
// branch's mlx Add of gv, each rounded to T
T cv = conv[i];
T sig = sigtab[as_type<ushort>(cv)];
T act = cv * sig;
T o = gv[i] + act;
