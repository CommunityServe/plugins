uint i = thread_position_in_grid.x;
if (i >= uint(N_size)) return;
uint d = i % uint(H);
uint row = i / uint(H);
// the gate's seven ops (pleGateChain) as MLX computes them, read from their table over every 16-bit pattern, then the
// gate times the value broadcast over the streams (mlx Multiply)
T gr = gate[row];
T g = gtab[as_type<ushort>(gr)];
gv[i] = g * value[(row / uint(HC)) * uint(H) + d];