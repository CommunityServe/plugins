uint i = thread_position_in_grid.x;
const uint D = uint(old_shape[2]);
if (i >= uint(old_shape[0]) * uint(RING) * D) return;
const uint d = i % D;
const uint r = (i / D) % uint(RING);
const uint b = i / (D * uint(RING));
const int keep = int(RING) - kr_shape[1];
T v;
if (int(r) < keep) {
  v = old[b * old_strides[0] + (uint(old_shape[1] - keep) + r) * old_strides[1] + d * old_strides[2]];
} else {
  v = kr[b * kr_strides[0] + (r - uint(keep)) * kr_strides[1] + d * kr_strides[2]];
}
out[i] = v + zero;
