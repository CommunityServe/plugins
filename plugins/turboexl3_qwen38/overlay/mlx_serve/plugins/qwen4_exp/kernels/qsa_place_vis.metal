const uint t = thread_position_in_grid.x;
const int kv = int(threads_per_grid.x);
const int blk = int(t) / RATIO;
bool visible = int(t) >= (kv / RATIO) * RATIO;
if (!visible) {
  int lo = 0;
  int hi = KB - 1;
  while (lo <= hi) {
    const int mid = (lo + hi) >> 1;
    const int v = blocks[mid];
    if (v == blk) { visible = true; break; }
    if (v < blk) lo = mid + 1; else hi = mid - 1;
  }
}
// headPlaceVisibility on this row: MLX's signed floor_divide of live_end by RATIO, then the recent and dead tests
const int le = live_end;
int eb = le / RATIO;
if (le % RATIO != 0 && (le < 0) != (RATIO < 0)) eb -= 1;
const bool recent = int(t) >= eb * RATIO;
const bool not_dead = (int(t) < le) || (int(t) >= dead_end);
mask[t] = (visible || recent) && not_dead;
