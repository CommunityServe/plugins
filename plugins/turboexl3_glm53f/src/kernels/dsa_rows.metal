// TURBOEXL3_GLM53F_DSA_ROWS (src/dsa_rows.zig): glm5_next's kernels/pooled_rows.metal with the forward's rows read from
// their parts. The stock forward first builds each row as the key (the LayerNorm's output `ikn`) and the gate (the f32
// projection `gate` cast to T by MLX's astype), concatenated; here the key half comes from `ikn` and the gate half is
// `T(gate)`, the astype's rounding, so every value the stock kernel reads is the same and so is every value it writes.
// Thread (d, r, b) writes channel d of row r: the key, then the gate or, on a row that completes a pool, the pool's key
// (the softmax over its rows' gates plus `ape`, weighting their keys). `prev` holds the prev_len cached rows before
// pos0 (key | gate, read only for a pool the forward completes).
const uint d = thread_position_in_grid.x;
const int r = int(thread_position_in_grid.y);
const int b = int(thread_position_in_grid.z);
const int S = int(threads_per_grid.y);
const int pos0 = dims[0];
const int prev_len = dims[1];
const size_t at = (size_t(b) * S + r) * DH + d;
device T* orow = out + (size_t(b) * S + r) * (2 * DH);
orow[d] = ikn[at];
const int pos = pos0 + r;
if ((pos + 1) % KPOOL != 0) {
    orow[DH + d] = T(gate[at]);
    return;
}
float lg[KPOOL];
T kv[KPOOL];
float mx = -INFINITY;
for (int j = 0; j < KPOOL; ++j) {
    const int q = pos - KPOOL + 1 + j;
    if (q >= pos0) {
        const size_t qa = (size_t(b) * S + (q - pos0)) * DH + d;
        kv[j] = ikn[qa];
        lg[j] = float(T(gate[qa])) + ape[j * DH + d];
    } else {
        const device T* row = prev + (size_t(b) * prev_len + (q - pos0 + prev_len)) * (2 * DH);
        kv[j] = row[d];
        lg[j] = float(row[DH + d]) + ape[j * DH + d];
    }
    mx = metal::max(mx, lg[j]);
}
float sum = 0.0f;
for (int j = 0; j < KPOOL; ++j) { lg[j] = metal::precise::exp(lg[j] - mx); sum += lg[j]; }
float acc = 0.0f;
for (int j = 0; j < KPOOL; ++j) acc += float(T(T(lg[j] / sum) * kv[j]));
orow[DH + d] = T(acc);
