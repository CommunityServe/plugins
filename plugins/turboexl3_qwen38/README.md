# turboexl3_qwen38

Serves the fully EXL3 Qwen3.8-Flash-Next packs K2 to K5 (OpensourceWTF/Qwen3.8-Flash-Next-EXL3-{2.05,3.05,4.05,5.05}
-mlx-serve) through turboexl3. It claims a `qwen4_exp` pack only when its config has a `trunk_quant` block.

    scripts/build.sh turboexl3_qwen38                                # from the repo root: tests, then ReleaseFast
    plugins/turboexl3_qwen38/scripts/prepare.sh                      # generate src/mlx_serve/ (build.sh runs it)
    zig build gpu-test -Dmlx-lib=<dir>      # in the plugin's folder, against the server's MLX (libmlx, libmlxc)

M5-class GPUs (NAX) only; any other GPU is refused by name at load. MTP needs two mlx-serve fixes, open in
[ddalcu/mlx-serve#1017](https://github.com/ddalcu/mlx-serve/pull/1017); with `--no-mtp` it serves on mlx-serve main.

- src/direct is the model: one Qwen3.8 forward for these packs, with turboexl3 doing the EXL3 math.
- src/mlx_serve (generated, not committed) holds the parts of mlx-serve the model calls: kernels, their launchers, the
  MTP round policy and shared types, cut to what the plugin's builds compile. scripts/prepare.sh writes it from mlx-serve
  at MLX_SERVE_REF and copies overlay/mlx_serve over it: the cut edges, and the files whose text at mlx-serve 1b3e624b
  (ddalcu/mlx-serve#1005's head) differs from the ref's. A build refuses a tree whose hash is not scripts/mlx_serve.sha256.
- MTP drafts with the pack's own head (src/spec/drafter.zig). The server's `mtp` load setting reaches it, and
  src/host_flags.zig reads the server's `--mtp-min-depth` and `--mtp-max-depth` (and the old `--mtp-depth`) off its
  arguments: equal values pin the depth every round drafts. It also reads `--max-mtp-ctx` (no drafting past that
  context), `--mtp-history-window` (a long prompt's head history from its last positions only) and `--warmup-eager` /
  `--no-warmup-eager` (whether the startup load of `--model` runs the MTP warm-up), as the built-in Qwen3.8 takes them.
- `--int8-qk`, read the same way, runs a dense prefill block of 64 rows or more (below the QSA budget) on int8 Q and K,
  as the built-in does (lossy).
- `--kv-quant 4` or `8` (a model's `kv_quant` in ~/.mlx-serve/model-settings.json first), `--kv-attn-mode` and
  `--mtp-head-kv-quant`, read the same way, keep the KV cache affine (group 64) and read it as the built-in does.
- The prefix cache's snapshots (src/cache/aux.zig: KV, recurrent state with the QSA history, the head's history) go to
  the server's disk tier with `--prefix-cache-disk`, and at exit within `--shutdown-persist-secs`, in
  src/cache/codec.zig's form; a later prompt, after a restart too, restores from there as it would from RAM.

## Measured

(final window: to fill in)
