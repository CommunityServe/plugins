# turboexl3_ds41

Serves DeepSeek-V4.1-Flash's EXL3 packs in the streaming repack
(OpensourceWTF/DeepSeek-V4.1-Flash-1017-streaming-repack-exl3-2.81bpw). The routed experts stream from SSD, and turboexl3
runs the EXL3 trunk. The plugin claims a `deepseek_v41` pack only when its config carries exllamav3's
`quantization_config` (`quant_method` exl3); the built-in deepseek_v41 keeps every other pack.

    scripts/build.sh turboexl3_ds41                                  # from the repo root: tests, then ReleaseFast
    plugins/turboexl3_ds41/scripts/prepare.sh                        # generate src/mlx_serve/ (build.sh runs it)

M5-class GPUs (NAX) only; any other GPU is refused by name at load.

- src/mlx_serve (generated, not committed) is mlx-serve's own DeepSeek-V4.1 model (plugins/deepseek_v41), its vision
  tower, and mlx-serve's expert streamer (sdk/weights/streamed, module `mlx_stream`), at MLX_SERVE_REF.
  scripts/prepare.sh writes it with scripts/sync-mlx-serve.sh, copies overlay/mlx_serve over it and applies
  patches/turboexl3.patch. A build refuses a tree whose hash is not scripts/mlx_serve.sha256.
- The patch is the EXL3 arm:
  - turboexl3 runs every trellis linear of the trunk, the LM head, the DSpark stages and the vision tower. `wo_a` runs
    as one grouped linear and the tower's biased linears through `linearBias`.
  - The stages' experts stay resident as one turboexl3 bank per stage, run by `moeClamped` (the pack's `swiglu_limit`).
  - The pack's plain f16 linears are cast to the activation dtype at load.
  - The load keeps suh/svh f16 and never reads a trunk layer's routed experts.
  - The stream budget stays within the host's ceiling.
- The routed experts are the repack's `experts.bin` bank. The streamer's own EXL3 kernels (mul1, K2 and K3) run them;
  `MLX_SERVE_DSV41_ROUTED=turboexl3` runs them on turboexl3's `moeClamped` instead (src/routed_t3.zig: each routed pair
  a row of a top-1 call, on the streamer's schedule without its banked waves and device route).
- The streamer's MLX shims (events, uninitialized buffers, the launch cache, the read pool's Metal signal) are the
  plugin's own, over the server's host table (src/host_shims.zig). The read pool is the streamer's C, compiled here.
- The DSpark stages draft as the model's native drafter (plugin.zig), greedy or under the engine's sampling. A round
  verifies the whole block.
- `MLX_SERVE_DSV41_COMPILE=0` runs every decode stretch eager instead of compiled.

## The pack

- `config.json` keeps exllamav3's `quantization_config` (codebook mul1) and names the Engram tables, which stay in
  `ngram_embedding.safetensors`: `omlx_deepseek_v41.engram_tables["language_model.layers.<l>.engram.embed"]`.
- `model.safetensors.index.json` names every non-routed tensor in the root's shards, and not the Engram file.
- `experts.bin`, `expert-manifest-v2.json` and `expert-manifest.json` hold the routed experts. The parse refuses a pack
  without them (`Exl3PackNotRepacked`).

## Measured

Nothing yet: no GPU run of this plugin has been made.
