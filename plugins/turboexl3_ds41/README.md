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
- The DSpark stages draft as the model's native drafter (plugin.zig), greedy or under the engine's sampling, with the
  two round features the engine gives its built-in model (src/rounds.zig, over the engine's own `mtp_lookup.zig`,
  `verify.zig` and `dflash/policy.zig` from `engines/mlx/spec`, copied by scripts/slice.txt). Both only choose what a
  round verifies; the trunk's accept decides what is kept.
  - Copy rounds: where the context agrees with the committed tail and `t1` for 16 tokens or more, its continuation is
    the draft (more than the block, up to the 7 a verify carries on the bank's decode lane), with no stage forward.
    `MLX_SERVE_DSV41_DSPARK_COPY=0` turns them off.
  - The verify policy: a stage round verifies the prefix of the block whose calibrated confidence buys the most tokens
    per ms of verify, on the verify ladder the plugin measures from its own rounds, and where drafting does not pay the
    round verifies `t1` alone (the engine's backoff). `MLX_SERVE_DSV41_DSPARK_POLICY=0` verifies the whole block every
    round.
  - The ladder (src/rounds.zig `Ladder`): the 1-row cell in ms, every other row as a ratio to it, so a level change
    the plain rounds see moves every row; each cell the lower median of its last 8 samples and the engine's prior
    ladder (one stall moves it a rank, a new level takes 4 or 5 samples); a cell unsampled for 64 rounds counts no
    dearer than the fresh rows around it, and a row never measured is the prior's shape bent to them. The two ends
    are refreshed once stale: the 1-row cell by a plain round, the whole block on a stage round (the backoff's probes
    included; the backoff does not judge a refresh). `refreshes=` in `dspark policy:` counts them; a stale cell in
    `ladder_ms` ends in `~`.
  - The context is the prompt the resume hands the plugin, then each forward's and round's tokens at their positions;
    a gap or a failed forward drops it, and with it the copies, until the next prompt.
  - Each request's rounds are logged beside the engine's `[spec-stats]` (`dspark rounds:`, `dspark policy:` with the
    measured ladder in `ladder_ms`); each round's phases go to `[dspark-prof]` (diag `dspark-profile`).
- `MLX_SERVE_DSV41_COMPILE=0` runs every decode stretch eager instead of compiled.

## The pack

- `config.json` keeps exllamav3's `quantization_config` (codebook mul1) and names the Engram tables, which stay in
  `ngram_embedding.safetensors`: `omlx_deepseek_v41.engram_tables["language_model.layers.<l>.engram.embed"]`.
- `model.safetensors.index.json` names every non-routed tensor in the root's shards, and not the Engram file.
- `experts.bin`, `expert-manifest-v2.json` and `expert-manifest.json` hold the routed experts. The parse refuses a pack
  without them (`Exl3PackNotRepacked`).

## Measured

Nothing yet on the served pack. The copy rounds, the verify policy and its ladder have not run on a GPU; their choices
are tested MLX-free (`zig build test`).
