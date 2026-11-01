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
  patches/turboexl3.patch, then the other patches/*.patch in name order. A build refuses a tree whose hash is not
  scripts/mlx_serve.sha256.
- The patch is the EXL3 arm:
  - turboexl3 runs every trellis linear of the trunk, the LM head, the DSpark stages and the vision tower. `wo_a` runs
    as one grouped linear and the tower's biased linears through `linearBias`.
  - The stages' experts stay resident as one turboexl3 bank per stage, run by `moeClamped` (the pack's `swiglu_limit`).
  - The pack's plain f16 linears are cast to the activation dtype at load.
  - The load keeps suh/svh f16 and never reads a trunk layer's routed experts.
  - The stream budget stays within the host's ceiling.
- The routed experts are the repack's `experts.bin` bank. The streamer's own EXL3 kernels (mul1, K2 and K3) run them;
  `MLX_SERVE_DSV41_ROUTED=turboexl3` runs them on turboexl3 instead (src/routed_t3.zig), on the streamer's schedule:
  its decode waves, banked and with the device route (patches/turboexl3-t3-banked.patch), on turboexl3's banked decode
  (`moeClampedBankedGateUp` / `moeClampedBankedDown`: a wave's pairs over the layer's three slot banks in one call, each
  pair `moeClamped`'s bytes at top-1 on its bank); its prompt calls on `moeClamped`, each routed pair a row of a top-1
  call.
- `MLX_SERVE_DSV41_ROUTED_ONE` (src/routed_one.zig, its MLX-free plan src/one_plan.zig, the hook in
  patches/turboexl3-wide-one.patch) runs a wide prompt call's resident routed experts as one turboexl3 `moeClamped`
  call over the layer's base slot bank in the router's [rows, top-k] layout, in place of the wide lane's DIG-X waves
  and per-pair join. Unset or `0`, nothing changes.
  - `1`: a call whose experts are all resident is that call. A call with streamed experts runs its resident pairs as
    that call (each streamed pair in it at weight 0) and its streamed experts on the streamer's schedule, a combine
    adding their weighted rows, when its resident pairs are at least half its pairs. Any other call keeps the lane.
  - `resident`: only the calls whose experts are all resident.
  - `check`: both ways on every call `1` takes, the lane's sum served; one `[routed-one] check` line a call against
    turboexl3's bar for two summation orders (the difference's rms within 2^-8 of the reference's, its largest within
    8e-2 of it). The sums agree within rounding (another GEMM order, f16 inner products), not bit for bit. The line
    also gives where the largest difference is (row, column, both values, the row's experts and weights), the largest
    reference element, the element-wise relative error (floored at the reference's rms) and how many elements pass
    1e-2 of it, and the largest difference beyond what the lane's bf16 pair rows may move an element (2^-8 of
    sum_j |w_j| |row_j|: `rounded=`). A call of 1024 rows or more is also compared to the exact sum on up to 8 rows:
    each row's experts in f64 from the base bank's own bytes (src/one_exact.zig, held to turboexl3's reference
    projection by src/one_exact_test.zig), against both paths' rows and each of the lane's pair rows (`[routed-one]
    exact` lines).
  - `[routed-one]` at each request's end counts the calls by path.
  - The prefill bill's routed term (plugins/deepseek_v41's `prefillBytes`) is, with the switch on, the one call's
    working set where that is more (src/one_plan.zig `callBytes`, every array alive at once: about 7.0 GB for a
    16K-row call with every expert resident, 7.8 GB with streamed experts, 19 GB under `check`).
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
- `MLX_SERVE_DSV41_STAGE_PROF=1` (src/stage_prof.zig, hooks in patches/turboexl3-stage-prof.patch) times each
  forward by stage, a sync at each stage's end, and logs at each request's end one `[stage-prof]` line per kind and
  rows (decode, verify by rows, prompt chunks by powers of two, the DSpark draft):
  `verify rows=6 forwards= layers= wall_ms= attn_core=total_ms/calls/ms_per_call ...`, a call a layer's (a
  forward's for `phase`, `embed`, `ds_commit`, `head`, `markov`). `=N` (N > 1) logs every N forwards as well. Synced
  stages lose their overlap: the times attribute a forward, they are not its served speed. With it on, the decode
  stretches run eager, the same ops on the same kernels, uncompiled; unset, nothing changes.

## The pack

- `config.json` keeps exllamav3's `quantization_config` (codebook mul1) and names the Engram tables, which stay in
  `ngram_embedding.safetensors`: `omlx_deepseek_v41.engram_tables["language_model.layers.<l>.engram.embed"]`.
- `model.safetensors.index.json` names every non-routed tensor in the root's shards, and not the Engram file.
- `experts.bin`, `expert-manifest-v2.json` and `expert-manifest.json` hold the routed experts. The parse refuses a pack
  without them (`Exl3PackNotRepacked`).

## Measured

Nothing yet on the served pack. The copy rounds, the verify policy and its ladder have not run on a GPU; their choices
are tested MLX-free (`zig build test`). The stage profiler has not run on a GPU either: its accounting and log lines are
tested MLX-free, its hooks only built.
