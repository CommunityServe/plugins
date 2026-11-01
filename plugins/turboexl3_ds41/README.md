# turboexl3_ds41

Serves DeepSeek-V4.1-Flash's EXL3 packs in the streaming repack
(OpensourceWTF/DeepSeek-V4.1-Flash-1017-streaming-repack-exl3-2.81bpw). The routed experts stream from SSD, and turboexl3
runs the EXL3 trunk. The plugin claims a `deepseek_v41` pack only when its config carries exllamav3's
`quantization_config` (`quant_method` exl3); the built-in deepseek_v41 keeps every other pack.

    scripts/build.sh turboexl3_ds41                                  # from the repo root: tests, then ReleaseFast
    plugins/turboexl3_ds41/scripts/prepare.sh                        # generate src/mlx_serve/ (build.sh runs it)

M5-class GPUs (NAX) only; any other GPU is refused by name at load.

- src/mlx_serve (generated, not committed) is mlx-serve's own DeepSeek-V4.1 model (plugins/deepseek_v41), its vision
  tower, and mlx-serve's expert streamer (sdk/weights/streamed, module `mlx_stream`, which mlx-serve absorbed from
  David Tai's mlx-stream: see NOTICE), at MLX_SERVE_REF.
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
  `MLX_SERVE_DSV41_ROUTED=turboexl3` runs their decode on turboexl3 instead (src/routed_t3.zig), on the streamer's
  schedule: its decode waves, banked and with the device route (patches/turboexl3-t3-banked.patch), on turboexl3's
  banked decode (`moeClampedBanked`: a wave's pairs over the layer's three slot banks in one call, each pair
  `moeClamped`'s bytes at top-1 on its bank). The prompt calls stay on the streamer's own DIG-X lane, whose working set
  the prompt bill covers. The banked kernels compile at load, one set per pair count (1 to 48) and code width (the
  full pack's K3 and K2: 96 calls); the first load on a box measured 14.2 s for 130 calls, the 96 of these plus the
  34 prompt calls of the route before it.
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
- The DSpark draft's Markov chain (src/markov.zig, its MLX-free plan src/markov_plan.zig, the kernels
  src/kernels/markov_*.metal, the hook in patches/turboexl3-z-markov.patch) runs in B + 1 launches a draft, in place of
  five op groups a position (a take, the [1, rank] x [rank, vocab] matmul, a cast, the add, an argmax), with the op
  chain's bytes. Position k is one launch: each threadgroup reduces the previous position's threadgroup bests to its
  token (MLX's argmax order: NaN first, then the larger value, then the lower index), sums its rows of the bigram bias
  as MLX's gemv sums them (lane and element order, the shuffle-down tree, the round to the activation dtype), adds the
  logits row in f32 and keeps its best (value, index) pair. A last launch gives the tokens and the confidence head's
  logit (MLX's dot_product, where every product is exact: the head's weights 16-bit floats' values, else MLX's ops
  give it). An odd position reads the rows from the far end, those the position before read last. Window 17 held the
  tokens, every biased row and the confidence logits bit for bit to MLX's op chain and to an exact emulation of its
  gemv (the TEST pack's Markov and confidence weights; ties across threadgroups, NaNs and -0 planted), and timed the
  chain at 575.6 us a draft (median, after a 512 MB read) against the op chain's 842.0 on the M5 Max. In the server
  (the TEST pack, full-block verifies), `check` found every token and all 204,908,800 biased row elements of 317 drafts
  equal, the greedy text was the op chain's, and the stage profiler's `markov` went from 0.896 to 0.637 ms a draft.
  `MLX_SERVE_DSV41_MARKOV=0` runs the op chain, `flat` the kernels without the zig-zag, `check` both every draft (the
  op chain's served) and logs `[markov]` at each request's end with what differed. The patch sorts last: its hunks sit
  on the stage profiler's in the draft.
- `MLX_SERVE_DSV41_COMPILE=0` runs every decode stretch eager instead of compiled.
- `MLX_SERVE_DSV41_STAGE_PROF=1` (src/stage_prof.zig, hooks in patches/turboexl3-stage-prof.patch) times each
  forward by stage, a sync at each stage's end, and logs at each request's end one `[stage-prof]` line per kind and
  rows (decode, verify by rows, prompt chunks by powers of two, the DSpark draft):
  `verify rows=6 forwards= layers= wall_ms= attn_core=total_ms/calls/ms_per_call ...`, a call a layer's (a
  forward's for `phase`, `embed`, `ds_commit`, `head`, `markov`). `=N` (N > 1) logs every N forwards as well. Synced
  stages lose their overlap: the times attribute a forward, they are not its served speed. With it on, the decode
  stretches run eager, the same ops on the same kernels, uncompiled; unset, nothing changes.
- `MLX_SERVE_DSV41_LOOKAHEAD` (src/stream_lookahead.zig, read by the copied stream through
  patches/turboexl3-zz3-lookahead.patch) sets the decode's expert lookahead, which reads the next layer's predicted
  experts ahead of its route. Unset, the streamer's constants: 2 records read ahead a layer call, idle rule 0.
  `<budget>[:<idle>]`: 0 to 4 records (0 turns the predictor off) and the idle rule 0 or 1 (1: a speculative thread past
  the first may start a chunk while one demand read runs). Budgets 0 to 2 keep the stream's sizing and its rows; 4 sizes
  it for 4. The served bytes are the same at every setting. On the full pack (code-ctx16384, greedy with the prefix
  cached, window w-20261101-0226), at full-block verifies the default verified 7.0 +/- 1.7 ms a round faster than 0
  (107.6 against 114.6 ms, faster in 89 of 107 rounds, the same greedy text), and budget 4 issued 57% more records for no
  more claimed ones.
- `[round-io]` (src/round_io.zig, on with the DSpark profile, `MLX_SERVE_DIAG=dspark-profile`) logs at each request's
  end the expert stream's traffic of the decode's rounds and of the rest (the prompt pass): unique experts missed and
  hit, the demand's bytes (and those copied out of lookahead records), the lookahead's bytes, its abandoned ones, its
  records issued and claimed, the read wall and wait times, and the process's disk reads and page-ins
  (`proc_pid_rusage`); then one line a round beside its verify time.
- The streamed bank's resident seed and usage record (mlx-serve's `expert.usage`, its settings src/expert_usage_cfg.zig, the
  hooks in patches/turboexl3-zz4-usage-seed.patch), as glm5_next: at the load, each layer's empty prompt rows are filled
  with its most used experts of `expert-usage-v1.json` (read once, before the first request; a seeded expert is a resident
  as a kept one is, the policy unchanged), when the pack has the file; `MLX_SERVE_DSV41_RESIDENT_SEED=none` turns it off,
  `usage` refuses a pack without it, and `MLX_SERVE_DSV41_EXPERT_USAGE_DIR` reads the file from another directory.
  `MLX_SERVE_DSV41_EXPERT_USAGE_RECORD=<absolute path>` counts each request's routed experts per layer (the prompt pass's
  and the decode routes') into that file at the request's end (`MLX_SERVE_DSV41_EXPERT_USAGE_LABEL` names the prompt
  set); the counts are routes, so the seed does not bias them. The prompt residents stay across requests (`keep`), and
  the rate tally stays off (mlx-serve's DeepSeek-V4.1 default, pending a measurement).
- `MLX_SERVE_DSV41_DEC_PREFETCH=S[:K]` (src/dec_prefetch.zig, the model's prediction and the stream's fills in
  patches/turboexl3-zz2-dec-prefetch.patch): at encoder layer S of a replaying prompt chunk, each CED decoder layer's
  router runs over the replay tail's rows, and from then on each routed call's end posts the next K decoder layers'
  experts (default 2), ranked by the tail rows' votes, into those layers' free rows: an empty row, else a free seed row
  (below). A fill never takes a row a route has used, so a warm layer gets few or no fills. Only the pool's reserved
  tickets are new memory. Cold code-ctx16384 prompt on the full pack (M5 Max, w-pio-20261101-0252), S=10: 963.8 and
  964.9 tok/s against 914.6 and 913.0, the decoder side 1.67 s against 2.75 s. Before fills were held to free rows, a
  warm server's fills evicted residents and its prompt lost 2 to 3%; with the rule a warm server's second prompt posted
  no fill (w-pio-20261101-0338). Over the usage seed it does not help: with SEED_FREE, S=10 took 984.7 tok/s against
  986.4 to 997.3 without it, its decoder misses 1404 against 1320.
- `MLX_SERVE_DSV41_SEED_FREE=1` (src/seed_free.zig, the policy in patches/turboexl3-zz5-seed-free.patch, tested by
  src/seed_free_test.zig): the rows the usage seed fills stay free until a route uses their expert. The prompt's
  read-ahead may take one for an expert its prediction ranks before the seeded one, so a layer's rows hold its call's
  hottest experts as they would have held them empty: a 16K-row prediction names nearly every expert, and without the
  rule a seeded server's read-ahead found no row to take (27 records posted against 2413 unseeded, w-20261101-0307).
  The decoder prefetch may take one whose expert its prediction does not name. A seeded row a route has used is a
  resident as any other. Cold code-ctx16384 prompt on the full pack (M5 Max, w-pio-20261101-0338, a usage profile
  recorded on prompts apart from the test case): 997.3 tok/s against 991.9 and 986.4 for the seed alone; the encoder's
  read-ahead read 11.5 GB against 0.36, its read waits 0.95 s against 1.11 and 1.14. Off by default.

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
