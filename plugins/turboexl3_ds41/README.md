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
  more claimed ones. The next layer's scores stay out of the routing barrier (patches/turboexl3-zz6-lookahead-late.patch):
  committed right behind the ids, read once the call's waves are built and the records issued then, so a layer call's
  barrier waits on its ids alone; the same records and claims. At full-block verifies (window w-20261101-0326) that
  verified 1.25 +/- 0.20 ms a round faster than the scores in the barrier (93 of 106 rounds; 1.42 +/- 0.23 ms, 132 of
  142, in the cold prompt's decode), the texts, rounds and misses the same.
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
- A diagnostic build (`zig build -Doptimize=ReleaseFast -Droute-rec=true`, never served) records the decode's routes:
  `MLX_SERVE_DSV41_ROUTES=<absolute directory>` arms mlx-serve's `expert.routes` recorder on the streamed bank
  (src/route_rec.zig, patches/turboexl3-zz7-route-rec.patch), and each request writes `ds41-routes-<pid>-<seq>.bin`
  (the decode handover's residents and prompt counts, then each step and each routed call: its ids, what served each,
  the lookahead's records) and `.steps.txt` (each step's rows' tokens, accepted drafts and emitted tokens), for replaying
  residency policies offline. A served build has no recorder branch and refuses the setting.
- A profile build (`zig build -Doptimize=ReleaseFast -Dplugin-profile=true`, never served) compiles the copied stream's
  profile probes in, and `MLX_SERVE_DSV41_TIMELINE=<absolute directory>` arms mlx-serve's decode verify timeline
  (src/verify_tl.zig, patches/turboexl3-zzz2-verify-timeline.patch) at each request's first DSpark round: every command
  buffer MLX commits, with its host commit time and Metal's GPU start and end, each verify's window and each routed
  layer's host stamps (the routing barrier's eval, the ids read, the route, the hit wave's submit, the call's end). At
  the request's end the log gets `VERIFY_GPU_TIMELINE` (per verify: the GPU's busy and idle time, the idle split at the
  routing barriers, before the layers and after the last, the hit waves, the gated waits on miss bytes) and
  `ds41-timeline-<pid>-<seq>.json` the raw rows and each round's rows, accepted drafts, misses and host times. The shim
  hooks the class of MLX's command buffers through a buffer of the plugin's own queue (src/tl_buffer.m). A served build
  has no hook and refuses the setting.
- `MLX_SERVE_DSV41_PROMPT_TL=<absolute directory>` (src/prompt_tl.zig, patches/turboexl3-zzz8c-prompt-timeline.patch;
  profile builds, with `MLX_SERVE_DSV41_PROMPT_IO`, refused with `MLX_SERVE_DSV41_TIMELINE`) arms the same store over
  each prompt pass: from its first stage to its end every command buffer MLX commits, tagged with the layer and the
  host's stage (`pre`: a wide layer's start, its Engram rows and read-ahead, to its routed call; `attn`: the call's
  barrier, its routes evaluated; `moe`: the call's reads and waves; `rest`: the layer's last eval), the stages' host
  marks and the read pool's event signals. No fence is added: the marks sit at `[prompt-io]`'s hooks. At the request's
  end `ds41-prompt-tl-<pid>-<seq>.json` holds the rows (`cycle` the layer, `phase` the stage) and the marks, all on
  Metal's clock.
- The decode's slot rows are split across the layers (src/decode_rows.zig, the hook in
  patches/turboexl3-zzz1-decode-rows.patch) at the flat bill's bytes or fewer, each layer between its prompt rows and
  256, by each layer's misses-a-row curve: the decode residency policy replayed over routes recorded on a development
  corpus kept apart from the code-ctx prompts (Python and Zig sources with tasks of their own, chat questions; 8
  requests, 1948 rounds). The curves fit this pack's geometry only (40 layers, 384 experts, its K3 and K2 records): on
  any other the rows stay flat (one info line). One load line gives the rows and the bytes against flat.
  `MLX_SERVE_DSV41_DECODE_ROWS=flat` keeps flat rows; `=split` requires the split (refused by name with log.err, rows
  flat, where it does not fit). On the full pack (code-ctx16384, fixed full-block verifies, the seed, SEED_FREE and
  ROUTED=turboexl3, window w-20261101-0510) the split decoded 46.1 and 46.0 tok/s greedy against flat's 44.6 and 43.9,
  39.5 and 39.6 on the cold prompt's decode against 37.9 and 37.2, with 7.7% fewer misses a round and the same texts.
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
- With `MLX_SERVE_DSV41_PROMPT_IO` (patches/turboexl3-zz8-prompt-wide.patch), a `[prompt-io] wide` line a wide layer
  splits the time before its routed call: the read-ahead's wait for its predicted routes (`ra_wait_ms`) and its host
  work after (`ra_host_ms`), then the host's build of each lazy graph step (the attention-side mix, the projections,
  the spans' attention, the output projection, the FFN-side mix, the router to the call); nothing evaluates between
  them. `=2` also splits the read-ahead's wait by an eval at each step (`ra_cat_ms` the stream's rows, `ra_col_ms`
  their collapse, `ra_route_ms` the router), two more syncs a wide layer. On the full pack's cold code-ctx16384 prompt
  (w-pio-20261101-0409) the waits summed 1.07 to 1.25 s over the encoder's layers 1 to 19, the host work 2 ms and the
  graph builds about 40 ms.
- `MLX_SERVE_DSV41_RA_FUSED` (src/ra_fused.zig, patches/turboexl3-zz9-ra-fused.patch; on, `=0` the op chain): a wide
  layer's read-ahead prediction on the fused hc kernels. The prediction is the router over the stream collapsed with the previous mix and
  normed for the FFN: the same collapse of the same rows with the same mix as the attention's input, normed by the other
  weight. The layer runs the attention-side mix kernel first and takes the prediction from one more pass of the norm
  kernel, where the op chain (the stream to f32, a batched matmul, a norm) built it. The attention's input and the routed
  calls are the same kernels over the same arrays; only the experts the read-ahead reads may round apart. Cold
  code-ctx16384 prompt on the full pack (M5 Max, w-pio-20261101-0433, the usage seed with SEED_FREE in every arm): 1019.0
  and 1021.9 tok/s against 984.1 and 988.0 with the op chain, the collapse step 14 to 18 ms a layer against 31 to 33, the
  seeded 32-token text the same in every arm.
- `MLX_SERVE_DSV41_LATE_GEMV` (src/late_gemv.zig, patches/turboexl3-zzz4-late-gemv-shared-hoist.patch; on, `=0` the f32
  product): the decode lookahead's next-layer scores (a verify's rows against the next layer's router weights, f16 in
  the pack) as a product in the weights' half type, which MLX runs as its wide GEMV, one weight pass for the rows. The
  f32 product took its tiled GEMM at a single row tile: 109 µs a layer on the verify's critical path when it did not run
  beside the hit wave (the verify timeline, w-20261101-0519). The scores only rank what the read-ahead reads: the served
  bytes are the same, near-ties may rank apart.
- `MLX_SERVE_DSV41_SHARED_HOIST` (src/shared_hoist.zig, the same patch; on, `=0` in the layer's last stretch): a decode
  layer's shared expert is committed on its own as soon as the routed call returns, so it queues right behind the device
  hit wave and runs while a layer with misses waits on their bytes, where it ran after them. The model's `ffn_post`
  stretch is split at the shared expert's down projection: the same kernels, so the same bytes.
- Both on against both off on the full pack (M5 Max, w-20261101-0543; code-ctx16384, card then greedy, fixed full-block
  verifies, the usage seed with SEED_FREE, ROUTED=turboexl3, A B B A): 40.0 and 40.1 tok/s on the card's decode against
  39.3 and 39.2, 47.0 and 47.0 greedy against 46.1 and 45.6, the misses a round the same (62.4 greedy, 82.3 card; 63.5
  and 83.1 in the last arm), the texts the same in every arm.
- `MLX_SERVE_DSV41_T3_GROUP` (src/t3_group.zig, patches/turboexl3-zzz8b-t3-group.patch; on with
  `MLX_SERVE_DSV41_ROUTED=turboexl3`, `=0` off): the t3 route's device hit wave on turboexl3's grouped banked decode
  (`Config.banked_group` 4, turboexl3 65c72d1). A verify's 36 routed pairs a layer name about 23 distinct experts
  (recorded routes: p10 18, p90 29). The banked GEMVs shared a weight decode between at most two row-slots of one
  expert; the grouped ones share it among up to four. DEVROUTE's LUT also sends a pair whose expert no bank holds to
  turboexl3's skip id, so its device row runs no expert, where it was computed from base row 0. The miss part's row
  replaces it in the join, as before, and every other row is the banked call's byte for byte (turboexl3's GPU test,
  window 32), so the served bytes are the same.
  - Window 32's microbench, 36 row-slots (34 hits in the recorded routes' p10, median and p90 profiles, plus 2
    misses), µs a call: 1024, 1090 and 1253 at K3 against 1112, 1176 and 1315 today; 983, 1062 and 1227 at K2 against
    1059, 1127 and 1263.
  - Served on the full pack (M5 Max, w-20261101-0641; code-ctx16384, card then greedy, fixed full-block verifies, the
    usage seed with SEED_FREE, ROUTED=turboexl3, A B B A): 39.2 and 39.8 tok/s on the card's decode against 40.1 and
    39.7, 46.9 and 46.6 greedy against 46.8 and 46.3, the texts the same in every arm. A wash at today's misses: the
    hit wave's saving turns into the miss parts' byte wait while the decode's reads are bound by the drive. It removes
    work and pays as misses fall.
- `MLX_SERVE_DSV41_WOA_ROWS` (src/woa_rows.zig, patches/turboexl3-zzz7-woa-rows.patch; on, `=0` off, `check` both): a
  wide layer's EXL3 `wo_a` over runs of 4096 rows of its attention spans. `=0`: each 256-row span runs its own grouped
  linear as its attention ends: at prefill rows turboexl3 decodes each of the 8 slices' trellis to W_hat and multiplies
  at M = 256, so a 16K-row layer decodes every slice 64 times and runs 512 small GEMMs. On, the spans' grouped heads are
  joined in runs of 4096 rows, one grouped linear a run: each slice decoded once a run, its GEMM at M = 4096. The
  prepare, the decode and the finish give the same words; MLX's GEMM at the other M rounds a row's sums in another
  order. `check` runs both on every run, serves the span-by-span result and logs `[woa-rows] check layer= runs= equal=
  differ= max_abs=` a layer. A run holds its spans' grouped heads, their join and the grouped linear's copy of them
  (about 0.8 GB), within the prompt bill's term for the chunk's grouped heads.
- On against `=0` on the full pack (M5 Max, w-pio-20261101-0624; code-ctx16384 cold, then a new 16K text, the usage seed
  with SEED_FREE, A B B A): 1028.0 and 1028.5 tok/s cold prefill against 1016.8 and 1010.9 (+1.4%); new text 1019.7 and
  1036.0 against 1017.7 and 1032.5. Not bit-exact, a reorder-level change: every layer's runs differ from span by span
  in about 2% of their words, by at most one bf16 ulp (max_abs 7.8e-3), and the seeded 32-token text changes (md5
  a323144e8b, "Return values without equality duplicates, keeping first-seen order.", becomes efeb6f4ecd, "Return values
  with equality-based duplicates removed, keeping first order."): the reference text from here on.
- `MLX_SERVE_DSV41_JOINLESS` (src/joinless.zig, patches/turboexl3-zzz8-joinless.patch; on, `=0` off, `check` both): a
  wide prompt call's routed sum and the layer's shared expert in one combine. `=0`: the streamer joins each bank call's
  wave outputs (a concatenate and a take, each evaluated inside the routed call), then the call's outputs again into
  [rows * 6, 5120] in routed order; the source widens them to f32, weights them and sums the six, and the model adds the
  shared expert in f32: about 18 GB of reads and writes a 16K-row layer, 14 GB of it in the layer's last eval. On keeps
  the waves' outputs as they are (the smallest concatenated into the 24th source above 24) and runs the kernel set's
  JOINLESS combine (`dsv41_jl_combine_bf16`): each routed row read where its wave wrote it, weighted, the six folded in
  MLX's column reduce order and the shared rows added, uncontracted: the op chain's f32 words. `check` builds both from
  the same outputs, serves the op chain's and logs `[joinless] check layer= rows= sources= words= differ= max_abs=` a
  wide call; it holds both paths' buffers at once (at 16K rows it lifted a window's accounted memory past 112 GiB), so
  it is a diagnostic for a test step or a small window, not a measured one. Refused with `MLX_SERVE_DSV41_ROUTED_ONE`.
- On against `=0` (w-pio-20261101-0630; as above, WOA_ROWS off in every arm): 1123.8 and 1123.3 tok/s cold prefill
  against 1016.2 and 1013.3 (+10.7%); new text 1132.8 and 1153.7 against 1017.6 and 1029.3 (+11%). Exact: the seeded
  text is the same in every arm (md5 a323144e8b), and `check` compared 52 wide calls word for word with none different.
- `MLX_SERVE_DSV41_EMBED_PREAD` (src/embed_pread.zig, patches/turboexl3-zzz8d-embed-vision.patch; on, `=0` resident):
  the token embedding table (`embed.weight`, bf16 [129280, 5120], 1.32 GB) stays on disk and its rows are read by pread
  at each lookup, each distinct id once, in parallel (GCD). A decode round looks up the verify's 6 ids and the draft's
  first id, a 16K prompt its distinct ids; the page cache keeps the hot rows. The rows are the table's bytes, so the
  served bytes are the same. A plain decode token still on the GPU is read back first.
- `MLX_SERVE_DSV41_VISION_LAZY` (src/vision_lazy.zig, the same patch; on, `=0` resident): the vision tower (`vision.*`,
  `aligner.*`, `image_*`, 0.94 GB) stays on disk; the media hook reads it for each image, encodes, evaluates the rows
  and frees the tower before it returns. Image encodes run between other slots' steps, so a bank holding its decode
  rows drops to its prompt rows while the tower is loaded (the prompt's working set, unused until a prompt's wide calls,
  holds it) and grows them back after. The tower's arithmetic is the resident tower's.
- The load also leaves on disk the tower's split EXL3 copies the repack ships beside the fused `attn.wqkv` and
  `mlp.w1` the tower reads (`attn.q_proj` / `k_proj` / `v_proj`, `mlp.gate_proj` / `up_proj`: 0.21 GB, never read).
- The streamed bank is sized from the memory in use when it opens, so those bytes become slot rows. Both on against
  both off on the full pack (M5 Max, w-20261101-0744; code-ctx16384, card cold then greedy, fixed full-block verifies,
  the usage seed with SEED_FREE, ROUTED=turboexl3, A B B A): 128 / 150 prompt / decode slot rows against 125 / 147;
  41.6 and 41.6 tok/s on the card's decode against 41.1 and 40.6, 48.0 and 47.9 greedy against 47.2 and 46.4; misses a
  round 78.4 / 60.0 (card / greedy) against 81.4 / 62.9; cold prefill 1136.1 and 1138.4 tok/s against 1132.3 (the
  window's first server, Engram-cold, read 1057.1). The card and greedy texts are the same in every arm. An image
  request (448x448, a red disc) returned HTTP 200 on the resident and the lazy tower alike, the same reasoning head;
  its 64 tokens all went to reasoning, so the answer itself was not read in that window.
- `MLX_SERVE_DSV41_WHOLE_STREAM` (src/whole_stream.zig, patches/turboexl3-zzz8e-whole-stream.patch; on, `=0` off): a
  wide prompt layer takes the last wide layer's whole stream and FFN mix as they are. `=0`: each wide layer starts by
  concatenating its spans' streams (`[L, hc, D]`) and mixes into fresh arrays, though the last wide layer's one MoE
  group left every span holding a row slice of one array: a copy and a fresh buffer a layer. On, the layer's last step
  also keeps that array and the mix (two held slots), and the next wide layer takes them when every span still holds its
  slice: not on an Engram layer (1 and 14 write rows per span), and not after an exchange or a layer of several MoE
  groups. The words are the concatenation's.
- On against `=0` (w-pio-20261101-0737; WOA_ROWS and JOINLESS on in every arm): 1132.3 and 1127.8 tok/s cold prefill
  against 1125.7 and 1126.8 (+0.3%); new text 1116.8 and 1137.2 against 1104.9 and 1130.3 (+0.8%). Exact: the seeded
  text is the same in every arm (md5 efeb6f4ecd).
- `MLX_SERVE_DSV41_STAGE` (src/stage.zig, patches/turboexl3-zzz8f-stage.patch; on, `=0` off): a wide prompt call's
  predicted transient loads read while its layer's attention runs. `=0`: P1's read-ahead fills the persistent rows the
  seed leaves free (about 50 records a layer at 16K rows) and the call's stream groups (about 240 transient misses a
  layer once the seed holds every persistent row) are read by their own routes while the groups before them compute. On,
  after P1 the copied stream reads the prediction's stream misses into the windows the call's first stream groups take
  (group g of a call takes window g mod 5); the stream's ticket ring doubles (2048) so their tickets stay unclaimed
  beside P1's and the call's own. The call's first stream groups become the staged experts it routes, each filled up to
  48 from the rest in hot order; a transient miss takes the row of its window holding its record and skips the read; a
  route with a transient load lands its window's staged reads first. `[prompt-io]` adds `staged=` and `adopted=` a
  layer, `[stream-stats prompt]` `stage_posted=`, `stage_adopted=` and `stage_gb=`.
- On against `=0` (w-pio-20261101-0821; WOA_ROWS, JOINLESS, WHOLE_STREAM, EMBED_PREAD and VISION_LAZY on in every arm):
  1171.4 and 1178.4 tok/s cold prefill against 1136.0 and 1120.5 (+4.1%); new text 1204.2 and 1202.6 against 1119.1 and
  1136.1 (+6.7%). Over the cold request's encoder the moe stage took 4470 and 4484 ms against 5476 and 5590, the host's
  waits on bytes 666 and 679 ms against 1516 and 1650; 4722 records staged, 4079 adopted (86%), the rest 7.9 GB of reads
  under the attention, which ran about 25 ms a layer longer with them in flight. Exact: the texts are the same in every
  arm (the cold one md5 efeb6f4ecd).
- `MLX_SERVE_DSV41_TAIL_FUSE` (src/tail_fuse.zig, src/kernels/tail_fuse.metal, patches/turboexl3-zzz8g-tail-fuse.patch;
  on, `=0` off, `check` the kernel checked): a wide prompt layer's tail in one kernel. `=0`: the routed call ends in the
  joinless combine (an f32 [rows, D] sum of the routed rows and the shared expert, itself widened to f32 for it), the
  model rounds the sum to the activation dtype and glm5's hc expand writes the new streams: the widened shared rows, the
  f32 sum and its rounded copy are GB-sized transients of a 16K-row layer (~840 MB), over four passes. On, the source
  runs one kernel over the same inputs (the waves' outputs where they wrote them, the weights, the shared expert as it
  came, the streams, each row's post and comb) and writes the new streams: the combine's fold under `fp contract(off)`,
  one rounding to the activation dtype, then the expand's fmas in its order, so the same words. `check` serves the
  chain's streams and runs the kernel over each wide call's first and last 2048 rows, compared word for word:
  `[tail-fuse] check layer= rows= words= differ= max_abs=` a call (the chain's memory and a few hundred MB). The kernel is
  the joinless combine's: unset, it follows `MLX_SERVE_DSV41_JOINLESS`; set with JOINLESS off, the load is refused.
- On against `=0` (w-pio-20261101-0828, on 7af6430: WHOLE_STREAM on; usage seed with SEED_FREE, A B B A): 1150.0 and
  1161.5 tok/s cold prefill against 1147.1 and 1136.6 (+1.2%); new text 1142.8 and 1157.4 against 1122.9 and 1142.1
  (+1.5%); the guard's sampled peak 105.3 GiB. Exact: the seeded cold text is the same in every arm (md5 efeb6f4ecd), and
  `check` (w-pio-20261101-0809) compared every wide call's first and last 2048 rows, 40 calls and 3,355,443,200 words,
  with none different.

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
