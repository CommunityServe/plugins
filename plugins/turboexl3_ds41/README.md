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
- The routed experts are the repack's `experts.bin` bank. Their decode runs on turboexl3 by default (src/routed_t3.zig;
  `MLX_SERVE_DSV41_ROUTED=streamer` keeps the streamer's own EXL3 kernels, mul1, K2 and K3, throughout), on the
  streamer's schedule: its decode waves, banked and with the device route (patches/turboexl3-t3-banked.patch), on turboexl3's
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
    per ms of verify, and where drafting does not pay the round verifies `t1` alone (the engine's backoff). By default
    the fixed policy, on sampled requests only: the cut and the backoff read a cost table by context (src/rounds.zig
    `Fixed`; the built-in `DEFAULT_COSTS`, fitted on the M5 Max, or `MLX_SERVE_DSV41_DSPARK_COSTS`'s), never the
    measured ladder or a round's wall time, so a request's text is the same run to run; a greedy request's rounds verify
    the whole block (`MLX_SERVE_DSV41_DSPARK_FIXED_GREEDY=1` cuts them too; off by default, aw1s).
    `MLX_SERVE_DSV41_DSPARK_POLICY=1` is the live policy on the ladder the plugin measures from its own rounds; `=0`
    turns it off: every round verifies the whole block.
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
- `MLX_SERVE_DSV41_LOOKAHEAD_WIDE` (src/la_wide.zig, patches/turboexl3-zzz9c-lookahead-wide.patch; on at 10, `=0` off,
  `=N` about N records a round, 1 to 64; with the predictor on): the decode lookahead's recall widened past each layer call's
  budget. On ctl26 a greedy round has 36.2 misses, 13.9 claimed from the lookahead and 22.3 read on demand (12.4 MB and
  about 1.0 ms of verify each); the lookahead reads 617 MB a round, 412 of it unclaimed, at no verify cost the round
  fit can see (-15 ms/GB beside the demand reads), and the drive moves 10.4 GB/s in decode against the ~13 GB/s it
  streams a prompt at: about 240 MB a round to spare. On, the stream is sized for the lookahead's widest budget (4:
  eight staging slots, 51 MiB more in the bill) and a call reads its budget, then while a credit lasts up to 2 more of
  the same ranking: the credit grows N / (layers - 1) at each predicting call and spends one a record read past the
  budget, at most 2 held. N = 10 is half the spare (the speculative class's queue rule still yields to demand reads).
  On or off, a `[lookahead] wide= p= r= xp= xr= p0= r0= ... xp7= xr7=` line at each request's end with the DSpark
  profile: the records read ahead, the budget's (p) and the extra ones (xp), by the top-k position each was chosen at,
  with those the target call routed (r, xr). Exact: the lookahead only reads ahead.
- Read as la2 (M5 Max, w-20261102-0435; code-ctx16384 card then greedy, a = ctl29, b = la2 at 10; a1 b1 b2 a2 b3 a3 a4
  b4): card 44.50 44.70 44.70 44.40 against 44.50 44.70 44.70 44.70 tok/s (+0.17%), greedy 57.10 56.10 57.20 57.10
  against 57.00 56.90 56.80 56.90 (+0.04%); the texts the same bytes and 150 / 179 rows in every arm. Merged as measured
  (both point estimates >= 0), not as a lever: the extra records were chosen (10.1 / 9.8 a round; 17% / 10% routed)
  but hardly read. The b arms' records issued rose by exactly them while records landed (2643 -> 2650 a card request),
  claims a round (20.14 -> 20.25 card, 14.16 -> 14.30 greedy) and the speculative bytes a round (+3 / +4 MB) did not
  move: the speculative class's two threads were busy with the budget's records (640 / 617 MB a round read, 369 / 409
  of it abandoned unrouted), so the extras waited queued behind them and were dropped unstarted at their call's settle,
  or, routed, cancelled by the demand read that needed them.
- `MLX_SERVE_DSV41_DEMAND_FIRST` (src/demand_first.zig, src/demand_first_test.zig; the copied read pool's
  `q3ld_demand_first`, ABI 2026110203, patches/turboexl3-zzz9u-demand-first.patch; on at `2` by default, `=1` the order
  alone, `=0` off): decode's lookahead reads behind its demand reads. The order (`=1`, and part of `=2`): an unclaimed
  lookahead record starts and continues only while no
  demand read is queued or running (on every lookahead thread; the lane's rule let thread 0 read beside one), in 1 MiB
  pieces (doubled until 64 cover a slot) in place of quarter-slot chunks, so a demand read that starts meanwhile waits
  behind at most one piece a thread and the record pauses at its next piece, its landed pieces kept. `=2` (the
  default): also, a claimed record's untaken pieces go to the demand ranges that need them: a deadline-free range whose
  record is claimed
  and not all in reads the record's untaken pieces under its own bytes itself, into the record's staging, beside the
  lookahead thread, and adopts the range once every piece under it is in (ahead of the record's landed prefix). A
  record whose pieces are all taken while demand still reads them is left to its last reader; a failed piece fails the
  record once no reader is left (its ranges then read the stock way). The source sets the pool's mode at each phase
  change (decode's the setting, the prompt's off); a record keeps the mode it started with. `[demand-first]` at each
  request's end: the pieces read, the stops for demand inside a lane chunk and those chunks' rest (MB), mode 2's
  demand-read runs (count, MB, ms), records landed by the demand, releases, early adoptions and failures, and the
  adoption waits and their time. The motivation (ctl48's profile, pt6d): 51% of the card's K = 3 gate/up ranges start
  beside an unclaimed lookahead chunk and take 1.82 ms against 1.59 ms clean, 58% of the lookahead's bytes are never
  claimed, and claimed records still reading cost 11 ms a round on the card, 7 greedy. Exact: the same file bytes into
  the same staging and rows; only which thread reads a piece, and when, changes. Its byte gate
  (src/demand_first_test.zig, the pool's C file alone with its test hooks, in `zig build test`): a record claimed in
  flight, a demand read beside an unclaimed record, a failed piece and rounds of random delays, at each mode, every slot
  row the file's bytes.
  - Read as df1d, `=1` alone (M5 Max, w-20261103-0004; code-ctx16384 card then greedy, a = ctl48 b528be70e900, b = df1
    7e1cad69a020 at 1, both on G_KITSP1 a34826975049 with keyed sampling; a1 b1 b2 a2 b3 a3 a4 b4): card 48.5 48.1 48.4
    48.0 against 47.8 46.9 47.7 47.9 tok/s (-1.40%, every pair negative), greedy 58.5 57.8 58.3 57.7 against 58.0 57.4
    57.9 57.9 (-0.47%); the texts the same bytes in every arm. The claims about the same (20.5 -> 20.4 a round), the
    abandoned records 373 -> 235, the card's verify 1.6 ms longer: with the lookahead yielding to every demand read the
    claimed records land later (`[demand-first]` adopt_waits 5468, adopt_wait_ms 3669 a card request). Not merged alone.
  - Read as df2o, `=2` against `=0` on one binary (df1 7e1cad69a020; the same config). Its first window
    (w-20261103-0019) was invalid on rows: 147 / 176 in a1 b1 b2 a2 b3, 145 / 174 in a3, 146 / 175 in a4 and b4 (the
    bank bill's ceiling moved at load), its numbers card +1.15%, greedy +1.81%. The rerun (w-20261103-0036): the rows
    146 / 175 in every arm, the texts the same bytes (card 8074f27375, greedy 0a1c2b8690); card 48.0 47.4 48.0 48.0
    against 47.9 48.2 48.4 48.3 tok/s (+0.73%; pairs -0.21, +1.69, +0.83, +0.62), greedy 58.0 57.3 58.0 58.0 against
    58.7 58.7 58.8 58.4 (+1.43%; pairs +1.21, +2.44, +1.38, +0.69). Merged on at 2 (zzz9u).
- `MLX_SERVE_DSV41_GU_SPLIT` (src/gu_split.zig; the copied read pool's `q3ld_gu_split`, ABI 2026110202,
  patches/turboexl3-zzz9h-gu-split.patch; on by default, `=0` off, `=check`): decode's demand reads with each missed
  record's gate/up range read as two range jobs of one ticket, side by side, split at the part boundary nearest the
  range's middle (a classic bank's gate/up boundary; an a4k bank's up code alone in the second). One range reads as
  one preadv at about 3.6 GB/s (dq1a's `[demand-split]`, w-20261102-0303: a K = 3 record's 8.88 MB gate/up span in
  2.44 / 1.88 ms on the card / greedy, its 4.44 MB down span in 1.35 / 1.20 ms beside it) against about 13 GB/s for
  the drive at four streams: a missed record's latency was its gate/up span's, and with the halves side by side it is
  about its down span's. Each half is a range of its own (the stock loop, an adoption out of a lookahead record, or a
  pre-range of its own: the pre-read queues the halves while the split is on); the ticket's status words are the
  halves' (the counts summed, the earlier start, the later end) and are published once, when both are in. The source
  sets the pool's mode at each phase change (decode's on, the prompt's off). `check` reads each split range again
  whole and compares it with the rows its halves wrote before the publish. `[gu-split]` at each request's end: split
  tickets, their halves read / adopted / pre-read, half pre-ranges queued, the mean span and both halves' mean overlap
  a ticket, and in check mode the tickets compared and those that differed. Exact: the same bytes into the same rows;
  only the preadv calls covering a range change. With it, dq1a's `[demand-split]` (src/demand_split.zig, ten status
  words a ticket: each range's lookahead flags and wait; dq1a's lane rule left out) and `[cold-start]`
  (src/cold_start.zig): both on with `[round-io]`; `[cold-start]` gives, per request, each layer's first three decode
  calls' distinct and missed experts and of the missed those the layer's last prompt call routed, those its last 8
  rows routed, those among its 6 / 12 / 24 / 48 most voted, and per layer the first call's misses / in the tail / in
  the top 12 (`r0_by_layer`): the size of a phase-change seed of the decode rows from the prompt's last routes.
- Read as dq1b (M5 Max, w-20261102-0820; code-ctx16384 card then greedy, a = ctl29 6e9c8b0a279a, b = dq1b
  301b9c8d08fe at 1, both on G_KIT1271; a1 b1 b2 a2 b3 a3 a4 b4): card 43.8 44.1 43.5 44.2 against 45.5 45.8 45.8 46.0
  tok/s (+4.27%; pairs +3.88, +3.85, +5.29, +4.07), a round 111.56 -> 107.08 ms; greedy 56.1 56.3 55.7 56.4 against
  57.0 57.0 56.5 57.0 (+1.34%; pairs +1.60, +1.24, +1.44, +1.06), a round 87.12 -> 85.88 ms; the texts the same
  bytes in every arm, check's split tickets all compared with no mismatch, the rows the same, the claims not
  collapsing. Merged on (zzz9h).
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
- `MLX_SERVE_DSV41_SCRATCH_WAVES` (src/scratch_waves.zig on src/scratch.zig's `Region`, the twins' texts by
  src/wave_twins.zig, their launches and the wiring in patches/turboexl3-zzz8n-scratch-waves.patch; on by default, `=0`
  off, `check`): a wide prompt call's DIG-X waves write their intermediates into per-prompt regions, not fresh buffers.
  A wave's launches (take2, the gate|up GEMM, onepass, the down GEMM; the GEMMs at K3 and K2) run as write-into twins
  of the stock texts, derived at load from the registry's pinned texts: each output becomes a `device` cast of a
  `<out>_dst` input (a whole region: an evaluated array, so no `contiguous` copy stands in for it), one carrier word
  is the only output, the launch before's carrier is bound as `dep`, every other word is the stock text's.
  rot_widen1 is a dep twin (the stock text with `dep` bound): its y stays MLX's output, the combine's part. Each anchor
  (the `fp contract` pragma where expected, every output named, no name the twin adds) is checked and every twin
  compiled by one launch at load; one that fails refuses the load by name. No manifest, enum or pin changes. Two sets
  of the intermediates (by the wave's parity, at the tier shape's 7168 rows, 2 x 459 MB; two waves in flight never
  share one) carry an uncarried call's waves; a carried call's and a solo wave stay fresh. `check` runs the stock
  launches beside the twins and compares every launch's output (words, those that differ, max abs, max ulps, over
  one ulp). On the full pack (M5 Max, the a4k canonical, --wired-margin-gib 3, EVICT=1; w-pio-20261101-1514, a1 b1 b2
  a2, each a cold code-ctx16384 card request then the new-text 16K slice 3): cold 1195.5 and 1208.0 tok/s against
  1183.4 and 1181.1 (+1.65%), new text 1239.1 and 1217.5 against 1201.8 and 1202.0 (+2.20%), the guard's accounted
  peaks 109.0 / 105.9 GiB on against 102.2 / 108.7 off; check (c1, margin 12): 88,177,669,376 words over 760 waves,
  every launch's output (x0, x1, zg, zu, hd, z, y) 0 words differing, max ulps 0. Exact: the texts identical in every
  arm.
- `MLX_SERVE_DSV41_M1` (src/m1.zig, patches/turboexl3-zzz9k-m1.patch: the four texts, their manifest entries and the
  manifest's pin, the registry tags, the self-check's twins, the quant's route; on by default, `=0` off, `=check`): a
  wide prompt call's DIG-X waves run their 128-row routed GEMMs (K3 and K2, gate|up and down) on texts with the B stage
  double-buffered (`dsv41_prefill_dig_gemm_*_m128d`): stage 0 decoded before the K loop, then per stage the MMA on its
  buffer and the next stage decoded into the other, one barrier a stage. Every output's A fragments, decoded B values
  and tile_matmad_nax sequence are the 128-row text's, so the same z words: the quant's `routeM1` self-checks the four
  texts at load (compile, composition, and a twin check against the 128-row text, bitwise), and a failure refuses the
  load by name. SCRATCH_WAVES' write-into twins derive from the double-buffered texts (the same anchors). `check`
  serves the 128-row texts, launches the double-buffered ones beside them on every wave's inputs (the waves off the
  scratch regions, so every GEMM is compared) and gives `[m1]` at each request's end (per gate|up and down: GEMMs,
  words, differing words, the largest ulp distance).
  - m1u's microbench (w-m1u-20261102-0007; 8 experts a wave, median µs a launch): every word of 28 wave records the
    128-row text's (204,365,824 words); the 16K shape's wave (8 x 256 rows) 1789.5 -> 1707.3 (-4.6%) on K3's
    gate|up, 886.6 -> 878.1 (-1.0%) on its down, 1755.6 -> 1702.4 (-3.0%) and 872.9 -> 861.8 (-1.3%) on K2's;
    every other wave -0.9% to -6.5%.
  - Read as m1 (M5 Max, w-pio-20261102-0941; a cold code-ctx16384 card then the new-text 16K slice 3, 32 tokens,
    a = ctl25 5ccb44791b50, b = m1 3466f8330cbd at 1, both on G_KIT1271; a1 b1 b2 a2): cold 1397.4 1399.6 against
    1397.1 1405.7 tok/s (+0.21%; pairs -0.02, +0.44); new text 1489.8 1474.6 against 1487.4 1501.7 tok/s (+0.83%;
    pairs -0.16, +1.84); the cold card's encoder MoE stage 4102 4097 against 4087 4057 ms; the texts the same bytes
    in every arm, check's `[m1]` differ 0 and max_ulps 0 over 1428 and 1334 GEMMs of each kind, the bank rows 149 /
    174 in every measured arm, the guard's accounted peak not up (b 106.7 against a 107.8 GiB). Merged on (zzz9k).
- `MLX_SERVE_DSV41_SEED_FREE` (src/seed_free.zig, the policy in patches/turboexl3-zz5-seed-free.patch, tested by
  src/seed_free_test.zig): the rows the usage seed fills stay free until a route uses their expert. The prompt's
  read-ahead may take one for an expert its prediction ranks before the seeded one, so a layer's rows hold its call's
  hottest experts as they would have held them empty: a 16K-row prediction names nearly every expert, and without the
  rule a seeded server's read-ahead found no row to take (27 records posted against 2413 unseeded, w-20261101-0307).
  The decoder prefetch may take one whose expert its prediction does not name. A seeded row a route has used is a
  resident as any other. Cold code-ctx16384 prompt on the full pack (M5 Max, w-pio-20261101-0338, a usage profile
  recorded on prompts apart from the test case): 997.3 tok/s against 991.9 and 986.4 for the seed alone; the encoder's
  read-ahead read 11.5 GB against 0.36, its read waits 0.95 s against 1.11 and 1.14. On by default (the measured
  policy, with the pack's usage profile), `=0` off.
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
- `MLX_SERVE_DSV41_SHFIN` (src/shfin.zig, src/shfin_t3.zig, patches/turboexl3-zzz9p-shfin.patch; turboexl3's
  `sharedSwiglu`: root.zig, `Dense.swigluPrep` and kernels/dense/swiglu_prep.metal; on, `=0` the host's chain,
  `=check`): a decode layer's shared expert with the gate's and up's finishes, the clamped SwiGLU and the down's prepare
  in one launch (F3). At a verify's rows (5 to 16: DEC-ONE keeps up to 4 as they are) the shared expert ran the gate's
  and the up's prepare, GEMV and finish, the host's clamped SwiGLU in f32 (one compiled elementwise launch in the
  stretch), then the down's prepare, GEMV and finish. On, the two finishes, the SwiGLU and the down's prepare are one
  launch (`dense.text_swiglu_prep`: for each 128-wide block, text_dfinish's statements on the gate's and the up's
  planes, MLX's Minimum, Maximum, Sigmoid and Multiply functors in clampedSwiGLU's order in f32, the activation dtype's
  rounding, then text_prep_s's statements, nothing between them stored): three launches and two dependent levels fewer
  a layer, the same words (turboexl3's CPU test holds the statements). The output is the chain's words, so what reads it
  is unchanged: SHARED_HOIST's `ffn_shared` stretch hands it to DECODE_FUSE's kernel or to `ffn_post_shared`, or
  `ffn_post`'s `withShared` adds it. `check` serves the host's chain and, for every shared expert F3 takes, runs F3 on
  the same input and compares the down's output word for word (after a compiled stretch, its body once more, eagerly).
  `[shfin] mode= calls= words=` at each request's end, since load (on, the calls F3's code ran: a compiled stretch's
  traces, its replays running F3 uncounted; under check every call, and `differ=`).
  - The probe (f3u, w-f3u-20261102-1410): every word the host chain's at rows 1 to 16 (taken at 5 to 16; bf16, and f32
    at 6 and 12); the 40-layer trunk chain at rows 5, 6 and 8 -0.056, -0.029 and -0.218 ms a verify against the host's
    chain (medians, A B B A in 3 rounds, every run the first's words).
  - Read as f3 (M5 Max, w-20261102-1413; code-ctx16384 card then greedy, a = ctl27 42a2a45fae3c, b = f3 at 1; a1 b1 b2
    a2 b3 a3 a4 b4): card 44.0 44.0 44.1 44.3 against 44.4 44.4 44.3 43.9 tok/s (+0.34%; pairs +0.91, +0.91, +0.45,
    -0.90; round 111.07 -> 110.75 ms); greedy 56.4 56.0 56.2 56.3 against 56.3 56.3 56.4 56.3 (+0.18%; pairs -0.18,
    +0.54, +0.36, 0.00; round 86.95 -> 86.85 ms); the texts the same bytes in every arm, check's `[shfin]` differ 0
    over 9503 calls (4899 by the card's end), the bank rows 147 / 176 in every measured arm. Measured on zzz8y (before
    DECODE_FUSE, GU_SPLIT, HW4B and HW2); merged on (zzz9p).
- `MLX_SERVE_DSV41_T3_GROUP` (src/t3_group.zig, patches/turboexl3-zzz8b-t3-group.patch; on with the t3 route, the
  default, `=0` off): the t3 route's device hit wave on turboexl3's grouped banked decode
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
- `MLX_SERVE_DSV41_HW4B` (src/hw4b.zig, patches/turboexl3-zzz9j-hw4b.patch; turboexl3's `Config.banked_m3`; on by
  default with the t3 route's grouped banked decode, `=0` off, `=check`): the t3 route's banked calls (the decode hit
  wave's and each miss part's) on turboexl3's grouped texts with an M = 3 member path. A lead with three row-slots of
  one expert runs it, where the grouped texts ran the M = 4 path with a fourth member's FMAs padding. Every member is
  fed in the one-member order (turboexl3's `turboexl3_gemv_group`), so every row-slot's words are the grouped texts'.
  Unset is off without the grouped banked decode (`MLX_SERVE_DSV41_ROUTED=streamer` or `MLX_SERVE_DSV41_T3_GROUP=0`);
  `1` or `check` without it is refused. `check` runs the grouped texts' call and the M = 3 one for every banked call,
  compares their f32 words on the device (a sync a call) and serves the grouped texts' rows. `[hw4b]` at each
  request's end: the mode, turboexl3's `banked_m3`, and in check the calls, words, differing words, the largest ulp
  distance and absolute difference.
  - The microbench (hw4u, w-hw4u-20261102-0359; hw5u, w-hw5u-20261102-0648, reproduced it): every word of hd, hs and y
    the grouped texts' at rows 1-64 with one id on 1 to 16 row-slots, K3 and K2; -1.5 to -16.8 µs a banked call on
    the recorded routes' profiles, the 3-member sweeps -9.8% to -10.9%, flat with no 3-member lead.
  - Read as hw4b (M5 Max, w-20261102-0923; code-ctx16384 card then greedy, a = ctl29 6e9c8b0a279a, b = hw4b
    502c5238bc37 at 1, both on G_KIT1271; a1 b1 b2 a2 b3 a3 a4 b4): card 44.1 44.2 44.0 44.3 against 44.0 44.4 44.4
    44.1 tok/s (+0.17%; pairs -0.23, +0.45, +0.91, -0.45); greedy 56.2 56.4 56.0 56.2 against 56.1 56.6 56.6 56.3
    (+0.36%; pairs -0.18, +0.35, +1.07, +0.18); the texts the same bytes in every arm, check's `[hw4b]` differ 0 and
    max_ulps 0 over 8498 and 6949 calls. Merged on (zzz9j).
- `MLX_SERVE_DSV41_HW2` (src/hw2.zig, patches/turboexl3-zzz9m-hw2.patch; turboexl3's `group_variants`,
  `Config.banked_variant_gu` and `banked_variant_dn`; on by default with the t3 route's grouped banked decode, `=0`
  off, `=check`): the t3 route's banked calls with the grouped gate|up and down GEMVs at hw2u's picked variant, t4 for
  both, on the grouped texts the calls run (hw4b's M = 3 texts with `MLX_SERVE_DSV41_HW4B` on). t4 is the texts with
  `EXL3_T1V 4`: a codeword's two pair sums become t as one half2 (1024 plus each pair's sum), widened, added and 2048
  subtracted, every step on integers below 2^24, so t is today's route's word for word, with one shift and one add a
  weight fewer. Every variant reads, decodes and sums each weight as today's texts do (the same codewords, decoded
  values, FMA sequence, accumulators and folds), so every row-slot's words are the same. Unset is off without the
  grouped banked decode; `1` or `check` without it is refused, and so is `check` beside `MLX_SERVE_DSV41_HW4B=check`.
  `check` runs the call at variant 0 and at the picks for every banked call, compares their f32 words on the device (a
  sync a call) and serves variant 0's rows. `[hw2]` at each request's end: the mode, the picks, and in check the calls,
  words, differing words, the largest ulp distance and absolute difference.
  - hw2u's microbench (w-hw2u-20261102-0023; the whole banked call at a verify's 36 pairs on the recorded p10, median
    and p90 profiles, chain medians, µs a call): today's 902.0, 1029.1 and 1239.3 at K3, 888.3, 981.6 and 1162.9 at
    K2; t4 at both GEMVs -24.3, -21.5 and -21.3 at K3, -17.4, -21.1 and -12.6 at K2; the ten other variants (tile
    pairs, a rolling prefetch, x held as half4, the four-member path as two passes, shuffled tile words, the other
    routes to t and three mixes) slower in every cell. Every variant's words today's (968 records, 100,859,088 words),
    and every route to t exact over all 65536 codeword states read from I16 and U16 words.
  - Read as hw2 (M5 Max, w-20261102-1025; code-ctx16384 card then greedy, a = ctl25 5ccb44791b50, b = hw2
    d43b09222a94 at 1, both on G_KIT1271; a1 b1 b2 a2 b3 a3 a4 b4): card 43.9 43.9 43.6 43.9 against 44.0 43.9 43.6
    44.1 tok/s (+0.17%; pairs +0.23, 0.00, 0.00, +0.46); greedy 55.8 55.9 55.6 55.6 against 56.0 56.1 55.2 55.8
    (+0.09%; pairs +0.36, +0.36, -0.72, +0.36); the texts the same bytes in every arm, check's `[hw2]` differ 0 and
    max_ulps 0 over 8531 and 7018 calls, the bank rows 150 / 174 in every arm. Measured on today's grouped texts
    (before hw4b); merged on both (zzz9m).
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
- `MLX_SERVE_DSV41_WOA_JOIN` (src/woa_join.zig, patches/turboexl3-zzz9b-woa-join.patch; on, `=0` off, `=check`
  checked): a wide layer's EXL3 `wo_a` run (WOA_ROWS' run of spans, read from its SCRATCH_HEADS region) without
  turboexl3's per-group copies, launches and decodes. turboexl3's grouped linear at prefill rows (root.zig
  `groupedPrefill`) runs `linear` on each of the 8 groups: a strided column slice of the run's rows, which MLX copies
  whole for the prepare; the prepare; the group's W_hat decoded from its trellis; MLX's matmul; the finish; then one
  concatenation: 41 launches a run, and each of a 16K layer's runs decoding the same 8 W_hat again. On: one prepare
  launch for every group reading the run's rows in place, each group's W_hat decoded once a layer (turboexl3's own
  `Dense.dequantize`) and kept across the layer's runs, MLX's matmul of each group on the same operands, one finish
  launch writing the run's joined rows: 10 launches a run, 18 at a layer's first. The prepare and finish are
  turboexl3's prep.metal and finish.metal at the pin with only their index expressions moved (no multiply-add a compiler
  could contract), so the same words. The kept W_hat (67 MB at the pack's shape) is billed in the prompt term only
  (`prefillBytes` reads the setting before the load, for calls wider than a span: decode's calls are never wide). A
  run another path would take (decode or register-path rows, turboexl3's fused-GEMM arm) keeps the stock call.
  `check` runs both on every run, serves the stock run and logs `[woa-join] check layer= runs= equal= words= differ=
  max_ulps=` a wide layer.
- On against `=0` (wj1, w-pio-20261102-0405, two binaries: a = ctl26 (2d0a33f, zzz8x), dylib 3030784171b2; b = this
  code on zzz8x with the switch off by default, dylib 061842336690, `=1` in the b arms; a warm server, the check arm
  at margin 3, then a1 b1 b2 a2): cold 16K a 1409.1/1406.1 -> b 1446.9/1440.6 tok/s (+2.57%, pairs +2.68% and
  +2.45%), new text (slice 3) 1496.6/1495.2 -> 1518.5/1536.0 (+2.10%, pairs +1.46% and +2.73%); 152 prompt rows in every
  arm; the guard's peak 111.0 GiB. That build also billed the kept W_hat in the decode term (ws_decode_mb 804 -> 884,
  178 decode rows against 179); here the prompt term only, so decode keeps 179. Exact: the texts the same bytes in
  every arm (cold d081845201, new-text slice 3 4d5b401f3c), and the check arm's 41 wide-layer lines compared
  4,026,531,840 words, none different (max_ulps 0).
- `MLX_SERVE_DSV41_PG2` (src/pg2.zig, src/pg2_t3.zig, patches/turboexl3-zzz9d-pg2-fused-finish.patch; turboexl3's
  `pgemm`: src/pgemm.zig, src/kernels/pg2/ and its hook in dense.zig's `linearOn`; on, `=0` off, `=check` checked): a
  prompt's dense EXL3 linears' matmul and finish as one launch. turboexl3's `linearOn` at prefill rows runs
  the prepare, the W_hat decode, MLX's f16 matmul (z half [rows, N] written: MLX's fused NAX kernel, on M5 Max at its
  64 x 128 tile, BK 256 and 2 x 4 simdgroups) and the finish (z read back, y written: f16(H128(f16(z)) * svh) per
  128-column block, kernels/dense/finish.metal). On, at a shape of turboexl3's `pgemm.table`, one kernel runs the two:
  MLX's fused NAX GEMM body and its gemm loop (MLX 0.32.3's steel_gemm_fused_nax.h `gemm` and gemm_nax.h `gemm_loop`,
  their statements: SK 32, the 16 x 32 x 16 multiply-accumulate on the f16 operands, f32 accumulation, k in MLX's
  order) at the tile the pg2u probe measured for that (K, N) on this machine, launched in MLX's own threadgroup shape
  (32, WN, WM), with the finish's statements as its epilogue: the accumulators rounded to half as MLX's store rounds
  them, staged in threadgroup memory, and the finish run per (row, 128-column block) on z's halves read from there. The
  tile only nests BK's k-steps and moves which simdgroup computes which outputs (each simdgroup's N extent a multiple of
  32: the MMA's column pairs as MLX's), and a call is taken only where MLX runs that regular kernel (no split K), so y
  is the chain's y word for word, and z (about 1 GiB at a 16K prompt's wq_b) is never allocated. The table (BM x BN,
  BK, WM x WN simdgroups, swizzle 2 throughout; the fused launch's time over the chain's at 16384 / 7168 / 2048 rows,
  w-pg2u-20261102-0609; one tile a shape whatever the rows): wq_b (K 1280, N 32768) 32 x 256, 256, 1 x 4 (0.866 /
  0.870 / 0.900); wo_b (8192, 5120) 32 x 256, 512, 1 x 4 (0.957 / 0.973 / 0.899); the shared expert's w1 and w3 (5120,
  2304) 32 x 256, 512, 1 x 4 (0.933 / 0.984 / 0.982); its w2 (2304, 5120) 32 x 256, 256, 1 x 4 (0.938 / 0.933 /
  0.950); wq_a (5120, 1280) 32 x 256, 512, 1 x 4 (0.963 / 0.920 / 0.963); wkv and the compressors' wkv and wgate (5120,
  512) 64 x 256, 512, 2 x 4 (0.986 / 0.951 / 0.962). `wo_a` (WOA_JOIN's runs, or turboexl3's grouped linear) is not in
  the table, and decode's register path never reaches the matmul. The table's kernels compile at load (the
  activations' dtype, and f32 for (5120, 512): the compressors run on f32 rows), so a prompt's first call compiles
  nothing. `check` serves the chain and, at every call pg2 would take, runs pg2's launch on the same operands and
  compares the two y word for word (one host read a call); it holds both outputs, so its arm runs at margin 3. A
  `[pg2] mode= launches= elems=` line at each request's end (counts since load; `checks= words= differ=` under
  check).
- On against ctl31 (pg2, w-pio-20261102-0633, two binaries on the G_KITSP1 host with keyed sampling: a = ctl31 (a407f1f,
  zzz9c), dylib 2c8d5488e421; b = this code on zzz9c, dylib 833ade5320fb, `=1` in the b arms; a warm server, the check
  arm at margin 3, then a1 b1 b2 a2): cold 16K a 1459.5/1460.3 -> b 1483.2/1490.7 tok/s (+1.85%, pairs +1.62% and
  +2.08%), new text (slice 3) 1551.4/1555.2 -> 1584.9/1584.7 (+2.03%, pairs +2.16% and +1.90%); 240 fused launches a
  16K card (16,310,922,240 output elements); 148 prompt and 178 decode rows in a1, b1 and b2 (a2 177 decode rows); the
  guard's accounted samples 98.9 and 99.9 GiB in the b arms against 108.9 and 105.3 in the a arms. Exact: the texts the
  same bytes in every arm (cold 65d438d647, new-text slice 3 83a88208e3), and the check arm's 480 calls compared
  31,172,213,760 words, none different.
- `MLX_SERVE_DSV41_DEC_INVARIANT` (src/dec_invariant.zig, src/gemv_twin.zig, src/kernels/gemv_twin.metal,
  patches/turboexl3-zzz9r-1-idx-twins.patch; on, `=0` off, `=check` checked): the indexer's plain bf16 linears (each
  index layer's head weights `weights_proj`, 32 x 5120, and each KV source's index key `wk`, 128 x 512) give a decode
  row the same words at any verify width. MLX runs them as its M=1 gemv for one row, gemv_wide for 2..15 rows and its
  GEMM past them, so a verify's rows round apart from the same rows decoded one at a time. On, a call of 2..6 rows runs
  on a batched twin of MLX's M=1 gemv (`GEMVKernel::run`'s body at MLX 64ea011): a thread's K elements, its products,
  its simdgroup shuffles and its threadgroup partials are MLX's one-row call's (fixed by MLX's sn, tn and bn), so each
  row is that call's words, and the weight is read once a launch. The free axes (the rows a threadgroup, a thread's
  rows, the simdgroup rows, the chunks' place in the grid) move no word; `served` picks them per weight shape and rows
  from this machine's timings. One row keeps MLX's gemv (the same words) and wider calls MLX's matmul. `check`
  compares every twin call with MLX's own M=1 matmul of each of its rows and logs `[gemv-twin] check N= K= calls= rows=
  words= differ= max_ulps=` a weight shape every 128 calls, at its first differing word and at the model's end, and
  the call's own matmul against the same words (`[gemv-twin] wide-vs-m1`, reported only).
  - Read as iv1, with `MLX_SERVE_DSV41_ENGRAM_TWIN` (both twins in one build; M5 Max, w-20261102-1818; a = ctl45
    d572b1c44375, b = iv1 69cc90910e93, both on G_KITSP1 with keyed sampling and the standard decode config, every round
    the whole block; chk (b, both twins at `check`) first, g_a / g_b at MTP:0, then a1 b1 b2 a2 b3 a3 a4 b4, each a
    code-ctx16384 card then greedy, 1024 tokens): exact. chk's check lines differ 0 / max_ulps 0 on every twin shape
    (weights_proj 1816 calls, 10896 rows; wk 911 calls, 3417 rows; the Engram wkv 456 calls, 2740 rows); every b greedy
    the 1-row text (4d36d6e8aa2a87ac, g_a's and g_b's), the control's DSpark greedy (5ab2578db940c51c) not; the card the
    keyed 1-row card (d63fabe7ba2cbb39) in all eleven arms; 6 rows every round in every arm. The card, the same text in
    both arms: 48.4 49.1 48.8 48.9 against 48.8 48.8 48.8 48.6 tok/s (-0.10%; pairs +0.83, -0.61, 0.00, -0.61), 122
    rounds in each arm, a round's wall ms 102.13 -> 102.23, misses a round 56.19 -> 56.56. The greedy, a different text
    by design (the control's DSpark greedy is not its 1-row text; b's is), is
    information only: 57.4 58.0 57.9 57.9 against 58.5 58.0 58.5 58.4 tok/s (+0.95%), 107 rounds against 105. A
    correctness fix (a verify row's words its own at any width), merged on its exactness gates. Measured on zzz9q;
    merged on (zzz9r).
- `MLX_SERVE_DSV41_ENGRAM_TWIN` (src/engram_twin.zig, on src/gemv_twin.zig;
  patches/turboexl3-zzz9r-2-engram-twin.patch, eg2; on, `=0` off, `=check` on and checked): an Engram layer's `wkv` (a
  plain half-type linear: bf16 [L, 6144] x [25600, 6144]^T) at 2 to 16 rows runs on the batched twin of MLX's one-row
  gemv, so a verify row's keys and value are the words that row gets decoded alone (MLX runs gemv_wide at 2 to 15 rows:
  ip1, w-20261102-0143, the only break at widths to 6); one row keeps MLX's gemv (the same words), a wider call its
  matmul. The twin's served launches for the shape (dvi4u): r2 t1, r3 t1h, r4 t1, r5 and r6 c3t2i, compiled at load.
  `check` serves the twin and compares every row with MLX's one-row matmul (`[gemv-twin] check N=25600 K=6144` lines);
  `[engram-twin] mode= served= declined=` (and in check the request's calls, rows, words, differ, max_ulps) at each
  request's end. No second weight is held in check.
  - Read as iv1 with `MLX_SERVE_DSV41_DEC_INVARIANT` (above: w-20261102-1818): exact, the Engram wkv's check lines
    differ 0 over 456 calls and 70,144,000 words, every b greedy the 1-row text. Measured on zzz9q; merged on (zzz9r).
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
- `MLX_SERVE_DSV41_DECODE_FUSE` (src/decode_fuse.zig, src/tail_fuse.zig's decode instance, patches/turboexl3-zzz9l-decode-
  fuse.patch; on by default, `=0` off, `=check` the chain served and the kernel compared; needs
  `MLX_SERVE_DSV41_SHARED_HOIST`, on by default): a
  decode-lane routed call's tail in TAIL_FUSE's kernel. A decode call (at most 48 routed ids) ends in the streamer's
  join (its outputs concatenated, then taken in routed order), the source's weight and sum (a multiply, then MLX's
  column reduce over the six) and the model's `ffn_post_shared` stretch (the shared expert added in f32, rounded,
  glm5's hc expand). TAIL_FUSE's kernel computes that chain's words for a wide call; on, the model builds the shared
  expert first and hands it to the source, which commits it behind the call's waves (where SHARED_HOIST does); the
  call ends at the join's rows and their (source 0, row) table (`Hook.routedHoistLocated`, JOIN_INDEXED's order) and
  the kernel writes the layer's new streams. Its decode instance (`tail_fuse.expandDecode`: NROWS fixed at 16, the
  grid's own bound never reached) is
  one pipeline for every decode call, compiled at load. `=check` serves the chain from the same rows and compares the
  kernel's streams with it on every decode call, word for word. `[decode-fuse] mode= calls= checks= words= differ=
  max_abs=` at each request's end. Exact: the chain's operations in its order, one launch.
- Read as jc1 (M5 Max, w-20261102-1043; code-ctx16384 card then greedy, a = ctl25 5ccb44791b50, b = jc1 3ba9ae0d352e
  at 1, chk at check; a1 b1 b2 a2 b3 a3 a4 b4): card 44.1 44.0 43.9 43.7 against 44.3 44.3 43.8 44.4 tok/s (+0.63%;
  pairs +0.45, +0.68, -0.23, +1.60), greedy 55.7 56.1 55.6 55.7 against 56.0 55.9 56.1 56.1 (+0.45%; pairs +0.54, -0.36,
  +0.90, +0.72); the texts the same bytes in every arm, chk's [decode-fuse] differ 0 over 4640 + 8960 checks, the
  kernel's calls in every b arm. Merged on (zzz9l).
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
- `MLX_SERVE_DSV41_WIDE_EARLY` (src/wide_early.zig, patches/turboexl3-zzz8y-wide-early.patch; on, `=0` off, `check`
  checked): what a wide prompt call's MoE does not read, committed behind its routes before the routing barrier waits on
  them (the TAIL_FUSE path). `=0`: the barrier evaluates the routes and waits; the next layer's predicted routes
  (STAGE_NEXT) start behind it once the ids are on the host, and the shared expert and the layer's attention state are
  evaluated by the layer's last eval, after the call's groups: the GPU idles through the barrier's host work and the
  call's first read wait (about 37 ms a 16K-row encoder layer, pt3 w-pio-20261101-2207), then runs the shared expert
  (about 25 ms) at the layer's end. On, the copied model commits the routes (cast to the int32 ids the streamer reads,
  its own cast made here), then the prediction, then the shared expert with the attention state, each its own commit;
  the barrier waits on the first only, and the layer's last eval is left the tail kernel. The same kernels over the same
  inputs, so the same words; the MoE stage holds the shared expert's evaluation (every array of it is made when the host
  encodes it ahead of the call: 1.03 GB at 16K rows), which the prompt bill counts (`prefillBytes` reads the setting
  before the load; only calls wider than a span, so not decode's bill), so the bank opens with the prompt rows the rest
  affords. `check` also builds the shared expert the old way and serves it (evaluated after the call's groups), compared
  with the early one word for word: `[wide-early] check layer= words= differ= max_ulps= max_abs=` a wide call.
- On against `=0` (hg1, w-pio-20261102-0114, two binaries: a = ctl25 (2ff23cb, zzz8w), b = this code on zzz8w, =1 in the
  b arms; a warm server and the check arm (margin 3), then a1 b1 b2 a2): cold 16K a 1414.0/1400.3 -> b 1418.6/1425.1
  tok/s (+1.04%, pairs +0.33% and +1.77%), new text (slice 3) 1490.4/1486.2 -> 1498.1/1502.6 (+0.81%, pairs +0.52% and
  +1.10%); the encoder's moe stage 4105/4147 -> 4054/4060 ms, its read waits 645/699 -> 616/627; 153 -> 150 prompt rows
  (the bill's 1.03 GB) and 177 decode rows in every arm; the guard's peak 110.9 GiB. On zzz8x the prompt bill takes the
  same bytes and the decode bill none (DECODE_STOP's cut is the decode budget's; decode's calls are never wide), so the
  16K cells' 150 prompt rows and zzz8x's 179 decode rows. Exact: the texts the same bytes in every arm (cold d081845201,
  new-text slice 3 4d5b401f3c), and the check arm compared 41 wide calls' shared experts, 3,205,017,600 words, none
  different (max_ulps 0).
- `MLX_SERVE_DSV41_REST_ASYNC` (src/rest_async.zig, patches/turboexl3-zzz9s-rest-async.patch; on, `=0` off): a wide
  prompt layer followed by another wide one has its last eval committed, not waited for. `=0`: `runLayers` evaluates
  each wide layer's held arrays and attention state and waits: since REST_EARLY and HC_FUSE the tail is ~17.5 ms of GPU
  work a 16K-row encoder layer and the host waits out most of it, then the next layer's start keeps the GPU idle 5.9 ms
  a regular wide layer (pt5 w-pio-20261102-1502; ~0.11 s a cold 16K card). On, the same arrays are committed
  (`mlx_async_eval`) when the next layer in the pass is wide too, so the next layer's graph and its read-ahead's first
  buffers are built under the tail; the pass's last wide layer is waited for as before. The same kernels in the same
  order on the one stream, so the same words (a text gate). The layer's teardown has already run, so the next layer's
  first arrays overlap only the tail's own inputs and outputs (no bill term; the window's peak gate checks it). With
  TRANSIENT_KEEP the next layer's start hands the kept projection buffers to MLX's cache while the tail runs: their
  readers (the attention) finished before the routing barrier, and MLX recycles a buffer only after its last command
  completes. Prompt only: decode's layers are never wide.
  - Read as ra2 (ra1's lines on zzz9r; M5 Max, w-pio-20261102-1943; a = ctl46 949ce432e7ff, b = ra2 at 1
    (830e9e1d94fe), both on G_KITSP1 with keyed sampling; a warm server, then a1 b1 b2 a2, each a cold code-ctx16384
    card request then the new-text 16K slice 3): cold 1604.0 1612.1 against 1617.2 1623.2 tok/s (+0.76%; pairs +0.82,
    +0.69), new text 1738.5 1700.7 against 1748.8 1721.2 (+0.90%; pairs +0.59, +1.21); the texts the same bytes in
    every arm (cold 65d438d647, new-text slice 3 83a88208e3), the bank rows 147 / 177 in every arm, the guard's
    accounted peak 103.8 / 100.6 GiB in the b arms against 104.7 / 106.6 in the a arms (step footprint 91.2 / 88.1
    against 91.4 / 93.9); the encoder's other stage past layer 0 4208 / 4189 ms against 4319 / 4294. Measured on
    zzz9r; merged on (zzz9s).
- `MLX_SERVE_DSV41_REST_EARLY` (src/rest_early.zig, patches/turboexl3-zzz9e-rest-early.patch; on, `=0` off): a wide
  prompt layer's new streams (its routed call's tail kernel, the TAIL_FUSE combine and hc expand, with its inputs' small
  ops) committed at the call's end, before the layer's teardown. `=0`: `wideLayer` returns after its last group's holds,
  its scopes free every intermediate the layer made (the attention's GB-size transients among them, mostly back to Metal
  past MLX's 2 GiB cache), and only then does `runLayers` evaluate the held arrays, the new streams first: the GPU idles
  about 20.7 ms a 16K-row encoder layer (pt4 w-pio-20261102-0613's pk arm: 5.8 ms from the call's end to the eval's
  first commit, 3.5 ms between its first two small kernels, 11.5 ms for the new streams' fresh allocation), then runs
  the tail kernel (4.8 ms). On, the last group commits the new streams (`mlx_async_eval`) right after its holds, so the
  GPU runs the tail kernel while the host tears the layer down. The same kernels over the same inputs, only committed
  earlier, so the same words (a text gate). The holds have just freed the previous streams, so the new ones take their
  place in the live set and the peak is the old flow's just before its holds (the bill's `held` term counts two stream
  sets a row).
- On against `=0` (re1, w-pio-20261102-0740, two binaries: a = ctl31 (a407f1f, zzz9c), b = this code on zzz9c, =1 in the
  b arms, both on G_KITSP1 with keyed sampling; a warm server, then a1 b1 b2 a2): cold 16K a 1451.8/1441.3 -> b
  1459.4/1446.4 tok/s (+0.44%, pairs +0.52% and +0.35%), new text (slice 3) 1543.8/1518.1 -> 1543.3/1584.0 (+2.14%,
  pairs -0.03% and +4.34%); 147 prompt and 176 decode rows in every arm; the guard's accounted peak 106.4/100.2 GiB in
  the b arms against 107.8/107.8 in the a arms. Exact: the texts the same bytes in every arm (cold 65d438d647, new-text
  slice 3 83a88208e3).
- `MLX_SERVE_DSV41_TRANSIENT_KEEP` (src/transient_keep.zig, patches/turboexl3-zzz9q-transient-keep.patch; on, `=0`
  off, `=check` on and counting): a wide prompt layer with another wide one after it keeps its attention projections'
  large outputs (the attention norm's rows, `wq_b`'s heads, their roped tail: 1.4 GB at 16K rows, each at least 64 MiB)
  past its teardown by an extra reference, and the next wide layer's start hands their buffers to MLX's cache. `=0`: the
  layer's scope frees them at its end, past MLX's cache limit (the routed call's freed transients already fill it), so
  they go back to Metal and the next layer allocates them anew as it encodes them, ~15 ms of host a GB in a host-bound
  stage (pt5 w-pio-20261102-1502's one-op-a-buffer arm: `wq_b`'s 1.07 GB output's encode 19.2 ms of host a regular wide
  layer, the GPU idle 5.1 ms of it). On, `release` lifts the cache limit, frees the references (each buffer to the cache,
  the newest there) and puts the limit back at once; the layer's first allocation trims the cache from its oldest
  buffers, and its same-shaped arrays take these. The same kernels over the same inputs, only the buffers move, so the
  same words (a text gate). The kept arrays were alive through the routed call either way, so after the teardown they
  sit beside only the tail's in-flight buffers; the cache is over its limit by their bytes only from the next layer's
  start (its graph build) to its first allocation. `[transient-keep] mode= layers= arrays= mb=` (`reused=` with check:
  the kept buffers the next layer's own projections took) at each request's end.
  - Read as tk1 (M5 Max, w-pio-20261102-1746; a = ctl44 ae169b3ed1b3, b = tk1 at 1 (2d9981be3fd8), both on G_KITSP1
    with keyed sampling; a warm server, then a1 b1 b2 a2, each a cold code-ctx16384 card request then the new-text 16K
    slice 3): cold 1597.9 1602.2 against 1612.1 1608.2 tok/s (+0.63%; pairs +0.89, +0.37), new text 1710.4 1712.0
    against 1732.7 1728.0 (+1.12%; pairs +1.30, +0.93); the texts the same bytes in every arm (cold 65d438d647,
    new-text slice 3 83a88208e3), the bank rows 148 / 178 in every arm, the guard's accounted peak 108.7 / 104.2 GiB in
    the b arms against 103.8 / 108.8 in the a arms (step footprint 96.6 / 92.1 against 91.7 / 96.7); `[transient-keep]
    layers=19 arrays=57` a request (mb=24909 the card, 22686 the slice). Measured on zzz9p; merged on (zzz9q).
- `MLX_SERVE_DSV41_DIRECT_PARTS` (src/direct_parts.zig, src/direct_parts_test.zig, patches/turboexl3-zzz8i-direct-parts.patch:
  the copied read pool's PARTS route; on, `=0` off): a record is
  read as its gate/up span and its down span, each in one F_NOCACHE preadv into a worker's 9 MiB staging buffer and a
  copy of every part into its slot row. On, the pool tests each part as it reads it: a part whose file offset, length
  and row address lie on the file's block (fstatfs f_bsize, 4,096 B) is read straight into its row (no staging, no
  copy), each run of the others staged and copied as before. On this pack only the gate's code passes (at offset 0 of a
  record padded to 4,096 B, 1,474,560 x K bytes; the up and down codes sit 2,560 and 1,024 B off the block): a third
  of every prompt-phase demand read's bytes no longer copied (its routes, read-ahead and staged loads, the CED
  decoder's replay reads); the decode phase's reads, the pre-read ranges and the lookahead's records stage as before
  (the source switches the route off at each decode phase: decode's misses wait on the drive, not the copy). Exact: the
  file's bytes land in the rows either way (`zig build test`'s direct_parts gate reads the pack's K = 3 and K = 2
  records both ways and across the phase switch: the same slot bytes and status words, the counters exactly the gate
  codes read while on). `[direct-parts] prompt|request parts= direct_gb= staged_gb=` at each prompt's and request's end.
- On against `=0` (dp1, w-pio-20261101-1006, on d451beb with the route in both phases; staging on, policy off, A B B A):
  1199.5 and 1200.4 tok/s cold prefill against 1195.7 and 1192.7 (+0.48%); the encoder's attention 268.0 and 267.5 ms a
  layer against 274.2 and 275.5 (-7.1 ms); 35.81 of the cold prompt's 107.8 GB read direct; the texts the same in every
  arm (the cold one md5 d081845201). Decode (dd1, w-20261101-1015, 16K card + greedy): the same misses a round (78.5 and
  60.0), verify 118.1 / 99.9 ms a round against 114.8 / 99.5 at 0 and decode 40.7 / 47.9 tok/s against 41.8 / 48.1: no
  gain there, so the route is the prompt's only.
- `MLX_SERVE_DSV41_Q_TAIL` (src/q_tail.zig, src/kernels/rope_tail_only.metal, src/kernels/attn_qk_t.metal,
  patches/turboexl3-zzz8h-q-tail.patch; on, `=0` off, `check` checked): a wide prompt layer's query heads are not copied
  whole to rope their last channels. `=0`: `qkvProj` writes the chunk's roped heads [rows, 64, 512] (bf16, 1.07 GB at
  16K rows) from `wq_b`'s output: the rope kernel copies the 448 unroped channels of every head as they are and ropes
  the last 64. On, the wide layer keeps `wq_b`'s output as the unroped heads and ropes only each head's last 64 channels
  into [rows, 64, 64] (rope_tail.metal's arithmetic on the same pairs); the gather's score kernel reads the unroped
  heads for its first seven 64-deep steps and the roped tail for the last: the same bf16 operands in the same steps on
  the same tensor op, so the same scores. A span the gather declines, and a trace, get the whole roped heads as before.
  `check` serves the whole roped heads and also ropes the tail apart and runs the two-part score kernel over every span,
  each compared word for word with the served ones: `[q-tail] check layer= tail_words= tail_differ= score_words=
  score_differ=` a wide layer (~0.13 GB and a span's scores more).
- On against `=0` (w-pio-20261101-0957, on d451beb; usage seed with SEED_FREE, the DSpark verify policy and copy rounds
  off, A B B A): 1206.3 and 1205.4 tok/s cold prefill against 1200.9 and 1195.1 (+0.66%); new text 1273.8 and 1270.0
  against 1243.7 and 1237.8 (+2.5%). Over the cold request's encoder the attention stage ran ~410 ms shorter and the
  moe's waits on bytes ~320 ms longer: the staged reads start where they did. The attention's fresh Metal buffers over
  the encoder: 205.3 GB on against 229.3 (w-pio-20261101-0953, a profile build). Exact: `check` (w-pio-20261101-0953)
  compared 2,564,014,080 tail words and 23,135,944,704 score words over both requests, none different, and every arm's
  texts are the policy-off references (cold d081845201, new-text slice 3 4d5b401f3c).
- `MLX_SERVE_DSV41_STAGE_TRIM` (src/stage_trim.zig, patches/turboexl3-zzz8z-stage-trim.patch; on, `=0` off, `=check`
  on and checked): a wide prompt call's staged records its routes do not take let go before the drive reaches them. On
  the cold code-ctx16384 card (pt3, w-pio-20261101-2207) the encoder staged 4,168 records and its routes took 3,373, 795
  read for nothing, 207 of them at layer 0 (of 323) and 115 at layer 1, whose staging is posted at the prompt's start
  with part of it still queued when layer 0's routes are known. On, a staged window's jobs go out in rank order
  (hottest first) rather than file order, and once the call has its routes and has taken its staged groups, each
  staged job holding none of the call's experts that no read worker has started is withdrawn from the read pool
  (`Pool.withdraw`: its tickets published unread, no row written), its rows let go; a started job reads as posted.
  `=check` reads each withdrawn ticket's status back. `[stage-trim] mode= parts= records= started= verified= bad=` at
  each request's end. Exact: only reads no route of the call takes are dropped.
- On against `=0` (tr1, w-pio-20261102-0152, two binaries: a = ctl25 (2ff23cb, zzz8w), dylib 5ccb44791b50; b = this
  code on zzz8w with the switch off by default, dylib 14e9e72ec2c9, `=1` in the b arms; a warm server, then a1 b1 b2
  a2): cold 16K a 1407.6/1407.4 -> b 1409.9/1408.2 tok/s (+0.11%, pairs +0.16% and +0.06%), new text (slice 3)
  1496.3/1492.5 -> 1498.7/1493.5 (+0.11%, pairs +0.16% and +0.07%); 152 prompt / 177 decode rows in every arm. No job
  was withdrawn: every b request's line reads parts=0 records=0 (started=0 on the cards, 1 on the slices), every staged
  job holding one of its call's experts, so the waste sits record by record inside them; the b arms posted in rank
  order, their stream stats the a arms' (read 101.75 / 70.41 GB, 6049 misses, 4228 staged, 3441 adopted). Exact: the
  texts the same in every arm.
- `MLX_SERVE_DSV41_STAGE_NEXT` (src/stage_next.zig, patches/turboexl3-zzz8j-stage-next.patch; on, `=0` off, `=own` as
  this patch left it; with `MLX_SERVE_DSV41_STAGE`): a wide prompt layer's read-ahead and predicted transient loads read during the call before
  it. `=0`: staging posts a layer's stream misses at the layer's start, so they read under its attention (~4.5 GB a
  16K-row layer, ~340 ms at the drive's rate against ~250 ms of attention) and its call waits on the rest, while the
  drive idles from the call before's last reads to this layer's start (~5.5 s of a cold 16K prompt's encoder): a shorter
  attention (Q_TAIL) turned into as much more waiting on bytes. On, the model predicts layer N+1's routes from layer N's
  FFN rows (N+1's gate over N's MoE input, its columns scaled by the two FFN norms' ratio) and hands the ids,
  unevaluated, to the source before N's call (`Routed.next_fn`, `NextAhead`), which starts them behind its routing
  barrier. Right after the call's last route the source reads them back (each expert's votes, hottest first), reads
  layer N+1's predicted seed ahead into N+1's persistent rows (`Stream.readAheadNext`: P1's row rule; N's call never
  reads those rows) and lays N+1's stream misses out as staging does (`Stream.stageNext`); each window's records are
  posted once no route holds the window (a group drains before its release) and the copier is done with it. Layer N+1's
  staging takes them as its own (the windows not posted yet posted then; none posted: staging as before), its call
  adopts them as before, and its own read-ahead reads whatever its prediction adds. The read pool is first come, first
  served and N's routes' reads are all queued first. The ticket ring doubles again (4096). `[prompt-io]` `early=` counts
  the records posted during the call before theirs (in that call's line), `[stream-stats prompt]` `stage_early=` the
  prompt's. The hook also takes a cap and DEC_PREFETCH's row rule (`fill`) and can skip the windows, for a caller that
  reads ahead into persistent rows only. Exact: the prediction picks which records are read ahead and into which rows,
  never the routes.
- On against `=0` (w-pio-20261101-1102, on 9f62348; usage seed with SEED_FREE, the DSpark verify policy and copy rounds
  off, A B B A): 1254.1 and 1259.3 tok/s cold prefill against 1211.6 and 1214.2 (+3.6%); new text 1287.3 and 1283.1
  against 1281.2 and 1280.6 (+0.3%). Over the cold request's encoder the moe stage took 4489 and 4431 ms against 4930
  and 4968, the host's waits on bytes 536 and 595 against 1048 and 1077; 4434 and 4427 of the 4638 and 4630 staged
  records were posted during the call before theirs, 3965 and 3944 adopted (85%). Exact: the texts are the policy-off
  references in every arm (cold d081845201, new-text slice 3 4d5b401f3c).
- `MLX_SERVE_DSV41_STAGE_NEXT=only` (patches/turboexl3-zzz8k-stage-next-only.patch; the default, also `=1`; `=own`
  every wide layer's own read-ahead too, `=0` off): a wide prompt layer the call before it read ahead for runs no
  read-ahead of its own (RA_FUSED's prediction from its input, read back at its start: ~17-20 ms of host wait a 16K-row
  layer, and its norm and router passes on the GPU); its attention's input comes from the unfused kernels (the same
  kernels over the same arrays), its staging is taken at its call's start (`Stream.promoteFor`) and its P1 records are
  the early ones. The first wide layer of a pass reads ahead as before.
- `only` against `own` (w-pio-20261101-1135, on 5ed23c2; usage seed with SEED_FREE, the DSpark verify policy and copy
  rounds off, A B B A): 1293.2 and 1285.8 tok/s cold prefill against 1259.6 and 1256.3 (+2.5%); new text 1307.0 and
  1306.6 against 1282.2 and 1283.6 (+1.9%). Over the cold request's encoder the moe stage took 4388 and 4404 ms against
  4438 and 4470, the host's waits on bytes 526 and 559 against 603 and 603, 91.86 GB read against 95.23 and 867
  read-ahead records posted against 1082; adoption held (3954 against 3944). Exact: the texts are the policy-off
  references in every arm (cold d081845201, new-text slice 3 4d5b401f3c).
- `MLX_SERVE_DSV41_STAGE_VOTES` (src/stage_votes.zig, patches/turboexl3-zzz9a-stage-votes.patch; on at the floor 8, `=N`
  the floor N, 1 to 65535, `=0` off; with `MLX_SERVE_DSV41_STAGE`): a wide prompt call's staging takes only the
  predicted stream misses that at least N rows of the prediction name. Staging reads the prediction's stream misses past
  the seed ranks, hottest first, so its tail is the experts a few rows name, the ones the call routes least often: on a
  cold 16K card (ctl25's a arms, w-pio-20261102-0114) 786 staged records were never adopted, 10.47 GB of the prompt
  pass's 102.6 GB. On, the plan (`stage_votes.plan`, for `stageGroups` and STAGE_NEXT's `stageNext`) stops at the
  first expert of the ranking under the floor; the read-ahead into the persistent rows keeps the whole ranking, and an
  expert the floor takes out that the call routes is read by its own route. On or off, every staged record keeps its
  votes: a `[stage-votes] floor=
  posted= adopted= cut= p1= a1= c1= ... p64= a64= c64=` line at the end of each prompt that staged, with
  `MLX_SERVE_DSV41_PROMPT_IO` (bucket N: N to 2N - 1 votes, the last 64 and more; p staged, a adopted by a route of the
  call, c taken out of a plan by the floor). `posted` is every record the prompt staged; `[stream-stats prompt]`'s
  counters start at the prompt's first routed call, after the first wide layer's own read-ahead has staged, so on a
  warm server its stage_posted is lower by that layer's staging (adopted agrees). Then the staged jobs by their read
  records no route adopted (`jobs= jwaste= jall= j0= j1= j2= j4= j8=`; a job STAGE_TRIM withdrew is not counted).
  Exact: the floor picks which records are staged, never the routes.
- The floor 8 against `=0` (sv1, w-pio-20261102-0252, two binaries: a = ctl27 (f6aa81b, zzz8y), dylib 42a2a45fae3c;
  b = this code on zzz8y with the floor off by default, dylib ac1a4670fddd, `=8` in the b arms; a warm server, then a1
  b1 b2 a2): cold 16K a 1419.7/1415.2 -> b 1429.0/1428.0 tok/s (+0.78%, pairs +0.66% and +0.90%), new text (slice 3)
  1500.8/1490.4 -> 1513.7/1497.5 (+0.67%, pairs +0.86% and +0.48%); 150 prompt / 179 decode rows in every arm. Read
  102.25 -> 100.27 GB cold, 70.32 -> 68.52 new; the staging's waste 775 -> 624 records cold, 212 -> 109 new. The floor
  took 957 records out of the cold card's staging (4268 -> 3311 posted), 806 of which the call adopted (3493 -> 2687):
  808 more demand reads, about 151 records saved. What stays wasted is mostly high-vote (64 and more: 750 of 1104
  adopted; 32 to 63: 151 wasted): the prediction's misses, which a higher floor does not reach. Exact: the texts the
  same in every arm.

- An a4k bank (patches/turboexl3-zzz8m-a4k.patch, src/mlx_serve/sdk/weights/streamed/expert/a4k.zig; the manifest declares
  it, `expert-manifest-v2.json`'s `sidecar.layout` "a4k"): each record's three codes start on a 4 KiB block, so the
  PARTS route reads every code straight into its slot row (99.7% of a record) where the classic layout reads the gate's
  code alone (33%). On disk a record is a zero lead of 3,072 B, the gate/up span [gate rout, gate rin, up rout, up rin,
  gate code, up code], then the down span [down code, down rout, down rin], padded to 4 KiB: 4,096 B more a record
  (+62.9 MB a bank). The segments' bytes, lengths and shapes are the classic ones; the bank parses the layout and checks
  its table against `a4k.geometry`, the read pool gets each record's rows and lengths in the order on disk
  (`expert_io.Options.disk_order`), the speculative class reads a record to its last segment and the pre-read class from
  its gate/up start. A classic bank runs the classic path; an older build refuses an a4k bank by name (its segment table).
  `zig build test`'s a4k gate reads the same records laid out both ways through every route of the copied pool and
  finds the same slot bytes (`zig build test -Da4k-gate=<classic>:<a4k>` adds two real packs).
- `MLX_SERVE_DSV41_ENGRAM_DIRECT` (src/engram_direct.zig, patches/turboexl3-zzz8o-engram-direct.patch; on, `=0`
  off): a prompt's Engram rows read once per table and past the page cache. The pack's tables (oMLX's lossless
  layout) keep a row's e4m3 codes (256 B) and e8m0 scales (8 B) in two regions of one file. Off, a prompt grid's rows
  are read span by span (256 tokens), each span's distinct rows by two preads a row through the page cache: a 16K-token
  prompt's 587,099 row reads (1,174,198 IOs) page in ~8 GiB of file cache (+7.97 GiB cold, +7.19 GiB new text), and
  with the pack's files evicted layer 1 waits ~1.1 s for its table. On, a prompt grid's table is read whole: every
  span's ids sorted and made distinct (310,174 rows for the two tables), the distinct rows read past the cache
  (F_NOCACHE, the file system blocks that hold them, APFS 4,096 B) on up to 64 threads, each a contiguous run of the
  sorted rows that keeps its last scale block for the rows after it (588,276 IOs, 2,373 MB), then each span's rows
  copied out of them; a decode token's rows as before. A `[engram] at= len= table0_ms= table1_ms= rows= distinct= ios=
  read_mb= wait0_ms= wait1_ms= direct=` line for each prompt grid either way. Exact: the same bytes land in the same
  rows. en2 (w-pio-20261101-1523, ABBA, margin 3, EVICT, FCACHE, texts identical in every arm): cold 1185.6 / 1256.0 /
  1252.6 / 1191.6 tok/s (a1 b1 b2 a2: +5.53%), new text 1213.3 / 1300.0 / 1299.5 / 1265.8 (+4.86%); the file cache
  grows +0.03 / 0.00 GiB on (+7.97 / +7.19 off), the live peak ~8 GiB lower; layer 1's wait 887-899 ms (1059-1174
  off), table 0 read in 1495-1505 ms (1783-1786). A repacked table (`engram/engram-manifest.json`, engram_repack.py:
  a row's codes then its scales in one 264 B record) is read a record a row, one pread through the blocks that hold it
  (two blocks for the 6.4% of records that straddle one; src/engram_direct.zig's `readRecord`), in a whole-table read
  and a decode token's alike (patches/turboexl3-zzz8t-engram-records.patch); a table in oMLX's layout keeps its two
  reads. Exact: the same bytes. rp1p (w-pio-20261101-2110, two binaries, ABBA, margin 2, EVICT, COOL; a: ctl21 on the
  canonical pack, b: this build on the repacked one; texts identical in every arm): cold 1347.0 / 1417.6 / 1416.4 /
  1349.9 tok/s (a1 b1 b2 a2: +5.08%), new text 1414.3 / 1501.1 / 1496.5 / 1425.4 (+5.56%); the b arms' ios equal
  their rows (310,174; 276,588), table 0 read in 951 ms (1532), layer 1's wait 365 ms (940).
- `MLX_SERVE_DSV41_ENGRAM_DECODE` and `MLX_SERVE_DSV41_ENGRAM_LRU_MB` (src/engram_decode.zig,
  patches/turboexl3-zzz8u-engram-lru.patch; on, `=0` off, `=check` on and checked; the LRU's MiB, 1 to 16384, 256
  unset): Engram rows kept across a conversation's turns. A DSpark verify's 6 rows read 24 n-gram ids a table past the
  page cache, on the prefetch thread from the forward's start, a row a GCD work item (about 9-10 reads in flight); the
  first Engram layer waits for the first table's: ~2.0 ms a round, the GPU idle ~1.0 ms of it (tp1, w-20261101-1811:
  0.98 ms median at layer 1's entry, the verify's largest gap). On, one LRU of rows over the tables (keyed by table and
  id; its rows and their index within the MiB, billed at the load beside the host's bytes, the load line giving the
  prompt / decode slot rows they cost), kept across requests: a prompt grid's distinct rows go in as ENGRAM_DIRECT's
  whole-table read hands its spans over (310,174 rows over the two tables for a 16K-token prompt), and so does every
  row a decode read reads; a decode read takes the rows the LRU holds and reads the others on a pool of 64 threads,
  each read (a repacked table's record, else a row's codes or its scales) a work item; a round's bonus token (`t1`,
  known before its draft runs) has its rows read when the round starts, on a GCD worker a table, into the LRU.
  `=check` reads every row of a decode read on the pool, compares the ones the LRU held and serves the read ones.
  `[engram-decode] mode= lru_mb= lru_rows= lru_cap= prompt_rows= inserted= evictions= hits= misses= early= early_busy=
  early_failed= checked= differ= pool=` at each request's end; `[round-io]` gives each round's `engram_hits=`,
  `engram_ios=` (the tables' read calls), `lru_evictions=` and `engram_wait_ms=` (the first Engram layer's wait) in
  both modes. Exact: a row's bytes are the table's whichever way they come. ed1 (w-20261101-2125, two binaries, ABBA +
  BAAB, margin 2, EVICT, COOL; a: ctl21, b: ctl21 + this code, both on the canonical pack in oMLX's layout; texts
  identical in every arm and mode, the check arm's differ 0): greedy 56.15 -> 56.53 tok/s (+0.67%, round 89.43 ->
  88.85 ms), card +0.23%, a second turn +0.65%; the first Engram layer's wait 0.016 ms a greedy round in the b arms
  (0.734 in the check arm, which reads every row); the bank's decode rows 178 -> 177 (the LRU's 256 MiB).
- `MLX_SERVE_DSV41_DRAFT_AHEAD` (src/draft_ahead.zig, patches/turboexl3-zzz8v-draft-ahead.patch; on, `=0` off,
  `=check` on and checked): a greedy round's next DSpark draft launched at the end of the round before it. A greedy
  round ends with its state committed and its next token known; with `=0`, the next round's draft goes to the GPU only
  after the host's bookkeeping, the engine's turn and the round's start (its checks and plan), the GPU idle meanwhile
  (tp1: 0.63 ms median from the verify's last kernel to the draft's first). On, `draftStep` launches the next round's
  draft from its next token right after its bookkeeping, when the next plan would be the stages' block
  (`rounds.Drafter.wouldStage`): the draft's graph built from the committed state and handed to the GPU, not read
  (`dsparkDraftLaunch`); the next round takes it when it fits (a greedy round, its plan the stages' block, the same
  token, position and state; `dsparkDraftCollect`), any other entry drops it. `=check` makes the round's own draft too,
  compares the ids and confidences word for word and serves its own. `[draft-ahead] mode= launched= taken= dropped=
  failed=` (and `checks= words= differ=`) at each request's end. Exact: the same ops on the same state, launched
  earlier. pd1 (w-20261101-2249, two binaries, ABBA + BAAB, margin 2, EVICT, COOL; a: ctl23, b: ctl23 + this code;
  texts identical in every arm and mode, the check arm's 106 compares 0 differing): greedy 56.03 -> 56.33 tok/s
  (+0.54%), card 44.33 -> 44.53 (+0.45%); 107 drafts launched ahead a greedy request, 106 taken. `[round-io]`
  `round_ms` (a round's draft start to its commit) drops more, 89.60 -> 86.79 ms greedy, as the launch now comes after
  the commit, outside that span: tok/s is the measure.
- `MLX_SERVE_DSV41_DECODE_WINDOW` (src/decode_window.zig, patches/turboexl3-zzz8w-decode-window.patch; on, `=0` off,
  `=check` on and checked): decode's window 0 at 4 rows, not the widest decode route's 48. At the phase change the
  stream frees the prompt's transient scratch and allocates window 0, the rows a decode plan's transient loads take:
  48 x (13.3 + 8.9 MB) = 1.066 GB on the 2.81 bpw pack, billed before the decode slot rows. A transient load is a miss
  the decode admission declines (it ranks below every resident it could evict); the 16K card's greedy decode declines
  none (ed1's b1: 4,165 misses, 4,165 evictions). On, window 0 holds 4 rows, a decode plan admits its declined misses
  past them over the lowest-ranked residents the route does not use (`LayerPolicy.admitPast`, the pinned last), and the
  bill gives the freed ~977 MB to the decode slot rows (about two more a layer); taken where every layer keeps 96
  decode rows or more, else the widest route's window. `=check` counts the plans left past the window (`overflow`, 0 by
  construction). A `[bank-bill]` load line gives the bill's terms (the working set, ceiling, margins, host and LRU
  bytes, memory in use, each phase's working set, windows, staging, rows and their remainders), and `[mem-decode]
  window= declined= forced= overflow= mlx_active_mb= mlx_cache_mb= mlx_peak_mb= footprint_mb=` each request's (MLX's
  peak over its decode, from the first round). Exact: a route's bytes are the records' wherever they sit; only which
  rows hold them moves, and only on a route past the window. wz1 (w-20261101-2308, two binaries, ABBA + BAAB, margin 2,
  EVICT, COOL; a: ctl23, b: ctl23 + this code; texts identical in every arm and mode, overflow 0, no declined miss):
  greedy 56.20 -> 56.93 tok/s (+1.29%), card 44.23 -> 44.68 (+1.02%); decode rows 176 -> 178 (a2 175), misses a
  greedy round 39.2 -> 37.7 (card 58.3 -> 57.3). At decode MLX holds 97,056 MiB active and ~0.1 GiB cached, its peak
  51 MiB over active; the process footprint 98,127 MiB (MLX's and ~0.95 GiB of the host's).
- `MLX_SERVE_DSV41_DECODE_STOP` (src/decode_stop.zig, patches/turboexl3-zzz8x-decode-stop.patch; on, `=0` off, `=check`
  as on): the decode phase's budget takes 1.0e9 B of the 2.0e9 B stop under the ceiling back for its slot rows, MLX's
  cache capped at 1 GiB from the phase change to decode (after the grow) and the prompt's cap restored at the change
  back (before decode's rows go). At decode MLX holds little beyond the bill: wz1's b arms (16K) cached 97 to 109 MiB
  with a peak 51 MiB over active; md1 (w-20261102-0003, 32K on ctl25) cached 0 with the peak 51 MiB over active, the
  footprint 3.51 GiB under the 102,400 MiB ceiling. The cut is two decode rows (470 MiB a row) at 16K and at 32K; the
  prompt's budget and rows and the 2 GiB wired margin are unchanged. `[decode-stop] mode= cut_mb= cache_mb=
  prompt_cache_mb= enters= exits=` at each request's end, `decode_cut_mb=` in `[bank-bill]`. Exact: a route's bytes are
  the records' in any row.
- On against `=0` (ns1, w-20261102-0057, two binaries: a = ctl25 (2ff23cb, zzz8w), b = this code, =1 in the b arms;
  chk then A B B A, B A A B): greedy 56.60 -> 57.08 tok/s (+0.84%, round 86.36 -> 85.53 ms, misses 38.5 -> 36.9 a
  round), card 44.40 -> 44.78 (+0.84%); the decode rows 177 -> 179 in every pair (a2 opened 152 prompt rows at its
  in_use of 102,183 MiB, the others 153); the texts the same in every arm (card f7dfb7a5d4, greedy fd06ee302a).
- `MLX_SERVE_DSV41_SCRATCH_ATTN` (src/scratch_attn.zig, src/attn_twins.zig, patches/turboexl3-zzz8p-scratch-attn.patch;
  on, `=0` off, `=check`): a wide prompt layer's attention spans write their scores (f32 [L, H, NK]) and probabilities
  (f16) into a ring of two per-prompt slots (S and P, 256 x 64 x 640 words a slot, 126 MB) instead of fresh MLX buffers:
  a layer's spans are encoded inside its one eval, so their arrays are live together and MLX's cache cannot hand one
  span's to the next at any cap (prf2: 66% of p missed cold, 82% new text, at 2 GiB as at 6 GiB). The gather's
  launches are write-into twins of their stock texts, derived at load (each rebound array declared at the slot's
  first word from a whole region, a one-word carrier output, a `dep` input) and compiled then by one launch; sm binds
  qk's carrier, pv sm's, and a slot's next qk the slot's last pv's (span k - 2's). `check` serves the fresh launches
  and compares each span's ring scores, probabilities and heads word for word (`[scratch-attn] check layer= s:words=
  ,differ=,max_abs=,max_ulps=,over1= p:... o:...`). Exact: the stock statements. sa1 (w-pio-20261101-1748, ABBA,
  margin 2, EVICT, on eeba5c8, texts identical in every arm): cold 1291.3 / 1303.5 / 1300.2 / 1296.4 tok/s (a1 b1 b2
  a2: +0.62%), new text 1329.1 / 1351.3 / 1338.8 / 1330.5 (+1.15%); its check arm compared 66.8G words over 41 wide
  layers' spans (s, p and the heads), 0 differing, max_ulps 0, no span fresh; the bank's rows 156 / 178 in every arm.
- `MLX_SERVE_DSV41_SCRATCH_HEADS` (src/scratch_attn.zig, patches/turboexl3-zzz8q-scratch-heads.patch; on, `=0`
  off, `=check`; with `MLX_SERVE_DSV41_SCRATCH_ATTN` on, the default): phase B of the attention arena. A wide layer's WOA_ROWS run
  (4096 rows: 16 spans) joined its spans' grouped heads with `concat` ([og, rows, in], 268 MB fresh a run) and
  turboexl3's grouped linear then copied that join's transpose into the `[rows, og, in]` it reads. Now pv's twin
  (derived at load: p from the ring's slot, the stock grouped store moved to the run region's `[rows, og, in]` at the
  span's rows, the same values) writes each span's heads straight into its run's region (two by the run's parity, 537 MB),
  the run's region is published (the host's event wait over it and the last pv carrier of each span parity) and its
  `wo_a` reads it as it is; the region's next writers bind that `wo_a`'s output (`dep2`). A run that is not full (the
  layer's last, 4095 rows at 16383) keeps the join. `check` serves the join and compares each full run's region with it
  word for word (an `h:` field on the `[scratch-attn] check` line). Exact: the same words. sb1 (w-pio-20261101-1835,
  ABBA, margin 2, EVICT, on f073b85, texts identical in every arm): cold 1302.6 / 1345.6 / 1345.3 / 1304.1 tok/s (a1 b1
  b2 a2: +3.23%), new text 1363.7 / 1428.0 / 1407.8 / 1346.6 (+4.63%); its check arm compared 402,653,184 words a wide
  layer (its three full runs' heads), 0 differing, over 41 check lines (120 runs on the regions, 40 joined); the bank's
  rows 156 / 178 in every arm.
- `MLX_SERVE_DSV41_ATTN_CORE` (src/attn_core.zig, patches/turboexl3-zzz9f-attn-core.patch; on, `=0` off, `=check`
  checked): a wide prompt layer's attention gather on two exact changes, each moving only which threads compute a
  word. pv's text at 64 output columns a threadgroup instead of 128 (derived at load from the stock attn_pv.metal: the
  tile constant, the staging registers, the fetch and the store at 32 bytes a thread): the 16 x 64 x 32 tensor op,
  twice the threadgroups, half the accumulator a lane; every output word is the same contraction (32 keys a step in
  the stock order, accumulated in f32), then the same epilogue. The SCRATCH_ATTN ring twin and SCRATCH_HEADS' pv twin
  derive from that text. And the gather's tensor-op launches (qk, Q_TAIL's qk_t and pv, fresh and as their twins) go
  as MLX's NAX GEMMs do, threadgroup (32, 1, 4) over (x / 4, rows, 4) threads: the same threadgroups, each thread's
  index and simdgroup the 1D launch's. pv1u (w-pv1u-20261102-0644, L5's late span, GROUPED as served): 0 differing
  words, 425.6 -> 332.1 us a call (1.285x); ls1u (w-ls1u-20261102-0630): the 3D launch alone 0 differing, qk 1.066x,
  qk_t 1.064x. `check` serves the stock text and launches and runs every gather call's scores and pv the new way
  beside them, compared word for word (`[attn-core] check kind=qk|qk_t|pv calls= words= differ= max_ulps=`, a kind
  every 128 calls, at its first differing word and at the model's end).
  ac1 (w-pio-20261102-0757, ABBA, margin 2, EVICT, on zzz9d 9dcd5f1, texts identical in every arm): cold 1479.7 /
  1451.1 / 1497.5 / 1463.0 tok/s (a1 b1 b2 a2: +0.20%, pairs -1.93% / +2.36%), new text 1571.4 / 1595.8 / 1587.7 /
  1570.7 (+1.32%, pairs +1.55% / +1.08%); its check arm 0 differing words and max_ulps 0 on every line (qk_t 2461
  calls, pv 2504, qk 43); the guard's peak not up (a 108.4 / 104.3, b 99.8 / 108.5); the bank's rows 147 / 176 in
  both a arms and b2 (b1 opened at 146, the bill's load-time noise).
- `MLX_SERVE_DSV41_SCRATCH` (src/scratch.zig: the regions; src/scratch_x1.zig, src/kernels/hc_expand_into.metal,
  patches/turboexl3-zzz8r-scratch-x1.patch; on, `=0` off, `check` checked): a wide prompt layer's streams after its
  attention (`x1`, [rows, 4, 5120] f32, 1.34 GB at 16K rows) written into one per-prompt buffer, not a fresh MLX
  allocation each layer. Off, `hcExpand`'s output is a new array each wide layer, and MLX's buffer cache (2 GiB) cannot
  hold it: each is a fresh `newBufferWithLength`. On, a prompt's first wide layer allocates the region (the host's
  `alloc_uninit`; kept while the chunk's rows hold, let go at the decode phase change). glm5's expand writes its words
  into it through an input binding (src/kernels/hc_expand_into.metal: the same loads, fmas and stores) behind the
  region's last reader (the layer before's MoE output, bound and unread), and declares a one-word carrier output. The
  norms and the MoE tail read the region through the host's event wait over an event already at 1, with the region and
  the carrier as tracked inputs (`Region.publish`): MLX fences the compute pass after it on the writer's pass, and its
  outputs alias the region. The regions' bytes stay within 4 GiB (a region past it takes the fresh path). `check` serves
  the fresh array and compares the region's with it word for word: a `[scratch] check region=x1 rows= words= differ=`
  line a wide layer. x1a2 (w-pio-20261101-1841, ABBA, margin 2, EVICT, on f073b85: zzz8p, before phase B; texts
  identical in every arm): cold 1270.5 / 1308.8 / 1305.4 / 1297.7 tok/s (a1 b1 b2 a2: +1.79%; a1 the slow arm, so the
  b2 / a2 pair's +0.59% is likelier x1's own), new text 1294.4 / 1351.4 / 1350.2 / 1345.7 (+2.33%; that pair +0.33%);
  the bank's rows 156 / 178 in every arm, the file cache's growth ~0. On zzz8q the regions together (the waves' 917.5
  MB, the ring's 125.8 MB, phase B's 536.9 MB, x1's 1342.1 MB at 16,383 rows: 2.72 GiB) stay within the cap.
- `MLX_SERVE_DSV41_DSPARK_POLICY` (src/rounds.zig `Fixed`, `DEFAULT_COSTS`; plugin.zig;
  patches/turboexl3-zzz8s-dspark-fixed.patch for env.zig's `MLX_SERVE_DSV41_DSPARK_COSTS`; unset or `fixed` the fixed
  policy on sampled requests, `=1` the live policy, `=0` off): the cut and the drafting backoff on a fixed cost table,
  never the clock. The tables by context (`U1:t1+U2:t2+t3`, a round at position n on the first table whose bound
  exceeds it; each `t` the verify ms by rows from one to N, then the draft's ms), built in as lad2's calibration on
  the a4k canonical pack (w-lad2-20261102-1707, zzz9t): on aw1k's build, whose trunk is batch-invariant (a forced
  width leaves the text alone), `MLX_SERVE_DSV41_DSPARK_WIDTH=sweep:6` forcing a greedy round's verify width 1..6 in
  turn on the warm request (a card first on the same prompt) at about 1K, 4K and 16K prompt tokens of two prompts of
  their own (Python's asyncio and email sources, apart from every benchmark prompt); row w the two prompts' median
  verify ms at w rows plus the bucket's per-round overhead (the median round less its verify), to 0.1 ms and strictly
  increasing, the draft's ms the greedy rounds':
  `2048:41.4,47.9,57.2,66.4,77.2,84.4/4.8+8192:40.1,47.1,55.0,64.9,73.2,85.1/4.8+41.0,47.9,57.9,65.8,76.7,85.9/4.9`.
  Until zzz9t the built-in tables were fp3's (a server on the live policy and one on the whole block at about 1K, 4K
  and 16K prompt tokens of Python's asyncio sources, each ladder fitted monotone and strictly increasing by
  fp3_costs.py:
  `2048:17.7,25.5,26.2,33.6,36.8,42.7/5.0+8192:42.8,60.6,74.5,84.4,88.2,96.3/5.0+37.1,70.2,70.3,104.5,104.6,104.7/4.9`);
  fp3's and kt2's runs below are on those.
  fp3 (w-fp3-20261101-1906, f073b85 + this code with the policy off by default; ABBA, margin 2, EVICT=1, copy rounds
  on; each arm qbench's ten reference rows sampled as the card, then the code-ctx16384 card sampled and greedy):
  sampled requests 28.45 / 30.97 / 31.24 / 28.98 tok/s (a1 the whole block, b1 b2 fixed, a2: +8.33%; pairs +8.88%,
  +7.80%), the ten rows alone +7.99%, the card +6.74%; a1 = a2 and b1 = b2 byte for byte on every request (the fixed
  policy is deterministic), the greedy card the same bytes and rounds in every arm (rows_hist 6:107). The greedy card
  also ran 5.41% faster in b, with nothing of the policy on it: the requests before it in its server left the bank
  warmer (5,401 misses against 5,994), so each server's history moves a later request by several percent and the
  sampled +8.33% holds some of it; the gain is sized on fresh servers next (fp4). The gain is the backoff's on long
  prose of low acceptance (row 0 skipped drafting in 3,413 of its 4,919 rounds); the code card keeps the whole block
  in most rounds (rows_hist 6:106 of 169). kt2 (w-kt2-20261101-1549, the 16K table alone): +14.4% against the whole
  block over the same rows, the row-by-row geometric mean +4.3%.
- `MLX_SERVE_DSV41_DSPARK_FIXED_GREEDY` (src/rounds.zig `Switches.fixed_greedy`, `fixedGreedy`, `wouldStage`;
  plugin.zig; patches/turboexl3-zzz9t-1-fixed-greedy.patch for env.zig; off by default, `=1` on): the fixed policy's
  cut and backoff on a greedy request's rounds too. On the batch-invariant trunk (zzz9r) a greedy round's width leaves
  its text the 1-row decode's, so the cut is text-safe; off, the policy stands aside on a greedy round (the whole
  block), as before. The same patch launches the next draft ahead (`MLX_SERVE_DSV41_DRAFT_AHEAD`) on a sampled round
  too, where the fixed policy owes no plain round (before, only where the plan was the whole block). Policy-on serving
  only: the served default (`MLX_SERVE_DSV41_DSPARK_POLICY` unset, the fixed policy) changes with zzz9t's tables and
  this patch; the standard decode cells run `MLX_SERVE_DSV41_DSPARK_POLICY=0` (every round the whole block) and do not
  move.
  - Read as aw1s (M5 Max, w-20261102-1840; the greedy cut alone on one text: one build in every arm, aw1kt
    9767770a470a, a = `=0`, b = the cut, both on G_KITSP1 with keyed sampling and the policy on; chk first, then a1 b1
    b2 a2 b3 a3 a4 b4, each a code-ctx16384 card then greedy): every arm's greedy the 1-row text (4d36d6e8aa2a87ac),
    the card the same in every arm; greedy 57.6 57.5 57.7 57.7 against 56.6 56.5 56.2 56.6 tok/s (-2.00%; pairs -1.74,
    -1.74, -2.60, -1.91): the cut's rounds 13.5 ms cheaper (73.06 against 86.60 wall ms a round) but 127 of them
    against 105 (rows_hist 2:29,3:12,4:24,5:7,6:55; accepts a round 3.13 against 4.00). So off by default.
  - Read as aw2 (M5 Max, w-20261102-2028; a = ctl46 949ce432e7ff, b = aw2 cc7e8b14c0f9, zzz9t's code on zzz9r with
    this switch off, both on G_KITSP1 with keyed sampling and the policy on, no `MLX_SERVE_DSV41_DSPARK_POLICY=0`; chk
    (b, the twins at `check`) first, g_a / g_b at MTP:0, then a1 b1 b2 a2 b3 a3 a4 b4, each a code-ctx16384 card then
    greedy): every arm's greedy the 1-row text (4d36d6e8aa2a87ac), every arm's card the keyed 1-row card
    (d63fabe7ba2cbb39), chk's twin checks differ 0. The card, the ruled cell (pre-registered; the same text in both
    arms): 47.5 47.3 47.1 47.2 against 48.1 48.1 48.2 48.3 tok/s (+1.90%; pairs +1.26, +1.69, +2.34, +2.33), 144
    rounds against 141 (rows_hist 1:15,3:40,6:89 on fp3's tables against 2:18,3:18,4:31,5:13,6:61 on lad2's; accepts a
    round 3.23 -> 3.32), a round's wall ms 89.31 -> 89.51: the gain is the tables' cut, not the draft launched ahead.
    The greedy, the same code in both arms with this switch off, reported and not ruled: 57.9 57.5 57.1 57.8 against
    57.7 57.5 57.5 57.7 tok/s (+0.04%), 105 rounds in each. Measured on zzz9r; merged on (zzz9t).
- `MLX_SERVE_DSV41_DSPARK_WIDTH` (src/rounds.zig `Width`, `WidthStats`; plugin.zig `forceWidth`;
  patches/turboexl3-zzz9t-2-dspark-width.patch for env.zig; a diagnostic setting, never served; off): a greedy stage
  round's verify width forced: `w` (rows 1..8), `random:S[:M]` (a width 1..M a round, keyed by S and the position, so
  the same each run), `sweep[:M]` (1..M in turn); past the block the drafts again from the first, as filler the
  trunk's argmax rejects. `[dspark-width]` at each request's end: per width its rounds, verify mean and median ms and
  drafts kept a round. rw1k (w-20261102-1654) checked the invariant trunk's greedy text at every width 1..6 with it,
  lad2 timed the built-in tables' ladder with it. Merged on (zzz9t).
- `MLX_SERVE_DSV41_BANK_POOL` (src/bank_pool.zig, the copied stream's `BankPool` in
  patches/turboexl3-zzz9g-bank-pool.patch; on by default, `=0` off, `=check`; needs the host's `HostGpu.alias`,
  mlx-serve's additive ABI entry: G_KITSP1 carries it, host-alias.patch 3bdf623e15f3; the plugin builds against the
  standard kit and reads the entry at its offset, bytes 96-104 of the host's HostGpu, so a host without it loads and
  serves with the pool off, a `[bank-pool] off` line at load naming the missing entry): the streamed bank's slot rows of
  both phases over one MLX buffer. Off, a phase change frees one phase's rows and freshly allocates the other's: prompt
  to decode frees the 5 prompt windows (5,082 MiB on the 2.81 bpw pack) and allocates window 0 and the ext rows
  (11,356 MiB), decode to prompt the reverse, about 13-15 ms a GiB
  (o2a1: 176-191 ms and 94-101 ms a change). On, the pool's prompt layout is the windows of each geometry and the two
  wave sets' regions (875 MiB, src/scratch_waves.zig), and its decode layout over the same bytes is window 0 of each
  geometry and then whole ext component arrays, largest first, while they fit; the other ext arrays are allocated at
  the grow as before. Each pooled array is its own MTLBuffer (MLX orders it apart, as a separate allocation), made at
  load and held; a phase installs references to its layout's. Neither phase holds more than without the pool (the
  prompt layout is what a prompt holds from its first wide call on; the decode layout is part of decode's own rows).
  `check` fills every range that changes hands with 0xFF at its install (`[bank-pool] check` lines). A `[bank-pool]
  plan` line at load. Exact: the same plans, rows, kernels and words; a pooled row is read only after its record
  landed, and a range changes hands only at a phase change, after the stream's drains and the source's synchronize.
  Measured (bp1, w-pio-20261102-0808: ctl28 3a0002cf7f13 against bp1 abe4a4b85246, both on G_KITBP1, two runs an arm):
  the cold 16K card 1405.2/1389.7 -> 1420.1/1416.6 tok/s (+1.50%; pairs +1.06/+1.94), the new-text slice
  1486.2/1495.7 -> 1508.6/1510.0 (+1.23%; pairs +1.51/+0.96); texts identical in every arm (check's 0xFF fills
  included), bank rows 147/176 in every arm, the guard's peak 104.2/106.3 against 107.5/110.1.
- `MLX_SERVE_DSV41_SCRATCH_TAIL` (src/scratch_tail.zig, TAIL_FUSE's twin in src/tail_fuse.zig,
  patches/turboexl3-zzz9i-scratch-tail.patch; on by default, `=0` off, `=check`): a wide prompt layer's new streams written in place
  over its x1 region (`MLX_SERVE_DSV41_SCRATCH`) instead of a fresh buffer. Off, TAIL_FUSE's kernel writes each layer's
  streams (671 MB of bf16 at 16K rows) into a fresh MLX buffer: MLX holds a buffer until its command buffer completes and reuses
  a cached one only within two pages of the size asked, under the kit's 8 GB pool, and a layer's transients pass
  through that pool between one stream buffer's release and the next one's allocation (pt2: 2.34 GB of fresh buffers a
  layer in the rest stage; pt4's read: the tail's output allocation, ~11.5 ms a wide layer of the rest stage's ~20 ms of
  exposed GPU idle). On, with the x1 region serving the layer, one MoE group over every span, and the next layer a wide
  one of the same call whose launches before its hc expand are the streams' only readers (no read-ahead of its own, no
  DSpark row), the kernel's write-into twin (its text derived from the stock one at its first use: the store over `res`,
  a one-word carrier; the stock kernel's 29 inputs and one output, 30 buffers of Metal's 31) writes the new streams over
  the region: each thread reads its row's copies of x1 and every other input before it stores the row's outputs, and
  nothing reads the region's x1 after the tail; the region is published as the layer's streams, and the next layer's hc
  expand reads and writes it in place the same way. A call's last wide layer keeps the fresh buffer (its streams are read
  after the call's last eval: the CED replay, the head, the next chunk). `check`, in the one routed call: the stock
  kernel's fresh streams first, then the twin, compared word for word (`[scratch-tail] check layer= words= differ=`), the
  stock's served. `zig build gpu-check -Dgpu-check-mlx=<host tree>/lib/mlx` builds src/gpu_check.zig, which compiles
  both in-place kernels on the GPU at the served template arguments and checks their words against the stock kernels'
  (run in a GPU window). Exact: the stock statements. Measured (sx4, w-pio-20261102-0840: ctl33 0262e23069e7 against sx4
  59fe2654e8b8, both on G_KITSP1, two runs an arm; its GPU check first: 30 buffers of 31, differ 0 on every tail,
  tail-check and hc line at 16,383 and 4,097 rows): the cold 16K card 1491.3/1487.6 -> 1518.9/1514.9 tok/s (+1.84%;
  pairs +1.85/+1.84), the new-text slice 1588.5/1587.6 -> 1596.1/1616.7 (+1.16%; pairs +0.48/+1.83); texts identical
  in every arm; chk's 38 `[scratch-tail]` check lines differ=0; bank rows 147/176, 146/175 in one arm of each binary
  (b1, a2: the load's); the guard's peak not up (b's largest 106.2 against a's 108.1).
- `MLX_SERVE_DSV41_ENGRAM_FIRST` (src/engram_first.zig, patches/turboexl3-zzz9n-engram-first.patch; on by default,
  `=0` off, `=check`): a prompt chunk's Engram rows read sooner and on more threads. Off, extendIds starts the chunk's Engram read
  thread after the phase change and the embedding, and the thread reads a prompt grid's table (about 155K distinct 4 KB
  rows a table on a cold 16K card, one direct pread a row) on at most 64 threads: the first Engram layer (L1) waits for
  table 0, 383-406 ms on the cold card and 290-328 ms on the new-text slice (w-pio-20261102-0840, -1002, -1013; table 0
  read in 925-936 / 752-770 ms), the GPU idle meanwhile. On, the ids are hashed and the thread started first, before the
  phase change, and a prompt grid's tables read on up to 128 threads (a machine parameter: a deeper queue for the small
  reads beside the expert stream's). `check` also reads table 0 at 64 threads and compares the bytes
  (`[engram-first] check table=0 rows= bytes= differ=`). Exact: the same rows, read into the same buffers and served the
  same. Measured (ew1, w-pio-20261102-1135: ctl39 ae17e03db754 against ew1 a2000ad09c62, both on G_KITSP1, two runs an
  arm): the cold 16K card 1550.0/1565.2 -> 1584.5/1580.7 tok/s (+1.60%; pairs +2.23/+0.99), the new-text slice
  1661.3/1631.5 -> 1665.3/1689.7 (+1.89%; pairs +0.24/+3.57); texts identical in every arm; chk's `[engram-first]`
  differ=0 (155,080 and 138,288 rows); bank rows 147/176 in every measured arm. The read: L1's wait0 398.0/400.1 ->
  158.9/166.0 ms on the card and 353.6/358.5 -> 180.7/189.0 ms on the slice (table 0 read in 940.9/943.8 -> 771.8/771.5
  and 765.6/771.2 -> 667.9/673.5 ms); the wider reads slow the expert stream beside them (L0's moe rwait 96.5/95.2 ->
  122.8/120.3 and L1's 69.1/72.2 -> 94.0/93.1 ms on the card; 22.9/22.9 -> 35.5/39.0 and 62.3/63.8 -> 81.1/83.3 on the
  slice). The card's L0 also shows other_ms 636 -> 408 ms (layer 963 -> 756), which the Engram wait does not explain (L0
  has none, and the slice's L0 does not show it): unexplained, not claimed. Memory (from the code): the same host
  buffers, allocated earlier: about 48 MB over the phase change and the embedding (the row ids, the thread's copy, table
  0's read buffers, 64 more threads' stacks), at most about 45 MB more at L1's activation peak; nothing in GPU/MLX memory
  or the page cache (direct reads).
- `MLX_SERVE_DSV41_HC_FUSE` (src/hc_fuse.zig, src/kernels/hc_mix_norm.metal, src/kernels/hc_expand_norm.metal,
  patches/turboexl3-zzz9o-hc-fuse.patch; on by default, `=0` off, `=check`): a wide prompt layer's hyper-connection
  stages with fewer passes over the stream ([L, 4, 5120] bf16, 671 MB at 16K rows). Off, the attention side's hc_mix
  reads the stream and hc_norm reads it again (RA_FUSED's read-ahead norm a third time); the FFN side's expand writes
  the new stream into the x1 region and hc_mix and hc_norm each read it back. On, the attention side is one launch
  (hc_mix's threadgroups, statement for statement, and beside each row group's a threadgroup of hc_norm's collapse and
  norm, the read-ahead's norm too under RA_FUSED) and the FFN side's expand and norm are one launch into the region
  (each thread expands its columns of every copy with the expand's fmas and collapses them from its registers), hc_mix
  then reading the region; both sides take the split from the mixes after with hc_split (hc_norm's split statements).
  Exact: every sum over the same `hc_tg` threads, the same simd sums and partials in the same order; a GPU that runs a
  fused kernel at fewer threads refuses it by name at load. `check` serves the stock kernels and compares every output
  word for word (`[hc-fuse] check layer= side=attn|ffn words= differ=`; the FFN side's fused launch evaluated before the
  served expand, which may overwrite the stream in place). src/gpu_check.zig's hc-fuse arm compiles the three fused
  kernels on the GPU at the served template arguments and checks every output against the stock chain's at 1, 2, 4,097
  and 16,383 rows. GPU check (o1g, gpu_check 0238e2fbbf45): every output word-equal to the stock chain's at all four
  sizes (hc_mix_norm with and without the read-ahead's norm, hc_expand_norm, hc_expand_norm_into out of place and in
  place, the in-place stream changed), DONE rc=0, peak 4,322 MB; the first build (o1a1) did not load: a row pointer of
  hc_expand_norm's was `const device float*` from `post`, which MLX binds in the constant space at the load probe's one
  row (`auto` since). Measured (o1a2, w-pio-20261102-1343: ctl42 05cb5ae13fd0 against o1a2 4f2d245ff5e9, both on
  G_KITSP1, two runs an arm): the cold 16K card 1586.7/1587.9 -> 1598.5/1599.7 tok/s (+0.74%; pairs +0.74/+0.74), the
  new-text slice 1705.3/1684.2 -> 1703.3/1704.6 (+0.54%; pairs -0.12/+1.21); texts identical in every arm, chk's
  included; chk's 80 `[hc-fuse]` check lines differ=0; bank rows 147/176 in every measured arm; the guard's peak
  103.7/99.7 GiB against 107.9/104.2 (not up: the fused launches make the same large arrays at the same points as the
  stock kernels, and no fresh stream).

## The pack

- `config.json` keeps exllamav3's `quantization_config` (codebook mul1) and names the Engram tables, which stay in
  `ngram_embedding.safetensors`: `omlx_deepseek_v41.engram_tables["language_model.layers.<l>.engram.embed"]`.
- `model.safetensors.index.json` names every non-routed tensor in the root's shards, and not the Engram file.
- `experts.bin`, `expert-manifest-v2.json` and `expert-manifest.json` hold the routed experts. The parse refuses a pack
  without them (`Exl3PackNotRepacked`).

## Measured

Nothing yet on the served pack. The copy rounds and the fixed verify policy ran on it in kt2 and fp3 (zzz8s's bullet
above), the live policy and its ladder only as the fixed tables' calibration; their choices are also tested MLX-free
(`zig build test`). The stage profiler has not run on a GPU either: its accounting and log lines are
tested MLX-free, its hooks only built.
