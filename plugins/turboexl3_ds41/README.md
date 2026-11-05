# turboexl3_ds41

Serves DeepSeek-V4.1-Flash's EXL3 packs in the streaming repack
(https://huggingface.co/OpensourceWTF/DeepSeek-V4.1-Flash-1017-streaming-repack-exl3/tree/2.81bpw).
The routed experts stream from SSD, and turboexl3 runs the EXL3 trunk. The plugin claims a `deepseek_v41` pack only
when its config carries exllamav3's `quantization_config` (`quant_method` exl3); the built-in deepseek_v41 keeps every
other pack.

    scripts/build.sh turboexl3_ds41                                  # from the repo root: tests, then ReleaseFast
    plugins/turboexl3_ds41/scripts/prepare.sh                        # generate src/mlx_serve/ (build.sh runs it)

M5-class GPUs (NAX) only; any other GPU is refused by name at load.

- src/mlx_serve (generated, not committed) is mlx-serve's own DeepSeek-V4.1 model (plugins/deepseek_v41), its vision
  tower, and mlx-serve's expert streamer (sdk/weights/streamed, module `mlx_stream`, which mlx-serve absorbed from
  David Tai's mlx-stream: see NOTICE), at MLX_SERVE_REF.
  scripts/prepare.sh writes it with scripts/sync-mlx-serve.sh, copies overlay/mlx_serve over it and applies the
  plugin's patches in name order. A build refuses a tree whose hash is not scripts/mlx_serve.sha256.
- The EXL3 arm:
  - turboexl3 runs every trellis linear of the trunk, the LM head, the DSpark stages and the vision tower. `wo_a` runs
    as one grouped linear and the tower's biased linears through `linearBias`.
  - The stages' experts stay resident as one turboexl3 bank per stage, run by `moeClamped` (the pack's `swiglu_limit`).
  - The pack's plain f16 linears are cast to the activation dtype at load.
  - The load keeps suh/svh f16 and never reads a trunk layer's routed experts.
  - The stream budget stays within the host's ceiling.
- The routed experts are the repack's `experts.bin` bank. Their decode runs on turboexl3 (src/routed_t3.zig), on the
  streamer's schedule: its decode waves, banked and with the device route, on turboexl3's banked decode
  (`moeClampedBanked`: a wave's pairs over the layer's three slot banks in one call, each pair `moeClamped`'s bytes at
  top-1 on its bank). The prompt calls stay on the streamer's own DIG-X lane, whose working set the prompt bill covers.
  The banked kernels compile at load, one set per pair count (1 to 48) and code width (the full pack's K3 and K2: 96
  calls); the first load on a box measured 14.2 s for 130 calls, the 96 of these plus the 34 prompt calls.
- The streamer's MLX shims (events, uninitialized buffers, the launch cache, the read pool's Metal signal) are the
  plugin's own, over the server's host table (src/host_shims.zig). The read pool is the streamer's C, compiled here.
- Every decode stretch runs compiled.

## DSpark

The DSpark stages draft as the model's native drafter (plugin.zig), greedy or under the engine's sampling, with the two
round features the engine gives its built-in model (src/rounds.zig, over the engine's own `mtp_lookup.zig`,
`verify.zig` and `dflash/policy.zig` from `engines/mlx/spec`, copied by scripts/slice.txt). Both only choose what a
round verifies; the trunk's accept decides what is kept.

- Copy rounds: where the context agrees with the committed tail and `t1` for 16 tokens or more, its continuation is
  the draft (more than the block, up to the 7 a verify carries on the bank's decode lane), with no stage forward.
  `MLX_SERVE_DSV41_DSPARK_COPY=0` turns them off.
- The verify policy: a stage round verifies the prefix of the block whose calibrated confidence buys the most tokens
  per ms of verify, and where drafting does not pay the round verifies `t1` alone (the engine's backoff). By default
  the fixed policy, on sampled requests only: the cut and the backoff read a cost table by context (src/rounds.zig
  `Fixed`; the built-in `DEFAULT_COSTS`, fitted on the M5 Max, or `MLX_SERVE_DSV41_DSPARK_COSTS`'s), never the
  measured ladder or a round's wall time, so a request's text is the same run to run. A greedy request's rounds verify
  the whole block: `MLX_SERVE_DSV41_DSPARK_FIXED_GREEDY=1` cuts them too (text-safe, since the trunk is batch-invariant
  and a greedy round's width leaves its text the 1-row decode's), but is off by default: its rounds were cheaper and
  more of them, about 2% slower greedy on the M5 Max. `MLX_SERVE_DSV41_DSPARK_POLICY=1` is the live policy on the
  ladder the plugin measures from its own rounds; `=0` turns it off: every round verifies the whole block.
- The cost tables (`MLX_SERVE_DSV41_DSPARK_COSTS`): `U1:t1+U2:t2+t3`, a round at position n on the first table whose
  bound exceeds it; each `t` the verify ms by rows from one to N, then the draft's ms. The built-in tables were
  calibrated on the a4k canonical pack with a greedy round's verify width forced 1..6 in turn at about 1K, 4K and 16K
  prompt tokens of two prompts of their own (Python's asyncio and email sources, apart from every benchmark prompt):
  row w the two prompts' median verify ms at w rows plus the bucket's per-round overhead, to 0.1 ms and strictly
  increasing, the draft's ms the greedy rounds':
  `2048:41.4,47.9,57.2,66.4,77.2,84.4/4.8+8192:40.1,47.1,55.0,64.9,73.2,85.1/4.8+41.0,47.9,57.9,65.8,76.7,85.9/4.9`.
  The fixed policy's gain is the backoff's, on long prose of low acceptance; a code request keeps the whole block in
  most rounds.
- The ladder (src/rounds.zig `Ladder`): the 1-row cell in ms, every other row as a ratio to it, so a level change
  the plain rounds see moves every row; each cell the lower median of its last 8 samples and the engine's prior
  ladder (one stall moves it a rank, a new level takes 4 or 5 samples); a cell unsampled for 64 rounds counts no
  dearer than the fresh rows around it, and a row never measured is the prior's shape bent to them. The two ends
  are refreshed once stale: the 1-row cell by a plain round, the whole block on a stage round (the backoff's probes
  included; the backoff does not judge a refresh). `refreshes=` in `dspark policy:` counts them; a stale cell in
  `ladder_ms` ends in `~`.
- The context is the prompt the resume hands the plugin, then each forward's and round's tokens at their positions;
  a gap or a failed forward drops it, and with it the copies, until the next prompt.
- The next draft is launched ahead: a greedy round's next draft, and a sampled round's where the fixed policy owes no
  plain round, goes to the GPU from the round's next token right after its bookkeeping, and the next round takes it
  when it fits (the same token, position and state), else drops it. Exact: the same ops on the same state, launched
  earlier.
- Each request's rounds are logged beside the engine's `[spec-stats]` (`dspark rounds:`, `dspark fixed policy:` or
  `dspark policy:` with the measured ladder in `ladder_ms`); each round's phases go to mlx-serve's `[dspark-prof]`
  (`MLX_SERVE_DIAG=dspark-profile`).
- The draft's Markov chain (src/markov.zig, its MLX-free plan src/markov_plan.zig, the kernels
  src/kernels/markov_*.metal) runs in B + 1 launches a draft, in place of five op groups a position (a take, the
  [1, rank] x [rank, vocab] matmul, a cast, the add, an argmax), with the op chain's bytes. Position k is one launch:
  each threadgroup reduces the previous position's threadgroup bests to its token (MLX's argmax order: NaN first, then
  the larger value, then the lower index), sums its rows of the bigram bias as MLX's gemv sums them (lane and element
  order, the shuffle-down tree, the round to the activation dtype), adds the logits row in f32 and keeps its best
  (value, index) pair. A last launch gives the tokens and the confidence head's logit (MLX's dot_product). An odd
  position reads the rows from the far end, those the position before read last. Exact: the tokens, every biased row
  and the confidence logits bit for bit the op chain's; 575.6 against 842.0 us a draft on the M5 Max.

## The served path

Every piece below is on and has no switch, except where a setting is named. "Exact" means the served words are the
stock chain's; the gains are the M5 Max's on a 16K prompt.

### The streamed bank

- The resident seed (mlx-serve's `expert.usage`, its settings src/expert_usage_cfg.zig), as glm5_next: at the load,
  each layer's empty prompt rows are filled with its most used experts of `expert-usage-v1.json` (read once, before
  the first request; a seeded expert is a resident as a kept one is, the policy unchanged), when the pack has the
  file. `MLX_SERVE_DSV41_RESIDENT_SEED=none` turns it off, `usage` refuses a pack without it, and
  `MLX_SERVE_DSV41_EXPERT_USAGE_DIR` reads the file from another directory. The prompt residents stay across requests
  (`keep`), and the rate tally stays off (mlx-serve's DeepSeek-V4.1 default).
- The seeded rows stay free until a route uses their expert: the prompt's read-ahead may take one for an expert its
  prediction ranks before the seeded one, so a layer's rows hold its call's hottest experts as they would have held
  them empty. A seeded row a route has used is a resident as any other.
- The usage record: `MLX_SERVE_DSV41_EXPERT_USAGE_RECORD=<absolute path>` counts each request's routed experts per
  layer (the prompt pass's and the decode routes') into that file at the request's end, and
  `MLX_SERVE_DSV41_EXPERT_USAGE_LABEL` names the prompt set; they mirror the in-tree archs' `expert_usage_record` and
  `expert_usage_label` model settings. The counts are routes, so the seed does not bias them.
- The decode's slot rows are split across the layers (src/decode_rows.zig) at the flat bill's bytes or fewer, each
  layer between its prompt rows and 256, by each layer's misses-a-row curve, fitted on routes recorded on a
  development corpus kept apart from the benchmark prompts. The curves fit this pack's geometry only (40 layers, 384
  experts, its K3 and K2 records): on any other the rows stay flat (one info line). One load line gives the rows and
  the bytes against flat. 7.7% fewer misses a round, greedy decode about 4% faster.
- An a4k bank (src/mlx_serve/sdk/weights/streamed/expert/a4k.zig; the manifest declares it, `expert-manifest-v2.json`'s
  `sidecar.layout` "a4k"): each record's three codes start on a 4 KiB block, so the direct route (below) reads every
  code straight into its slot row (99.7% of a record) where the classic layout reads the gate's code alone (33%). On
  disk a record is a zero lead of 3,072 B, the gate/up span [gate rout, gate rin, up rout, up rin, gate code, up code],
  then the down span [down code, down rout, down rin], padded to 4 KiB: 4,096 B more a record (+62.9 MB a bank). The
  segments' bytes, lengths and shapes are the classic ones; the bank parses the layout and checks its table against
  `a4k.geometry`. A classic bank runs the classic path; an older build refuses an a4k bank by name. `zig build test`'s
  a4k gate reads the same records laid out both ways through every route of the copied pool and finds the same slot
  bytes (`zig build test -Da4k-gate=<classic>:<a4k>` adds two real packs).
- Direct reads: a record is read as its gate/up span and its down span. In the prompt phase, a part whose file offset,
  length and row address lie on the file's block (4,096 B) is read straight into its row, no staging and no copy;
  decode's reads, the pre-read ranges and the lookahead's records stage as before. Exact: the file's bytes land in the
  rows either way.

### The prompt pass

Staging and reads:

- The predicted transient loads of a wide prompt call read while its layer's attention runs: the stream reads the
  prediction's stream misses into the windows the call's first stream groups take; the call adopts them, and a
  transient miss takes the row of its window holding its record and skips the read. Exact. +4.1% cold prefill.
- The next layer's reads start during the call before: the model predicts layer N+1's routes from layer N's FFN rows
  (N+1's gate over N's MoE input, its columns scaled by the two FFN norms' ratio) and the source reads layer N+1's
  predicted seed ahead into its persistent rows and lays its stream misses out as staging does, behind N's routes. A
  wide layer the call before read ahead for runs no read-ahead of its own; the first wide layer of a pass reads ahead
  as before, its prediction on the fused hc kernels. The prediction picks only which records are read ahead and into
  which rows, never the routes. +3.6% cold prefill, and +2.5% more from dropping each layer's own read-ahead.
- Staging takes only the predicted stream misses that at least 8 rows of the prediction name; the read-ahead into the
  persistent rows keeps the whole ranking, and an expert the floor leaves out that the call routes is read by its own
  route. A staged window's jobs go out hottest first, and once the call has its routes, a staged job holding none of
  its experts that no reader has started is withdrawn. Exact: these pick which records are staged, never the routes.
- A persistent load whose record a staged window row already holds is filled by copying that row, not read from the
  file again. Exact: a record is the same bytes in any row. +1.95% cold prefill.
- The prompt's Engram rows: see Engram below.

The wide layers' kernels:

- `wo_a` runs over runs of 4096 rows of the attention spans, one grouped linear a run (each slice's trellis decoded
  once a run, its GEMM at M = 4096), in place of a grouped linear a 256-row span. NOT bit-exact: a reorder-level
  change; MLX's GEMM at the other M rounds a row's sums in another order, so about 2% of words differ, by at most one
  bf16 ulp. +1.4% cold prefill.
- A `wo_a` run skips turboexl3's per-group copies and launches: one prepare launch reads the run's rows in place,
  each group's W_hat is decoded once a layer and kept across its runs (67 MB, billed in the prompt term), and one
  finish launch writes the joined rows: 10 launches a run where there were 41. Exact. +2.6% cold prefill.
- A prompt's dense EXL3 linears run their matmul and finish as one launch: MLX's fused NAX GEMM body with the finish
  as its epilogue, at a tile measured per shape on this machine, only where MLX runs its regular kernel, so z (about
  1 GiB at a 16K prompt's `wq_b`) is never allocated. Exact. +1.85% cold prefill.
- The wide router's f32 logits run as one launch of MLX's fused NAX GEMM body, reading x and the gate in their stored
  types and casting as they load, only where MLX runs its regular kernel (1,707 rows and up). Exact. +0.65% cold
  prefill.
- A wide call's DIG-X waves run their 128-row routed GEMMs (K3 and K2, gate|up and down) with the B stage
  double-buffered; the quant self-checks the four texts at load against the 128-row texts, bitwise, and a failure
  refuses the load by name. Exact.
- A wide call's routed sum and the layer's shared expert are one combine: each routed row read where its wave wrote
  it, weighted, the six folded in MLX's column reduce order and the shared rows added. Exact. +10.7% cold prefill.
- The layer's tail is one kernel: the combine's fold, one rounding to the activation dtype, then glm5's hc expand,
  writing the new streams. Exact. +1.2% cold prefill.
- The hyper-connection stages make fewer passes over the stream: the attention side's mix and norm are one launch,
  and the FFN side's expand and norm are one. Exact; a GPU that runs a fused kernel at fewer threads refuses it by
  name at load. +0.74% cold prefill.
- A wide layer takes the last wide layer's whole stream and FFN mix as they are, where every span still holds its
  slice, in place of a concatenation. Exact.
- What a wide call's MoE does not read (the routes' cast, the next layer's prediction, the shared expert with the
  attention state) is committed before the routing barrier waits on the routes; the prompt bill counts the shared
  expert's evaluation (1.03 GB at 16K rows). Exact. +1.0% cold prefill.
- A replaying prompt chunk's last decoder layer routes its last row only (its other rows are read by nothing but the
  logits); its router and shared expert run over every row. The served words were measured equal to the 128-row
  call's. +0.7% cold prefill.
- The index layers evaluate their spans' top-k picks in groups of spans whose pending picks stay within 2.5 GiB (every
  span up to a 16K prompt, then 35, 17, 8 and 4 spans at 32K, 64K, 128K and 256K) (src/idx_group.zig). Exact: the
  same picks in the same order.
- A KV-source layer runs its compressor for every span before the span loop, so the chunk's rows append to one cache
  buffer and each span reads its causal prefix of it, where each span's append copied the whole cache
  (src/comp_once.zig). Exact. The prompt's peak over the load peak, 16K / 32K / 64K / 128K: 2.97 / 3.25 / 3.67 / 3.02
  GiB, where it was 3.02 / 3.60 / 4.56 / 5.43; 256K served at a 99.50 GiB peak.

Attention:

- The query heads rope only their last 64 channels; the score step reads the unroped heads for its first seven
  64-deep steps and the roped tail for the last. Exact: the same scores. +0.66% cold prefill, +2.5% on new text.
- `MLX_SERVE_DSV41_FLASH_ATTN` (on by default): a wide prompt layer's attention span runs as one flash kernel
  (src/flash_attn.zig, src/flash_plan.zig, src/kernels/attn_flash.metal) in place of the gather's three launches. One
  threadgroup is a query row and its 64 heads, which share the row's keys; the keys go in blocks of 128, the
  probabilities in f16 against a running max, the accumulator in f32; the epilogue inverse-ropes the last 64 dims and
  stores the heads into the run's region. No scores or probabilities are written. NOT exact: the probabilities round to
  f16 before the division and the sum's order differs, so 3.8-7.5% of a wide layer's bf16 head words differ, by at
  most one bf16 ulp, and a text can part from the exact one after a near tie (KL at a prompt's last row 3.8e-3 to
  4.5e-2). +0.58% cold prefill, +2.1% on new text. `=0` serves the exact gather; a span the kernel cannot take gets
  the gather.
- The gather (with `MLX_SERVE_DSV41_FLASH_ATTN=0`, or a span the flash kernel declines) writes its scores and
  probabilities into a ring of two per-prompt slots (126 MB) instead of fresh MLX buffers, its pv at 64 output columns
  a threadgroup and its tensor-op launches in MLX's NAX GEMM threadgroup shape. Exact.
- The attention's heads are written straight into the `wo_a` run's region (two by the run's parity, 537 MB), which
  `wo_a` reads as it is; a run that is not full keeps the join. Exact. +3.2% cold prefill.

Buffers and commits:

- Per-prompt regions (src/scratch.zig) in place of fresh MLX buffers: the waves' intermediates (two sets by the wave's
  parity, 2 x 459 MB, write-into twins of the stock texts derived and compiled at load; a twin that fails refuses the
  load by name), the streams after the attention (`x1`, 1.34 GB at 16K rows), and the new streams written in place
  over the x1 region where the next layer is a wide one of the same call. The regions together stay within 4 GiB.
  Exact. +1.65%, +1.8% and +1.84% cold prefill.
- A wide layer's new streams are committed at the call's end, before the layer's teardown, and when the next layer is
  wide too its last eval is committed, not waited for. Exact. +0.44% and +0.76% cold prefill.
- A wide layer with another after it keeps its attention projections' large outputs (1.4 GB at 16K rows) past its
  teardown, and the next layer's start hands their buffers to MLX's cache. Exact. +0.63% cold prefill.
- MLX's command-buffer split is raised to 500 MB during prompts and put back to MLX's (50 MB on the M5 Max) at the
  change to decode. A user's `MLX_MAX_MB_PER_BUFFER` wins. It needs mlx-serve's additive
  `HostGpu.command_buffer_limits`; on an older host, MLX's own split in every phase (the load line says so). Exact.
  +3.1% cold prefill, +4.1% on new text.
- The token embedding table (1.32 GB) stays on disk and its rows are read by pread at each lookup, each distinct id
  once, in parallel. The vision tower (0.94 GB) stays on disk too: each image reads it, encodes and frees it, and a
  bank holding its decode rows drops to its prompt rows while the tower is loaded. The tower's split EXL3 copies the
  repack ships beside the fused ones are never read. The bank is sized from the memory in use, so those bytes become
  slot rows. Exact.

### Decode

- The expert lookahead reads the next layer's predicted experts ahead of its route: 2 records a layer call, then while
  a credit lasts up to 2 more of the same ranking (about 10 extra records a round), the stream sized for a budget of
  4. The next layer's scores run as a product in the router weights' half type (MLX's wide GEMV) and stay out of the
  routing barrier: read once the call's waves are built. Exact: the lookahead only reads ahead. 7.0 ms a round faster
  than no lookahead, and 1.25 ms more from the late scores.
- Demand reads first: an unclaimed lookahead record reads only while no demand read is queued or running, in 1 MiB
  pieces, and a claimed record's untaken pieces go to the demand reads that need them. Exact: the same bytes into the
  same rows. +0.73% card decode, +1.43% greedy.
- A missed record's gate/up range is read as two range jobs side by side, split near its middle, so a miss waits about
  as long as its down span. Exact. +4.3% card decode, +1.3% greedy.
- The device hit wave runs on turboexl3's grouped banked decode: up to four row-slots of one expert share a weight
  decode, a lead with three takes an M = 3 path, the GEMVs take turboexl3's t4 variant, and a pair whose expert no
  bank holds runs no expert (its miss part's row replaces it). Exact: every row-slot's words are the banked call's.
- The shared expert is committed as soon as the routed call returns, so it runs while a layer with misses waits on
  their bytes; at a verify's 5 to 16 rows its gate's and up's finishes, the clamped SwiGLU and the down's prepare are
  one launch. Exact.
- A decode call's tail (the join, the weighted sum, the shared expert and glm5's hc expand) is one kernel
  (src/tail_fuse.zig), compiled at load. Exact. +0.63% card decode.
- The indexer's plain bf16 linears (at 2 to 6 rows) and the Engram layers' `wkv` (at 2 to 16 rows) run on a batched
  twin of MLX's one-row gemv (src/gemv_twin.zig); one row keeps MLX's gemv and wider calls MLX's matmul. So a verify
  row's words are the ones it gets decoded alone, at any width (the batch-invariant trunk).

### Engram

- A prompt's Engram tables are read once per table and past the page cache: every span's ids sorted and made
  distinct, the distinct rows read with F_NOCACHE on up to 128 threads, then each span's rows copied out; a repacked
  table (`engram/engram-manifest.json`, a row's codes then its scales in one 264 B record) is read a record a row, one
  pread. The ids are hashed and the read started before the phase change. Exact: the same bytes land in the same rows.
  +5.5% cold prefill from the direct read, +5% more from the repacked table, +1.6% from the earlier start.
- An LRU of Engram rows over the tables is kept across requests and a conversation's turns (src/engram_decode.zig): a
  prompt's distinct rows go in, and so does every row a decode read reads; a decode read takes the rows the LRU holds
  and reads the others on a pool of 64 threads; a round's bonus token has its rows read when the round starts.
  `MLX_SERVE_DSV41_ENGRAM_LRU_MB` sizes it, 1 to 16384 MiB, 256 by default, billed at load (the load line gives the
  prompt and decode slot rows it costs). Exact. +0.67% greedy decode.

### Memory

- The streamed bank's bill: the slot rows come from each phase's budget at the bank's open, MLX's recommended working
  set less the wired margin, the host's and the Engram LRU's bytes, the memory in use (every process's, read live) and
  the phase's working set. A `[bank-bill]` load line gives the bill's terms (the working set, ceiling, margins, host
  and LRU bytes, memory in use, each phase's working set, windows, staging, rows and their remainders, the decode cut).
- `MLX_SERVE_DSV41_BILL_CAL` (on by default): the bill calibrated against measured peaks (src/bill_cal.zig). Each
  phase's budget is MLX's recommended working set less 410 MiB, the memory in use and the phase's measured working set
  at the context (prompt 12.32 GiB for one 16K call up to 13.60 at 256K; decode 0.16 GiB at 1K up to 2.58 at 256K), so
  the box's non-file baseline plus the server's peak lands 1.5 GiB under a 112 GiB guard; the wired margin, the
  host's, the Engram LRU's and the decode stop's terms are inside what was measured. On only under the calibration's
  conditions (`--prefill-chunk 16384`, MLX_SERVE_CACHE_LIMIT 2 GiB, an Apple M5 Max with 128 GB, a 112 GiB wired
  limit), else the plain bill and a load line naming what differs; `=0` the plain bill. `[bank-bill]` gains `cal=` and
  the bank-open readings `self_mb`, `spec_mb` and `purg_mb`. Every server's peak measured within -0.14/+0.20 GiB of
  its prediction. Rows (prompt / decode) at 16K and 256K: 2.81 146/183 and 144/179, 2.25 200/247 and 197/240, 1.68
  267/324 and 263/316, 2.68 160/199 and 158/195.
- Window 0: decode's window holds 4 rows, not the widest route's 48; a decode plan admits its declined misses past
  them over the lowest-ranked residents the route does not use, and the bill gives the freed ~977 MB to the decode
  slot rows, where every layer keeps 96 decode rows or more (else the widest route's window). Exact. +1.29% greedy.
- The decode stop: the decode budget takes 1,536,870,912 B of the 2.0e9 B stop under the ceiling back for its slot
  rows, with MLX's cache capped at 512 MiB from the change to decode and the prompt's cap restored at the change back.
  The prompt's scratch regions are let go before the change to decode. The decode phase's runtime floor is 128 MiB
  (a prompt pass's is 512 MiB). The prompt's budget and rows are unchanged. Exact: only the decode rows' count moves.
- The bank pool (src/bank_pool.zig): the slot rows of both phases over one MLX buffer, so a phase change installs the
  other phase's layout instead of freeing one and allocating the other (about 13-15 ms a GiB). Neither phase holds more
  than without the pool. It needs mlx-serve's additive `HostGpu.alias`; a host without it loads and serves with the
  pool off and a `[bank-pool] off` line at load. Exact. +1.5% cold prefill.

### Logs

- `[bank-bill]` at load, the bill's terms.
- `[spec-stats] dspark rounds:`, `dspark fixed policy:` / `dspark policy:` at each request's end, and `[stream-stats]`.
- `[bank-pool] off` at load on a host without `HostGpu.alias`.
- `[dspark-prof]` under `MLX_SERVE_DIAG=dspark-profile`.

## Settings

- `MLX_SERVE_DSV41_DSPARK_COPY`: copy rounds; `=0` off.
- `MLX_SERVE_DSV41_DSPARK_POLICY`: unset, the fixed verify policy; `=1` the live policy; `=0` off (every round the
  whole block).
- `MLX_SERVE_DSV41_DSPARK_COSTS`: the fixed policy's cost tables.
- `MLX_SERVE_DSV41_DSPARK_FIXED_GREEDY`: `=1` the fixed policy cuts greedy rounds too; off by default.
- `MLX_SERVE_DSV41_ENGRAM_LRU_MB`: the Engram LRU's size, 1 to 16384 MiB, 256 by default, billed at load.
- `MLX_SERVE_DSV41_FLASH_ATTN`: the prompt's flash attention, on by default and not exact; `=0` serves the exact
  gather.
- `MLX_SERVE_DSV41_BILL_CAL`: the calibrated bill, on only under its measured conditions (else the plain bill, with a
  load line); `=0` the plain bill.
- `MLX_SERVE_DSV41_RESIDENT_SEED` and `MLX_SERVE_DSV41_EXPERT_USAGE_DIR`: the resident seed (`none` off, `usage`
  required).
- `MLX_SERVE_DSV41_EXPERT_USAGE_RECORD` and `MLX_SERVE_DSV41_EXPERT_USAGE_LABEL`: record a usage profile; they mirror
  the in-tree archs' `expert_usage_record` and `expert_usage_label` model settings.
- `MLX_SERVE_DIAG=dspark-profile`: mlx-serve's own DSpark profile, `[dspark-prof]`.
- `MLX_MAX_MB_PER_BUFFER`: a user's value wins over the plugin's prompt split.

## The pack

- `config.json` keeps exllamav3's `quantization_config` (codebook mul1) and names the Engram tables, which stay in
  `ngram_embedding.safetensors`: `omlx_deepseek_v41.engram_tables["language_model.layers.<l>.engram.embed"]`.
- `model.safetensors.index.json` names every non-routed tensor in the root's shards, and not the Engram file.
- `experts.bin`, `expert-manifest-v2.json` and `expert-manifest.json` hold the routed experts. The parse refuses a pack
  without them (`Exl3PackNotRepacked`).

## Measured

Nothing yet on the served pack. The copy rounds and the fixed verify policy ran on it during development (the verify
policy above), the live policy and its ladder only as the fixed tables' calibration; their choices are also tested
MLX-free (`zig build test`).
