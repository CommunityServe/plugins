# turboexl3_glm53f

Serves GLM-5.3-Flash's EXL3 packs in the streaming repack of turboderp's self-calibrated branches
(https://huggingface.co/turboderp/GLM-5.3-Flash-exl3: sc_2.10bpw, sc_2.60bpw, sc_3.10bpw, sc_4.10bpw). The routed
experts stream from SSD, and turboexl3 runs every EXL3 linear and every routed expert. The plugin claims a `glm5_next`
pack only when its config carries exllamav3's `quantization_config` (`quant_method` exl3) and an `expert_streaming`
block with `family` `turboexl3`; the built-in glm5_next refuses that block by name and keeps every other pack.

    scripts/build.sh turboexl3_glm53f                                  # from the repo root: tests, then ReleaseFast
    plugins/turboexl3_glm53f/scripts/prepare.sh                        # generate src/mlx_serve/ (build.sh runs it)
    zig build gpu-test-build -Dmlx-lib=<dir of the server's libmlx>    # in the plugin: turboexl3's GLM tests, compiled
    TURBOEXL3_GPU=1 zig-out/bin/gpu-test                               # the same on the Metal device (a GPU run)
    zig build kda-gpu-test-build -Dmlx-lib=<dir of the server's libmlx>  # in the plugin: the KDA texts' parity test
    TURBOEXL3_GLM53F_KDA_GPU=1 zig-out/bin/kda-gpu-test                  # the same on the Metal device (a GPU run)
    zig build router-gpu-test-build -Dmlx-lib=<dir of the server's libmlx>  # in the plugin: the router's parity test
    TURBOEXL3_GLM53F_ROUTER_GPU=1 zig-out/bin/router-gpu-test               # the same on the Metal device (a GPU run)
    zig build smallop-gpu-test-build -Dmlx-lib=<dir of the server's libmlx>  # in the plugin: the merged small ops' check
    TURBOEXL3_GLM53F_SMALLOP_GPU=1 zig-out/bin/smallop-gpu-test               # the same on the Metal device (a GPU run)
    zig build draft-gpu-test-build -Dmlx-lib=<dir of the server's libmlx>  # in the plugin: the head's last-row check
    TURBOEXL3_GLM53F_DRAFT_GPU=1 zig-out/bin/draft-gpu-test                  # the same on the Metal device (a GPU run)

M5-class GPUs (NAX) only; any other GPU is refused by name at load.

## What is copied

src/mlx_serve (generated, not committed) is mlx-serve's own GLM-5.3-Flash model (plugins/glm5_next), its weight loader
(sdk/weights/load.zig) and Sushi's tensor moves (sdk/quant/exl3/sushi_pack.zig), at MLX_SERVE_REF. scripts/prepare.sh
writes it with scripts/sync-mlx-serve.sh (turboexl3_ds41's script; its kernel scan also copies the CUDA texts a copy
embeds beside its Metal arm), copies overlay/mlx_serve over it and applies patches/*.patch (one an area:
turboexl3-model.patch, turboexl3-sdk.patch). A build refuses a tree whose hash is not scripts/mlx_serve.sha256.

The GLM-5.3-Flash EXL3 bank is not copied: src/streamed/ and src/exl3/ (module `mlx_stream`, over its own `sdk` in
src/streamed/sdk/) are mlx-serve's GLM bank, its scheduler and MiMo's stream bridge, moved out of mlx-serve into this
plugin at bc6c5f2c (mlx-serve no longer carries them; see NOTICE). They stream through the kit's expert stream
(`kit.streamed`: the stream, its residency policy and its read pool, whose C the build compiles in, `streamSources`).

The plugin's own code:

- config.zig: the claim, the pack check (the mul1 codebook, the bank block's fields, both banks' files) and the
  built-in's parse over the config without the bank block, then the built-in's facts for a streamed pack.
- plugin.zig: the load. The host loads none of the shards (`keep_key`); the plugin reads them (src/pack_weights.zig), runs
  the built-in's trunk load over them, binds the MTP layer's bank, builds the readout head and opens the streamed bank.
  Its bill is the built-in's plus one prompt's (`Bill.prefill_bytes`). The load line `streamed experts, N prompt / M
  decode slot rows` also prints the room and where it came from (`host` or `own`), the memory total and available,
  and the free bytes the rows came from.
- src/stream_room.zig: the prompt bill and the slot rows. A host gives a streamed plugin `LoadCtx.stream_room`: the
  GPU ceiling less one 16K + 1K prompt and the wired margin. mlx-serve after 26.11.1 prices an outside plugin's prompt
  through its `prefill_bytes`; 26.11.1 prices only its built-in GLM pack and gives an outside plugin 0, and every layer
  then held top_k (8) rows. So the plugin uses the host's room when it gives one, else its own: the host's ceiling
  (`MLX_SERVE_GPU_CEILING_MB`, else Metal's recommended working set, else 75% of the RAM) less the same prompt bill and
  `LoadCtx.wired_margin_bytes`. `prefillNeed` is `glm5PrefillMemoryNeeded` at the host's defaults.
- src/pack_weights.zig: the copied loader over the pack's nonexpert shards, then the moves the engine makes for a GLM pack
  in turboderp's names: `hc_attn_*` / `hc_ffn_*` to `attn_hc.` / `ffn_hc.`, layer 45 to `mtp.0` and `mtp.0.block`,
  `kv_b_proj` split into `embed_q` / `unembed_out`. The vision tower is not read.
- src/routed_t3.zig: the streamer's math seam with every routed call on turboexl3 (below). It slices each slot bank's
  scale arrays to the code array's rows: the stream floors every slot array at one VM page, so a bank of 1 to 3 rows (a
  grown layer's ext rows) has longer scale arrays, which turboexl3 refuses. A turboexl3 refusal is logged by name (the
  routed seam hands the host EngineFailed).
- src/mtp_bank.zig: the MTP layer's 288 experts from `mtp-experts.bin` (its v2 manifest checked against the record
  geometry its K gives) as one resident turboexl3 bank.
- src/bank_layers.zig: each routed layer's slot-row bytes (its record's segments) from the pack's
  `expert-manifest-v2.json`, for the per-layer bill (`TURBOEXL3_GLM53F_LAYER_BILL`).
- src/options.zig: every `TURBOEXL3_GLM53F_*` switch the plugin reads at load (67), as the manifest's options: its kind
  (60 `bool`, 2 `number`, 5 `text`: BANKED, DEC, PF_OPTS, PG2's tile codes and LOGITS_DUMP's directory), the
  plugin's own value when none is set (`defaultOf`: the served configuration's, under Switches below) and one line of
  help. A kit whose `sdk.Plugin` has `options`
  (mlx-serve #1131, the 26.11.3 kit) exports them in the dylib's manifest, so the app shows them on the plugin's page and
  sets a saved value before it opens the plugin; an older kit (26.11.2) builds the plugin without them, unchanged. Its
  MLX-free test checks each name against the kit's option name rule and each default against its kind, where the kit has
  them (`abi.dylib.optionNameOk`, `valueOk`).
- src/stream_stats.zig: `[stream-stats]`, the streamed bank's counters by phase in each request's summary (the kit's
  `meters`, beside `[spec-stats]`): one `prompt:` and one `decode:` line, each the stream's own counters (routes, resident
  hits and misses, evictions, loads, bytes read, the read pool's busy wall and the host's read waits, the read-ahead and
  the lookahead) since the phase's first forward, and the decode's forwards by mode. The decode line splits the
  lookahead's unclaimed records by why: `spec_expired`, `spec_abandoned` and `spec_discarded` (dropped while queued,
  dropped while read, read and never served: wrong predictions) and `spec_cancelled` (still queued when their call's
  demand read them: right, not started); `spec_inflight` (claimed while still in flight), `spec_gb` and
  `spec_abandoned_gb` (the speculative reads' and the abandoned records' bytes); after the forwards, `adopt_copy_ms`
  (the adopted ranges' copies into their rows, summed, on the read pool's clock). The counters of the switches whose
  mechanism is a seam of the kit's stream (`split`, `adopt_waits`, `help`, `adopt_split`, `ahead_picks`, `zc_*`, the I/O
  shape) print only where the kit's `Stats` has them (none in 26.11.3's). Where it lacks them, the decode line ends with
  the read pool's own counters: `pool_adopt_waits` and `pool_adopt_wait_ms` (the demand ranges that waited for a
  lookahead record in flight, and that wait), `split_pieces` (`DEMAND_SPLIT`'s ranges split / pieces) and
  `adopt_pieces` (`ADOPT_PIECES`' ranges copied in pieces / pieces another worker copied), each 0 on a kit stream
  without its seam. Nothing else is timed for it.
- src/box_memory.zig: the box's non-file memory (wired, anonymous and compressor pages) for `TURBOEXL3_GLM53F_BILL_FIT`'s
  check when the bank opens.
- src/host_shims.zig: the C symbols the kit's stream and the GLM bank call that mlx-serve serves from C++ shims over
  MLX's private headers (events, uninitialized buffers, the bank pool's alias, the launch cache, the read pool's Metal
  signal, buffer addresses), over the server's host table. turboexl3_ds41's, with the GPU-address shim.

## Switches

Read at load. Unset, each switch takes the frozen exact configuration's value (src/options.zig's default: on, the
number and text switches at their accepted values), so a plain install serves that configuration: the 64-entry control
of 2026-11-10 with `DSA_HOLD` and `PF_TAIL_EVAL` on (frozen 2026-11-12: plugin ac87fe1, the standard 16K coding
prompt's greedy text c3c346222307), but for three switches whose mechanism is a seam of the kit's stream
(`PROMPT_KEEP`, `DEMAND_SPLIT`, `ADOPT_PIECES`; src/streamed/glm5n/experts.zig `seams`): on a kit whose stream lacks
the seam (26.11.3's lacks all three) each is off, and setting it refuses the load by name (`Glm5NextStreamSeam`). The
control's `ADOPT_ZC=2` is gone (no switch). Four switches are off unset: `PF_INT8` (not exact: a server-start
option, 1 to 4), `LOGITS_DUMP` (a check, never served), `PF_RUN_TRIM` (exact, rejected in its A/B) and `DEMAND_HELP`
(exact; on the kit stream's own `demandFirst` it was rejected in its A/B, g53g2h1: decode 40.575 -> 39.30 tok/s). `0` turns a
switch off, `1` on, any other value refuses the load by name (`Glm53fSwitchValue`). A switch that needs another ("With X") is refused while that one
is set to 0 and it is not. Each keeps the served words: the bill switches move only slot-row counts (a row holds its
record's bytes in any row), the grow switch only how new rows are allocated, the residency switches only which
records stay in which rows (a record's row, persistent or transient, runs the same kernel on the same bytes), the seed
switches only which records start in the rows, the lookahead and read switches only which records are read ahead and
when, and the hoist switch only when the shared expert's ops are committed. Each prints a load line that starts with its
`NAME=1: `.

- `TURBOEXL3_GLM53F_LAYER_BILL=1`: the slot rows billed at each routed layer's own row (src/bank_layers.zig), the
  transient and window rows of both of the bank's geometries counted (src/stream_room.zig `rowsByLayer`), where the
  default bills the widest row (an SC pack's layer 3, one K above the rest) in all 42 layers. sc_3.10 at g53smd's room:
  130 prompt / 133 decode rows a layer, where the default gives 100 / 102 in the same bytes.
- `TURBOEXL3_GLM53F_DECODE_ROOM=1`: the decode phase's rows also in the prompt's bill less the decode's
  (`decodeNeed`: the prompt's KV and checkpoints, a generation's growth and the runtime floor at the prompt bill's 5/4,
  plus the host's 2 GiB hot prefix cache), which the phase change frees; the stream grows to them at the first decode
  call and shrinks back before the next prompt. sc_3.10 at g53smd's room: 134 decode rows (16,707,181,312 B more), 174
  with the per-layer bill. A host that prices the plugin's prompt in its admission (after 26.11.1) sees these rows in
  use until that shrink.
- `TURBOEXL3_GLM53F_GROW_UNFILLED=1`: the decode grow (at each change into the decode phase) takes its new slot rows
  as MLX buffers without the zero fill, the kit stream's `GrowFill.unfilled` (src/streamed/glm5n/experts.zig
  `servedStreamOptions`), where the default evaluates MLX zeros over every new row. Exact as the stream states it: a
  row is read only through a slot whose meta says its record landed, and every record lands by its read first.
- `TURBOEXL3_GLM53F_PROMPT_KEEP=1` (a seam of the kit's stream: refused on a kit without it): no prompt-phase admission
  evicts a resident (the stream's `prompt_keep` on every layer's policy). A prompt miss takes an empty prompt row, else
  a transient row, and is not kept; the call's seed marks no expert, and the read-ahead takes empty rows only. The
  default lets each miss and each predicted record take the coldest unprotected resident's row (a 995-token probe: 3,989
  evictions), so the hot set a decode left is gone by the next decode; with the switch it stays, and the prompt reads
  more.
- `TURBOEXL3_GLM53F_SPEC_PICK1=1`: the decode lookahead reads ahead one record a layer call, the selector's first
  non-resident prediction, where the stream reads 2 (the kit stream's selector budget set to 1 once the stream is built,
  src/streamed/glm5n/experts.zig). The read pool keeps its 4 speculative slots, its 2 threads and its bytes, which the
  stream sized from the options' budget, so no slot or row count changes. A decode route settles the records read ahead
  for it right after its claims, so a wrong guess is dropped as soon as its routes are known; what it read by then is
  the waste: the int10cp2 probe's control abandoned 10,737 records, 75.8 GB of its 156.5 GB read ahead, while 7,808 of
  its 8,110 claims found their record still in flight. A call's second guess shares the drive with its first and with
  the demand reads. Only reads move: every range is read into the same row with the same bytes.
- `TURBOEXL3_GLM53F_DEMAND_SPLIT=1` (a seam of the kit's stream and its read pool: refused on a kit without it): a
  decode call that reads one record reads its gate/up span (6,316,032 B on
  sc_3.10's K3 layers) as one page-aligned piece per read pool worker but one (3 at the default 4 workers), each a pool
  job read straight into its rows, beside its down span, where the copy reads the span as one job and leaves the other
  workers idle. The span keeps its one ticket: the pool publishes it, and
  releases the GPU's gate on it, after its last piece landed, its status and counts the pieces' added up. Only on the
  direct route (the served one), and never for a span a record read ahead holds (copied out whole). No slot or row count
  changes. Only reads move: every byte lands in the row the whole span's read writes it to. A call that reads two or
  more records is read as before. `[stream-stats]` counts the split spans and their pieces (`split_pieces=`), and its
  `preadv` count rises by the pieces.
- `TURBOEXL3_GLM53F_SHARED_HOIST=1`: a decode or verify call (8 rows or fewer) builds its layer's shared expert before
  the routed call and hands it to the streamed source, which commits it behind the call's hit wave and ahead of its
  miss waves when the call misses (patches/turboexl3-sdk.patch: routed.zig's `Routed.source_hoist`). The
  GPU then runs the shared expert while the misses' bytes land, where the default runs it after the routed call, behind
  the last miss wave. The same ops on the same rows and the same add, so the same words. A call with no miss leaves it
  to the layer's evaluation, where the add reads it. The first decode call handed its hoist prints
  `TURBOEXL3_GLM53F_SHARED_HOIST engaged`. No slot or row count changes.
- `TURBOEXL3_GLM53F_READ_AHEAD3=1`: a decode or verify window (8 rows or fewer) scores the first routed layer's router
  (model layer 3) on the last dense layer's MLP input and commits the scores; once layer 3's attention is committed
  behind them, the scores are read back and that layer's likely records are read ahead before its route, as the
  lookahead reads every later layer's records on the call before (src/streamed/glm5n/experts.zig `speculateAhead`:
  the kit stream's lookahead step for that layer over its selector and read pool, at the selector's budget, tagged for
  the coming route; the stream's pick-precision counts are not fed). No routed call comes before
  layer 3, so by default its misses are read only on demand. Speculative reads only: the routes, the ops and the
  outputs are the same. The first records read ahead print `TURBOEXL3_GLM53F_READ_AHEAD3 engaged`. No slot or row
  count changes.
- `TURBOEXL3_GLM53F_DEMAND_HELP=2` (off by default: rejected in its A/B on the kit stream, g53g2h1, decode -3.1%; the
  control's 2 was this plugin's own read pool help, and the mechanism is now the kit stream's `demandFirst(2)`, as
  turboexl3_ds41 sets it at its phase change): at each
  decode phase the read pool puts decode's demand first (`Stream.demandFirst(2)`, the read pool's `q3ld_demand_first`;
  back to 0 at each prompt phase): an unclaimed lookahead record starts and continues only while no demand job is queued
  or running, read in pieces (1 MiB, doubled until 64 cover a slot), so it stops at the next piece once demand starts;
  and a demand range that finds its record claimed and still in flight reads the record's untaken pieces itself, into
  the record's staging. A record keeps the mode it started with. No slot or row count changes. Only reads move: the
  same file bytes into the same staging and the same rows. Any other value (`1` among them) refuses the load by name.
- `TURBOEXL3_GLM53F_ADOPT_PIECES=1` (a seam of the kit's stream and its read pool: refused on a kit without it): a
  demand read served from a lookahead record (adopted) copies its bytes from the record's staging into its rows in up to
  as many pieces as the read pool has workers (each at least 1 MiB), and the workers with no read queued copy some of
  them, ahead of the pre-ranges, where the copy is one memcpy by the worker that adopts the range (the stream's
  `Lookahead.adopt_pieces`, the read pool's `q3ld_adopt_pieces`). The copy sits on the gate's path of every range a
  claimed record serves (about 17 records a round). The adopting worker copies the pieces nobody took and waits for the
  others before it publishes the range; the record stays held until every piece is copied. No slot or row count changes.
  Only the copy moves: the same bytes into the same rows. `[stream-stats]`'s decode line counts the copies' time
  (`adopt_copy_ms`) and the ranges copied in pieces (`adopt_pieces=`).
- `TURBOEXL3_GLM53F_BILL_FIT=1`: the slot rows are fixed by the box's ceiling and the server's context limit, not by the
  memory available at load (src/stream_room.zig `fitPlan`). The plan keeps the box's non-file memory (wired, anonymous
  and compressor: the account the GPU window's guard counts) at or under the ceiling for any request up to the context
  limit. On ai-server the ceiling is 110 GiB, 2 GiB (the host's wired margin) under the guard's 112 GiB trip. Each phase
  takes a measured base: the guard's 110.0 GiB decode peak (g53fzr3, 212 decode rows) less its bank gives the decode
  base, 32,064,681,984 B; the server's 95.5 GB prompt peak (g53lb1, 157 prompt rows) less its bank, plus the box's
  memory outside the server at the guard's peak, gives the prompt base, 45,220,334,746 B. Each phase then adds the
  engine's exact terms from 16K + 1K to the context limit at its 5/4, the 2 GiB hot prefix cache, and the read pool
  (`readPoolBytes`: 117,950,464 B on sc_3.10, the kit's pool; the evidence's own, in the bases, 117,917,696 B). The
  context limit is the load settings' `ctx_size` (`--ctx-size`); a load without one is refused by name
  (`Glm53fBillFitContext`). The bases are sc_3.10's, so another pack is refused (`Glm53fBillFitPack`). When the bank
  opens, the plugin reads the box's non-file memory outside the server; any part over the fit's 14,233 MiB comes off
  both phases, and the load line names it. At ai-server's ceiling and 32,768 tokens: 167 prompt / 204 decode rows with
  `LAYER_BILL` and `DECODE_ROOM`, at any available memory (today: 155/200 to 168/213).
- `TURBOEXL3_GLM53F_FIT_INUSE=1` (with `BILL_FIT`; alone it is refused by name, `Glm53fFitInUseAlone`): the fit's
  allowance for the box's memory outside the server is what is in use there when the bank opens, plus a 1 GiB margin,
  in place of the fixed 14,233 MiB. In use is the non-file memory less its purgeable and reusable pages (freed malloc
  pages the kernel takes back on demand; src/box_memory.zig `inUseNonFileBytes`, the A/B harness's serve gate
  measure), less the server's footprint. The margin is about three times what the guard's account grew past the
  window's start and the server's footprint in the control's windows (0.22-0.36 GiB). Nothing is over that allowance,
  so each phase takes the fixed allowance (with any excess over it) less the new one: at the fit's evidence plan, 10,
  12 and 13 GiB in use give 174/212, 169/207 and 166/204 rows against 167/205 (src/stream_room.zig). The plan then
  holds the box's in-use non-file memory at the ceiling; the guard's raw account also counts the purgeable and
  reusable pages outside the server (about 5.5 GiB on ai-server today), so a window that runs it needs that much more
  guard ceiling. Only the rows move.
- `TURBOEXL3_GLM53F_TRUNK_F16_FREE=1`: the trunk's plain f16 matrices let go of their f16 sources once their bf16 copies
  are evaluated (src/f16_free.zig, patches/turboexl3-model.patch: trunk.zig's `Loader.lin`). The
  pack stores a KDA layer's b, f_a, g_a, f_b and g_b, a DSA layer's embed_q and unembed_out (its kv_b,
  split) and the indexer's wk and weights_proj in f16, and the loader serves each as a bf16 copy. Without the switch the
  model also keeps a reference on each f16 matrix it cast, so about 0.65 GB of f16 sources stay allocated for as long
  as the model is loaded (the shards are read into MLX buffers, not mapped). With the switch the loader hands each
  source to src/f16_free.zig instead. After the trunk's load the plugin releases the pack's source map, frees the
  sources and clears MLX's cache before the bank opens. The load line counts the sources, their stored bytes and the
  bytes freed: MLX's active memory before the release less after it, so a source that something else still reads (a
  router's raw table) is not counted. The model reads the same bf16 copies, so the words do not change. With
  `BILL_FIT` the freed bytes come off both phases' bases (src/stream_room.zig `fitPlanFreed`), so both phases' rows take
  them under the same ceiling. The bases' evidence (g53fzr3,
  g53lb1) held these sources in both phases, so no new fit is measured. Without `BILL_FIT`, the memory available when
  the bank opens includes them. Its line starts with `TURBOEXL3_GLM53F_TRUNK_F16_FREE=1: `, and any value other than 0
  or 1 refuses the load by name (`Glm53fTrunkF16FreeSwitch`).
- `TURBOEXL3_GLM53F_SEED_DECODE=1`: the resident seed reads the pack's `expert-usage-decode-v1.json` (a decode-only
  usage record) in place of `expert-usage-v1.json` (src/streamed/glm5n_source.zig `seedDecision`, the kit's usage
  decision over that file). A pack without the file refuses the load by name before the pack is read
  (`Glm53fSeedDecodeMissing`). The load line names the file, its sha256's first 12 hex digits and its size. The seed
  only fills the empty prompt rows with the layer's most used experts before the first request, so only which records
  start resident moves.
- `TURBOEXL3_GLM53F_KEEP_WARM=1`: the keep-warm engine on the expert stream's waits (src/streamed/mimo/gpu_warm.zig at
  `waits`, MiMo's X12). While a gated wait (a decode miss part, a prompt group's
  reads) is below the stream's event, or the host waits on a prompt read-ahead, a parked thread commits a one-simdgroup
  FMA chain in ~0.5 ms chunks on a command queue of its own, so the GPU clock is not the one an idle gap leaves behind;
  it stops committing at the event (at most the two chunks in flight run on) and gives up on an event standing still for
  200 ms. The queue, kernel and 32 KiB buffer come from this plugin's own Metal shim (src/csrc/glm53f_gpu_warm.m: the
  server's csrc/mimo_gpu_warm.inc, which the host table does not give a plugin, over Metal's own API, its symbols hidden
  in the dylib, which links Metal and Foundation for it). No model array, graph, MLX queue or event value is touched
  (the event is only read), so the served words do not move; it costs GPU power in the waits and its 1 MiB bill term
  (the slot rows are sized from that much less). It does not run in host gaps without a pending wait. `zig build test`
  runs the engine's own CPU proofs on its trace backend (src/keep_warm_test.zig).
- `TURBOEXL3_GLM53F_HEAD_LAST_ROW=1`: the MTP head's merged first draft step (the engine's one forward over the
  committed rows' history and the draft row) runs its routed experts for the last row alone
  (patches/turboexl3-model.patch and patches/turboexl3-sdk.patch: plugins/glm5_next/head.zig `step`, glm5_next/moe.zig
  `forwardLastRow`, sdk/bases/blocks/moe.zig `mlpLastRow`). Only the last row's output is read; the history rows need
  their cache entries alone, which the attention appends as before. The routing and the shared expert still run over
  every row (the same launches over the same rows), the resident bank's experts (turboexl3's `moeClamped`) over the
  last row, then the same add and final norm: the last row's bytes are the whole call's, since turboexl3's decode
  kernels sum a row in an order that never depends on how many rows share the call (src/draft_gpu_test.zig holds
  it). The other rows' experts are not read: up to 8 K4 records (12.6 MB each, resident) a history row; the first step
  ran ~1.47 ms of GPU against the one-row step's ~0.80 ms in the D0 decode profile. A history append (no output
  wanted) runs as before. A block that takes another arm (no resident EXL3 bank, a gated shared expert) and a step over
  more than turboexl3's 16 decode rows (the whole call would take its prefill chain, whose bytes are not the decode
  form's) run whole. `TURBOEXL3_GLM53F_HEAD_LAST_ROW: <n> merged draft steps ran their routed experts for the last row
  alone, <m> ran whole (declined)` at each request's end.

## The split prompt forward

The split prompt forward runs a prompt's last two forwards as one, so each routed layer reads its expert records once
for both. With SSM checkpointing on, mlx-serve's prompt loop holds back the prompt's last 30 tokens. The loop's snapshot
of the recurrent state, taken at its end, then lands where the next turn's prefix match can reach it. A final forward
runs those 30 tokens and the last token. A 16K prompt is thus the loop's last chunk of 16,356 rows (span A), then 31
rows (span B). Span B's routed layers read again records that span A's layers read and the stream has since evicted.

A host whose plugin ABI has `Forward.split_rows` and `Forward.split_states` can send both spans as one prefill forward.
It sends one only to a model that declares `Caps.split_prefill`: mlx-serve with the split prompt forward (on by default
where a plugin declares the cap, `MLX_SERVE_PREFILL_SPLIT=0` turns it off), which no release has yet. `split_rows` is span A's row count. `split_states` is a second
cache from the host. The host takes its end-of-loop snapshot from it, because span B replaces span A's states in the
slot's own cache.

At each layer the forward runs these steps:

1. Span A's attention sublayer.
2. At a KDA layer, a copy of span A's recurrent states (`conv`, `ssm`) into `split_states`. A DSA layer keeps no
   recurrent state.
3. Span B's attention sublayer, which reads span A's states and KV rows in the slot's cache, then replaces the states.
4. Both spans' MLP sublayers: at a MoE layer, one routed call over both spans' rows; at a dense layer, one span at a
   time.

The logits are span B's, as a forward of span B alone returns them. When the host asks for hidden rows, it gets span A's
rows, then span B's.

The code makes each span compute what its own forward computes, for these reasons:

- Each op except the routed call reads only its span's rows and the caches. These ops are the embedding, the
  hyper-connections, the norms, the attention, the routers, the shared experts, the dense MLPs and the head. Each span
  routes alone, because a router's output can change with its row count.
- Each span takes the arms (the code paths an op picks, mostly by row count) that its own forward takes. The plugin
  refuses a span short enough to take another arm (below).
- Each layer keeps its own KV and recurrent state (`CacheOps`). Span A, then span B, at each layer leaves the caches as
  span A's forward and then span B's forward leave them.
- The routed call takes the bank's wide prompt lane, as each span's own call does. The combine sums each row's routes in
  router order. The call puts rows in other turboexl3 calls and tiles than each span's own call would. It also changes
  residency (which records the stream keeps) and reads. turboexl3 states that a routed row's bytes depend on neither the
  call's other rows nor the row's tile. The plugin's served prompt path already relies on that.

turboexl3's chunk test (`fused_test.zig` on its glm53f-fused branch) holds calls of 2,048 to 16,384 rows to 4,096-row
calls byte for byte. In it, the rows near each chunk edge change tiles. In span B's own calls, every row is in a
register-path tile of 32 rows or fewer. No test moves such a row into a 64-row tile of a large call.

The plugin refuses a split forward by name, with a `log.err` line, before it touches a cache. The host then runs the two
spans as ordinary forwards. The plugin refuses:

- a forward that is not a prompt (`Glm53fSplitMode`), or one of more than one slot (`Glm53fSplitSlots`);
- DFlash capture layers (`Glm53fSplitCapture`), media rows (`Glm53fSplitMedia`) or no `split_states`
  (`Glm53fSplitStates`);
- an empty span B, host tokens other than the forward's rows, or a span B past the last position (`Glm53fSplitRows`);
- a span of 16 rows or fewer (`Glm53fSplitShortSpan`), where a sublayer runs `hcPre`'s one-dispatch hyper-connection,
  which the split's MLP step lacks;
- a MoE layer off the streamed bank's `.source` arm, or a span its `fused` arm would take (`Glm53fSplitRoutedPath`). No
  layer of this plugin trips these guards.

The plugin's own forward calls `host.splitRefusal` before it marks `[stream-stats]`, so a refused split does not reach
it. Checks inside the forward (`Glm53fSplitStep`, `Glm53fSplitNoState`, `Glm53fSplitShortSpan` and
`MoeSplitShape`, `MoeSplitArm`, `MoeSplitSpans`) fail it (`failed`). The refusals above make them unreachable.

Built against the pinned kit (`KIT_VERSION` 26.11.1), which has neither field, the plugin declares nothing and runs no
split. Built against a kit from an mlx-serve with the fields, it declares `Caps.split_prefill`. src/split_test.zig
checks the refusals, the spans and the fields' size gate without MLX. The standard tests under Measured ran on mlx-serve
9a173773 with the split prompt forward PR and gave the frozen configuration's text.

## What is patched, and why

The edits to the copies are two patches, one an area of the generated tree, each its diff against the copied files,
so no patch depends on another: patches/turboexl3-model.patch (plugins/glm5_next/: the GLM-5.3-Flash model) and
patches/turboexl3-sdk.patch (sdk/bases/, sdk/blocks/, sdk/util/ and sdk/weights/: the shared blocks, the weight loader,
`Routed` and the streamed weights' GLM wiring). The GLM bank (src/streamed/, src/exl3/) is this plugin's own source; its
changes since bc6c5f2c are in the table and the notes below, at their new paths. The EXL3 arm's edits:

| file | what |
|---|---|
| plugins/glm5_next/trunk.zig | an EXL3 linear (`.trellis/.suh/.svh`) loads as a `Linear` with `x3` set; `Model.mm` runs it on turboexl3's `linear`; KDA's one `qkv_proj` beside the joined conv; the EXL3 `lm_head` and MTP `eh_proj`; the MTP layer's experts left to the plugin's bank; no resident Sushi bank (`expert_quant`: turboexl3's Spec has no codebook table); the plain f16 matrices cast to bf16 at load |
| sdk/bases/blocks/linear.zig | `Linear.x3`, and its arm in `matmul` (the shared expert and the gate logit go through here) |
| sdk/bases/blocks/moe.zig | `Moe.Weights.x3`: a resident EXL3 bank run by turboexl3's `moeClamped` (the MTP layer's experts) |
| sdk/weights/load.zig | an EXL3 linear's `suh` and `svh` stay f16 (the engine narrows every 1-D f16 table to bf16) |
| sdk/util/shared.zig | the copies' globals export under the plugin's own prefix |
| src/streamed/expert/bank.zig | a v2 layer's K read as a number: whole, or exllamav3's half-integer rate (`n` halfwords a tile), per layer (turboexl3_ds41's change, without its a4k layout) |
| src/streamed/glm5n/exl3_bank.zig | the bank admits K 2, 3, 4, 5 and the half-integer widths 40 and 56, one K a layer; the geometry is the widest layer's; two code widths keep separate transient scratches |
| src/streamed/glm5n/experts.zig | the math is src/routed_t3.zig's; the prebuild also builds the second code width; the stream's options (`servedStreamOptions`) turn prompt lever 1's relocation off on a bank of two code widths, which the kit's stream refuses it on (InvalidOptions); `seams` and `missingSeam`, the switches whose mechanism is a seam of the kit's stream; `checkStreamOptions`, the construction's checks (the reader's staging, the direct reads' record ranges) without slot memory or GPU; `servedLookahead`, the read pool the kit's stream starts (`expert.io.poolOptions`, `poolBytes`); `speculateAhead`, READ_AHEAD3's read-ahead over the kit's stream |
| src/streamed/glm5n/settings.zig | the read pool's staging 16 MiB: a K5 record's gate/up span is 10.5 MB |
| src/streamed/glm5n_source.zig | the keep-warm queue only with `TURBOEXL3_GLM53F_KEEP_WARM` (its Metal shim this plugin's own); `checkStreamOptions` over a pack's bank and `checkStreamOptionsN` over an in-memory bank of given tile widths, `slotArrayRows` (a slot array's rows under the stream's page floor) and `planRows` (the per-layer rows), for the MLX-free tests; `seedDecision` over the kit's usage parse (`decideFile`) |
| src/streamed/sdk_ext.zig | `expert` is the kit's stream (`kit.streamed.expert`); its `bank_file` is this plugin's (src/streamed/expert/bank_file.zig, turboexl3_ds41's) |

The built-in serves one GLM EXL3 pack: a K3 bank on its own GLM texts with FP8 residents. The SC packs store every
linear in EXL3 and route a layer at K 2 to 5 (layer 3 one step above the rest), so the edits move each EXL3 tensor to
turboexl3 and admit per-layer K.

dsa-head (patches/turboexl3-model.patch): dsa_nax.zig gains `sparseLatentOneP` (the sparse kernel without MSV_DSA_TWO_P,
its own kernel and engaged
line) and table_forward.zig's `dsaAttn` takes `dsa_prompt.sparseOneP` for a prompt pass's rows
(`TURBOEXL3_GLM53F_DSA_ONEP`); the stock path runs wherever it returns null.

grow-unfilled (src/streamed/glm5n/experts.zig): `servedStreamOptions` sets the kit stream's
`grow_fill` to `.unfilled` while `grow_unfilled` is on (`TURBOEXL3_GLM53F_GROW_UNFILLED`, set by the plugin before the
bank opens through `setGrowUnfilled`), and `servedFill` / `servedFillN` read the served options back for the MLX-free
test.

kda-prompt (patches/turboexl3-model.patch): kda.zig imports src/kda_prompt.zig; in a prompt pass's forward (a prefill
without a verify's capture) the
op-chain prologue takes `kda_prompt.prologue` first (where the fused prework did not run), and the gated norm
`kda_prompt.normGate` before the rms_norm, sigmoid and multiply (where the fused prework's norm-gate did not run); the
chain runs where they return null (the switch off, or
a shape or dtype the text does not take). Where `kda_prompt.percoreServes` (TURBOEXL3_GLM53F_KDA_PERCORE installed for
the layer's heads), the prompt pass skips the gate and runs `kda_prompt.percoreRecur` in place of the recurrence.

prompt-keep (a seam of the kit's stream, `seams.prompt_keep`): a stream `prompt_keep` option that sets each layer
policy's, under which a prefill miss may not evict (`mayEvictInPrefill`, so the seed marks none) and the read-ahead's
admission and a seed promotion take empty rows only; `servedStreamOptions` sets it while `prompt_keep` is on
(`TURBOEXL3_GLM53F_PROMPT_KEEP`, through `setPromptKeep`) where the kit's stream has the option. 26.11.3's has not.

prompt-stage (src/streamed/experts.zig): the scheduler's wide prompt lane takes `Wide.stage` (`runWideCoreAt`): a prompt
call of at least 8 rows per expert routes its layer's misses (the stream's `stageMisses`, the experts no persistent row
holds, in expert order) into the transient windows before its routing barrier, with every resident held (the kit
stream's `holdResidents`, released after), then runs those groups first and its other experts hottest first; glm5n's
`stage_misses` sets it (`TURBOEXL3_GLM53F_PF_STAGE`, through `setStageMisses`).

prompt-drain-lag (src/streamed/experts.zig): the scheduler's wide prompt lane takes `Wide.drain_lag`
(`runWideCoreAt`, with the feed): a group's one drain, its call's release and the route `depth` groups on run once
the next group's first chunk is committed, in place of before the next group is encoded. The math's `LagPending`
(src/routed_t3.zig `LagPendingOf`) holds the pending group; the scheduler arms its hook before a group's banks,
`prefillParts` runs it after its first commit, and a group that commits nothing runs it after its banks. glm5n's
`drain_lag` sets it (`TURBOEXL3_GLM53F_PF_DRAIN_LAG`, through `setDrainLag`).

seed-decode (src/streamed/glm5n_source.zig): `decideFile` is the kit's `expert.usage.decide` over any file name in the
pack (the kit's `parse` reads it); `seedDecision` is the decision `Source.open` makes, over `seedFile()`:
`expert-usage-decode-v1.json`, the seed named, while `seed_decode` is on (`TURBOEXL3_GLM53F_SEED_DECODE`, through
`setSeedDecode`), else `expert-usage-v1.json` as before.

split (patches/turboexl3-model.patch and patches/turboexl3-sdk.patch): the split prompt forward (above). In
plugins/glm5_next/, host.zig reads the two fields only from a kit and a host whose `Forward` has them (`splitOf`). Its
`forwardOne` hands a split forward to `forwardSplit`, which refuses by name (`splitRefusal`) or runs the two spans
(`splitOne`, `finishSplit`). table_forward.zig has the checks (`splitSpans`, `splitServes`) and the layers
(`hiddenSplit`, `mlpSublayerSplit`). slots.zig copies a state to another cache (`Slots.copyState`) and names the states
a prompt leaves (`prompt_state_slots`). moe.zig runs a layer's MoE block over both spans (`forwardSplit`). The MoE
block, sdk/bases/blocks/moe.zig, has `splitServes` and `mlpSplit`. `hidden`, `sublayer` and `mlp` are unchanged.

demand-split (a seam of the kit's stream and its read pool, `seams.demand_split`; 26.11.3's has not): a read pool
with `q3ld_submit_split`, `q3ld_submit` of one record whose gate/up range is
queued as up to `pieces` page-aligned pieces, a job each; the range's ticket is published by its last piece, its
words the pieces' added up. It splits only on the direct route, for a range no pre-range or live speculative record
holds, while the pieces fit the queue. A stream `demand_split` option makes the lone-record submit call it for a decode
call that reads one record; `servedStreamOptions` sets it to `demandSplitPieces(io_workers)` while `demand_split` is
on (`TURBOEXL3_GLM53F_DEMAND_SPLIT`, through `setDemandSplit`) where the kit's stream has the option.

hc-fuse (patches/turboexl3-model.patch): a
prompt sublayer's expand and the next sublayer's collapse as one text where `hc_fuse.expandCollapse`
(TURBOEXL3_GLM53F_HC_FUSE, below) serves. `sublayer` takes the collapse made ahead (`ahead`) in place of its own and,
at its end, asks the text for the expand and the collapse of the next sublayer's hyper-connection (`nextHc`: the
layer's MLP's after its attention, the next layer's attention's after its MLP, none after the last); the split
forward's `mlpSublayerSplit` does the same for both spans. Where the text returns null (the switch off, a span under
its rows, the last layer) the copy's collapse and expand run as before. A one-token forward drops a collapse made
ahead (none is: the text serves prompt rows only).
shared-hoist (patches/turboexl3-sdk.patch and src/streamed/): routed.zig's `Routed`
has `source_hoist`, the `source` math with a hoist. moe.zig's `mlp` builds an ungated shared expert before the routed
call when the streamed source takes a hoist and the call has 8 rows or fewer (`hoistsShared`, `hoist_max_rows`), hands
it through `routedExperts` and `streamedRouted` to `source_hoist`, and adds it after the call in place of building it
there. glm5n_source.zig's `Source.rowsHoist` passes it to the decode lane's forward (`decodeLaneAt`; its first such
call prints the engaged line), whose `runEnd` commits it behind the hit wave only when the call has miss parts
(experts.zig `post_hit`); streamed.zig's `glm5nRouted` sets `source_hoist` while `shared_hoist` is on
(`TURBOEXL3_GLM53F_SHARED_HOIST`, through `setSharedHoist`).

read-ahead3 (patches/turboexl3-sdk.patch, patches/turboexl3-model.patch and src/streamed/): routed.zig's `Routed`
has `ahead_score_fn` and `ahead_issue_fn`. table_forward.zig's `branch` calls the first on the last dense layer's MLP
input in a window of 8 rows or fewer, and the second after the first routed layer's attention, once that is committed
(`commitArrays`). glm5n_source.zig's `Source.aheadScore` scores the first routed layer's router on that input
(experts.zig `firstScores`, the lookahead's `scoresOf`) and commits the scores; `Source.aheadIssue` reads them back and
calls `speculateAhead` (src/streamed/glm5n/experts.zig: the kit stream's `speculate` step for routed layer 0 with no
route of its own, over the stream's selector and read pool, its records tagged for the coming route; the stream's
precision counts, its private `notePending`, are not fed). streamed.zig's `glm5nRouted` sets
both while `read_ahead3` is on (`TURBOEXL3_GLM53F_READ_AHEAD3`, through `setReadAhead3`).

demand-help: `TURBOEXL3_GLM53F_DEMAND_HELP` sets `experts.demand_first` (through `setDemandFirst`); `beginDecodeTimed`
calls the kit stream's `demandFirst(demand_first)` before the decode grow, and `resumePrefill` calls `demandFirst(0)`
before the prompt's shrink (turboexl3_ds41's `setPhase` does the same). The mechanism is the kit read pool's df1
(`q3ld_demand_first`): no seam.

kda-fold (patches/turboexl3-model.patch): in a prompt
pass on the chunkwise route, `kda_prompt.foldWindow` (TURBOEXL3_GLM53F_KDA_FOLD, below) runs before the prologue text:
where it serves it keeps the conv window as the text keeps it and returns the window before the forward
(`fold_before`), the text and the chain do not run, and the recurrence takes `kda_prompt.foldRecur` (the chunkwise
launch over qkv and that window) before the per-core hooks. Where it returns null (the switch off, the chunkwise route
not installed for the layer's heads, a shape or dtype the text does not take) the prologue runs as before.

async-eval (patches/turboexl3-model.patch): `Slots.evalWith` takes `async_eval` and runs `mlx_async_eval` over the same
arrays with it, else `mlx_eval`;
a prompt forward's two cadence evals (the forward's and the split forward's) pass `async_eval.take()`
(TURBOEXL3_GLM53F_ASYNC_EVAL, above): false with the switch off.

keep-warm (src/streamed/glm5n_source.zig): the bank's keep-warm route is `waits` where `keep_warm` is set
(`TURBOEXL3_GLM53F_KEEP_WARM`, through
`setKeepWarm`, above) and `off` otherwise, where the EXL3 arm's edit (the table above) had it off for good;
`keep_warm_bill_bytes` gives the
plugin the engine's bill term.

adopt-pieces (a seam of the kit's stream and its read pool, `seams.adopt_pieces`; 26.11.3's has not): a read pool
with `q3ld_adopt_pieces`, set on a stopped pool (reset at stop). With 2 or
more, `try_adopt` posts an adopted range's copy as a task of pieces (`cp_post`: at least 1 MiB each, 64 KiB aligned; a
task slot a worker, else the one memcpy), the worker loop takes a piece of a live task when no job is queued, ahead of
the pre-ranges (`cp_pick`, `cp_run`), and the adopting worker copies the pieces left and waits for the ones in flight
(`cp_own`) before it drops the record and publishes the range. A stream `Lookahead.adopt_pieces` passes it to the
pool at `Pool.start`, and `servedStreamOptions` sets it to the pool's workers while `adopt_pieces` is on
(`TURBOEXL3_GLM53F_ADOPT_PIECES`, through `setAdoptPieces`) where the kit's stream has it.

pf-reuse (patches/turboexl3-model.patch): `hidden`
and `hiddenSplit` call `pf_reuse.begin` at a forward's start (a prompt forward's rows, both spans' in a split forward),
`pf_reuse.end` in a defer, and `pf_reuse.layer` before each layer's attention (this layer KDA, the forward's next layer
KDA) (TURBOEXL3_GLM53F_PF_REUSE, below): each a no-op with the switch off.

## The EXL3 arm

- turboexl3 (pinned in build.zig.zon at 1db354d, a fast-forward of its main c7ce66c) runs every trellis linear of the trunk, the LM head and the
  MTP layer: KDA `qkv_proj` and `o_proj`, DSA `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `o_proj`, the indexer's
  `wq_b`, the shared experts, the dense layers' MLP, `eh_proj`. Its tests at every one of these shapes and rates:
  turboexl3's `src/models/glm53f/`.
- The routed experts (`experts.bin`, model layers 3-44): on the streamer's schedule (its decode waves, prompt waves,
  lookahead and read-ahead), every call on turboexl3 (src/routed_t3.zig):
  - decode waves: `moeClampedBanked`, a wave's pairs over the layer's three slot banks in one call, each pair
    `moeClamped`'s bytes at top-1 on its bank;
  - a prompt call's rows and a cold-row chunk: sorted by slot, `moeClamped` at top-1 and weight 1 over the bank's slot
    arrays, at most 4,096 rows a call, a call of 16 rows or fewer padded to 17 so every row takes the prefill GEMM.
  - the clamp is `swiglu_limit` 10. Each row comes back bf16 at weight 1; the streamer joins and weighs.
- The MTP layer's experts (`mtp-experts.bin`, model layer 45): one resident turboexl3 bank, `moeClamped` in the head.
- The engine's MTP readout reads the trunk head through `trunk_weight`, as a dense or affine weight: the plugin gives it
  the EXL3 head decoded once at load (its identity rows through turboexl3's `linear`) and quantized to 8-bit affine,
  group 64. Drafts only; the served logits are turboexl3's head.
- Not served: the vision tower (`model.visual.*` stays on disk; no images are advertised).

## Fused paths

Each is on by default (unset or 1; 0 turns it off; any other value refuses the load by name), read once when the
streamed bank opens, and prints its line at load. Each runs a turboexl3 entry that gives the default path's bytes,
held so by turboexl3's own `src/models/glm53f/fused_test.zig` at GLM-5.3-Flash's routed geometry and every SC rate
(they need a turboexl3 with those entries: `--fork=<turboexl3 checkout>` until the pin moves). The standard test decides
whether one is kept.

| switch | what changes | what it saves |
|---|---|---|
| `TURBOEXL3_GLM53F_PF_ROWS=1` | a prompt call and a cold-row chunk run turboexl3's `moeClampedRows`: the chain's prepare reads each row-slot's x row through the call's row map, and its finish stores bf16 | per call, the gather of its rows (`take`), their f32 copy and the cast of the f32 y to bf16: four launches and about 80 KB a routed pair of memory traffic (a 16K prompt routes 131,096 pairs a layer) |
| `TURBOEXL3_GLM53F_DEC_BF16=1` | a decode wave runs turboexl3's `moeClampedBankedAs` at bf16: the banked down finish stores bf16 | the cast of the wave's f32 y: one launch a routed layer a forward |

## Schedules

The same kernels' arithmetic on another schedule, so the same bytes; each on by default (unset: its value in the table below; 0 the default schedule) and refused by name on a value
it does not take, with its line at load. turboexl3's `src/models/glm53f/fused_test.zig` holds each schedule's rows to
the default's at GLM-5.3-Flash's routed geometry and every SC rate.

| switch | what changes | what it saves |
|---|---|---|
| `TURBOEXL3_GLM53F_PF_FEED=1` | a prompt call returns with its last chunks in flight; the scheduler evaluates a group's outputs once before its slots go back, and that one wait covers every chunk | the wait at the end of each prompt call of a group but its last (a group's base, ext and transient calls run back to back) |
| `TURBOEXL3_GLM53F_PF_STAGE=1` | a prompt call of at least 8 rows per expert routes its layer's misses before its routing barrier, as many as the transient windows hold (96), with every resident held so no staged miss evicts one (src/streamed/experts.zig: the scheduler's `Wide.stage`); the call runs those groups first, then its other experts hottest first, every group on the stream's stock route | the misses' reads run while the barrier's evaluation runs the layer's attention, when the drive is otherwise idle, in place of after it |
| `TURBOEXL3_GLM53F_PF_DRAIN_LAG=1` | with `PF_FEED`: a prompt MoE group's one drain, its slots' release and the next route run after the next group's first chunk is committed (or its banks, when it commits none), not before the next group is encoded (src/streamed/experts.zig: the scheduler's `Wide.drain_lag`; src/routed_t3.zig `LagPendingOf`); the same kernels over the same rows in the same order, each group's slots back after its own drain | the GPU's idle at each group boundary: it runs the group's last chunks while the host builds the next group's first (about six groups of 48 experts a layer at 16K) |
| `TURBOEXL3_GLM53F_PF_ALIGN=1` | a prompt call's chunks end where no expert's 128-row run is split (routed_t3 `alignedEnd`): at an expert boundary, or a multiple of 128 rows into the expert crossing the chunk's rows at most; every chunk but the call's last holds between that many less 127 rows and that many, so no array grows. turboexl3's `src/models/glm53f/chain_time_test.zig` holds the aligned chunks to fixed 4096-row ones byte for byte | each expert takes its one-call tiles: a fixed cut through an expert's run adds a part-full 128-row tile or a mixed-launch remainder on each side of it |
| `TURBOEXL3_GLM53F_BANKED=g4m3.t4.t4` | the banked decode's schedule through `t3.configure`, set before any of its kernels is built: turboexl3's grouped texts (`banked_group` 4) with `banked_m3`, the grouped gate&#124;up and down GEMVs at turboexl3's variant `t4` | a weight's read and decode shared by up to four row-slots of one expert (two by default): a verify's rows often route the same expert |
| `TURBOEXL3_GLM53F_PF_OPTS=bk128` | the routed prefill's options through `t3.configure`, set before any prefill engine is made: `bk128` (turboexl3's `Config.pf_bk128`: the down GEMM's K in 128-deep stages, two 64-deep sub-stages a buffer between barriers) | half the down GEMM's barriers and stage-loop turns |
| `TURBOEXL3_GLM53F_PG2=<codes>` | the trunk's linears at prefill rows on turboexl3's pg2: one tile code a linear of `t3.glm53f.trunk` (11, in order: `0` the chain, `1`-`9` then `a`- turboexl3's `pgemm.candidates`), installed as turboexl3's host table (`pgemm.host_table`) with pg2 on, its kernels compiled at load; the codes come from turboexl3's timing on this machine (`TURBOEXL3_GLM_PG2_CODES`), and its `fused_test.zig` holds every candidate at every trunk linear to the chain's words | the matmul and the finish in one launch (no f16 z written and read back) at a tile fitted to this machine, where MLX runs its regular kernel |
| `TURBOEXL3_GLM53F_DEC=g1` | the dense decode GEMV (turboexl3's DEC-ONE: every trunk linear, the head and the MTP layer's linears at decode rows) through `t3.configure`, set before any dense engine is made: turboexl3's variant `g1`, GLM-5.3-Flash's own GEMV at its trunk shapes, each lane's tile word offsets taken once and the tile and input pointers stepped a constant (any other shape keeps today's); turboexl3's `src/models/glm53f/dense_dec_test.zig` holds it to the served configuration's bytes at each sc_3.10 dense shape and decode rows 1 to 4, and times it over one forward's trunk at a verify's rows | the trunk's 5.9 GB a verify streams at a higher share of the memory's bandwidth, where a lane's tile loads or its decode instructions hold the GEMV back |

## Opt-in int8 routed prefill (not exact)

`TURBOEXL3_GLM53F_PF_INT8=1` or `2` runs the routed prefill's gate|up and down GEMMs on int8 (routed_t3 `PfInt8`:
turboexl3's `Config.pf_int8` through `t3.configure`, set before any prefill engine is made). The prepare stores each
activation row as int8 in 128-wide blocks, mid stores the down GEMM's input the same way, and each GEMM decodes its
weights' trellis to int8 in its stage and multiplies int8 by int8 into int32 sums, the 64-row table's tiles and the
128-row tiles alike. `1` gives each activation 128-block its own scale (each block's sums folded into f32 at it); `2`
gives a row one scale (one fold). The route and the finish are the exact chain's. It is not exact: its rows agree with
the exact chain's to the int8 rounding, not bit for bit. turboexl3 takes it beside the 128-row tiles on the deferred
MUL1 map (the served options), with or without `TURBOEXL3_GLM53F_PF_REUSE`'s size classes, and declines it elsewhere
(the exact chain runs). turboexl3's `src/models/glm53f/fused_test.zig` holds both schemes, with the size classes, to
the exact chain on the device at every SC rate, on the gathered and the row-map prompt paths
(`TURBOEXL3_GLM_PF_INT8_AGREE`: cosine at least 0.99, the difference's rms at most 0.1 of the chain's, no nonfinite
value), and times them against it (`TURBOEXL3_GLM_PF_INT8_TIME` and `_STAGE`). Unset or `0` the exact chain; any other
value refuses the load by name (`Glm53fPfInt8Switch`). It prints its line at load, and turboexl3 prints its own engaged
line at the first int8 engine. With `TURBOEXL3_GLM53F_PF_MOE_HOLD` the hold takes the int8 chain's arrays (below).

`TURBOEXL3_GLM53F_PF_INT8=3` is `1` with the trunk's KDA qkv (4096 x 24576) and o_proj (8192 x 4096) at prefill rows
on int8 too (turboexl3's `Config.pf_trunk_int8` at routed_t3's `trunk_int8_shapes`, its `trunk_q8.zig`): in
`Dense.linearOn`'s matmul path the prepare stores each row as int8 at one scale (its amax / 127 over text_prep's f16
values), the trellis is decoded to int8 once a call (round((t - 510) / 4) of each MUL1 codeword's byte sum, one scale
for every weight), and one int8 GEMM (mpp matmul2d, int32 sums) runs the routed int8's affine epilogue and pg2's finish.
`TURBOEXL3_GLM53F_KDA_PIPE`'s spans decode the int8 weights once a forward and run each span's prepare and GEMM (a
row's y depends on its own row alone, so the spans give the unspanned forward's bytes; src/kda_gpu_test.zig holds
them to it). turboexl3's `fused_test.zig` holds every trunk linear on int8 to the chain at 100 to 8200 rows
(`TURBOEXL3_GLM_TRUNK_INT8_AGREE`: cosine at least 0.99, rms at most 0.1, finite), the switch off to the chain word for
word (`_OFF`) and a call's rows split over two calls to the one call (`_SPLIT`), and times each against the linear as
the control serves it (`_TIME`, `_STAGE`). The plugin prints a second line at load, and turboexl3 its trunk int8 engaged
line at the first int8 call. The routed hold is the int8 chain's, as under `1`.

`TURBOEXL3_GLM53F_PF_INT8=4` is `3` at every trunk linear of `t3.glm53f.trunk` (routed_t3's `trunk_int8_all`, 11
shapes): the KDA qkv and o_proj, the DSA layers' q_a, q_b, kv_a and o_proj, the indexer's wq_b, the shared experts'
gate|up and down and the dense layers' gate|up and down, each through `Dense.linearOn`'s matmul path at prefill rows
(decode and verify rows keep their GEMVs). Its second load line counts the linears.

`TURBOEXL3_GLM53F_LOGITS_DUMP=<absolute directory>` is a check, never served: each main-model forward also asks for its
final-normed hidden, projects it through the LM head 256 rows at a time after the forward and writes each row's top-64
log-probabilities, with its input id and position, to `<dir>/logits-<pid>.bin` (src/logits_dump.zig has the record's
layout). The logits and hidden the engine asked for are the forward's own, so the served bytes are unchanged (a probe
holds a dump-on server's text and end-of-prompt snapshot to a dump-off one's); it costs the LM head again over every
row and a wait a forward. A forward of more than one slot is counted, not written. The prefill bill adds the dump's head
blocks while it is set. A relative directory, or one the file does not open in, refuses the load by name.

## KDA prompt texts

A KDA layer's prompt pass runs kda.zig's op chain on ai-server: the copy's per-core recurrence plans one threadgroup a
GPU core at 128 value rows at most, and GLM-5.3-Flash's 64 x 128 value rows need 64 cores where an M5 Max has 40, so the
chain's materialized f32 gate, its prologue ops, the stock per-token recurrence and the gated norm's three kernels run.
Each switch below runs one piece of that chain as one MLX custom kernel (src/kda_prompt.zig, src/kernels/kda_*.metal and
the copy's norm-gate text; kda.zig through patches/turboexl3-model.patch), each step the kernel the chain runs (the
server's MLX 64ea011c's, the copy's own recurrence) or a table of its outputs, in the chain's operand order and types,
so the outputs are the chain's bits. src/kda_gpu_test.zig holds each to the chain on the device, bit for bit
(`zig build kda-gpu-test`: kda.zig's forward over GLM-5.3-Flash's KDA geometry, cold chunks of 17, 995 and 16,356 rows
each with a warm 31-row chunk after it, every output, conv window and state, and each text's launches counted). Each
runs only in a prompt pass's forward (not a verify's) and only on the shapes and dtypes it is written for, else the
chain runs. Each is on by default (unset or 1; 0 turns it off), refuses any other value by name, and prints its line at load and an
engaged line at its first launch. The standard test decides whether one is kept.

| switch | what changes | what it saves |
|---|---|---|
| `TURBOEXL3_GLM53F_KDA_PROLOGUE=1` | the prologue (`conv1d.run`'s concat, depthwise conv and SiLU, the q / k / v slices, the two rms_norms and their scales, v's copy) in one pass that reads qkv once; the conv window kept as the chain keeps it | about 11 GB of a KDA layer's memory traffic at 16K rows (1.6 GB left) and about ten launches |
| `TURBOEXL3_GLM53F_KDA_NORMGATE=1` | the gated norm after the recurrence (the rms_norm over each 128-wide value head, the sigmoid of the output gate, their product) in one pass that writes the out projection's input: the copy's norm-gate text (kernels/gdn/gdn_normgate.metal, which the copy's KDA layer runs only beside its per-core route), its sigmoid a table of MLX's own op over every 16-bit value | about 1.1 GB of a KDA layer's memory traffic at 16K rows (0.8 GB left), two launches and two 0.27 GB arrays |

`TURBOEXL3_GLM53F_KDA_PERCORE=1` is not exact: it runs a prompt pass's recurrence and its bounded gate on the copy's
per-core kernel (sdk/blocks/kda/recurrence.zig's text, kernels/kda/kda_percore.metal, oMLX's), the gate built inside
the kernel, at the 1086 tok/s cell's plan (mlx-stream cf30601's `planFor`): one threadgroup a GPU core where 128 rows a
threadgroup carry the value rows, else two (the 40-core M5 Max: GLM-5.3-Flash's 64 x 128 value rows as 80 threadgroups
x 104 rows), where the copy's own plan, one a core, declines. Its delta-rule folds are another summation order than the
stock kernel's, so its outputs are the chain's within rounding, not its bits: a prefill kernel judged on prefill (David,
2026-11-07). At load it plans for the GPU's cores and launches once over zeros (Metal caps the threadgroup by register
use); no plan or launch refuses the load by name (`Glm53fKdaPercoreCores`, `Glm53fKdaPercoreThreads`, ...). Its line
names the plan. It replaces the gate and the recurrence; `KDA_PROLOGUE` still feeds it. `zig build kda-gpu-test`
reports its largest differences from the chain
(`TURBOEXL3_GLM53F_KDA_DIFF`: max_abs, the chain's largest magnitude and their ratio), failing only on a nonfinite value
or a ratio past 1e-2.

`TURBOEXL3_GLM53F_KDA_CHUNK=1` is not exact either: it runs a prompt pass's recurrence and its bounded gate as the
chunkwise delta rule (fla's chunk_kda form, src/kernels/kda_chunk.metal) in 16-row chunks. A chunk's rows go through
one triangular solve, T = (I + diag(beta) A)^-1 diag(beta) with A the chunk's decayed key products, and the state is
read and written once a chunk in f32 simdgroup matrices, not once a row: the per-core kernel's 16,356 dependent steps of
a 16K prompt become 1,023 steps of 8x8 matrix work. Each decay factor is taken against the chunk's middle row, so at
GLM-5.3-Flash's gate bound of -5 none passes e^40 or e^-40 (no f32 overflow, and no flushed term that counts). A
threadgroup carries one value head's 64 value rows (8 simdgroups of 8; 128 threadgroups for 64 heads) and builds the
gate inside the kernel from the gate input, dt_bias and A_log by kdaGateChain's statements. It takes kda.zig's
per-core hooks (`percoreServes`, `percoreRecur`), so kda.zig is unchanged, and runs before the per-core route where both
switches are on. At load it checks the layout (key and value heads alike, 128 wide) and the bound (-5 to 0) and
launches once over zeros; any of them refuses the load by name (`Glm53fKdaChunkLayout`, `Glm53fKdaChunkBound`,
`Glm53fKdaChunkThreads`, ...). `zig build kda-gpu-test` reports its largest differences from the chain and from the
per-core route as served (`TURBOEXL3_GLM53F_KDA_DIFF ... vs=off` and `vs=percore_prologue`, the same 1e-2 bound), and
holds the norm-gate text and the per-core switch beside it to its bits.

`TURBOEXL3_GLM53F_KDA_FOLD=1` (exact, with `KDA_CHUNK=1`) folds the prologue into the chunkwise route's launch
(src/kernels/kda_fold.metal, patches/turboexl3-model.patch). Before a chunk's gate walk each simdgroup makes
two of the chunk's q rows and two of its k rows from qkv and the conv window before them by the prologue text's
statements (the conv over the taps, the SiLU, the rms over the 128-wide head by simd_sum over the same lanes, the
scale), and each lane makes its own elements of v the same way; each value is rounded to the activation dtype where the
text rounds it. So q, k and v (0.8 GB a KDA layer at 16K rows) are never written and read back, the prologue's launch
goes, and the outputs are `KDA_CHUNK`'s over `KDA_PROLOGUE`'s bit for bit (the conv window too: kept as the text keeps
it). Both threadgroups of a head make its q and k, and a chunk takes one more barrier. Without `KDA_CHUNK=1` the load is
refused by name (`Glm53fKdaFoldNeedsChunk`); at load it launches once over zeros at the layer's conv taps
(`Glm53fKdaFoldTaps`, `Glm53fKdaFoldThreads`). `zig build kda-gpu-test` holds it bit for bit to the chunkwise route with
the prologue text (`TURBOEXL3_GLM53F_KDA_BYTES ... vs=chunk_prologue`): alone, beside control step 14's prologue and
per-core switches, and before the norm-gate text; and counts its launches (`fold=` in `TURBOEXL3_GLM53F_KDA_ENGAGED`).
On by default (unset or 1; 0 turns it off), any other value refused by name (`Glm53fKdaFoldSwitch`).

`TURBOEXL3_GLM53F_KDA_PIPE=1` (exact, with `KDA_CHUNK=1`) runs a KDA layer's prompt forward of one slot, 8,192 rows or
more over an exllamav3 q|k|v linear, as 4,096-row spans (src/kda_pipe.zig,
patches/turboexl3-model.patch): a remainder under 2,048 rows joins the last span, and past 16 spans the
forward runs unspanned. Each span's q|k|v rows are turboexl3's `linear` on its matmul path (the prepare, the W_hat
decoded once a layer, pg2's fused launch or MLX's matmul, the finish), recorded on a second MLX stream on the model
stream's GPU, with its own command queue. Each span's prologue text or fold, then its chunkwise launch, runs on the
model's stream over the conv window and the f32 state the span before it left. The chunkwise launch is latency-bound, so
the next span's GEMM fills the GPU beside it; MLX orders the two streams with its fences. The z, a and b projections and
the prompt tail (the norm-gate text or the chain, then the out projection) run over every row, as kda.zig runs them.
Each value is the unspanned forward's bit for bit: the prepare and the finish are per row, MLX's GEMM gives a row the
same words at every row count from 2,048 to 16,384 (where pg2's probe matched it word for word), every span but the last
ends on a 16-row chunk boundary, and the state stays f32 between spans as it does in the launch. Without `KDA_CHUNK=1`,
or with neither `KDA_PROLOGUE=1` nor `KDA_FOLD=1`, the load is refused by name (`Glm53fKdaPipeNeedsChunk`,
`Glm53fKdaPipeNeedsPrologue`), and so is a model stream that is not a GPU's (`Glm53fKdaPipeStream`).
`zig build kda-gpu-test` holds the spanned forward bit for bit to the unspanned one
(`TURBOEXL3_GLM53F_KDA_BYTES ... set=pipe_<set>`) at 8,197 and 16,356 rows and in the warm 31-row chunk after them,
beside control step 14's prologue and per-core switches, on the chunkwise route and with the fold, each with and without
the norm-gate text, and counts the spans' launches (`TURBOEXL3_GLM53F_KDA_PIPE_ENGAGED`). On by default (unset or 1; 0 turns it off),
any other value refused by name (`Glm53fKdaPipeSwitch`).

`TURBOEXL3_GLM53F_HC_FUSE=1` (exact) runs a prompt sublayer's hyper-connection expand and the next sublayer's collapse
as one text (src/kernels/hc_fuse.metal, src/hc_fuse.zig, patches/turboexl3-model.patch), over each 8 rows:
the copy's expand, mixes and sinkhorn_collapse texts (plugins/glm5_next/kernels) in one dispatch, each phase their
statements in their order and types (the mixes' 8-row blocks and 8 simdgroups splitting K, the gates of each row on
its own simdgroup). At a 16K prompt a sublayer boundary (two a layer, 90 a prompt) moves the 0.54 GB bf16 stream five
times as three dispatches (the expand reads it and writes the new one, the mixes and the collapse each read the new
one); the text writes the new stream once and reads its own rows back from the cache. It serves at 512 rows or more,
where the copy's mixes is its text (`MIXES_KERNEL_MIN_ROWS`; below, MLX's rms_norm and matmul, whose sums it does not
reproduce), so a split's 31-row tail keeps the copy's three. `zig build kda-gpu-test` holds it bit for bit to the
copy's expand and collapse at 16,356, 995 and 512 rows (the new stream, the next input and its post and comb gates) and
checks that it declines 511 and 31 rows (`TURBOEXL3_GLM53F_HC_BYTES`, `TURBOEXL3_GLM53F_HC_DECLINED`,
`TURBOEXL3_GLM53F_HC_PARITY`). On by default (0 turns it off); it refuses any value but 0 and 1 by name, prints its line at load and an
engaged line at its first launch.

`TURBOEXL3_GLM53F_ASYNC_EVAL=1` (exact) hands a prompt forward's eval at the host's cadence (`Forward.eval_every`: the
stream and every conv window the slots hold, after every 4 layers and the last, the copy's `Slots.evalWith`) to the GPU
without the host's wait (`mlx_async_eval`, the sdk base's `pipeline_every` form; src/async_eval.zig,
patches/turboexl3-model.patch). The same arrays run the same kernels in the same order on the same
stream, so nothing served moves and the graph is cut where it was. The host builds the next sublayer's graph while the
GPU runs the layer's tail, where the blocking eval left the GPU idle while the host did: about 8 ms an eval at a 16K
prompt, about 90 ms over its 12 evals. It is not billed. A tail's buffers are held by its command buffers until they
complete, and the next sublayer's arrays can be made while the tail still runs: at a 16K prompt at most the tail's
transients beside the stream it replaces (the old 0.54 GB stream, the shared expert's and the combine's outputs), about
1 GiB, which MLX's buffer cache can also absorb; the A/B's peak column decides. On by default (unset or 1; 0 turns it off), any other
value refused by name (`Glm53fAsyncEvalSwitch`); the load line, and an engaged line at its first eval.

`TURBOEXL3_GLM53F_PF_TAIL_EVAL=1` (exact) hands a prompt layer's tail (the routed call's combine, the shared expert and
the hyper-connection's stream update) to the GPU without the host's wait at the layer's end, where the cadence's eval is
not due (src/tail_eval.zig, patches/turboexl3-model.patch in `hidden` and `hiddenSplit`). Between two cadence evals a
layer's tail is otherwise committed by the next layer's first eval: a KDA layer's routing barrier, or (with
`DSA_HOLD`) a DSA layer's q_abs, after that layer's start and its attention graph. The routed call's last chunks have
completed by then, so the GPU waits for the host: g53hmt (PF_INT8=4's host-marks probe, a 16K prompt) put 0.77 ms
(median) of GPU idle at each KDA layer's barrier, and a DSA layer's start behind its cache trim. With the switch the
tail runs while the host builds the next layer: 33 more async evals at GLM's 45 layers (every 4th and the last keep the
cadence's own). The same arrays by the same kernels in the same order, so the same words. Not billed: as
`ASYNC_EVAL`'s, a tail's buffers can be held while the next layer's are made (about 1 GiB at most at a 16K prompt); the
A/B's peak column decides. On by default (unset or 1; 0 turns it off), any other value refused by name
(`Glm53fPfTailEvalSwitch`); the load line, and an engaged line at its first commit. An MLX-free test holds the layers it commits.

`TURBOEXL3_GLM53F_PF_REUSE=1` (exact) makes a prompt pass reuse its buffers (src/pf_reuse.zig,
patches/turboexl3-model.patch). g53prof8 counted about 375 GB of fresh buffers in a 16K prompt pass
(a KDA layer 6.2 to 7.6 GB, the MoE stage 110 GB in 1,558 tiles) and 1.4 s of GPU idle, mostly while the host
allocated: MLX reuses a cached buffer only within two pages of a request, and the prompt's 2 GiB cache releases what is
freed past it. Two parts:
- A KDA run (a KDA layer whose next attention layer is KDA, to the run's last KDA layer) holds its buffers. At the run's
  first layer MLX's cache limit rises by the bill's MLP envelope a forward row (stream_room `envelopeRowBytes`: 884,736
  B, 13.5 GiB at a 16K prompt) and stays there through the run, so what a layer frees stays cached and the next KDA
  layer takes it back. The limit is not lowered inside a run: MLX trims a cache over its limit from its oldest
  buffers, the held ones. At the next DSA layer and at the forward's end the limit goes back to its value before the
  run, and MLX trims the cache at its next allocation, so the DSA layers see today's cache. A 16K prompt has 11 runs
  and 22 layers that take a run's buffers back.
- turboexl3's routed prefill allocates a call's row-slot arrays at its row-slots rounded up to a multiple of 256
  (`Config.pf_size_class`), so successive calls' buffers match each other in the cache.

Only which buffer an array uses moves, so the same words: turboexl3's GLM test (`TURBOEXL3_GLM_PF_SIZE_CLASS_BYTES`)
holds the size classes to the bytes without them at row-slot counts that are not multiples of 256. Not billed: through a
run's MoE stages the cache holds the run's KDA working set, below the DSA layers' peak; the probe pair's and the A/B's
peak columns decide. A forward's counts print at its end (`TURBOEXL3_GLM53F_PF_REUSE pass rows= runs=
holds= drops= held_mb= cache_mb_max= cap_mb= ...`). On by default (unset or 1; 0 turns it off), any other value refused by name
(`Glm53fPfReuseSwitch`); the load line, and an engaged line at a run's second layer.

`TURBOEXL3_GLM53F_PF_MOE_HOLD=1` (exact, needs `PF_REUSE`) gives a DSA layer's routed prefill its first two chunks'
arrays from MLX's cache (src/pf_reuse.zig, src/routed_t3.zig `prefillParts`). g53pflt (the 62-entry control, a 16K
prompt): at each DSA layer the MoE stage's first chunk took 13-18 ms of host time before its first commit (0.11-0.19 ms
at a KDA layer), and the GPU idled 144 ms over the 11 DSA layers. The run's limit goes back at the DSA layer, and its
attention allocates and frees far more than the 2 GiB cache keeps, so no chunk-sized buffer is left for its MoE stage.
- At a DSA layer's start, before the run's limit goes back, one launch that writes nothing makes two 4096-row chunks'
  arrays at their size class, those of the chain turboexl3 runs: 26 on the exact chain (`chunkBytes`, 482.5 MiB at
  GLM-5.3-Flash's geometry), 32 on `TURBOEXL3_GLM53F_PF_INT8`'s int8 chain (`chunkBytesI8`: its int8 copies, their sums
  and scales in place of the f16 copies, 407.6 MiB), each at its MLX buffer's bytes (whole pages). They take what the
  cache holds of those sizes, and the layer's attention cannot trim them.
- At the layer's first routed call they go back to the cache under a limit raised by their bytes, and the call's first
  two chunks take them. The next layer's start puts the limit back (or the forward's end), and lets go of a hold that
  no routed call took.

Only which buffer an array uses changes, and nothing reads the held arrays, so the same words. `zig build
moe-combine-gpu-test` (src/pf_hold_gpu_test.zig) holds the sizes to the chain's own, on the exact chain and on the int8
chain under both schemes: two chunks on `moeClampedRows` at the size classes take every array from a hand-back and
allocate under 1 MiB more, where they allocate the hold's bytes from an empty cache. Not billed: 482.5 MiB (407.6 MiB
on the int8 chain) more through each DSA layer's attention, the prompt's peak stage; the A/B's peak column decides. The hold's counts print at a forward's end (`TURBOEXL3_GLM53F_PF_MOE_HOLD pass rows= hold_mb=
taken= used= unused= failed= from_cache_mb= first_chunks_measured= fresh_kb_max= ...`). On by default (unset or 1; 0 turns it off),
any other value refused by name (`Glm53fPfMoeHoldSwitch`), and refused without `PF_REUSE`
(`Glm53fPfMoeHoldWithoutReuse`); the load line with the hold's MiB, and an engaged line at its first hand-back.

`TURBOEXL3_GLM53F_PF_MOE_HOLD_TOP=1` (exact, needs `PF_MOE_HOLD`) makes the hold's hand-back cache every held buffer.
g53pf63t (the 63-entry control's timeline, a 16K prompt): at every DSA layer the cache held 2858-2921 MiB at the routed
call, over the hand-back's 2530 MiB limit (the 2048 MiB limit and the hold). MLX caches a freed buffer whole while the
cache is under its limit, and trims only at its next allocation, so the layer's attention left the cache over its limit.
A buffer freed then is released: the hand-back released all 26 held buffers (3.7-8.8 ms of host time), and chunk 0
allocated its arrays afresh while its first allocation trimmed the cache (its commit 2.7-9.7 ms, 0.1 ms at a KDA
layer).
- The hand-back's limit is max(the limit, the cache's bytes) + the hold, so every held buffer is cached and the call's
  first two chunks take them, with nothing to trim. The next layer's start puts the limit back, as before.
- A forward's DSA layers before its first routed call hold nothing: the dense layers before them leave no chunk-sized
  buffer to take.

Only which buffer an array uses changes, so the same words. Memory: through the DSA layer's MoE stage the cache keeps
what it held over the limit (about 0.5-1 GB at a 16K prompt) instead of trimming it at the stage's start, and the next
KDA run's raise carries it to the next DSA layer; the DSA attention sees what it sees with PF_MOE_HOLD alone. Not
billed: the A/B's peak column decides. `zig build moe-combine-gpu-test` (src/pf_hold_gpu_test.zig) hands the hold back
into a cache over its limit: without TOP every held buffer is released; with it every one is cached and the two chunks
take them, with the same bits. With PF_MOE_HOLD, each count of the take read around one step prints at a forward's end
(`TURBOEXL3_GLM53F_PF_MOE_HOLD take rows= top= handbacks= handback_buffers= cached= released= still_in_flight=
over_limit= excess_mb= first_alloc_trims= first_alloc_cache_drop_mb= chunk01_commits= commit_cache_drop_mb=
commit_active_mb= outputs_from_hold=`, src/pf_reuse.zig's doc). On by default (unset or 1; 0 turns it off), any other value refused by
name (`Glm53fPfMoeHoldTopSwitch`), and refused without `PF_MOE_HOLD` (`Glm53fPfMoeHoldTopWithoutHold`); the load line.

`TURBOEXL3_GLM53F_DSA_HOLD=1` (exact, a memory trade) gives a DSA layer's absorbed queries and sparse attention output
the buffers of the previous DSA layer's (src/dsa_hold.zig, patches/turboexl3-model.patch in `dsaAttn`). g53q8pt
(PF_INT8=4's prompt timeline, a 16K prompt): the GPU idled about 9.6 ms before each DSA layer's sparse attention, where
the host allocates q_abs and o_lat afresh (both [1, 64, rows, 512], 2 x 0.94 GB at 14,305 sparse rows); MLX reuses a
cached buffer only at its size, and the 2 GiB cache is long past them by the next DSA layer.
- A trunk DSA layer's prompt span (2,048 sparse rows or more, the sparse path) keeps both for the forward's next DSA
  layer. The MTP head's call and a split forward's short span neither keep nor release.
- At that layer, just before its q_abs is evaluated, the limit in force is read by raising it to the most, the held
  arrays are freed into the cache, and the limit is set to the limit read plus their bytes. PF_REUSE has put the KDA
  run's limit back at the layer's start, so the next allocation still trims what the run left, oldest first; the
  released buffers are the newest. q_abs and then o_lat are evaluated without the host's wait, each taking a released
  buffer of its size, and the limit goes back to exactly the limit read. PF_MOE_HOLD's hold is taken before (at the
  layer's start) and handed back after (at its first routed call), so TOP's raise reads the limit put back.
- A decode or verify forward drops anything left and prints the pass's counts (`TURBOEXL3_GLM53F_DSA_HOLD pass:
  releases= hits= misses= kept= dropped= held_max_mb= mlx_peak_mb=`).

Only which buffer an output uses moves, and when its work is committed: the same kernels on the same rows, so the same
words. Memory: the held 1.9 GB (at 16K rows) stay live through the three KDA layers and four MoE stages between; not
billed: the A/B's peak column decides. On by default (unset or 1; 0 turns it off), any other value refused by name
(`Glm53fDsaHoldSwitch`); the load line, and an engaged line at the first release with the held MiB. An MLX-free test
holds the holder's keep, release under the most limit, the limit in force plus the held bytes, its exact restore, the
hit counts and the drops.

`TURBOEXL3_GLM53F_PF_RUN_TRIM=1` (exact, needs `PF_REUSE`) trims a KDA run's cache at the run's last layer, not at the
next DSA layer (src/pf_reuse.zig). g53hmt (PF_INT8=4's host-marks probe, a 16K prompt): from L7 on, each DSA layer spent
1.3-8.8 ms (median 6.1) of host time between its start and its KV append with the GPU idle, where a KDA layer spends
0.04 ms and L3, with no run before it, 0.23 ms. PF_REUSE puts the run's limit back at the DSA layer's start with the
cache holding up to 8.4 GB, and MLX's first allocation in the attention releases everything past the limit before
anything is queued.
- Once the run's last KDA layer's routed call has two chunks in flight (`prefillParts`; at its end with one), the limit
  before the run is put in force, one small allocation makes MLX trim the cache to it (oldest buffers first, while the
  GPU runs the chunks), and the run's limit goes back.
- The chunks' own buffers are the newest, so they stay. What the layer frees after is cached as before (its later
  chunks take their predecessors', a DSA layer's PF_MOE_HOLD its last chunks'), and the DSA layer's drop finds only
  that.

Only which buffers MLX releases when moves, so the same words. Memory: the cache holds less, sooner. A forward's end
prints `TURBOEXL3_GLM53F_PF_RUN_TRIM pass rows= trims= trimmed_mb= drop_excess_mb_max= drop_excess_mb_sum=` (what the
DSA layers' drops still found past the limit). Off by default (unset or 0), any other value refused by name
(`Glm53fPfRunTrimSwitch`), and refused without `PF_REUSE` (`Glm53fPfRunTrimWithoutReuse`); the load line. An MLX-free
test holds the trim at the run's last layer's second chunk (at the call's end with one), the limits in order, the
drop's excess, and no change when off.

`TURBOEXL3_GLM53F_KDA_VERIFY=1` runs the prologue, gate and gated-norm texts above in a KDA layer's verify forward too
(kda.zig's `capture`: the trunk forward of every decode round, which keeps every row's state for a partial accept's
rollback, so the fused decode route never takes it, and an exllamav3 pack has no row-joined projection for the copy's
one-dispatch step), whatever the prompt switches. Without it each KDA layer of a verify runs the chain's prologue, gate
and gated-norm ops a launch each (about 19 launches of 1 to 4 rows, x 34 layers a round); with it, three texts and the
capture's conv input kept as the chain's own MLX concatenate of [window | qkv] (src/kda_verify.zig,
patches/turboexl3-model.patch). The capture recurrence stays the chain's. `zig build kda-gpu-test` holds a
cold 17-row prompt then two verifies of 1, 2, 3 and 4 rows in a row to the chain bit for bit (each verify's output,
conv window, state, conv input and the per-row states the capture writes: the first T - 1, the last row being the
capture kernel's unwritten one, which no rollback reads) and counts the three texts' launches
(`TURBOEXL3_GLM53F_KDA_VERIFY_ENGAGED`). On by default (unset or 1; 0 turns it off), any other value refused by name
(`Glm53fKdaVerifySwitch`).

`TURBOEXL3_GLM53F_KDA_LAZY_SEQ=1` makes a KDA layer's verify write only its output and final state, and its per-row
states (the capture a partial accept's rollback reads) a launch of their own that MLX runs only when something reads
them (src/kda_lazy.zig, patches/turboexl3-model.patch in kda.zig's capture branch). The
verify runs the stock recurrence launch without the per-row stores (`gated_delta_step_vec`: the capture launch's text
but for them, so the same output and state bits), and the per-row states come from the stock capture launch
(`gated_delta_step_seq_vec`) as a primitive of its own over the same inputs and initial state, in no eval set. A full
accept keeps the final state and frees the per-row states unrun; a partial accept's rollback slice (the host's
`rollbackSsmFromCapture`) runs the capture launch when the next verify reads the state, and its rows are the capture
launch's bits. A full-accept verify writes 8 MB a KDA layer less at 3 rows (34 layers); a partial accept's next verify
runs the capture launch once more; the initial state is held until the next verify (4 MB a layer). `zig build
kda-gpu-test` holds two verifies in a row to KDA_VERIFY's bit for bit at widths 1 to 4 (`set=verify_lazy vs=verify`,
every output, window, state, conv input and per-row state), checks the per-row states stay uncomputed while the rest is
read (`TURBOEXL3_GLM53F_KDA_LAZY_ENGAGED ... seq_computed_unread=0`), and holds a verify from each partial accept's
rolled-back state to KDA_VERIFY's (`set=verify_lazy_rollback<a>`). Each request's end logs
`TURBOEXL3_GLM53F_KDA_LAZY_SEQ verifies=...`. On by default (unset or 1; 0 turns it off), any other value refused by name
(`Glm53fKdaLazySeqSwitch`).

`TURBOEXL3_GLM53F_ROUTER=1` routes a routed layer's 2- to 4-row decode call (an MTP or prompt-lookup verify) in two
launches where the stock route spends four (src/fused_router.zig, patches/turboexl3-sdk.patch). The
stock route casts x to f32 and runs `mlx_matmul` over the router's f32 table, which MLX (TF32 on, its default) takes to
its split-K NAX GEMM (K split at 2048; per 32-expert block and partition one chain of 64 dependent load-then-MMA steps),
then its split-K accumulate, then mlx-serve's fused router kernel (sigmoid, the correction bias, the top 8 by the
lowest-index tie-break, the weights normalized and scaled). Phase 1 runs the same chains with MLX's own NAX tile matmul
(the kit's steel headers), the same `matmul2d` op on the same operands in the same k order, its weight reads prefetched
by the whole threadgroup and the weight read as stored (the pack's f16, or bf16, which the f32 table only widens); phase
2 is mlx-serve's
router text with its logits read as the two partials summed as MLX's accumulate sums them. Every other call routes as
before: 1 row (the MTP head) or more than 4 (a prompt pass), a layer without its raw router registered, another
geometry, layout, routing mode or dtype, or TF32 off. `zig build router-gpu-test` holds the logits, the routes and the
weights to the stock route's bit for bit for f16 and bf16 routers at 2, 3 and 4 rows, 512 calls each, and checks 1 and 5
rows route as before.
Each request's end logs the routes taken by layer (`TURBOEXL3_GLM53F_ROUTER routes=... per_layer_min=...
per_layer_max=...`: equal counts mean every routed layer of every verify took the plan). On by default (unset or 1; 0 turns it off),
any other value refused by name (`Glm53fRouterSwitch`).

`TURBOEXL3_GLM53F_PREDICT=1` scores the decode lookahead's next layer in two launches where the stock predictor spends
six (src/fused_router.zig `predict`, the first statement of the predictor function src/streamed/glm5n/experts.zig
`scoresBf16`, which every caller of the bank's `scoresOf` reaches). The stock predictor multiplies bf16 x by the stored
f16 router's transpose, so MLX promotes both to f32: it casts x, copies the whole router to f32 (2.4 MB read and 4.7 MB
written a call), runs the router's split-K NAX GEMM (the copy keeps the transposed layout) and its accumulate, then
MLX's Sigmoid and the correction add. The plan runs the router's phase 1 over the next layer's router as stored, then
mlx-serve's router text's sigmoid (MLX's `Sigmoid`, `precise::exp`) over the summed partials and the correction add in
one launch. f16 routers only (the pack's): with a bf16 router the stock matmul stays bf16 (its logits rounded to bf16),
so such a call scores as before. The scores only pick what is read ahead; `zig build router-gpu-test` holds them to the
stock predictor's bit for bit (f16 routers, 2, 3 and 4 rows, 512 calls each, x 2-D and 3-D) and checks a bf16 router's
calls decline, so the same records are read. Each request's end logs `TURBOEXL3_GLM53F_PREDICT predictions=...
declined=... other=...`. On by default (unset or 1; 0 turns it off), any other value refused by name
(`Glm53fPredictSwitch`).

`TURBOEXL3_GLM53F_ROUTER_ONE=1` runs each call `TURBOEXL3_GLM53F_ROUTER` or `TURBOEXL3_GLM53F_PREDICT` takes in one
launch where it spends two, at GLM-5.3-Flash's dims (K 4096, N 288: the 18 chains, constants of the texts). Phase 1's
text runs as it is but for its partial's store, an atomic store of the same f32 bits. Each threadgroup then fences its
stores at device scope and counts its arrival on the call's own counter, a fresh zeroed array a call
(src/kernels/router_one_arrive.metal); the threadgroup that counts last runs phase 2 and the other 17 return, so no
group waits on another. Phase 2 reads the partials with atomic loads and runs the stock texts' statements: the router's
score loop over every row's (row, expert) pairs, then each row's selection on its own simdgroup
(src/kernels/router_one_route.fmt.metal over mlx-serve's router text, its read and its indexing of the score, weight
and selection arrays by a row's slice edited, each edit found once at compile time), and the predictor's sigmoid and
correction a pair (src/kernels/router_one_predict.fmt.metal). The same chains, sums, sigmoids and selections in the same
order, so the same routes, weights and scores: `zig build router-gpu-test` holds them to the stock route's (f16 and bf16
routers) and the stock predictor's (f16) bit for bit at 2, 3 and 4 rows, 512 calls each, eight calls evaluated together
(`TURBOEXL3_GLM53F_ROUTER_ONE_BYTES`). It saves one dependent launch a routed layer's router and one its lookahead
scores (two a layer of a verify with both on). A call the plan does not take routes and scores as before. Each request's
end logs `TURBOEXL3_GLM53F_ROUTER_ONE routes=... predictions=...` (equal to ROUTER's routes and PREDICT's predictions).
On by default (unset or 1; 0 turns it off), any other value refused by name (`Glm53fRouterOneSwitch`); on with neither ROUTER nor
PREDICT, the load is refused (`Glm53fRouterOneAlone`).

`TURBOEXL3_GLM53F_MOE_COMBINE1=1` ends a decode or verify call's routed experts (src/streamed/glm5n/experts.zig's decode
lane) with one launch where the stock lane runs six: the join's take of each routed position's row from the call's
concatenated outputs, then `combineStock`'s astype f32, the multiply by the f32 route weights, the sum over the 8 routes
and astype bf16, and before them the source's f32 cast of the router's bf16 weights. The call ends before the take
(`routedHoistIndexed`, JOIN_INDEXED: the concatenated outputs and the join's order) and src/kernels/moe_combine1.metal
reads each position's row in place through the order and runs the stock chain's arithmetic: each product f32(row) *
f32(w) rounded to f32 (MLX's Multiply, never contracted), added to the +0 init, folded k = 0..7 in router order (MLX's
col_reduce_small at threadgroup y = 8), bf16 once, as the prompt pass's combine text (mimo_prefill_combine.metal) states
it; the weights are read in their stored type (their widening is exact). GLM-5.3-Flash's dims only (hidden 4096, top-8):
any other call takes the stock take and combine over the same rows. `zig build moe-combine-gpu-test` holds the output to
the stock chain bit for bit at 1 to 4 rows, over joined outputs as long as the positions and 24 rows longer, bf16 and
f32 weights, rows with exact +0 and -0 (`TURBOEXL3_GLM53F_MOE_COMBINE1_BYTES ...`), and checks a 2048-wide row, an int32
order and f16 weights are declined. Each request's end logs `TURBOEXL3_GLM53F_MOE_COMBINE1 calls=...`. On by default
(unset or 1; 0 turns it off), any other value refused by name (`Glm53fMoeCombine1Switch`).

`TURBOEXL3_GLM53F_ROUTE_VIEW=1` hands a decode or verify call's route ids to the routed call as an int32 view of the
router's uint32 ids (src/route_view.zig, the same patch) where the stock lane casts them (`astype(ids, int32)`, a copy
launch every MoE layer, between the router and the hit wave's LUT gather). An id is below the 288 routed experts, so
its uint32 and int32 words are the same bits, and MLX's View at the same item size shares the buffer: no launch. Ids
of another type are cast as before. The same device step holds the view to the cast's values and checks it shares the
router's buffer at 1 to 4 rows (`TURBOEXL3_GLM53F_ROUTE_VIEW_BYTES ... shared=1`). Each request's end logs
`TURBOEXL3_GLM53F_ROUTE_VIEW views=...`. On by default (unset or 1; 0 turns it off), any other value refused by name
(`Glm53fRouteViewSwitch`).

`TURBOEXL3_GLM53F_MTP_ROUTER=1` (with ROUTER) registers the MTP head's router with ROUTER's plan too: the head's raw
router (`mtp.0.block.mlp.gate.weight`, bf16 [288, 4096]) is loaded with the head only with this switch
(patches/turboexl3-model.patch) and registered after the trunk's layers, so a draft step of 2
to 4 rows (step 0) routes as a verify's layer does (ROUTER_ONE's one launch when on), the same routes and weights (the
router's device step). ROUTER's request line counts the head's routes apart (`head_routes=`), so its per-layer counts
stay the trunk's. On by default (0 turns it off); any other value refused by name (`Glm53fMtpRouterSwitch`), and without ROUTER the load
is refused (`Glm53fMtpRouterAlone`).

`TURBOEXL3_GLM53F_ROUTER_ONE1=1` (with ROUTER) runs a registered layer's one-row router call in one launch where the
stock route takes three (x cast to f32; MLX's gemv over the router's f32 copy,
`gemv_float32_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby0`; the stock router kernel). With MTP_ROUTER, these are the MTP head's
one-row draft steps. Each of the launch's 18 threadgroups runs MLX's own `GEMVKernel::run` at the stock gemv's template
arguments. MLX's gemv.h is vendored unchanged in src/vendor/mlx/ (NOTICE; scripts/sync-mlx-gemv.sh copies it and checks
its pin), and src/fused_router.zig makes four edits to it at compile time, each found once: the two loaders and `run`
take the operands' stored types (x bf16, the raw router f16 or bf16), widened to f32 by the loaders' own casts, which
are the exact values of the stock cast and copy; and each logit is stored with an atomic store of the same f32 bits.
Then comes ROUTER_ONE's arrival: the threadgroup that counts last runs the stock router text, its logits read with
atomic loads (src/kernels/router_one1.fmt.metal). The loads, multiply-adds, lane reductions, sigmoids and selections
are the same and in the same order, so the logits, routes and weights are the same: `zig build router-gpu-test` holds
all three to the stock route's bit for bit, 512 calls for each router type, x 3-D and 2-D, eight calls evaluated
together (`TURBOEXL3_GLM53F_ROUTER_ONE1_BYTES`). It also checks that the launch refuses 2 rows and that an unregistered
layer's call routes as before. The launch reads the raw router (2.4 MB) where the stock gemv reads the f32 copy
(4.7 MB). Each request's end logs `TURBOEXL3_GLM53F_ROUTER_ONE1 routes=... head_routes=...`. On by default (unset or
1; 0 turns it off), any other value refused by name (`Glm53fRouterOne1Switch`); on without ROUTER, the load is refused
(`Glm53fRouterOne1Alone`).

`TURBOEXL3_GLM53F_SMALLOP_FUSE=1` merges the small ops around the decode and verify forwards' big launches where a kernel
gives the same bits (src/smallop.zig, patches/turboexl3-model.patch and patches/turboexl3-sdk.patch). At decode and
verify rows (1 to
16):
- a MoE layer's routed sum and shared expert leave the MLP unadded (moe.zig's `pair_slot`, glm5_next/moe.zig
  `forwardPair`); the next sublayer's hcPre runs its defer2 text, which reads each stream element as the defer text does
  with the two terms added in T as `mlx_add` adds them (`msv_hc_hval2`): one add less a MoE layer. The defer2 text runs
  at the defer text's threadgroup once the stock texts' first launches have set it; where Metal caps it below that, it
  declines for the process (logged) and the add and the defer text run;
- a DSA layer's q_a and kv_a (4096 -> 1536 and 4096 -> 512) run in one DEC-ONE launch, each at its own K split (8 and
  32), suh and svh (turboexl3's `glm53f.pair_ks.linear2`), the pairs of the trunk's 11 DSA layers and the MTP layer's
  registered when the bank opens (their svh joined, 4 KB a layer); then their two RMSNorms in one launch (`normPair`);
- a dense layer's gate and up (4096 -> 12288 each) run in one DEC-ONE launch (turboexl3's `glm53f.shared_gu.gateUp`,
  each member at its own K split, suh and svh), the pairs registered when the bank opens (their svh joined, 48 KB a
  layer);
- the MTP head's draft step at one row: its embedding row, both norms and their concatenation in one launch (`front`),
  its first residual and post-norm through the kit's addNorm, and its last residual (the MoE's two terms, the residual)
  and final norm in one launch (`add2Norm`).
The norms mirror MLX's rms_norm single-row kernel as the kit's add_norm does. Each piece declines (the stock ops run) off
its rows, widths or dtypes. `zig build smallop-gpu-test` holds every piece to the stock ops bit for bit: hcPre's normed
input, gates and rebuilt stream and the deferral's materialize at rows 1, 2, 3, 4, 8 and 16; the norm pair at rows 1 to
4 and 16; the DSA q_a and kv_a at K6 and K4, rows 1, 2, 3, 4, 8 and 16; the front (int32 and uint32 ids); the last
residual and the kit's addNorm at rows 1, 3 and 4; the dense gate and up at rows 1 to 8 and 16 (K6) where turboexl3's plan
takes them, and checks the declines. Each request's end logs `TURBOEXL3_GLM53F_SMALLOP_FUSE hc_pairs=... defer2=...
norm_pairs=... dsa_qkv=... dense=... fronts=... tails=... declined=... dense_layers=... dsa_layers=...`. On by default
(unset or 1; 0 turns it off), any other value refused by name (`Glm53fSmallopSwitch`).

`TURBOEXL3_GLM53F_DRAFT_S0=1` (with `SMALLOP_FUSE`) runs the MTP head's merged first draft step on SMALLOP's texts
(src/draft_s0.zig, patches/turboexl3-model.patch: plugins/glm5_next/head.zig `step`). That
step (step 0) takes the committed rows' history and the draft row, 2 to 4 rows; SMALLOP takes only the one-row step:
- the front: each row's embedding row and hidden row under their norms, and the concatenation, in one launch
  (src/kernels/draft_s0_front.metal, smallop_front.metal's statements for each row);
- the first residual and the post-norm through the kit's addNorm at the step's rows;
- the last residual and the final norm at the last row, the only row the step returns, in one launch. Without
  `HEAD_LAST_ROW` the launch adds the MoE's two terms and then the residual (`add2NormLast`); with it, the residual's
  last row and the MoE's last-row output (`addNormLast`). A history append (no output read) runs as before.
Each kernel does the stock ops' arithmetic on the same values in the same order for each row, so the draft tokens do not
change. Each piece declines (the stock ops run) off 2 to 4 rows of 4096 bf16. The front also declines a quantized or
non-bf16 table and ids other than int32 or uint32. `zig build smallop-gpu-test` holds each piece to the stock ops bit
for bit at 2, 3 and 4 rows, and checks the declines at 1 and 5 rows. Each request's end logs `TURBOEXL3_GLM53F_DRAFT_S0
fronts=... posts=... tails=... tails_last_row=... declined=...`. On by default (unset or 1; 0 turns it off), any other value refused by
name (`Glm53fDraftS0Switch`); on without `SMALLOP_FUSE` the load is refused by name (`Glm53fDraftS0Alone`).

Two switches let a verify linear's producer write the linear's prepared input itself (src/dec_prep.zig,
patches/turboexl3-model.patch). At 2 to 4 rows, turboexl3's DEC-ONE prepares the input of a
projection wider than 1024 outputs in a launch of its own (prep_sc: the input rounded to f16, times suh, through the
128-point Hadamard and its scale, stored f16 in lane order, each 128-block's sum, the GEMV's arrival counters zeroed),
then runs its GEMV. Here the producer's own text runs prep_sc's statements on its values in the same lanes, and
turboexl3's `decPrepared` runs the GEMV alone: one dependent launch fewer a call, the same bytes. Each text is
GLM-5.3-Flash's, its widths constants; `decPrepared` and `decPrepSpec` (what the GEMV reads at a row count) are
turboexl3's generic entries.
- `TURBOEXL3_GLM53F_KDA_OPREP=1`: a KDA layer's gated norm in a verify (64 value heads of 128, the norm-gate text's
  statements) writes the out projection's (8192 -> 4096) prepared input (`kdaOut`, src/kernels/kda_oprep.metal): 34
  launches fewer a verify.
- `TURBOEXL3_GLM53F_DSA_OPREP=1`: a DSA layer's out projection (16384 -> 4096) prepares its input from the attention's
  per-head values [1, 64, rows, 256] where they lie (`dsaOut`, src/kernels/dsa_oprep.metal): the copy into row order and
  prep_sc's launch become one, 11 fewer a verify.
Each producer declines (the stock ops run) at one row, past four rows, off GLM-5.3-Flash's widths or bf16, and where a
DEC-ONE variant reads f32 prepared rows. `zig build dec-prep-gpu-test` holds each to the stock GLM chain bit for bit at
2, 3 and 4 rows under the default and g1 GEMVs, and checks the declines. Each request's end logs
`TURBOEXL3_GLM53F_DEC_PREP kda_oprep=... dsa_oprep=... declined=...`. Each is on by default (unset or 1; 0 turns it off), any other
value refused by name (`Glm53fKdaOprepSwitch`, `Glm53fDsaOprepSwitch`).

## Launch trims

Four switches remove launches a verify (every decode round's trunk forward) spends on a copy, on a value it makes
again every call, or on zeroing counters. Each is exact: the same values reach the same kernels. Each is on by default
(unset or 1; 0 turns it off), refuses any other value by name (`Glm53fKdaConvinSwitch`, `Glm53fKdaExpsigSwitch`,
`Glm53fCachedScalarsSwitch`, `Glm53fHostZeroSwitch`), prints its line at load, and logs its counts at each request's end.

| switch | what changes | how it is checked | launches it removes |
|---|---|---|---|
| `TURBOEXL3_GLM53F_KDA_CONVIN=1` | with `KDA_VERIFY`: the verify's prologue text also writes the capture's conv input [window \| qkv] (src/kernels/kda_convin.metal after kernels/kda_prologue.metal: each thread copies its 4 channels of its row, the first row's threads the window's rows), where `KDA_VERIFY` makes it with MLX's concatenate; the layer's new window is that input's last 3 rows (src/kda_trim.zig) | `zig build kda-gpu-test`: two verifies in a row against `KDA_VERIFY`'s, every output, window, state, conv input and per-row state bit for bit at widths 1 to 4 (`set=verify_convin vs=verify`) | the concatenate's two copies a KDA layer (68 a verify), and at a 2-row verify the window's own concatenate |
| `TURBOEXL3_GLM53F_KDA_EXPSIG=1` | a verify's gate text reads exp(A_log) made once a layer (the chain's exp of A_log as f32, evaluated and kept); beta's sigmoid is made in the capture recurrence's launch: the copy's stock texts (`gdnKernelSource`, both the capture launch and, beside `KDA_LAZY_SEQ`, its launch without the per-row stores and the lazy per-row states' launch) with the one beta read taking MLX's own sigmoid of b, tabulated over every 16-bit value, at b's bits (src/kda_trim.zig, patches/turboexl3-model.patch in kda.zig) | `zig build kda-gpu-test`: the same verifies (`set=verify_expsig vs=verify`), both trims beside `KDA_LAZY_SEQ` (`set=verify_trim_lazy vs=verify`, the per-row states uncomputed while the rest is read) and the verify after each partial accept's rollback (`set=verify_trim_lazy_rollback<a>`); `TURBOEXL3_GLM53F_KDA_TRIM_ENGAGED` counts each trim's work | exp(A_log) and beta's sigmoid, one launch each a KDA layer (68 a verify) |
| `TURBOEXL3_GLM53F_CACHED_SCALARS=1` | the clamped SwiGLU's limit (`fusedClampedSwiGLU`: every shared expert's and dense layer's MLP) made once a device and dtype, evaluated and kept, where every call cast it from f32 (src/cached_scalars.zig, patches/turboexl3-sdk.patch in fused_activations.zig) | `zig build smallop-gpu-test`: the SwiGLU with the kept limit against the per-call one at a shared expert's and a dense layer's widths, 1 to 4 rows (`TURBOEXL3_GLM53F_TRIM_BYTES piece=cached_limit_*`) | one cast a call (45 a verify, and the MTP layer's) |
| `TURBOEXL3_GLM53F_HOST_ZERO=1` | DEC-ONE's arrival counters a new array of zeros copied from the host a call (turboexl3's `dense.dec_host_zero`, src/host_zero.zig), where a call whose prepare runs inside its launch, and the joined q_a \| kv_a and gate \| up launches, zero them with `turboexl3_dense_dec_zero` | `zig build smallop-gpu-test`: each such call with the launch and with host-made zeros, bit for bit at 1 to 4 rows: a DSA layer's q_a and kv_a at K6 and K4, a dense layer's and a shared expert's gate and up, kv_a and q_a alone, each call's counter arrays counted (`TURBOEXL3_GLM53F_TRIM_BYTES piece=host_zero_*`) | one launch a call (56 a verify) |

`KDA_CONVIN` without `KDA_VERIFY` would never run: that load is refused (`Glm53fKdaConvinNeedsVerify`).

## DSA texts

Through patches/turboexl3-model.patch. Each is on by default (unset or 1; 0 turns it off), refuses any other value by name, and
prints its line at load (the DSA texts also an engaged line at their first launch). The standard test decides whether
one is kept. `zig build dsa-gpu-test -Dmlx-lib=<dir>` (TURBOEXL3_GLM53F_DSA_GPU=1) runs src/dsa_gpu_test.zig on the
device: the one-P attention against the plain form at the main chunk's and the tail's shapes (finite, the largest
difference at most 1e-2 of the largest output).

| switch | what changes | exact? | what it saves |
|---|---|---|---|
| `TURBOEXL3_GLM53F_DSA_VATTN=1` | a DSA layer's verify window (2 to 8 rows) attends without the gather: `sparseLatentAttention`'s two matmuls (`latentAttendFolded`) run as MLX's regular NAX GEMM loop (src/kernels/dsa_vattn.metal: the same `tile_matmad_nax` steps on the same operands in the same k order, the same past-the-end zeros) with each key row read from the latent at its row's clamped index, where MLX reads the gathered copy (src/dsa_vattn.zig, patches/turboexl3-model.patch at its `latentAttendFolded` call); the scale's multiply, the mask and the softmax stay MLX's; `zig build dsa-gpu-test` holds the output to the gather's bit for bit at 1 to 4 rows (src/dsa_vattn_gpu_test.zig) | yes: the same output | the gather (a `maximum` and a `take` writing n x 2,051 x 512 keys, 8.4 MB at 4 rows) and its read by both matmuls, 2 launches a DSA layer |
| `TURBOEXL3_GLM53F_DSA_VSEL=1` | a DSA layer's verify window (2 to 16 rows) selects in two launches, every row's indexer scores in one and their top-k and expansion in one (src/dsa_verify.zig, patches/turboexl3-model.patch at `selectRows`' verify branch): glm5_next's one-query texts (kernels/dec_score.metal, kernels/topk_expand.metal) with the row from the grid and each row's pointers and position offset by it, edited at compile time (each edit found its expected number of times) | yes: each row runs the one-query texts' arithmetic as it is; src/dsa_gpu_test.zig checks the tokens bit for bit at 1 to 4 rows | 8 launches a DSA layer to 3 at 3 rows (the per-row scores and top-k and the join), 55 a verify round over 11 layers |
| `TURBOEXL3_GLM53F_DSA_GLM=1` | a DSA layer's prompt pass on GLM-5.3-Flash kernels, the model's dims fixed at compile time (src/dsa_glm.zig, patches/turboexl3-model.patch): `selectRows`' prompt branch (2 rows or more, on NAX) as one launch (src/kernels/dsa_glm_index.metal: 32 heads of 128, pools of 4, 512 pools, 2,051 slots) that scores each query row with the NAX indexer's MMAs and statement in its order (the row's 32 heads the MMA rows), keeps the scores in a scratch row of its threadgroup's (rows g, g + G, ...: 32 MiB of rows at most, 256 at least) and takes the 512 best with a radix select and ordered compaction (src/kernels/dsa_topk_header.metal's); and `dsaAttn`'s dense and sparse outputs concatenated as their transposes, the output projection's layout in one copy. Any other shape declines to the stock path | yes: `zig build dsa-gpu-test` holds the tokens to the stock selection's and the scores to the NAX indexer's bit for bit (16K main call and tail, 995 and 17 rows after it, 32K, 64K and 128K contexts, zero weights, repeating keys; cold 995 and 17 decline) and the output to the concatenate and transpose's (src/dsa_glm_gpu_test.zig) | the indexer's scores sheet, negation, argpartition temporaries, slice, cast, sort and expansion (about 1.7 GB of fresh buffers a DSA layer at 16K rows, 0.94 GB of them the merge sort's own) and one 536 MB copy a DSA layer at a 16K prompt's main call |
| `TURBOEXL3_GLM53F_DSA_LFUSE=1` | a DSA layer's verify window (2 to 8 rows, one chunk) attends in two launches at GLM-5.3-Flash's dims (64 heads, 512-wide latent rows, 2,051 slots a row, constants in the texts; any other shape keeps the stock chain): src/kernels/dsa_lfuse_scores.metal reads the absorbed queries at the window row in place of the q4 copy, multiplies each by the scale in the activation dtype as MLX's multiply does, runs the first matmul as MLX's NAX GEMM loop over the keys read from the latent at the selected rows (as DSA_VATTN) and stores -inf at the unused slots (the mask's where); src/kernels/dsa_lfuse_pv.metal takes each head's max and sum as MLX's block softmax reduces them (a lane for each of its threads in turn: the same simd_max and simd_sum) and writes each p inside the second matmul's NAX GEMM loop (src/dsa_lfuse.zig, patches/turboexl3-model.patch at the `latentAttendFolded` call; DSA_VATTN where it declines) | yes: each step the stock op's arithmetic in its order; `zig build dsa-gpu-test` holds the output to the stock chain's bit for bit at 1 to 8 rows, alone and beside DSA_VATTN (src/dsa_lfuse_gpu_test.zig) | the q4 copy, the scale's multiply, the mask, the where and the softmax: 7 to 9 launches a DSA layer to 2, about 70 a verify round over 11 layers, and their intermediate arrays |
| `TURBOEXL3_GLM53F_DSA_ONEP=1` | a DSA layer's prompt sparse latent attention with P meeting the values as one fp16 piece (dsa_nax.zig `sparseLatentOneP`: the kernel without MSV_DSA_TWO_P), not the plain form's hi and lo pieces | no: P rounded once to fp16 (rounding-class) | a third of the kernel's MMAs: up to about 35 ms of a DSA layer's 103 ms at 16K rows |

## The pack

The converter's layout spec (LAYOUT.md of the GLM-5.3-Flash SC repack):

- `config.json`: turboderp's, with exllamav3's `quantization_config` (codebook mul1) and the `expert_streaming` block
  (`family` turboexl3, `quant` exl3-mul1-k-by-layer, the four manifest names).
- `model.safetensors.index.json` over the nonexpert shards; every trunk tensor in turboderp's name, dtype and bytes.
- `experts.bin`, `expert-manifest-v2.json`, `expert-manifest.json`: the routed layers' bank, K per layer.
- `mtp-experts.bin`, `mtp-expert-manifest-v2.json`, `mtp-expert-manifest.json`: the MTP layer's.
- The parse refuses a pack without them (`Glm53fPackNotRepacked`), another bank block (`Glm53fBankBlock`) or codebook
  (`Exl3CodebookUnsupported`).

## Measured

The maintainer measured this plugin and its earlier configurations from 2026-11-10 to 2026-11-12 on one Apple M5 Max with 128 GB of unified memory.
Every number below uses the sc_3.10bpw pack, one of turboderp's self-calibrated ("SC") branches in our streaming
repack. Other packs, other Macs and other memory sizes are not measured.

A window is one scheduled test session on our M5 Max. Its ID, for example g53g2p2, names its logs. An A/B compares two
server builds, A and B, over 8 runs in the order ABBA then BAAB. Its two groups of four runs are its blocks.

Each table names the build that it ran on:

- The "plugin" rows ran plugin commit 969a2bc, built with turboexl3 95e98dc through `--fork`. turboexl3 is
  published as 1db354d: the same commits signed, with the same trees.
- The frozen-line rows, the int8 speed results and the context sweep ran on the frozen line (defined below), not on
  969a2bc.
- The int8 quality results and HumanEval ran on the frozen line's parent configuration, the 64-entry control.

This PR's code is 969a2bc plus three changes: documents and the defaults test; the read pool's counters on the
`[stream-stats]` decode line (src/stream_stats.zig, `poolCounters` in src/streamed/glm5n/experts.zig, one line of
plugin.zig); and the turboexl3 pin at 1db354d. The head built and passed its tests on 2026-11-12 on an M5 Max (macOS
26.6) against two kits, kit7 (444c83de) and kit6main (4ac98da1): 125 of 126 tests (1 skipped), the 101 stream tests,
the link checks and the install receipt. The head has not been served or measured. The counters read at the same
points as the existing ones and change no output.

### Builds and defaults

The frozen line is the configuration that we froze on 2026-11-12, before we moved the plugin onto stock mlx-serve.
This plugin's defaults copy it. The frozen line sets 65 plugin switches and two server switches,
`MLX_SERVE_PREFILL_SPLIT=1` and `MLX_SERVE_MTP_PIN=depth=2`. The 64-entry control, our control configuration of
2026-11-10, has 63 plugin switches and `MLX_SERVE_PREFILL_SPLIT=1`. The frozen line adds `DSA_HOLD` and `PF_TAIL_EVAL`
to it.

Three server builds appear below:

| Name | mlx-serve | Plugin |
|---|---|---|
| kit6main | main 9a173773 + the split prompt forward + the pinned-depth prompt-lookup gate (two draft PRs) | 969a2bc at its defaults |
| kit7 | kit6main + the three expert-stream seams (three draft PRs) | 969a2bc at its defaults, which turn the seams on |
| frozen line | kit5main, our build of main 66aad949 with local patches | ac87fe1 with 65 plugin switches and the two server switches above |

The plugin declares 67 options. Unset, each option takes the frozen line's value, except for these:

| Switch | Frozen line | Plugin default |
|---|---|---|
| `ADOPT_ZC` | 2 | Removed, no switch (commit 8c5f3d2) |
| `DEMAND_HELP` | 2, on its own read-pool code in kit5main | 0. On the kit stream's `demandFirst(2)` it made decode 3.1% slower in g53g2h1, our A/B of it on 2026-11-12. |
| `PROMPT_KEEP`, `DEMAND_SPLIT`, `ADOPT_PIECES` | 1 | 1 where the kit's stream has the seam, else 0 |

`PF_INT8`, `LOGITS_DUMP` and `PF_RUN_TRIM` are off in both. `KDA_PERCORE`, `KDA_CHUNK` and `DSA_ONEP` are on in both.
They change rounding compared with the op-by-op path of the built-in glm5_next. "Exact" below means `PF_INT8=0`, not
bit-exact against the built-in.

The plugin's memory plan (`BILL_FIT`, on by default) sizes its expert slot rows. It keeps the machine's non-file
memory at or under the GPU ceiling for any request up to the context limit. Our tests set the ceiling to 110 GiB
(`MLX_SERVE_GPU_CEILING_MB=112640`). Smaller Macs are not measured. The plan's bases are measured on sc_3.10bpw, so at
its defaults the plugin refuses other packs by name (`Glm53fBillFitPack`) and a load without `--ctx-size`.

### How we measured

Each speed result comes from one standard test. The prompt is CPython's shlex.py plus a coding task, 16,387 prompt
tokens, and the server writes 1,024 tokens. An A/B uses seeds 1234 and 4321. Each run starts a fresh server, waits
until the SoC is below 40 C, and sends one measured request. Each arm gets one unmeasured warm-up server before the A/B. Sampling is greedy
(temperature 0) unless a table says otherwise. Card sampling means the model card's settings, temperature 1.0 and
top_p 0.95.

Both arms of every A/B use the same box settings: `MLX_SERVE_CACHE_LIMIT` 2 GiB, the 110 GiB GPU ceiling,
`--ctx-size 32768`, `--prefill-chunk 16384` and `--wired-margin-gib 2`. Both arms pin the MTP draft depth at 2. The
frozen line does this with `MLX_SERVE_MTP_PIN=depth=2`, and the plugin with `--mtp-min-depth 2 --mtp-max-depth 2`.

The six columns are these:

- Prefill tok/s is prompt tokens divided by the server's own prompt time.
- Decode tok/s is the server's decode rate over the 1,024 written tokens.
- TTFT and wall time are client-side, from the request to the first and the last token.
- Spec hit is accepted draft tokens divided by drafted tokens. The server drafts with the model's MTP layer and with
  prompt lookup. Each table header names which drafts it counts.
- Peak is the server's peak memory footprint. The A/B tables give GB (10^9 bytes, so 105.1 GB is 97.9 GiB). The
  sweep tables give GiB.

One setting favors this prompt. The pack's seed files (`expert-usage-v1.json`, and `expert-usage-decode-v1.json`,
which `SEED_DECODE` reads by default) were recorded from one request of this same coding prompt. They choose which
experts are resident before the first request. Another prompt can start with fewer resident hits. The results here
do not measure that effect.

### Speed at the defaults (standard test, greedy, 2026-11-12)

Each row is the mean of 4 runs. All 24 runs of g53g2p2, g53g2s1 and g53g2h1 wrote the same text (sha256 prefix
c3c346222307).

| A/B | Arm | Build | Prefill tok/s | Decode tok/s | TTFT s | Wall s | Spec hit %, MTP + lookup (MTP only) | Peak GB |
|---|---|---|---|---|---|---|---|---|
| g53g2p2 | A | frozen line | 1,238 | 41.15 | 13.25 | 38.20 | 73.1 (69.2) | 105.1 |
| g53g2p2 | B | plugin on kit6main | 1,240 | 40.58 | 13.23 | 38.54 | 73.1 (69.2) | 105.1 |
| g53g2s1 | A | plugin on kit6main | 1,240 | 40.75 | 13.23 | 38.43 | 73.1 (69.2) | 105.4 |
| g53g2s1 | B | plugin on kit7 | 1,240 | 41.20 | 13.23 | 38.15 | 73.1 (69.2) | 105.4 |

- In g53g2p2, the plugin on kit6main writes the frozen line's text. Prefill is level (+0.2%). Decode is 1.4% lower, and
  lower in both blocks (-1.0% and -1.8%). Both arms do the same compute (416 verify rounds in each run).
- Our reading of the -1.4%, from the expert-stream counters below, is that the time goes to the stream's reads and
  copies. The frozen line's adopt copies took 83.9 ms a run, and the plugin's took 2,481 ms. On kit6main the plugin
  has no `ADOPT_ZC` (removed) and no `ADOPT_PIECES` (its seam is not in stock mlx-serve).
- In g53g2s1, the three seams make the plugin's decode 1.1% faster (blocks +1.2% and +1.0%).
- No A/B compares the plugin on kit7 with the frozen line.
- In g53g2h1, our `DEMAND_HELP` A/B, `TURBOEXL3_GLM53F_DEMAND_HELP=2` made decode 3.1% slower, so it stays off.

### Opt-in int8 prefill (not exact)

`TURBOEXL3_GLM53F_PF_INT8` (0 to 4, default 0) runs prefill GEMMs on int8. It changes the output text. We measured
it on the frozen line and its parent, not on plugin 969a2bc. The build window of 969a2bc served `PF_INT8=4` once and
checked that it engaged. That window did not time it.

| Setting | Prefill tok/s | TTFT s | Prompt-row NLL change vs exact, nats (se) | HumanEval, last block |
|---|---|---|---|---|
| exact (0) | 1,239 | 13.24 | 0 | 152/164 |
| int8=1 | 1,380 | 11.89 | +0.00133 (0.00041) | 152/164 |
| int8=3 | 1,554 | 10.56 | +0.00094 (0.00048) | 155/164 |
| int8=4 | 1,665 | 9.86 | +0.00037 (0.00052) | 152/164 |

- The speed columns come from three card-sampling A/Bs on the frozen line, 2026-11-12. The exact row is the A arm of
  the int8=4 A/B. Card decode was 39.6 to 42.2 tok/s across all runs. The arms wrote different text, so their decode
  numbers do not compare.
- The NLL and HumanEval columns come from the 64-entry control, 2026-11-10 to 2026-11-12. The NLL column covers the
  prompt rows of 32 documents (521,504 rows). A rounding-only change (`DSA_ONEP=0`) moves NLL by +0.00086 (0.00037).
  Each int8 level is inside two standard errors of that change.
- int8=2 has speed in the sweep below, but its quality is not measured.

### Context sweep, 1K to 256K (frozen line, 2026-11-12)

The sweep ran on the frozen line, not on plugin 969a2bc. It has 9 prompt sizes and 6 configurations, with 2 runs each
(108 runs). The prompts are the same code task built to each size, so the 16K prompt differs from the standard test's.
The four rows below are means of 2 greedy runs. The full tables are in the blocks below.

| Prompt tokens | Exact prefill tok/s | int8=4 prefill tok/s | Exact decode tok/s | Exact TTFT s | Exact peak GiB |
|---|---|---|---|---|---|
| 1K | 285.3 | 288.1 | 44.10 | 3.60 | 98.1 |
| 16K | 1,236 | 1,669 | 40.90 | 13.27 | 98.5 |
| 128K | 1,077 | 1,414 | 36.70 | 121.8 | 97.2 |
| 256K | 1,012 | 1,302 | 30.70 | 259.0 | 94.2 |

Charts show prefill, decode, TTFT and wall time by size:

![prefill by context size](https://raw.githubusercontent.com/davidtai/plugins/9ffd2062ce43958ebe2b746c9bfa730c79027392/turboexl3_glm53f/prefill.png)

![decode by context size](https://raw.githubusercontent.com/davidtai/plugins/9ffd2062ce43958ebe2b746c9bfa730c79027392/turboexl3_glm53f/decode.png)

![TTFT by context size](https://raw.githubusercontent.com/davidtai/plugins/9ffd2062ce43958ebe2b746c9bfa730c79027392/turboexl3_glm53f/ttft.png)

![wall time by context size](https://raw.githubusercontent.com/davidtai/plugins/9ffd2062ce43958ebe2b746c9bfa730c79027392/turboexl3_glm53f/wall.png)

<details>
<summary>Per-run tables: g53g2p2, g53g2s1, g53g2h1 (2026-11-12)</summary>

Every arm except g53g2p2's A (the frozen line, plugin ac87fe1) ran plugin 969a2bc with turboexl3 95e98dc. In every
run, 608 of 832 drafted tokens were accepted: MTP 486 of 702 and prompt lookup 122 of 130. The texts were the same, so
the drafts were the same.

**g53g2p2, 2026-11-12 06:16-06:32 UTC.** A = frozen line. B = this plugin at its defaults on kit6main.

| run | arm | seed | prefill tok/s | decode tok/s | TTFT s | wall s | spec hit %, MTP + lookup | peak GB | text |
|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 1234 | 1,235 | 40.70 | 13.28 | 38.48 | 73.1 | 105.1 | `c3c346222307` |
| 2 | B | 1234 | 1,239 | 40.60 | 13.24 | 38.56 | 73.1 | 105.1 | `c3c346222307` |
| 3 | B | 4321 | 1,241 | 40.60 | 13.21 | 38.52 | 73.1 | 105.1 | `c3c346222307` |
| 4 | A | 4321 | 1,237 | 41.30 | 13.25 | 38.12 | 73.1 | 105.1 | `c3c346222307` |
| 5 | B | 1234 | 1,242 | 40.50 | 13.21 | 38.54 | 73.1 | 105.1 | `c3c346222307` |
| 6 | A | 1234 | 1,239 | 41.30 | 13.24 | 38.08 | 73.1 | 105.1 | `c3c346222307` |
| 7 | A | 4321 | 1,240 | 41.30 | 13.23 | 38.10 | 73.1 | 105.1 | `c3c346222307` |
| 8 | B | 4321 | 1,239 | 40.60 | 13.24 | 38.54 | 73.1 | 105.1 | `c3c346222307` |

| metric | A mean | B mean | B vs A | block 1 | block 2 |
|---|---|---|---|---|---|
| prefill tok/s (higher is better) | 1,238 | 1,240 | +0.2% | +0.3% | +0.1% |
| decode tok/s (higher is better) | 41.15 | 40.58 | -1.4% | -1.0% | -1.8% |
| TTFT s (lower is better) | 13.25 | 13.23 | -0.2% | -0.3% | -0.1% |
| wall s (lower is better) | 38.20 | 38.54 | +0.9% | +0.6% | +1.2% |
| peak GB (lower is better) | 105.1 | 105.1 | 0.0% | 0.0% | 0.0% |

**g53g2s1, 2026-11-12 08:25-08:44 UTC.** A = this plugin on kit6main. B = this plugin on kit7 (seams on).

| run | arm | seed | prefill tok/s | decode tok/s | TTFT s | wall s | spec hit %, MTP + lookup | peak GB | text |
|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 1234 | 1,238 | 40.70 | 13.25 | 38.50 | 73.1 | 105.1 | `c3c346222307` |
| 2 | B | 1234 | 1,239 | 41.30 | 13.24 | 38.12 | 73.1 | 105.5 | `c3c346222307` |
| 3 | B | 4321 | 1,243 | 41.20 | 13.20 | 38.13 | 73.1 | 105.1 | `c3c346222307` |
| 4 | A | 4321 | 1,239 | 40.80 | 13.24 | 38.41 | 73.1 | 105.5 | `c3c346222307` |
| 5 | B | 1234 | 1,238 | 41.30 | 13.25 | 38.09 | 73.1 | 105.5 | `c3c346222307` |
| 6 | A | 1234 | 1,241 | 40.70 | 13.21 | 38.43 | 73.1 | 105.5 | `c3c346222307` |
| 7 | A | 4321 | 1,240 | 40.80 | 13.23 | 38.38 | 73.1 | 105.5 | `c3c346222307` |
| 8 | B | 4321 | 1,242 | 41.00 | 13.21 | 38.27 | 73.1 | 105.5 | `c3c346222307` |

| metric | A mean | B mean | B vs A | block 1 | block 2 |
|---|---|---|---|---|---|
| prefill tok/s (higher is better) | 1,240 | 1,240 | +0.1% | +0.2% | -0.1% |
| decode tok/s (higher is better) | 40.75 | 41.20 | +1.1% | +1.2% | +1.0% |
| TTFT s (lower is better) | 13.23 | 13.23 | -0.1% | -0.2% | +0.1% |
| wall s (lower is better) | 38.43 | 38.15 | -0.7% | -0.9% | -0.6% |
| peak GB (lower is better) | 105.4 | 105.4 | 0.0% | 0.0% | 0.0% |

**g53g2h1, the `DEMAND_HELP` A/B, 2026-11-12 05:56-06:13 UTC.** A = this plugin on kit6main. B = A +
`TURBOEXL3_GLM53F_DEMAND_HELP=2`. Rejected.

| run | arm | seed | prefill tok/s | decode tok/s | TTFT s | wall s | spec hit %, MTP + lookup | peak GB | text |
|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 1234 | 1,238 | 40.60 | 13.24 | 38.56 | 73.1 | 105.1 | `c3c346222307` |
| 2 | B | 1234 | 1,243 | 39.30 | 13.19 | 39.30 | 73.1 | 105.1 | `c3c346222307` |
| 3 | B | 4321 | 1,238 | 39.30 | 13.25 | 39.36 | 73.1 | 105.1 | `c3c346222307` |
| 4 | A | 4321 | 1,238 | 40.60 | 13.25 | 38.53 | 73.1 | 105.1 | `c3c346222307` |
| 5 | B | 1234 | 1,239 | 39.30 | 13.24 | 39.35 | 73.1 | 105.1 | `c3c346222307` |
| 6 | A | 1234 | 1,243 | 40.50 | 13.20 | 38.54 | 73.1 | 105.1 | `c3c346222307` |
| 7 | A | 4321 | 1,239 | 40.60 | 13.24 | 38.54 | 73.1 | 105.1 | `c3c346222307` |
| 8 | B | 4321 | 1,243 | 39.30 | 13.20 | 39.31 | 73.1 | 105.1 | `c3c346222307` |

| metric | A mean | B mean | B vs A | block 1 | block 2 |
|---|---|---|---|---|---|
| prefill tok/s (higher is better) | 1,239 | 1,241 | +0.1% | +0.2% | -0.0% |
| decode tok/s (higher is better) | 40.58 | 39.30 | -3.1% | -3.2% | -3.1% |
| TTFT s (lower is better) | 13.23 | 13.22 | -0.1% | -0.2% | +0.0% |
| wall s (lower is better) | 38.54 | 39.33 | +2.0% | +2.0% | +2.0% |
| peak GB (lower is better) | 105.1 | 105.1 | 0.0% | 0.0% | 0.0% |

</details>

<details>
<summary>Expert-stream counters behind the decode differences (means of 4 runs, 2026-11-12)</summary>

The plugin's `[stream-stats]` lines give these counts. They show where the seams change the work.

| A/B | arm | prompt evictions | decode misses | decode reads (preadv) | decode GB read | adopt copy ms |
|---|---|---|---|---|---|---|
| g53g2p2 | A frozen line | 0 | 11,105 | 19,942 | 105.6 | 83.9 |
| g53g2p2 | B kit6main | 895 | 11,369 | 14,931 | 108.1 | 2,481 |
| g53g2s1 | A kit6main | 843 | 11,150 | 14,621 | 106.0 | 2,462 |
| g53g2s1 | B kit7 | 0 | 10,627 | 19,151 | 101.1 | 1,905 |

Our reading of these counters follows. Without `PROMPT_KEEP`, the prompt evicts residents that decode then reads
again. Without `ADOPT_PIECES`, one worker copies each adopted read-ahead record. With the seams, prompt evictions go to
0 and decode reads 5.0 GB less.

</details>

<details>
<summary>Card-sampling A/Bs, exact vs int8=1, 3 and 4 (frozen line, 2026-11-12)</summary>

Card sampling: temperature 1.0, top_p 0.95. A = frozen line (`PF_INT8=0`). B = A + `PF_INT8=N`. The arms wrote
different text. The spec hit rate was not recorded for these runs.

**g53crd1b, 00:39-00:56 UTC, int8=1.**

| run | arm | seed | prefill tok/s | decode tok/s | TTFT s | wall s | spec hit %, MTP + lookup | peak GB |
|---|---|---|---|---|---|---|---|---|
| 1 | A | 1234 | 1,236 | 40.00 | 13.27 | 38.95 | Not recorded | 105.1 |
| 2 | B | 1234 | 1,381 | 41.60 | 11.88 | 36.54 | Not recorded | 105.2 |
| 3 | B | 4321 | 1,377 | 42.20 | 11.91 | 36.27 | Not recorded | 105.2 |
| 4 | A | 4321 | 1,241 | 40.20 | 13.21 | 38.78 | Not recorded | 105.2 |
| 5 | B | 1234 | 1,379 | 41.60 | 11.90 | 36.55 | Not recorded | 105.2 |
| 6 | A | 1234 | 1,238 | 40.00 | 13.24 | 38.92 | Not recorded | 105.1 |
| 7 | A | 4321 | 1,237 | 40.20 | 13.26 | 38.79 | Not recorded | 105.2 |
| 8 | B | 4321 | 1,382 | 42.20 | 11.87 | 36.22 | Not recorded | 105.2 |

Means A / B: prefill 1,238 / 1,380 tok/s (+11.4%; blocks +11.3% and +11.5%); TTFT 13.25 / 11.89 s; decode
40.10 / 41.90 tok/s (different text).

**g53crd3b, 00:59-01:16 UTC, int8=3.**

| run | arm | seed | prefill tok/s | decode tok/s | TTFT s | wall s | spec hit %, MTP + lookup | peak GB |
|---|---|---|---|---|---|---|---|---|
| 1 | A | 1234 | 1,237 | 40.00 | 13.26 | 38.95 | Not recorded | 105.1 |
| 2 | B | 1234 | 1,553 | 39.60 | 10.56 | 36.48 | Not recorded | 105.1 |
| 3 | B | 4321 | 1,554 | 40.40 | 10.56 | 35.95 | Not recorded | 104.7 |
| 4 | A | 4321 | 1,235 | 40.30 | 13.28 | 38.78 | Not recorded | 105.2 |
| 5 | B | 1234 | 1,555 | 39.60 | 10.55 | 36.45 | Not recorded | 105.1 |
| 6 | A | 1234 | 1,235 | 40.00 | 13.28 | 38.92 | Not recorded | 105.1 |
| 7 | A | 4321 | 1,237 | 40.20 | 13.26 | 38.80 | Not recorded | 105.2 |
| 8 | B | 4321 | 1,554 | 40.60 | 10.56 | 35.87 | Not recorded | 105.1 |

Means A / B: prefill 1,236 / 1,554 tok/s (+25.7%); TTFT 13.27 / 10.56 s; decode 40.13 / 40.05 tok/s
(different text).

**g53crd4b, 01:18-01:34 UTC, int8=4.**

| run | arm | seed | prefill tok/s | decode tok/s | TTFT s | wall s | spec hit %, MTP + lookup | peak GB |
|---|---|---|---|---|---|---|---|---|
| 1 | A | 1234 | 1,240 | 39.90 | 13.22 | 38.93 | Not recorded | 105.1 |
| 2 | B | 1234 | 1,665 | 40.20 | 9.86 | 35.39 | Not recorded | 105.1 |
| 3 | B | 4321 | 1,667 | 41.20 | 9.84 | 34.73 | Not recorded | 105.1 |
| 4 | A | 4321 | 1,239 | 40.20 | 13.24 | 38.81 | Not recorded | 105.2 |
| 5 | B | 1234 | 1,662 | 40.20 | 9.87 | 35.41 | Not recorded | 105.1 |
| 6 | A | 1234 | 1,240 | 40.00 | 13.23 | 38.91 | Not recorded | 105.1 |
| 7 | A | 4321 | 1,235 | 40.20 | 13.28 | 38.82 | Not recorded | 105.2 |
| 8 | B | 4321 | 1,665 | 41.30 | 9.85 | 34.70 | Not recorded | 105.1 |

Means A / B: prefill 1,239 / 1,665 tok/s (+34.4%); TTFT 13.24 / 9.86 s; decode 40.08 / 40.73 tok/s
(different text).

</details>

<details>
<summary>int8 quality: 32 documents, NLL and KL (64-entry control, 2026-11-10)</summary>

One server per arm scored the same 32 documents (chat, code and prose). E is the exact configuration. I1, I3 and I4
are E with `PF_INT8=1`, 3 and 4. F is E with `DSA_ONEP=0`, a change that only moves rounding; we use it as the noise
floor. NLL rows are the prompt rows (521,504); E's NLL is 1.045. Each cell is the pooled difference (block se).

| category | I4 - E | I3 - E | I1 - E | F - E |
|---|---|---|---|---|
| all | +0.00037 (0.00052) | +0.00094 (0.00048) | +0.00133 (0.00041) | +0.00086 (0.00037) |
| code | +0.00102 (0.00039) | +0.00093 (0.00040) | +0.00114 (0.00040) | +0.00061 (0.00031) |
| prose | -0.00027 (0.00063) | -0.00000 (0.00061) | -0.00046 (0.00056) | -0.00006 (0.00050) |
| chat | +0.00023 (0.00147) | +0.00190 (0.00135) | +0.00338 (0.00109) | +0.00211 (0.00102) |

KL over each row's top 64 log-probabilities, against E: I1 0.01441, I3 0.01716, I4 0.01817, F 0.009430. Before the
run we set a KL bar of 0.00574 nats, 10% of a 0.0574 value from turboderp's own trace. Every arm, the floor F
included, is above that bar, so the bar does not separate the int8 levels. I1, I3 and I4 each lie inside F - E plus or
minus two standard errors in every category. int8=2 was not scored.

</details>

<details>
<summary>HumanEval (64-entry control, 2026-11-11 to 2026-11-12)</summary>

164 problems, pass@1, greedy, thinking off, max_tokens 1,024, openai/human-eval b9453e8f. "Last block" scores the
reply's last fenced code block; "first block" scores its first. The arms are those of the quality block. KX is E with
the stock KDA prompt route (`KDA_PERCORE`, `KDA_CHUNK`, `KDA_FOLD` and `KDA_PIPE` off).

| arm | last block | first block | replies identical to E | completion tokens |
|---|---|---|---|---|
| E | 152 | 145 | not applicable | 44,642 |
| F | 152 | 145 | 164 | 44,642 |
| I1 | 152 | 150 | 76 | 42,042 |
| I3 | 155 | 147 | 68 | 41,714 |
| I4 | 152 | 147 | 73 | 43,333 |
| KX | 153 | 150 | 75 | 40,600 |

Some greedy replies are anomalous. They restart with "Wait", split their reasoning, or run to the 1,024-token cap.
For example, HumanEval/147 has stray words inside its code. Six problems are anomalous in E, F, I1, I3 and KX, and
five of them also in I4. We read these as near-tie token choices. We have not found their root cause.

</details>

<details>
<summary>Context sweep: every size and configuration (frozen line, g53fsw1-4, 2026-11-12)</summary>

Windows g53fsw1 (1K-4K), g53fsw2 (8K-32K), g53fsw3 (64K-128K) and g53fsw4 (256K). max_tokens 1,024; every run wrote
to the cap. Each run used a fresh server with a context of the prompt plus 1,024. Before each run, the harness evicted
the pack's files from the page cache and waited for the 40 C cool gate. Then it sent one request. "exact card" uses
card sampling. Peak is the server's peak footprint in GiB. Each cell is the mean of 2 runs (seeds 1234 and 4321).

Prefill tok/s:

| size | exact greedy | exact card | int8=1 | int8=2 | int8=3 | int8=4 |
|---|---|---|---|---|---|---|
| 1K | 285.3 | 284.2 | 287.3 | 289.2 | 247.1 | 288.1 |
| 2K | 524.3 | 523.9 | 543.0 | 547.5 | 543.1 | 545.5 |
| 4K | 884.6 | 884.7 | 942.0 | 959.5 | 947.7 | 958.5 |
| 8K | 1,225 | 1,223 | 1,360 | 1,398 | 1,436 | 1,489 |
| 16K | 1,236 | 1,236 | 1,381 | 1,423 | 1,546 | 1,669 |
| 32K | 1,171 | 1,172 | 1,303 | 1,344 | 1,461 | 1,573 |
| 64K | 1,123 | 1,123 | 1,242 | 1,278 | 1,391 | 1,495 |
| 128K | 1,077 | 1,074 | 1,182 | 1,215 | 1,319 | 1,414 |
| 256K | 1,012 | 1,012 | 1,102 | 1,129 | 1,220 | 1,302 |

Decode tok/s:

| size | exact greedy | exact card | int8=1 | int8=2 | int8=3 | int8=4 |
|---|---|---|---|---|---|---|
| 1K | 44.10 | 42.10 | 44.70 | 44.10 | 43.30 | 43.80 |
| 2K | 41.90 | 41.70 | 41.85 | 42.90 | 41.85 | 43.05 |
| 4K | 43.55 | 42.60 | 42.40 | 43.85 | 42.90 | 42.15 |
| 8K | 42.10 | 43.30 | 40.70 | 43.05 | 40.45 | 41.85 |
| 16K | 40.90 | 41.40 | 43.45 | 43.00 | 42.90 | 40.85 |
| 32K | 42.10 | 41.10 | 40.45 | 40.20 | 40.90 | 41.80 |
| 64K | 39.95 | 38.95 | 39.25 | 38.80 | 39.55 | 40.00 |
| 128K | 36.70 | 36.20 | 35.80 | 35.55 | 35.55 | 35.55 |
| 256K | 30.70 | 29.80 | 30.20 | 30.25 | 30.30 | 30.70 |

TTFT s:

| size | exact greedy | exact card | int8=1 | int8=2 | int8=3 | int8=4 |
|---|---|---|---|---|---|---|
| 1K | 3.60 | 3.62 | 3.57 | 3.55 | 4.26 | 3.56 |
| 2K | 3.92 | 3.92 | 3.78 | 3.75 | 3.78 | 3.76 |
| 4K | 4.64 | 4.63 | 4.36 | 4.28 | 4.33 | 4.29 |
| 8K | 6.70 | 6.71 | 6.04 | 5.87 | 5.71 | 5.51 |
| 16K | 13.27 | 13.27 | 11.88 | 11.52 | 10.61 | 9.83 |
| 32K | 28.01 | 27.98 | 25.17 | 24.39 | 22.45 | 20.85 |
| 64K | 58.38 | 58.39 | 52.77 | 51.31 | 47.12 | 43.84 |
| 128K | 121.8 | 122.1 | 111.0 | 107.9 | 99.40 | 92.74 |
| 256K | 259.0 | 259.0 | 238.1 | 232.2 | 214.9 | 201.4 |

Wall time s:

| size | exact greedy | exact card | int8=1 | int8=2 | int8=3 | int8=4 |
|---|---|---|---|---|---|---|
| 1K | 26.91 | 28.05 | 26.55 | 26.82 | 27.98 | 27.01 |
| 2K | 28.40 | 28.54 | 28.34 | 27.69 | 28.34 | 27.63 |
| 4K | 28.21 | 28.74 | 28.55 | 27.69 | 28.27 | 28.65 |
| 8K | 31.09 | 30.44 | 31.27 | 29.73 | 31.09 | 30.02 |
| 16K | 38.37 | 38.05 | 35.50 | 35.41 | 34.55 | 34.96 |
| 32K | 52.38 | 52.98 | 50.55 | 49.92 | 47.56 | 45.41 |
| 64K | 84.09 | 84.78 | 78.97 | 77.79 | 73.12 | 69.55 |
| 128K | 149.8 | 150.5 | 139.6 | 136.9 | 128.3 | 121.7 |
| 256K | 292.5 | 293.5 | 272.1 | 266.2 | 248.8 | 234.9 |

Spec hit %, MTP only:

| size | exact greedy | exact card | int8=1 | int8=2 | int8=3 | int8=4 |
|---|---|---|---|---|---|---|
| 1K | 80.3 | 79.8 | 81.5 | 75.4 | 71.9 | 79.4 |
| 2K | 77.3 | 74.5 | 78.8 | 74.3 | 75.1 | 77.2 |
| 4K | 81.5 | 79.0 | 73.8 | 81.4 | 75.3 | 73.2 |
| 8K | 73.8 | 79.1 | 70.7 | 81.5 | 69.5 | 75.1 |
| 16K | 68.9 | 77.8 | 81.5 | 79.5 | 78.4 | 69.1 |
| 32K | 79.8 | 78.5 | 72.1 | 74.4 | 74.9 | 80.7 |
| 64K | 81.8 | 75.5 | 79.4 | 71.0 | 79.9 | 81.8 |
| 128K | 79.4 | 77.7 | 76.9 | 73.6 | 76.9 | 75.5 |
| 256K | 78.4 | 76.9 | 80.4 | 73.1 | 78.4 | 76.5 |

Peak GiB (server):

| size | exact greedy | exact card | int8=1 | int8=2 | int8=3 | int8=4 |
|---|---|---|---|---|---|---|
| 1K | 98.1 | 98.1 | 97.9 | 98.2 | 98.2 | 98.1 |
| 2K | 98.2 | 98.2 | 98.2 | 98.2 | 98.2 | 98.2 |
| 4K | 98.3 | 98.2 | 98.2 | 98.2 | 98.3 | 98.0 |
| 8K | 98.3 | 98.3 | 98.3 | 98.3 | 98.3 | 98.3 |
| 16K | 98.5 | 98.3 | 98.4 | 98.3 | 98.5 | 98.5 |
| 32K | 98.3 | 98.3 | 98.4 | 98.3 | 98.3 | 98.3 |
| 64K | 98.6 | 98.5 | 98.3 | 98.7 | 98.7 | 98.3 |
| 128K | 97.2 | 97.2 | 97.0 | 97.2 | 97.2 | 97.2 |
| 256K | 94.2 | 94.2 | 94.2 | 94.2 | 94.2 | 94.2 |

Every greedy configuration wrote the same text with both seeds. One run is slow: int8=3 at 1K, seed 4321, prefilled at
206.4 tok/s against 287.9 for seed 1234 with the same text. Both runs read the same 49.25 GB of experts; the slow one
read them at about 9.6 GB/s instead of 13.4 GB/s. The 1K int8=3 means include it.

Memory plan. The memory plan keeps the machine's non-file memory at or under the 110 GiB GPU ceiling. The peak is in
the decode phase at every size. From 64K up, the plan leaves room unused: 3.1, 4.6 and 7.5 GiB in the decode phase at
64K, 128K and 256K.

| size | rows a layer, prompt / decode | server peak GiB | box peak, prompt phase GiB | box peak, decode phase GiB |
|---|---|---|---|---|
| 1K | 185 / 221 (one run 185 / 220) | 98.2 | 97.9 | 108.8 |
| 2K | 185 / 221 | 98.2 | 98.8 | 109.1 |
| 4K | 185 / 221 (one run 185 / 220) | 98.3 | 99.7 | 108.9 |
| 8K | 185 / 221 | 98.3 | 102.0 | 109.2 |
| 16K | 185 / 221 (two runs 184 / 220) | 98.5 | 107.3 | 109.5 |
| 32K | 181 / 219 | 98.4 | 105.9 | 109.6 |
| 64K | 174 / 215 | 98.7 | 104.4 | 106.9 |
| 128K | 158 / 206 | 97.2 | 100.0 | 105.4 |
| 256K | 127 / 189 | 94.2 | 92.1 | 102.5 |

The per-run table of all 108 runs (six columns and each text's sha256) is in the sweep report
([final-sweep-report.md](https://github.com/davidtai/plugins/blob/9ffd2062ce43958ebe2b746c9bfa730c79027392/turboexl3_glm53f/final-sweep-report.md), with [all_runs.tsv](https://github.com/davidtai/plugins/blob/9ffd2062ce43958ebe2b746c9bfa730c79027392/turboexl3_glm53f/all_runs.tsv)).

</details>

<details>
<summary>Tests (plugin 969a2bc, 2026-11-12)</summary>

`zig build test` runs with no MLX and no model. Its tests, one file to a line:

- src/stream_room.zig: the prompt bill against mlx-serve's own GLM bill test, the rows that its room leaves, and
  `BILL_FIT`'s bases, plan and context limit.
- src/banked_rows_test.zig: the slot banks' array rows under the stream's page floor, and the fit of the math to the
  code's rows.
- src/stream_options_test.zig: the stream's construction checks over the two code widths of every SC pack, and
  `BILL_FIT`'s read pool against the kit stream's own. It also checks `SEED_DECODE`'s profile over a pack's files and
  the seam switches' refusals on a kit without their seams. `zig build test -Dglm53f-packs=<pack dir>,...` adds the
  packs' own banks.
- src/row_plan_test.zig: the switches' plans on the bank's own split, and the bytes of its slot arrays.
- src/kda_prompt.zig: the layouts of the KDA prompt kernels and the per-core plan.
- src/bank_layers.zig: each layer's row bytes from a manifest.
- src/stream_stats.zig: the `[stream-stats]` lines by phase.
- src/split_test.zig: the split prompt forward's refusals, spans, field size check and routed arm.
- src/frac_test.zig: the tile gate. turboexl3's reference decode of 84 real routed-expert tiles equals exllamav3's rule
  bit for bit.
- The claim, the parse and its refusals, the MTP manifest's geometry and refusals, and the loader's dtype rule.

The device tests build against the server's MLX and run on the GPU. In this section, "kernel" means a Metal kernel;
the plugin README calls these kernels "texts".

- `zig build gpu-test` builds turboexl3's own GLM-5.3-Flash tests (its `testModule` root filtered to `glm53f`) over
  this plugin's `sdk`, with src/t3_test_host.zig as `mlx_host`. `--fork=<turboexl3 checkout>` builds them from a tree
  other than the pinned one.
- `zig build kda-gpu-test` builds src/kda_gpu_test.zig as a test rooted at plugin.zig with this plugin's modules. Under
  `TURBOEXL3_GLM53F_KDA_GPU=1`, it runs kda.zig's forward with each KDA prompt kernel on, against the op chain. It
  prints each output's differing elements (`TURBOEXL3_GLM53F_KDA_BYTES` lines, all 0 to pass).
- The router, small-op and draft tests (commands at the top of this README) and the DSA test (`zig build
  dsa-gpu-test`) build the same way.

Build window g53g2f1 (2026-11-12 04:55-05:00 UTC; plugin 969a2bc, kit6main, turboexl3 95e98dc) ran these:

| Test | Result |
|---|---|
| Build steps, MLX-free tests | 9/9 steps, 101/101 tests |
| Options in the dylib manifest | 67 (60 bool, 2 number, 5 text) |
| turboexl3 GLM byte tests on the device (wave3) | 780/780 |
| KDA prompt and verify kernels on the device | 470/470 |
| DSA kernels on the device | 116/116 |
| Router; router prediction; one-launch router, two forms | 6/6; 3/3; 9/9 and 2/2 |
| MoE combine | 16/16 |
| Merged small ops, launch trims, MTP draft step 0 (`DRAFT_S0`) | 83 lines and 0 fails, 38/38, 18/18 |
| turboexl3 routed int8 agreement (int8=1 and 2) | 24/24 |
| turboexl3 trunk int8: agree, off, split | 44/44, 44/44, 44/44 |
| `PF_INT8=4` served once (not timed) | loads, engages routed and trunk int8, replies |

Build window g53g2k7f (kit7) ran the build and the MLX-free tests: 9/9 steps, 101/101 tests, 67 options. It ran no
device tests.

</details>

<details>
<summary>Long-context check before the sweep (g53cal2, frozen line, 2026-11-12 00:05-00:37 UTC)</summary>

One greedy run a size, seed 1234. Each cell is exact / int8=4. The sweep's exact greedy and int8=4 texts at 32K to 256K
match the texts of these runs.

| size | prefill tok/s | decode tok/s | TTFT s | wall s | spec hit %, MTP only | peak GB |
|---|---|---|---|---|---|---|
| 32K | 1,172 / 1,573 | 42.20 / 41.80 | 27.97 / 20.85 | 52.31 / 45.41 | 79.8 / 80.7 | 105.6 / 105.6 |
| 64K | 1,121 / 1,494 | 39.90 / 39.90 | 58.49 / 43.88 | 84.23 / 69.64 | 81.8 / 81.8 | 105.9 / 105.6 |
| 128K | 1,071 / 1,413 | 36.60 / 36.00 | 122.4 / 92.80 | 150.5 / 121.3 | 79.4 / 75.5 | 104.3 / 104.4 |
| 256K | 1,009 / 1,300 | 30.60 / 30.80 | 259.8 / 201.8 | 293.4 / 235.1 | 78.4 / 76.5 | 101.2 / 101.1 |

</details>

### Not measured

- The other three packs (sc_2.10bpw, sc_2.60bpw, sc_4.10bpw). At the defaults the memory plan refuses them; they need
  `BILL_FIT=0` and `FIT_INUSE=0`.
- This plugin on mlx-serve main 9a173773 without the two draft PRs.
- The plugin's int8 prefill speed, card sampling and context sweep. These ran on the frozen line.
- int8=2 quality and HumanEval.
- Any Mac other than one M5 Max with 128 GB, and any memory size other than 128 GB.
- More than one request at a time.
- A diff against `--no-plugins`. The built-in glm5_next refuses these packs by name, so there is no built-in output to
  compare. The same-text check against the frozen line stands in for it.
- The head commit served. It is built and tested (above), but not served or measured.
