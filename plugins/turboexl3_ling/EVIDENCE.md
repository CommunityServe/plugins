# turboexl3_ling: evidence

This file holds the evidence behind the turboexl3_ling [README](README.md). It gives the receipts, the A/B tables with
each run, the gate logs' results and the parity record. It also gives the bill's derivations, RQ's plans, the KV growth
replay, the rules table and the gaps for a merge. We, the plugin's maintainers (davidtai, plugin.json), wrote it on
2026-11-10 and 2026-11-11, and times are in UTC. The source is `plugins/turboexl3_ling` on our unpushed branch
`turboexl3_ling-rq` of CommunityServe/plugins. Since 08:08:00Z on 2026-11-11 we serve dylib 4856f028, built from commit
fb966cc with turboexl3 4bf2df8 and a kit from mlx-serve 66aad949. Before that we served 01ca9769, built from bd00b1a.
Read the README first. It says what the plugin serves and how to run it.

How to read this file:

- Code is cited by symbol and commit, for example src/bill.zig `echoIdle` at fb966cc. A plugin path with no commit is at
  fb966cc, the served dylib's commit. A citation of 01ca9769's code names bd00b1a. An mlx-serve path with no commit is
  at 66aad949, and an MLX path is at c69303b0.
- A run record keeps its path. A line number points into a run record, such as a log or a script kept as it ran. A run
  record does not change.
- A quoted section name, such as "Memory", names a section of this file. "README, ..." names a section of the README,
  and a step, such as step 6, is one of the README's "Install and serve" steps.
- Each number has a kind. It is measured in our runs, calculated from the code, set (a setting or the model's size), or
  fitted to measured peaks (MEM_REFIT). "Inferred" marks a cause that no record states. "Receipts" gives each number's
  record. Our measured records are on our Mac and are not published.

## Contents

- [Builds](#builds): the dylibs that the README and this file name.
- [The copy of mlx-serve's code](#the-copy-of-mlx-serves-code): the layer-major pass and the plugin's parts.
- [Memory](#memory): the bill's terms, RQ's plans for each request, the 32K checks, the KV past the bill's term, the
  quantized KV and the prompt log-probability limit.
- [The served stack](#the-served-stack): the optimizations, with KDA_GATE_EPI, KDA_NORM_PREP and EMBED_ROWS in detail.
- [Logs: the run-path lines](#logs-the-run-path-lines): which paths a request names.
- [Tests](#tests): the gates, the test-only commits, the GPU byte tests and the restore check.
- [Parity](#parity): the parity test, its statistics and the changes since.
- [Measured](#measured): every run of the standard A/B tests, the 32K check and the context ladder, and how we measured.
- [Gaps for a merge](#gaps-for-a-merge): what a merge needs, the repo's rules one by one, and the smaller gaps.
- [Receipts](#receipts): the record of each number.
- [Terms](#terms)

## Builds

Gates are our build and test run of a dylib before we serve it ("Tests"). The 32K check is our memory check ("Memory").
A standard A/B test compares two dylibs, arm A and arm B, on our benchmark workload ("Measured"). Builds of the same
code can differ in hash. The served code is fb966cc on our branch turboexl3_ling-rq. That branch starts at e44e00d on
turboexl3_ling-l5, whose code is bd00b1a's with one comment changed in src/bill.zig (186eae1). Its commits from 1875c04
to fb966cc add RQ and the KV-quant warning ("RQ", "The quantized KV"). The commits after fb966cc on turboexl3_ling-rq
change only the README, this file, build.zig.zon's `.paths` and a comment in scripts/test_reach.py.

| Dylib | Commit | Role | Measured |
|---|---|---|---|
| 4856f028 | fb966cc | served from 08:08:00Z on 2026-11-11; 89be02b1's code with RQ built in and its option removed (44ba83b, fb966cc) | gates, 32K check, arm A of SPEC1_BUSY's A/B test |
| 89be02b1 | f6f3415 | RQ as an option, `TURBOEXL3_LING_RQ=1` | gates, RQ's safety test, arm B of RQ's A/B test, which kept it |
| 01ca9769 | bd00b1a | served from 20:59:54Z on 2026-11-10 to 08:08:00Z on 2026-11-11; 7f053407's code with KDA_NORM_PREP always on and its option removed (32f7c02, a6ea7ee) | gates, 32K check, the final context ladder, arm A of the PRED_F16, SPEC0_IDLE and RQ A/B tests |
| 63668c19 | 1dffdcd | served from 18:19:14 to 20:59:54Z | gates, arm A of KDA_NORM_PREP's A/B |
| 7f053407 | b0079f1 | KDA_NORM_PREP as an option, `TURBOEXL3_LING_KDA_NORM_PREP=1` | gates, arm B of KDA_NORM_PREP's A/B |
| 583523d4 | 4c13ec7 | served from 17:44:49 to 18:19:14Z; its 15 Mach-O sections are byte for byte 63668c19's, so its 32K check also applies to 63668c19 | gates, 32K check |

The README and this file also name these other dylibs:

- 090d2b0f (165d8ca) and 38f84c44 (2773ee6), the arms of EMBED_ROWS' A/B;
- 7e446929 (1a16e40) and 73db9cb6 (859cd93), the arms of KDA_GATE_EPI's A/B;
- c8d54a83 (0e1f5c0) and fdd4e0d8 (0ebec77), the arms of the re-fit's A/B, with c8d54a83 the earlier context ladder's
  build;
- 901a8e4a, the check of an earlier re-fit (6bb90b8) that we did not keep;
- 7b1843f1, the build that the parity test measured on 2026-11-08;
- 97c37406 (on turboexl3_ling-m2, with turboexl3 a0176ae), arm B of PRED_F16's A/B, which we rejected;
- 965ef461 (9788a4f, on turboexl3_ling-spec0 from 186eae1), arm B of SPEC0_IDLE's A/B, which we rejected;
- 23731721 (859f3d6, on turboexl3_ling-spec1 from fb966cc), arm B of SPEC1_BUSY's A/B, which we rejected.

The branch contains CommunityServe main cfc3704, so it builds against ddalcu/mlx-serve#1131's interfaces. The repo's
other three plugins, sushi_ds41, turboexl3_ds41 and turboexl3_qwen38, built against our kit from mlx-serve 66aad949 at
ecc4a1c (measured, `~/lingp/logs/portcs-cs1.log`).

## The copy of mlx-serve's code

The copy comes from mlx-serve commit bc6c5f2c, with this plugin's own files (overlay/) and four patches (patches/).
Since ddalcu/mlx-serve#1131, a quant plugin can give mlx-serve's own model its EXL3 weights and expert bank instead, as
turboexl3_ds41 does. That form would drop this plugin's Ling-specific paths, so we kept the copy.

For the whole-prompt pass, mlx-serve names the spans that its chunk loop would have run (`Forward.prompt_spans`), and
the layer-major pass reads them. It runs each operation whose bits depend on its batch of rows one span at a time, as
the chunk loop would (plugin.zig `spanMajor`). The code is plugin.zig `hostSpans` and `spanMajorOr`, and the comment on
`layer_major_rows`. The routed experts' bits do not depend on the call's width. So one call runs a routed layer's
experts over up to 16,384 prompt rows (token positions). A last part under 2,048 rows joins it, for up to 18,431
(plugin.zig `layer_major_rows`, `layer_major_tail_rows` and `callRows`). It reads each expert it needs once, not once
for each 4,096-token chunk. The call width is a constant (plugin.zig `layer_major_rows`), and the chunk width is step
6's `--prefill-chunk`.

- src/mlx_serve holds the copy. scripts/prepare.sh generates it, and git does not track it. It holds mlx-serve's own
  Ling model (plugins/bailing_hybrid, over plugins/glm5_next's trunk and KDA) and mlx-serve's expert streamer
  (sdk/weights/streamed, the module `mlx_stream`). Both come from the commit in the plugin's MLX_SERVE_REF (bc6c5f2c),
  which prepare.sh checks out in `.mlx-serve-src/`.
- prepare.sh copies the files that scripts/slice.txt and scripts/slice-stream.txt name, through
  scripts/sync-mlx-serve.sh. It then copies overlay/mlx_serve beside them and applies the patches. A build refuses a
  tree whose hash is not the one in scripts/mlx_serve.sha256 (scripts/check-mlx-serve.sh).
- overlay/ holds 17 files at mlx-serve paths, and the sync script never overwrites them. Two replace mlx-serve's files
  at those paths (mlx/kernels.zig and sdk/quant/exl3/format.zig), and 15 are new. Five of the new files are MLX 0.32.5's
  `sdpa_vector` Metal kernels, cut down (NOTICE). The other 10 new files and the two replacements are this plugin's own
  code.
- patches/ holds four patches, one for each area of the copy: ling-trunk.patch, ling-kda.patch, ling-mla.patch and
  ling-stream.patch. They are the only edits to the copies other than the sync script's own. NOTICE gives the copies'
  source.
- In the trunk, turboexl3's `linear` runs every exllamav3 projection (`<name>.trellis`, `.suh` and `.svh`, with its
  `mul1` checked). The output head is K6 (the config's `head_bits`). The query, key and value projections are
  exllamav3's one `qkv_proj`. The plugin casts the few plain f16 projections (b_proj, kv_b_proj and the MLA gate) to
  bf16 at load. The router keeps the pack's f16 weight, and each router call widens it to f32 (`router_dtype` fp32;
  src/ling_router.zig).
- The trunk and the MTP head held 14,734 MiB when the bank opened (measured on dylibs 01ca9769 and 4856f028, load logs).
  The embedding table is not resident (EMBED_ROWS, "The served stack").
- The bank holds experts at Ling's geometry: hidden 4096, width 1280, 512 experts, top-8, 68 layers from layer 4. The
  streamer schedules the reads, and turboexl3 runs every call (src/ling_math.zig). Decode calls run on `moeLingBanked`,
  a wide prompt call on `moeLingRows`, and any other routed call on `moeLing`. The bank reads each layer's K from the
  pack. Every routed layer is K 2.5 (40 halfwords a tile) in the 2.60bpw pack and K 2 (32) in the 2.10bpw pack (each
  pack's LING-REPACK-RECEIPT.json).
- The plugin reads only exllamav3's mul1 codebook, which turboderp's Ling packs use (config.zig `checkPack`).
- The converter runs on the CPU and does not requantize. Its plan step reads the branch's file list from huggingface.co.
  It writes each routed expert into `experts.bin` as one record, padded to a multiple of 4,096 bytes, with its codes and
  scales unchanged. It writes the other tensors into `*-nonexpert.safetensors` and `mtp-nonbank.safetensors`. We made
  our pack in the script's default layout, classic. The script in our converter clone has sha256 a135306e79949e13
  (calculated); the pack's receipt does not record the script's hash.
- Ling-3.1 clamps the SwiGLU of some layers (`expert_swiglu_limit_list` on layers 65-71,
  `share_expert_swiglu_limit_list` on 64-71, limit 10). The plugin computes min(silu(g), L) * clamp(u, -L, L),
  exllamav3's form. turboexl3's Ling kernels do this for the routed experts, and the plugin does it in f32 for the
  shared expert (src/clamps.zig, src/ling_exl3.zig).
- The MTP head (`model.layers.72`) always loads.

## Memory

The README's "Memory" gives the memory model in brief. This section gives each term's derivation, RQ's plans for each
request and the 32K checks. It then gives the KV's growth past the bill's term, the quantized KV, and the prompt
log-probability limit with its refusal test.

### The bill's terms and the plan's ceiling

The bill is the memory that the plugin asks mlx-serve to hold back from the bank (src/bill.zig). At load, the
prompt-pass bill has five terms:

| Term | MiB | Kind |
|---|---|---|
| The KV, the attention cache, for 32,768 tokens | 9,288 | calculated |
| `prompt_work`, the pass's working set | 2,048 | set |
| The layer-major pass's extra buffers | 4,797.7 | calculated |
| `free_ram`, the reserve for the rest of the Mac | 4,096 | set |
| MEM_REFIT, what the other terms miss: 989 for the pass and 544 for the server | 1,533 | fitted to measured peaks |
| Total | 21,763 | calculated |

mlx-serve takes its 8 GiB margin and this bill from the GPU wired limit of 112 GiB (step 6). That leaves the plugin
84,733 MiB, the room (calculated). In 01ca9769 and 4856f028, the trunk and the MTP head held 14,734 MiB of the room. The
streamer's read windows and staging take 1,245 MiB of it (calculated). Each slot of the bank holds one expert, and a
slot row, 320.8 MiB, is one slot in each of the 68 routed layers. The bank held 214 slot rows in the prompt pass, 214 of
each layer's 512 experts (measured).

The plan, the bill's layout for 32,768 tokens, has a ceiling of 101,856 MiB, the most that MLX may hold in the prompt
pass (calculated). It is the 112 GiB limit less the 8 GiB margin, the 4 GiB reserve and MEM_REFIT's 544 MiB server part.
A remainder is MLX's peak minus the trunk, the bank's slot rows and read windows, the KV that mlx-serve allocated,
`prompt_work` and the layer-major term.

The decode bill leaves out the layer-major term and all of MEM_REFIT, so the bank opens with 236 decode slot rows
(measured). 01ca9769 held them in every decode. On 4856f028, RQ plans each request's decode rows from there ("RQ"). No
decode term holds the server's own memory or mlx-serve's rerank head, about 269 MiB (calculated), so both can take part
of the reserve. On 4856f028, RQ's O counts the rerank head ("RQ"). The rerank head is mlx-serve's 3-bit copy of the
output head's draft readout, for MTP drafts.

mlx-serve's 8 GiB margin holds MLX's buffer cache, which mlx-serve caps at 8,192 MiB, and other processes' wired GPU
memory. The bank figures above are the load lines of 01ca9769 and 4856f028. src/bill.zig's EMBED_ROWS test computes
their 214 and 236 slot rows, with the windows and staging of its MEM_REFIT test. Without the 1,245 MiB of read windows
and staging, the prompt budget would fit 218. Decode keeps 1 of the 5 read windows, which also adds decode slot rows. At
0ebec77 and 165d8ca, with the embedding table resident, the bank got 68,770 and 75,101 MiB, for 210 and 233 slot rows
(measured). At c8d54a83 the room was 82,116 MiB, for 202 prompt slot rows (src/bill.zig's MEM_REFIT test). The five
terms are src/bill.zig's declarations from `context` to `budget`, and plugin.zig `prefillBytes`. The KV is 297,216 B for
each token: the trunk's 296,064 and the MTP head's 1,152 (calculated).

MEM_REFIT is the bill's last term, fitted to measured peaks. Since 0ebec77 it holds 989 MiB for the pass and 544 MiB for
the server's own memory that MLX does not count (src/bill.zig `mem_refit_pass` and `mem_refit_server`). The 544 MiB
includes the read pool's staging. Before the re-fit, the pass part was 3,606 MiB, fitted on 2026-11-08 at 18,380 tokens.
Decode takes back the layer-major term and all of MEM_REFIT, its server part included (plugin.zig `openBank`). So in
decode MLX may hold up to 102,400 MiB (calculated).

The pass part is the largest remainder in 13 runs at 31,652 prompt tokens (src/bill.zig's MEM_REFIT test). The read
windows are the streamer's scratch, 5 x 48 expert records of MLX memory (1,132 MiB, calculated), which the bank's prompt
budget holds. The earlier context ladder's 8 runs, on dylib c8d54a83, peaked at 98,807 MiB on 202 slot rows, 987 MiB
more than those terms. An earlier re-fit, commit 6bb90b8, counted the windows twice, and we did not keep it. Its check,
dylib 901a8e4a with 206 slot rows, peaked at 100,091 to 100,092 MiB in a warm-up and 4 runs, 988 to 989 MiB more than
its terms. The final ladder's 8 runs at 31,652 tokens, on 01ca9769, stayed within the pass part ("Measured").

The re-fit leaves the plan's ceiling unchanged. The refusal test's second and third runs record it as 101,855 MiB, 1 MiB
less, because the third run's script reads the room and the layer-major term in whole MiB from the load lines (lines
56-64 of r7e-echo.py, in the run's record). Every gap here uses 101,856 MiB.

### RQ: the bank's rows for each request

RQ plans the bank's slot rows for each request, for the KV that mlx-serve allocates for it (src/ling_rq.zig). It is
built into the served dylib, 4856f028, and it has no option. It plans at a request's first prompt forward pass and at
its first decode forward pass (plugin.zig `forward`, `phaseTo` and `rqForward`). It plans again, a trim, at any forward
pass whose append would pass the plan's KV. Each plan prints an `RQ prompt`, `RQ decode` or `RQ trim` line (plugin.zig
`rqLine`).

mlx-serve grows each KV cache when an append passes its capacity. The new capacity is the larger of the rows it needs
and its size plus a quarter, rounded up to 256 tokens. The quarter is at least 256 and at most 8,192 tokens
(src/ling_rq.zig `nextCapacity`, after mlx-serve's src/sdk/kv/cache.zig `KVCache.nextCapacity`). RQ replays that
schedule over the request's appends: the prompt's spans, the MTP head's history, then decode's rows (`replay`,
`grownTo`). A plan bills the trunk's MLA layers and the MTP head's latent cache at their capacities (`kvBill`). It also
bills one MLA layer's old buffer at the plan's last growth. mlx-serve tells the plugin no max_tokens. So a decode plan
holds the KV of the request's position and 4,096 more tokens, capped at the bill's 32,768-token context (`headroom`,
`decodeEnd`). A request that decodes past its plan takes a trim and a new plan.

O is the other MLX memory beside the request's KV and the plugin's own memory. At each plan, RQ synchronizes MLX and
clears its cache. It then reads MLX's active bytes, less the plugin's memory when the bank opened and the request's KV
as allocated (plugin.zig `rqPlan`). In a decode plan, the request's KV and O replace the bill's 32,768-token KV term.
Decode gets the whole slot rows that its budget then holds (src/ling_rq.zig `rowsAt` and `planDecode`).

Stage 1 keeps the prompt pass at the bank's base, the 214 slot rows that the bank opens with (`planPrompt`). The copied
streamer allocates the base at open and changes only the rows past it. Where a plan holds fewer rows than the base, RQ
logs `RQ over plan` and keeps the base (plugin.zig `rqLine`). The table gives RQ's plans at four prompt sizes, from the
RQ lines of 4856f028 and 89be02b1. 89be02b1 ran the same code with RQ as an option. All numbers are measured.

| Prompt tokens | Runs | Prompt plan: KV capacity, O | Plan under the base | Decode plan: KV capacity, O | Decode slot rows |
|---|---|---|---|---|---|
| 1,120 | the safety test, 89be02b1 | 1,280 tokens, 269 MiB | no | 6,144 tokens, 548 MiB | 258 |
| 18,380 | RQ's A/B test, arm B, 89be02b1, 4 runs; SPEC1_BUSY's A/B test, arm A, 4856f028, 4 runs | 20,480 tokens, 269 MiB | no | 25,600 tokens, 1,222 MiB | 237 |
| 31,652 | 4856f028's 32K check, 5 runs | 32,000 tokens, 269 MiB | yes, 211 rows | 40,192 tokens, 1,730 MiB | 221 |
| 32,000 | the safety test, 89be02b1 | 32,000 tokens, 269 MiB | yes, 211 rows | 40,192 tokens, 1,733 MiB | 221 |

01ca9769 held 236 decode slot rows at every prompt size. Near 32,000 tokens the prompt pass's plan holds 211 slot rows,
3 under the base, but MLX's peak still stayed under the plan's ceiling. At 31,652 tokens it was 101,430 MiB on 4856f028,
426 MiB under the ceiling. Its remainder, 988 MiB, stayed within MEM_REFIT's 989 MiB pass part. At 32,000 tokens it was
101,318 MiB on 89be02b1, 538 MiB under the ceiling. Our ruling accepts this for stage 1: the bank keeps its base, and
the line names the case.

Our root-cause analysis of O, from the code and the logs on 2026-11-11, accounts for the decode plans' O to within about
1 MiB. O holds mlx-serve's rerank head, 268.63 MiB, and the request's KDA state, 134.86 MiB, which are live in decode.
It also holds two idle items that grow with the prompt. The first is mlx-serve's set of copies of the prompt's KDA
checkpoints, 1, 5 and 8 copies of 134.86 MiB at the three sizes. The second is the prompt's last hidden rows, 8 KiB a
token. mlx-serve keeps both alive into decode. Their sums are 547.09, 1,221.38 and 1,732.36 MiB at 1,120, 18,380 and
32,000 tokens, against the logged 548, 1,222 and 1,733 MiB (calculated). These items were also alive in decode on
01ca9769, where no bill term held them. If an mlx-serve change freed the two idle items, decode would get 2 more slot
rows at 18,380 tokens and 4 more at 32,000 (calculated).

In decode, the first forward pass after a 32,000-token prompt grows each MLA layer's KV from 32,000 to 40,192 tokens.
The nine old buffers, 1,004 MiB each, wait in MLX's buffer cache until mlx-serve's next cache clear. RQ bills one old
buffer (`kvBill`), and the cache's cap of 8,192 MiB, which mlx-serve's 8 GiB margin holds, bounds the rest (calculated
and inferred). "The KV past the bill's term" measured this rise on 01ca9769.

RQ refuses a prompt forward pass by name, before any append, if mlx-serve's schedule would grow its KV past 32,000
tokens. The code is src/ling_rq.zig `prompt_capacity_max` and `promptRefusal`, and their call in plugin.zig `forward`.
From a cold prompt in 4,096-token spans, the schedule runs 25,600, 32,000 and 40,192 tokens. So 32,000 is the last
capacity that the bill's 32,768-token term holds. A longer prompt would grow the KV to 40,192 tokens inside the
layer-major pass, about 2 GB over the plan at the base rows (calculated). RQ's safety test, on 2026-11-11 on 89be02b1,
sent prompts of 1,120 and 32,000 tokens, then a 32,198-token prompt. The plugin refused the third by name in 0.018 s,
with HTTP 500. The 32,000-token prompt decoded 768 tokens, to the 32,768-token context. The memory guard stopped no run,
memory pressure stayed at 1, and swap stayed at 0.127 GB (measured).

RQ's slot table keeps up to 8 requests' capacities (src/ling_rq.zig `SlotTable`). A forward pass of one slot after
another slot's marks the request as interleaved, once a request, and prints `RQ interleaved` (plugin.zig `rqAtForward`).
O then holds the other request's KV as it is now, not its growth (`SlotTable.touch`). No test has run two long requests
at once.

### The 32K checks

Before we serve a dylib with changed code, our 32K check sends it a warm-up and 4 greedy requests of 31,652 tokens. It
passes when the remainder stays within MEM_REFIT's 989 MiB pass part, memory pressure stays normal and swap stays flat.
The remainder covers the prompt pass only, and pressure and swap cover the whole run. On 2026-11-11 the served dylib,
4856f028, passed with a remainder of 988 MiB in each run, and MLX's peak stayed 426 MiB under the ceiling. Its peak
footprint, which also counts MLX's buffer cache and the server's own memory, stayed 1,111 to 1,286 MiB short of the
reserve (calculated). On 2026-11-10 01ca9769 passed with the same remainders and peak, and its peak footprint went 4 to
13 MiB into the reserve (calculated). The check reports the footprint but does not test it.

The served dylib's 32K check ran on 2026-11-11, from 07:48:23 to 08:08:00Z, each run on a new server with 214 prompt
slot rows. MLX's peak was 101,430 MiB in all 5 runs, 988 MiB more than its terms of 100,442 MiB, as in 01ca9769's check.
The server's peak footprint was 114.62 to 114.80 GB. The highest, 114.80 GB, was 1.178 GB under the highest in
01ca9769's 32K check, 115.98 GB (calculated). In decode the bank held 221 slot rows, not 236 ("RQ"). Each run decoded
1,024 tokens. Free RAM fell to 0.059 to 1.681 GB a run, and each run had 253 to 408 pageouts. Memory pressure stayed at
1, swap stayed at 0.127 GB, and the memory guard stopped no run. All 5 runs decoded the same text, 919cf0fb, the text of
01ca9769's check. "Measured" gives each run's speeds.

01ca9769's 32K check ran on 2026-11-10, from 20:40:29 to 20:59:54Z, each run on a new server with 214 prompt slot rows.
MLX's peak was 101,430 MiB in all 5 runs, 988 MiB more than its terms of 100,442 MiB. That remainder is within
MEM_REFIT's 989 MiB pass part. 583523d4's check had the same terms and peaked at 101,429 to 101,430 MiB, with remainders
of 987 to 988 MiB. The server's peak footprint was 115.97 to 115.98 GB. As each prompt pass ended, MLX's buffer cache
held 8,126 to 8,190 MiB, nearly all of the 8 GiB margin. Each run decoded 1,024 tokens, so prompt and output passed
32,000 tokens (calculated). Free RAM fell to 0.19 to 1.10 GB a run, and each run had 347 to 639 pageouts. Memory
pressure stayed normal, swap stayed at 0.127 GB, and the memory guard stopped no run. All 5 runs decoded the same text,
919cf0fb. The 32K checks of 583523d4 and of EMBED_ROWS' dylib 7e446929, and the context ladder's greedy runs at that
size, decoded it too. We quote no speed from this check, because it checks memory only.

The re-fit's own check, dylib fdd4e0d8 (0ebec77) with the table resident and 210 slot rows, peaked at 101,373 MiB in a
warm-up and 4 runs at 31,652 tokens. That is 987 MiB more than its terms and 483 MiB under the plan's ceiling
(calculated). At 18,380 tokens, in the B arm of the re-fit's A/B ("Measured"), the remainder was −2,484 to −2,320 MiB.
From there to 31,652 tokens it rose 0.25 to 0.26 MiB for each token. The final ladder's remainders, −3,130 to −3,128 MiB
at 16,377 tokens and 987 to 988 MiB at 31,652, rose 0.27 MiB for each token (calculated). A longer prompt is not
measured on 01ca9769. A straight line from these rises does not predict it, because a prompt past 32,000 tokens grows
the KV by 8,192 tokens in one step. RQ now refuses such a prompt, and its safety test measured a 32,000-token prompt
("RQ").

### The KV past the bill's term

This section measured 01ca9769, before RQ. On 4856f028, RQ bills the KV that mlx-serve allocates, growth included, and
it refuses a prompt whose KV would pass 32,000 tokens ("RQ").

The bill's KV term holds 32,768 tokens, but mlx-serve grows a full KV by a quarter of its size, 256 to 8,192 tokens.
After most prompts of about 16,000 tokens or more, the grow step at token 32,001 takes the KV 2,104 MiB (2.2 GB) past
the term. After other prompts, a long output takes it past the term at an earlier, smaller grow step, or never (all
calculated; below). No bill term holds that excess, so it takes room from decode's working set or the 4 GiB reserve.

In 01ca9769's 32K check and the context ladder's 8 runs at 31,652 tokens ("Measured"), decode went past 32,000 tokens
(calculated). In those ladder runs, the GPU's memory in use, which counts MLX's buffer cache, rose about 6 GB for under
15 s. It ended decode 2.0 to 2.3 GB higher than it started, and free RAM stayed at 1.04 GB or more, at normal pressure
(all measured). We infer that the grow step caused both. The extra 3.4 to 4.2 GB fell at about mlx-serve's next
buffer-cache clear (the fall measured, its timing estimated from decode speed). mlx-serve clears that cache every 256
decoded tokens (read from the code). So MLX's buffer cache, which the 8 GiB margin holds, likely kept the old KV buffers
until then.

mlx-serve 66aad949 reserves no KV for a whole-prompt pass. For a prompt longer than one chunk, its chunk loop reserves
the prompt's KV and one chunk more before the first chunk (src/engines/mlx/generate/init.zig `initWithOptions`, its
chunk loop's `ctx.cache.reserve`; prefill_chunk.zig `reservedPrefillTokens`). The whole-prompt path never runs that
loop: `initWithOptions` sets `final_start` to 0 for it and runs `wholePromptPlan`'s spans in one forward. The KV grows
as its rows arrive instead. Each grow takes the KV to the larger of the rows it needs and its size plus a quarter,
rounded up to 256 (src/sdk/kv/cache.zig `KVCache.nextCapacity`). The quarter is at least 256 and at most 8,192 tokens.
So the grow step that passes the bill's term depends on the size that the prompt left. At 31,652 prompt tokens it holds
32,000 tokens, the KV that memfit counts. The next token past 32,000 grows it to 40,192 tokens (calculated). That is
7,424 tokens past the bill's 32,768-token term, 2,104 MiB at 297,216 B a token. Each MLA layer's grow copies its old
buffer, 1.05 GB (32,000 x 32,896 B), into a larger one (src/sdk/kv/cache.zig `KVCache.updateDense` and
`KVCache.growQuantBuf`).

We replayed the rule over the prompt pass's spans, then one token at a time, for every prompt from 32 to 32,767 tokens.
The spans end every 4,096 tokens up to 30 tokens before the prompt's last token, and then at its end. A last part under
512 tokens joins the span before it (prefill_chunk.zig `nextChunkEnd` and `wholePromptPlan`). The replay gives memfit's
KV at all six ladder sizes. The table gives the first grow step past the term, all calculated:

| Prompt tokens | KV after the prompt pass, tokens | The grow step, tokens | At token | Past the term |
|---|---|---|---|---|
| 1,120 | 1,280 | 30,208 to 37,888 | 30,209 | 1,451 MiB |
| 2,042 | 2,048 | 27,904 to 35,072 | 27,905 | 653 MiB |
| 4,094 and 8,185 | 4,096 and 8,192 | 31,744 to 39,680 | 31,745 | 1,959 MiB |
| 10,272 to 10,527, 16,416 to 16,671 and 20,768 to 21,022 | 10,496, 13,312, 16,640, 20,992 or 26,368 | 26,368 to 33,024 | 26,369, the earliest | 73 MiB |
| 16,160 to 16,415, 16,927 to 20,511 and 21,023 to 32,000, with 16,377 and 31,652 | 16,384, 20,480, 25,600 or 32,000 | 32,000 to 40,192 | 32,001 | 2,104 MiB, the most |
| 16,672 to 16,926 | 16,896 or 21,248 | 26,624 to 33,280 | 26,625 | 145 MiB |
| 20,512 to 20,767 | 20,736 or 26,112 | 26,112 to 32,768, which ends at the term | 26,113 | none |
| 32,001 or more | 40,192 | 32,000 to 40,192 | in the prompt pass | 2,104 MiB |

A prompt under 16,160 tokens goes past the term in decode, at one of eleven smaller grow steps. It does so once prompt
and output reach 26,369 to 31,745 tokens.

The final ladder's 8 runs at 31,652 tokens passed 32,000 tokens at the 349th decoded token, after the prompt pass
(calculated). In each run, the GPU's memory in use was between 108.1 and 108.3 GB at decode's start. Within 20 s it rose
to between 114.0 and 114.6 GB. It stayed there for one or two 5-second samples, and then it fell to between 109.2 and
110.6 GB until decode ended. In decode, free RAM fell to 1.04 to 1.36 GB, memory pressure stayed at 1 and swap at 0.127
GB (all measured). The KV's growth caused the rise (inferred). The rise came in the same window in all 8 runs. mlx-serve
clears MLX's buffer cache every 256 decoded tokens (66aad949 src/engines/mlx/generate/generator.zig
`Generator.advanceStep`, prefill_chunk.zig `CACHE_CLEAR_INTERVAL`). MLX keeps freed buffers in that cache until then
(the comment on `Generator.advanceStep`), and ioreg's in-use count includes them (the harness's `mem()` in
run1-mlxserve.sh, in our results folder `ladder-01ca9769/scripts/`). In every run, the high samples began within a
sample of the 349th decoded token (estimated from each run's decode speed). They ended within a sample of the 512th, the
next clear. The 4 card runs also dipped to about 107.0 GB near the 256th decoded token, the clear before. From decode's
first sample to its last, the memory in use rose 2.0 to 2.3 GB (measured). That is close to the 2.4 GB that the larger
KV adds (calculated), but no log records the growth. A prompt over 32,000 tokens would grow the KV inside the prompt
pass. We have not measured that on 01ca9769, and 4856f028 refuses such a prompt ("RQ").

### The quantized KV

`--kv-quant` is off by default. mlx-serve sets each request's cache scheme from the request's own `kv_quant`, then
model-settings.json's, then `--kv-quant` (66aad949 src/engines/mlx/scheduler.zig `Scheduler.submit`). It never tells
this plugin the scheme. Its one call of the plugin's bill passes only the context (src/engines/mlx/plugin/driver.zig
`streamRoomAt`). Its load settings carry `kv_quant` only for a plugin that owns its decode state (driver.zig
`loadSettings`), and this one does not. On each append, a 4- or 8-bit cache returns dense, dequantized K and V
(src/sdk/kv/cache.zig `KVCache.updateAffine`), and the plugin's MLA layers read them. V alone, [1, 1, T, 16,384] bf16,
is about 1.0 GiB for each MLA layer and call at 32,768 tokens (calculated, not measured). No bill term holds that copy,
and the bill's KV term stays bf16. The quantized cache stores 1 or 0.5 B a value (src/sdk/blocks/attention/kv_quant.zig
`KVQuantConfig.affine`). Each group of 64 values adds a scale and a bias in the dtype that the cache was fed
(kv_quant.zig `dequantizeAffine`). That is about 53% (8-bit) or 28% (4-bit) of the bf16 term (calculated). For
`--kv-quant` or model-settings.json, mlx-serve also puts the scheme on the model's own cache at load
(src/engines/mlx/load.zig `doLoadOnInferenceThread`, its `kv_quant_config`). The plugin then first sees it at layer 7's
append, the first MLA layer, in the load's warm-up (the same function's `warmup_eager` branch), which is on by default
(src/main.zig `main`, its `warmup_eager`). A refusal there only makes mlx-serve log "Warmup failed" and serve on. A
request's own `kv_quant` shows only partway through that request's first forward pass, where a refusal fails the
request. mlx-serve's rule for drafting paths is that a path that refuses a KV scheme is gated at load, or implemented
(src/engines/mlx/AGENTS.md, its spec-decode rule on KV schemes). We apply the same reasoning here. A 4- or 8-bit KV is
also lossy, so the decoded text can differ from our runs (not measured).

Since fb966cc the plugin logs one warning a load, when an MLA append's view first shows a quantized cache. The code is
src/ling_kv_scheme.zig `atView` and `warning`, called from the copied mla.zig's `rope1Attn` in patches/ling-mla.patch.
The warning reads:

    [turboexl3_ling] KV_SCHEME_DENSE_COPIES: the KV cache is quantized at <bits> bits (--kv-quant, the model's kv_quant setting or a request's kv_quant). Each MLA layer reads the dense dequantized K/V copies mlx-serve makes at every append: <n> MiB a layer a call at <tokens> cached tokens (calculated), which no term of this plugin's bill holds. Refusing the scheme needs a host change; keep --kv-quant off and send no kv_quant.

At the bill's 32,768 tokens the warning prices the copies at 1,028 MiB a layer and call (calculated; the tests in
src/ling_kv_scheme.zig). It refuses nothing and changes no byte, and a dense cache's view costs one branch. We have not
measured the warning.

### Prompt log-probabilities

Prompt log-probabilities (`echo` with `logprobs`) run one mlx-serve step over every prompt row at once, about 165 GB at
32,768 tokens (calculated). The plugin serves the step only in memory that sits idle between the prompt pass and decode,
for whole prompts up to 3,316 tokens. 01ca9769's limit was 3,317 tokens. A longer request of this kind gets HTTP 500
(below).

The whole-prompt step is the `promptLogprobs` call after the last forward in mlx-serve 66aad949
src/engines/mlx/generate/init.zig `initWithOptions`. mlx-serve's own account of the step's peak is 22 B for each row and
vocabulary entry: the bf16 logits and five 4-byte arrays (sampling/logprobs.zig `rowLogprobs`, and the comment on
prefill_interleave.zig `PROMPT_LOGPROBS_CHUNK`). MLX 0.32.5 (c69303b0) holds six. ArgPartition allocates its own output
beside the f32 input and the sort's four scratch arrays (mlx/backend/metal/sort.cpp `multi_block_sort` and
`ArgPartition::eval_gpu`). That makes 26 B. 2 B more are the plugin's own. `spanMajorOr` returns the spans' logits
joined but not evaluated (the logits' join in plugin.zig `spanMajorOr`), and MLX frees an op's inputs only when its
command buffer completes (mlx/backend/metal/eval.cpp `eval`). So the spans' copy stays alive through the step. We cannot
explain the other 1.8 to 3.6 B (the three runs below; src/bill.zig's comment on `echo_entry_bytes` gives 2.5 to 3.6 B
from the first two). The chunk loop runs the step on at most 256 rows in each chunk (prefill_interleave.zig
`PROMPT_LOGPROBS_CHUNK` and `promptLogprobsWidthCap`). Its last chunk can absorb a tail of up to 511 more rows
(generate/prefill_chunk.zig `nextChunkEnd`, sdk/bases/run.zig `TAIL_MERGE_MAX`).

The limit is the longest whole prompt whose step fits the bill's idle room (src/bill.zig `echoRowsMax`). The idle room
is the plan's KV for the tokens after the prompt, plus the idle terms, minus what stays resident. The idle terms are
`prompt_work` (2 GiB), MEM_REFIT's pass part (989 MiB) and the layer-major term (4,797.7 MiB). They are idle from the
end of the prompt forward pass until decode's first forward pass, and mlx-serve's step runs in that gap. At that pass
the bank takes back the layer-major term and all 1,533 MiB of MEM_REFIT, and decode's 2 GiB working set replaces
`prompt_work`. mlx-serve's 3-bit rerank head, 281,673,728 B ("The served stack"), stays resident through the step. So
the room leaves it out (src/bill.zig `echoIdle` and `rerankHead`; plugin.zig `prefillBytes`, `promptOnlyBytes`, and
`echo_plan` in `openBank`). At 165d8ca and at bd00b1a, (32,768 x 297,216 + 7,933,640,704) / (157,184 x 32 + 297,216)
gives 3,317 tokens (calculated). The vocabulary has 157,184 entries. At fb966cc the room counts the KV that mlx-serve
allocates for the prompt, its capacity, in place of the prompt's positions. The code is src/bill.zig `echoRowsMax` and
`echoRefusalAt`, and src/ling_rq.zig `wholePromptCapacity`. mlx-serve allocates 3,328 tokens of KV for a whole prompt of
3,316 or 3,317 tokens. At 3,328 tokens the room, 16,683,679,744 B, holds 3,316 rows' step and not 3,317 rows', so the
limit is 3,316 tokens (calculated).

Commit 3419552 added the layer-major term to the room, because that term is idle at the step. Commit 165d8ca then took
the rerank head out. The head stays resident from load on, and MEM_REFIT's measured pass part includes it. At 165d8ca's
limit the room is 15,914 MiB (16,686,949,120 B at 3,317 tokens, calculated). The refusal test's third run served that
limit with MLX's peak 1,359 MiB under the plan's ceiling (below).

The decode-only parity test sets the option `TURBOEXL3_LING_LAYER_MAJOR` to 0. At that value the chunk loop asks 256
rows at a time, and its last chunk can take up to 511 more. The room then has neither MEM_REFIT's part nor the
layer-major term, and it still leaves out the rerank head. A 256-row chunk fits at every position, and a 767-row chunk
fits wherever it ends at or before position 26,065. So at bd00b1a every prompt-log-probability request up to 26,065
tokens is served (calculated). Past that, a last chunk of over 256 rows can be refused by the same rule. We have not
measured the speed at that value.

src/bill.zig's rule-7 tests at fb966cc check these cases:

- At the served plan, a 3,316-token prompt is served and a 3,317-token prompt is refused (src/bill.zig's test `rule 7 at
  the served plan ...`). At bd00b1a the limit was 3,317 tokens.
- A prompt forward pass that asks for the last row or no logits is never refused, at any rows and position up to the
  context. Neither is a verify pass or a decode step (its test `rule 7: a last-row (or no-logits) prompt forward ...`).
- An all-rows forward pass of 1,577 rows or fewer is never refused at any position. One of 1,578 rows that ends at the
  context is refused (its test `rule 7: an all-rows prompt forward of 1,577 rows or fewer ...`). So the warm-up's
  forward passes and the chunk loop's 256-row chunks pass.
- The parity request's 1,146-token prompt passes, with or without the layer-major pass (its test `rule 7: the parity
  request's prompt ...`).
- The idle room counts MEM_REFIT's part and the layer-major term only with the layer-major pass. It leaves out the
  rerank head at both values of the option (its test `rule 7: the plan's idle room ...`).
- The limit follows the plan. A test plan of 16,384 tokens gives 2,394 tokens, and a room of the KV alone, with no idle
  terms, gives 1,800 (its test `rule 7: the limit follows the plan's context and room ...`). At bd00b1a they were 2,403
  and 1,828. The served plan always holds 32,768 tokens, at any `--ctx-size` (src/bill.zig `context`, and `echo_plan` in
  plugin.zig `openBank`).

The refusal test ran twice on 2026-11-09 and once on 2026-11-10. Each run used one new server, started with step 6's
command and `--port 11391`. It sent two non-streamed prompt-log-probability requests, each with `logprobs: 1` and
`max_tokens: 1` (the third run's r7e-echo.py, line 44). The first asked for one token over the limit. The second asked
for one token under it, and in the third run for the limit itself. In the first run the plugin counted mlx-serve's own
22 B, and MLX went 1,238 MiB over the plan's ceiling. Since 0e1f5c0 the plugin counts 32 B, the first run's measured
31.6 B rounded up. The third run checks the room of 165d8ca.

| Reading | First run, at 9700aa6 (22 B) | Second run, at 0e1f5c0 (32 B) | Third run, at 165d8ca (32 B) |
|---|---|---|---|
| Limit in the load line | 4,172 tokens | 2,941 tokens | 3,317 tokens |
| Refused, HTTP 500 | 4,173 tokens | 2,942 tokens | 3,318 tokens |
| Served, HTTP 200 | 4,171 tokens in 12.3 s | 2,940 tokens in 11.13 s | 3,317 tokens in 11.54 s |
| MLX active before the request, and at its peak | 82,174 and 103,094 MiB | 82,176 and 96,444 MiB | 84,742 and 100,497 MiB |
| The step: the peak minus the reading before and the request's KV | 19,738 MiB, 31.6 B for each entry | 13,435 MiB, 30.5 B for each entry | 14,815 MiB, 29.8 B for each entry |
| MLX's peak against the plan's ceiling, 101,856 MiB | 1,238 MiB over | 5,412 MiB under | 1,359 MiB under |
| Memory pressure and swap | normal at 8 readings, swap flat | normal at 7 readings, swap flat | normal at 8 readings, swap flat |

The server's peak footprint (`time -l`, GB = 10^9 B) was 115.1 GB in the first run. It was 110.5 GB in the second and
114.6 GB in the third. The 31.6, 30.5 and 29.8 B are single measurements at three lengths, not bounds. The first run's
"before" reading is the load's last `[mem]` line, and the others come from `/props`. mlx-serve answers every refused
forward pass with HTTP 500 (66aad949 src/host/http/errors.zig `slotFailure`, and the `error.GenerationFailed` case of
`mapGenerationError`). A streamed request whose stream has started gets an error event in its place (errors.zig
`sendGenerationError`).

mlx-serve 66aad949 does answer with a named HTTP 400 when a forward pass returns fewer logit rows than the prompt has
(src/host/http/errors.zig, the `error.PromptLogprobsUnavailable` case of `mapGenerationError`; the check after
`promptLogprobs` in src/engines/mlx/generate/init.zig `initWithOptions`). The plugin refuses instead, before any work
(the rule-7 check in plugin.zig `forward`). Fewer rows would cost the whole prompt pass, and the 400's message, "not
supported on this model" (src/host/api/completions.zig `PROMPT_LOGPROBS_UNAVAILABLE_MSG`), would name the wrong cause.

The third run's refused request, as mlx-serve logged it:

    [plugin] turboexl3_ling: forward failed: turboexl3_ling: prompt log-probabilities for a 3318-row prompt forward from position 0 need 15916 MiB, past the 15913 MiB this plan leaves at that length; it serves them for whole prompts up to 3317 tokens

## The served stack

All but two of the served dylib's optimizations decoded our coding prompt to the same text as the code they replaced.
The exact verify computes each verify row as a one-row decode does, so in greedy decoding, MTP drafts do not change the
text. MLA_FLASH, a one-launch MLA attention for prompts, met the parity test's four statistics ("Parity"). KDA_GATE_EPI
computes a KDA layer's forget gate inside its prompt projections' GPU launch, and KDA_NORM_PREP computes its gated norm
inside its output projection's first launch. Where their launches run, KDA_GATE_EPI never writes the forget gate's input
to GPU memory, and KDA_NORM_PREP never writes the gated norm. Both keep the same bits (below). RQ, kept on 2026-11-11,
changes only which experts are resident, so it keeps the text by construction ("RQ"). Leave the plugin's one option,
`TURBOEXL3_LING_LAYER_MAJOR`, at 1, its default, which keeps the layer-major pass on.

### The optimizations, part by part

`TURBOEXL3_LING_LAYER_MAJOR` is a Manifest option, and its default, 1, turns the layer-major pass on. Only the
decode-only parity test, a check of decode's kernels alone, sets it, to 0, which turns the layer-major pass off
("Parity"). mlx-serve shows the option in the app's Models > Plugins and in `/props`, and sets it in the environment
before the plugin loads. The plugin's other environment reads are mlx-serve's and MLX's own variables, read as they read
them, for example `MLX_SDPA_BLOCKS` in overlay/mlx_serve/plugins/bailing_hybrid/verify_rows.zig. The tests read their
own `MLX_SERVE_LING_*` variables.

Our ledger records five optimizations kept by a ruling although at least one measure was slower ("Receipts"). They are
COMBINE, MTP drafting, LAYER_MAJOR, ROUTER_FUSED and MEM_REFIT, a memory calibration.

The MTP head's optimizations change only which tokens it drafts. The exact verify makes each verify row equal to a
one-row decode, so in greedy decoding, drafting must not change the text. It did not in our greedy runs. A sampled
request accepts each draft with a probability (`specAcceptProb` in mlx-serve's src/engines/mlx/spec/mtp/depth.zig), so
its text can differ with and without drafts. A parity run of 2026-11-08 on dylib 7b1843f1 decoded the same 1,024 tokens
with and without MTP drafting. That run also had MLA_LATENT and MLA_FUSED on, which we rejected. Every run of the A/Bs
in "Measured" decoded the same text, 09571f6a. The run-path line (README, "Logs") can show 41 names on the served dylib,
the entries of `Path` in src/ling_ran.zig. It could show 40 on 01ca9769 and 39 on 63668c19.

- In decode, the plugin queues the GPU work for the routed experts already in the bank before the CPU reads the routing
  (DEVROUTE). A KDA layer's decode step runs in one launch for up to 8 rows (KDA_DECODE_FUSED). That launch also writes
  inputs that later launches made before, such as o_proj's prepared rows (KDA_EPILOGUE). A KDA call that this launch
  does not run, such as a prompt's, runs its prework in a packed kernel (KDA_PREWORK). Its gate runs in the prompt
  projection's launch (KDA_GATE_EPI) or in a launch of its own (KDA_GATE_FUSED). The trunk's one-row projections use
  turboexl3's one-row table and pair launches (ONE_ROW_TABLE, PAIR). The KDA projections, the shared expert, the router
  and the MoE tail each run in one or two launches. Their names are KDA_PROJ_FUSED, KDA_BPROJ, SHARED_F3, ROUTER_FUSED,
  ROUTER_F16, ROUTER_ROWS, MOE_TAIL and ROUTED_JOIN. The routed GEMVs run on turboexl3's grouped kernels at variant t4
  (BANKED_T4).
- KDA_PROJ_FUSED is not a per-request path. Only the load's 8-row warm-up forward pass runs it. KDA_BPROJ's launch
  declines 8 rows outside an exact verify, and KDA_PROJ_FUSED's launch takes 1 to 16 rows. A request runs KDA_BPROJ at
  decode and verify widths and KDA_PROJ_PROMPT on its wide spans. Its 31-token snapshot tail, the last 30 prompt tokens
  that mlx-serve holds back plus the last token, runs each projection's own launch.
- The exact verify is the kit's `Caps.exact_drafted`, which the plugin declares (config.zig). Its KDA and MLA rows run
  as one-row decodes do (VERIFY_KDA_ROWS, VERIFY_MLA_ROWS).
- The plugin loads a 4-bit draft readout of the output head (MTP_HEAD_Q4). It takes 345 MiB in place of the dense bf16
  1,228 MiB, which frees 883 MiB for the bank (calculated). The load line prints 882, because it cuts off the fraction.
  mlx-serve builds a 3-bit/gs64 coarse copy from that readout after the bank opens, the rerank head, about 269 MiB
  (calculated). Each greedy draft takes the rerank head's top 32 tokens and re-scores them through the 4-bit readout's
  own rows. mlx-serve also builds a 3-bit/gs64 draft-only copy first, and drops it once the rerank head exists. The
  peaks that MEM_REFIT fits include the rerank head.
- The MTP head's MLA layer caches its latent, 1,152 bytes for each token (MTP_HEAD_LATENT). That layer attends on NAX
  (MLA_LATENT_NAX), and on a staged launch at the calls NAX does not run (MLA_LATENT_STAGE). A call that neither launch
  runs uses the composed chain (MTP_HEAD_LATENT_COMPOSED); the served shapes never need it.
- In the prompt pass, the whole prompt runs one layer at a time (LAYER_MAJOR). A wide routed call gathers its rows in
  turboexl3's prepare and stores its expert outputs as f16 (ROWS_PATH, PF_F16_PAIRS), and one launch combines them
  (COMBINE). PF_DFIN runs the down projection's finish inside the GEMM. A call takes it at a mean of 80 or more rows an
  expert, which is 5,120 rows at top-8 over 512 experts (calculated). The KDA recurrence runs on a blocked kernel
  (KDA_PREFILL_FUSED, KDA_REC_SHAPE), and the KDA projections in one launch (KDA_PROJ_PROMPT), which also computes the
  forget gate (KDA_GATE_EPI). A KDA layer's gated norm runs in o_proj's prepare launch (KDA_NORM_PREP). The shared
  expert's and the dense layers' SwiGLU runs in the up projection's finish (SWIGLU_EPI). In the layer-major pass, each
  layer decodes its EXL3 weights, other than the routed experts', one time for all its wide spans (SHARED_LAYER_WHAT,
  LAYER_WHAT). A MoE layer ends each span in one launch (PF_MOE_TAIL). That launch does the routed sum, the shared
  expert's add, the residual add and the next layer's input norm. The MLA attention runs in turboexl3's one causal flash
  launch (MLA_FLASH), and a call that the launch refuses runs in spans (MLA_SPANS).
- Each MLA layer caches kv_b's rows and the roped key one time for each token (MLA_ROPE_ONCE). The 9 MLA layers of the
  trunk then take 296,064 bytes for each token, not 368,640 (calculated). The cache's SSD namespace (the code's layout
  namespace) is `ling-mla-rope1-mtp-head-latent-v1`, so a prefix that an earlier layout stored never restores into it.
- The read pool runs 8 workers (READS_WIDE). The read-ahead predicts with Ling's own router rule (PREDICTOR) and reads
  up to 2 records for each row in decode (LOOKAHEAD_ROWS). In the layer-major pass, it reads each next layer's predicted
  experts ahead (PROMPT_AHEAD, PROMPT_PARK, PROMPT_AHEAD_BF16, PROMPT_AHEAD_FILL). In a prompt pass, a missing expert
  never takes the bank slot of an expert that the same call uses (PROMPT_KEEP). The bill gives the decode phase the
  memory that only the layer-major pass uses (LM_DECODE_ROWS). It fits the prompt pass's last term to peaks that we
  measured on an M5 Max with 128 GiB (MEM_REFIT, "Memory").
- The plugin always sets the last-row flag. mlx-serve then asks a prompt for its last row only, and the output head runs
  over the last span (LAST_ROW_LOGITS). For prompt log-probabilities, mlx-serve asks for every row ("Memory").
- RQ plans the bank's decode rows for each request, for the KV that mlx-serve allocates (RQ, "RQ"). It refuses a prompt
  whose KV would pass 32,000 tokens.

### KDA_GATE_EPI

KDA_GATE_EPI computes a KDA layer's forget gate, an input of its recurrence, in the launch that runs the layer's prompt
projections. A launch is one GPU kernel dispatch. At a prompt span of more than 32 rows, that launch computes the
layer's q|k|v (query, key and value), f and g projections. f's output is the forget gate's input, and g's is the output
gate's. KDA_GATE_EPI computes the gate from f's output with the same bits. So f's output never goes to GPU memory, and
the gate's own launch does not run.

The projections' launch is KDA_PROJ_PROMPT, and the gate's own launch is KDA_GATE_FUSED. turboexl3's
`ling.kdaProjPromptGate` computes the gate in the loop that stores the launch's outputs (src/ling_kdaproj.zig
`projPromptGate`, and plugins/glm5_next/kda.zig in patches/ling-kda.patch). In the layer-major pass, the launch reads
the layer's kept W_hats (LAYER_WHAT, through `ling.kdaProjPromptWGate`). The gate does KDA_GATE_FUSED's operations in
the same order on the same values. The packed prework (KDA_PREWORK) still takes an f input, which it reads only when it
computes a gate itself. The layer gives it a second reference to q|k|v instead, which it does not read.

The layer takes the gate this way only when all of these hold:

- The gate is bounded: config.json's `kda_lower_bound`, the code's `kda_gate_lower_bound`, is not 0.
- The forward is a prompt's, not a verify's.
- f has no low-rank stage.
- The per-core recurrence does not run the layer. That recurrence reads f's output itself.

turboexl3 also declines a call of 32 rows or fewer, and a call off NAX. It declines gate inputs that are not f32 with
one value for each of f's 8,192 channels, and a bound that is 0 or not finite. A declined call runs the gate in its own
launch (KDA_GATE_FUSED), as before. The standard request's 31-row span is such a call. For the standard request,
KDA_GATE_EPI saves 315 launches: 63 KDA layers with 5 wide spans each. It also saves the write and the read of f's bf16
output, 64 MiB for each layer and 4,096-row span. Both figures are calculated. It adds no bill term. The launch came
with turboexl3 aa0ce54 (branch ling-l6-kdagate), which adds it to c2783f5.

KDA_GATE_EPI joined the served stack on 2026-11-10 at 17:44:49Z, in dylib 583523d4. Its A/B, an earlier one in
"Measured", ran it as an option, `TURBOEXL3_LING_KDA_GATE_EPI=1` in dylib 73db9cb6 (commit 859cd93). Every B request
line named KDA_GATE_EPI, and each B log printed `[turboexl3_ling] KDA_GATE_EPI engaged: S=4096`. Before the A/B,
turboexl3's own timing test (`TURBOEXL3_LING_TIME`, a microbenchmark that only guides) gave the launch 0.987 to 0.993
times the time of the kernels it replaces at every prompt-like shape. That is about 0.09 s less for each prompt. Dylib
73db9cb6 passed its gates: the MLX-free tests 76 of 76, the GPU byte tests 50 of 50 with 645 PASS lines and no FAIL
line, and the restore check's part B, 24 checks BYTES_OK. Its byte test of KDA_GATE_EPI printed 25 lines, none failed,
with 12 red checks caught and 937,988,992 words compared. The dylibs that always run it, 583523d4, 63668c19, 01ca9769
and 4856f028, passed their own gates, and 583523d4, 01ca9769 and 4856f028 their 32K checks ("Tests", "Memory").

A call that takes the gate marks KDA_PROJ_PROMPT and KDA_GATE_EPI on the run-path lines (README, "Logs"). The first such
call logs `[turboexl3_ling] KDA_GATE_EPI engaged: S=<rows>`. A request whose prompt pass has no KDA projection of more
than 32 rows takes no gate here, so its request line does not name KDA_GATE_EPI. The MLX-free tests are in
src/ling_switches.zig and src/ling_ran.zig. The GPU byte test, src/ling_kda_gate_epi_test.zig, compares the plugin's
call with the path it replaces, word for word, at 33 to 4,096 rows, per call and over kept W_hats. It also checks the
declines, and that the prework's stand-in input changes no value. Its red checks plant three faults in the gate's kernel
and give wrong gate inputs. Each must change the gate's values or be declined. Its lines start with
`TURBOEXL3_LING_KDA_GATE_EPI`.

### KDA_NORM_PREP

KDA_NORM_PREP computes a KDA layer's gated norm inside o_proj's prepare launch, the first launch of the layer's output
projection, with the same bits. In 63668c19, which we served before 01ca9769, the norm-gate launch wrote the gated norm
to GPU memory at each KDA prompt span over 32 rows. o_proj's prepare launch then read it back. A call that KDA_NORM_PREP
declines runs the norm-gate and then o_proj, as before.

At prompt rows, a KDA layer's gated norm is the RMS norm of the recurrence's output times the sigmoid of the gate
projection's output. mlx-serve's norm-gate, `gdnNormGateFused` in decode_fused.zig, writes it to GPU memory in bf16. The
copy's KDA layer calls `gdnNormGateFusedTab`, which patches/ling-kda.patch adds to the copied decode_fused.zig to take
the sigmoid from MLX's own table. o_proj's prepare launch then reads it back and prepares it for the matrix multiply.
turboexl3 4bf2df8's `ling.kdaNormGatePrep` does the norm-gate's operations inside that prepare launch, where the prepare
loads its input (src/ling_kda_norm_prep.zig `oProj`). o_proj's matrix multiply then reads the prepared rows and the
layer's kept W_hat (LAYER_WHAT). The layer-major pass decodes that W_hat once for all of the layer's wide spans.
o_proj's finish follows (src/ling_layer_what.zig `linearPrepared`). The launch does the norm-gate's operations and the
prepare's in the same order on the same values.

The copied KDA layer (plugins/glm5_next/kda.zig in patches/ling-kda.patch) tries the launch first at its norm-gate site,
inside `fused_prework` (`normPrep`). It takes the launch only when all of these hold:

- The forward is a prompt's, at the norm-gate's prompt rows (`gdnPrefillFusedFor`) and on its sigmoid table (TAB).
- The gate is the sigmoid, and the caller wants o_proj's output, not an exact verify's raw row (`raw`).
- The layer-major pass holds o_proj in the layer's LAYER_WHAT scope, on its matmul path at the call's rows, with
  turboexl3's pg2 GEMM off (src/ling_layer_what.zig `preparedDense`). On NAX that path takes more than 32 rows.
- The stream is a GPU stream, o_proj is EXL3, and turboexl3's `kdanorm.shapeOk` and `kdanorm.takes` accept the operands.

Every check comes before the launch, so a declined call launches, decodes and counts nothing. The norm-gate and o_proj
then run as in 63668c19. The standard request's 31-row span is such a call. So is every prompt span when the layer-major
pass is off.

For the standard request, KDA_NORM_PREP saves 315 launches: 63 KDA layers with 5 wide spans each. It never writes flat,
the norm-gate's bf16 output of 16 KiB for each row. That is 64 MiB for each layer at a 4,096-row span and 30.7 MiB at
1,965 rows. Without flat's write and read, the standard prompt moves 37.9 GB less through GPU memory. These figures are
calculated. KDA_NORM_PREP adds no bill term, and it allocates nothing that the two launches do not. By our reading of
the code, the prompt pass's peak cannot rise, and it can fall by at most 64 MiB (calculated, not measured). In our 32K
checks, the peaks of 01ca9769 and 4856f028 stayed within the range of 583523d4's runs ("Memory").

KDA_NORM_PREP joined the served stack on 2026-11-10 at 20:59:54Z, in dylib 01ca9769. Its A/B ("Measured") ran it as an
option, `TURBOEXL3_LING_KDA_NORM_PREP=1` in dylib 7f053407 (commit b0079f1). Every B request line named KDA_NORM_PREP,
and each B log printed `[turboexl3_ling] KDA_NORM_PREP engaged: S=4096`. Dylib 7f053407 passed its gates. Its MLX-free
tests passed 104 of 104, and its GPU byte tests 59 of 59, with 653 PASS lines and no FAIL line. The GPU count is 59
because, while the option existed, the GPU binary also ran src/ling_switches.zig's 7 MLX-free tests. The restore check's
part B passed its 24 checks, BYTES_OK. Its byte test of KDA_NORM_PREP printed 8 lines, none failed, with 1 red check
caught and 41,603,072 words compared. After the A/B, commits 32f7c02 and a6ea7ee took the option out, and 45ab5ba added
a check to the GPU byte test. Dylib 01ca9769, built from bd00b1a with turboexl3 4bf2df8 (README, "Requirements"), passed
its gates and its 32K check ("Tests", "Memory"). Its byte test of KDA_NORM_PREP gave the same counts as 7f053407's.
turboexl3's own byte test of the launch found no differing word at 33 to 16,384 rows (measured, 2026-11-10).

A call that takes the launch marks KDA_NORM_PREP, the 40th of the 41 `Path` names at fb966cc, on the run-path lines
(README, "Logs"). The first such call logs `[turboexl3_ling] KDA_NORM_PREP engaged: S=<rows>`. A request whose
layer-major pass has no KDA span of more than 32 rows takes no launch, so its request line does not name KDA_NORM_PREP.
Two MLX-free tests cover it. src/ling_switches.zig checks that no option, environment read or test hook is left for it.
src/ling_ran.zig checks that it comes right after KDA_GATE_EPI. At bd00b1a it was the last run-path name. The GPU byte
test, src/ling_kda_norm_prep_test.zig, compares the plugin's call with the two launches, word for word, over the kept
W_hat at 4,096 and 1,965 rows. It also checks the declines at 31 rows, outside a LAYER_WHAT scope and with pg2 on. After
the pg2-on decline, the two launches in the same scope must give the per-call words. Its red check plants a fault in the
launch's kernel, which must change o_proj's output. Its lines start with `TURBOEXL3_LING_KDA_NORM_PREP`.

### EMBED_ROWS

The plugin keeps no embedding table in GPU memory, and the CPU reads each forward pass's rows from the pack
(src/ling_embed.zig). That frees 1,228 MiB, which gives the bank 4 more prompt and 3 more decode slot rows. The CPU must
then wait for each decode round's token ids, which the GPU computes, before it reads their rows. We have not measured
that wait on its own. In EMBED_ROWS' A/B, decode went from 25.70 to 25.85 tok/s (below).

Any drafter model is declined at load, named in the log, and the server runs without it (below). The MTP head is not a
drafter model, and it still drafts.

The bank sets its number of slot rows from MLX's memory in use when it opens, at the end of the load. The table is not
in that memory, so the bank's prompt and decode budgets each grow by 1,228 MiB, 3.83 slot rows (calculated). Each budget
holds only whole slot rows, so the prompt pass gains 4 and decode 3 (src/bill.zig's EMBED_ROWS test).

EMBED_ROWS joined the served stack on 2026-11-10 at 15:23:50Z, in dylib 7e446929. Before that, our standard A/B test ran
it as an option, `TURBOEXL3_LING_EMBED_ROWS=1` in dylib 38f84c44 (commit 2773ee6). Arm A was dylib 090d2b0f (165d8ca).
Both arms decoded the same text, 09571f6a. The means of 4 runs, all measured:

| Arm | Dylib | Decode tok/s | Prefill tok/s | TTFT s | Wall s | MTP accepted / drafted | Server's peak footprint GB (`time -l`) |
|---|---|---|---|---|---|---|---|
| A | 090d2b0f | 25.70 | 695.7 | 26.43 | 66.34 | 1,753 / 2,783 = 63.0% | 108.4 |
| B, with the option | 38f84c44 | 25.85 | 694.9 | 26.46 | 66.18 | 1,805 / 2,913 = 62.0% | 108.5 |
| B against A (calculated) | | +0.58% | −0.12% | +0.11% | −0.23% | −1.0 points | +0.09% |

Dylib 7e446929 passed the MLX-free tests (73 of 73), the GPU byte tests (48 of 48) and the restore check (part B,
BYTES_OK). In a run with `MLX_SERVE_LING_PACK` set, EMBED_ROWS' byte test read the pack's own table and found no
differing word (23 lines, none failed). Its 32K check gave a remainder of 988 MiB in all 5 runs.

Each decode round waits for its token ids. With a resident table, the GPU gathered the rows (MLX's take), so the CPU
could queue a forward pass before its ids existed. Now the CPU must have the ids to read their table rows. The verify
and each head step after a round's first embed drafts that the GPU is still computing, so the CPU waits for them. A
prompt's ids are already on the CPU.

The plugin reads each distinct table row once, on parallel threads, and copies each repeat (src/ling_embed_rows.zig). It
copies the rows into one bf16 array, so a forward pass gets the stored values, as from a resident table. The GPU byte
test compares them with MLX's take over a resident table, on a test pack and on the pack that `MLX_SERVE_LING_PACK`
names ("Tests").

The plugin refuses a pack at load in four cases, each with a named error (README, "Logs"):

- config.json sets `tie_word_embeddings`, so the output head reuses the table;
- the output head is not the pack's own EXL3 head (`lm_head.trellis`);
- the pack has no `word_embeddings` table, or the table is not bf16 [`vocab_size`, `hidden_size`];
- the index does not name the table's shard, or the shard does not open, or its header is unreadable or cut short.

The log then shows a `[turboexl3_ling] EMBED_ROWS ...` line and `[plugin] turboexl3_ling: load failed: <reason>`. We
have not loaded the 2.10bpw pack with EMBED_ROWS.

mlx-serve loads a drafter model with `--drafter <dir>` or from the pack's folder. Its drafters borrow the table as an
unevaluated MLX array. mlx-serve 66aad949 reads only its pointer, dtype and shape, and the MTP head reads its rows
through the trunk. A DFlash drafter would take rows from the table, so MLX would load 1,228 MiB outside the bill.
mlx-serve binds a drafter only when the plugin declares DFlash's layer capture (66aad949
src/engines/mlx/plugin/driver.zig `loadTensors`, its `Bound.drafters`), which this plugin's parse clears (config.zig
`served`). Without it, mlx-serve prints this line and loads no drafter (66aad949 src/engines/mlx/load.zig
`doLoadOnInferenceThread`, its `drafter_dir`):

    [plugin] turboexl3_ling: the drafter at <dir> needs the plugin's trunk_weight and layer capture: served without it

If MLX still loads the table, the plugin logs `[turboexl3_ling] EMBED_ROWS: the embedding table the trunk lends was
evaluated` once, when decode next starts, and keeps serving.

## Logs: the run-path lines

In 4856f028's runs, in its 32K check and in SPEC1_BUSY's A/B test, a request's `(request, decode)` line named 36 of the
41 paths (measured). So did each request line of RQ's A/B arm B. RQ was among the 36. In the runs of 01ca9769 in
"Measured", the request line named 35 of the 40 paths (measured). In KDA_NORM_PREP's A/B, the A runs named 34 of 39, and
the B runs those 34 and KDA_NORM_PREP. No request line names the other 5. KDA_PROJ_FUSED and MEM_REFIT run only at load:
the warm-up's 8-row forward pass takes KDA_PROJ_FUSED, and MEM_REFIT marks when the bank opens. MLA_SPANS and
MTP_HEAD_LATENT_COMPOSED are fallbacks that the served shapes do not take. PREDICTOR's own call did not run, because
ROUTER_FUSED's launch scores the read-ahead by the same rule (src/ling_router.zig).

## Tests

Our gates are the ReleaseFast build and four checks: the MLX-free tests, the GPU byte tests, the restore check and
scripts/test_reach.py. A dylib passes them before we serve it. The MLX-free tests (`zig build test`) need no GPU, MLX or
pack, and most check the claim and parse, the bill or the streamer. Each GPU byte test checks that a path we keep as
bit-exact gives the same bits as the path it replaces. The served dylib, 4856f028, passed its gates on 2026-11-11, with
all 144 MLX-free and 53 GPU byte tests (measured). 01ca9769 passed its own on 2026-11-10, with all 104 and 52. A dylib
with changed code also passes our 32K check ("Memory"). A change to the decoded text also needs a parity test
("Parity").

With the pull request checked out, you can run the MLX-free tests with step 3's kit and no turboexl3 clone (build.zig
`build`: the `test` step's binaries take no turboexl3 module). We ran them that way at ecc4a1c. Run step 4's commands
from `git clone .../plugins` through `zig build --fetch`. Then run `../../.zig-toolchain/zig build test` with no
`--fork`.

Commits 7178ba2 and af601d8, merged as 1dffdcd, change only tests. 7178ba2 takes out four test blocks that no test
binary ran. One is plugin.zig's `test { _ = config; }`. The other three are the unnamed tests of three copies:
plugins/bailing_hybrid/plugin.zig, sdk/quant/gguf/claim.zig and sdk/weights/streamed/sdk/root.zig.
scripts/sync-mlx-serve.sh now takes those out too, as it takes out the copies' named tests (scripts/sync-mlx-serve.sh
`unrun_tests`). So at 1dffdcd scripts/mlx_serve.sha256 holds 2ffe2271, in place of 922e3f76 at 4c13ec7. At bd00b1a it
holds 2fac1c06, because ling-kda.patch adds KDA_NORM_PREP's call site. At fb966cc it holds 26b576c4, because RQ and the
KV-quant warning change ling-mla.patch, ling-stream.patch and the overlay's ling_source.zig. af601d8 adds
scripts/test_reach.py, which names any test block that no test binary reaches. It reads the test binaries' roots from
build.zig and follows their imports, with no build. The 15 Mach-O sections of dylib 63668c19 are byte for byte those of
583523d4 (measured). A later test-only commit, 1212e31, corrects test_reach.py's model: it now counts 24 tests for the
restore check's part A, as many as part A ran in 583523d4's gates (measured). At bd00b1a it counts 104 MLX-free tests,
52 GPU byte tests and 26 in part A, with KDA_NORM_PREP's tests. At fb966cc it counts 144, 53 and 31, with RQ's tests. In
the gates of 01ca9769 and 4856f028, each of the three binaries ran as many tests as the model counts (measured). Before
those gates, commit 32f7c02 added src/ling_switches.zig's test that no option is left for KDA_NORM_PREP. It failed as
planned at 32f7c02, before a6ea7ee took the option out (measured). Before the gates of 4856f028, commit 44ba83b added a
test that RQ is built in, with no option left. It also set rule 7's tests to the served limit, 3,316 tokens. They failed
as planned at 44ba83b: 138 of 144 passed, with 6 error records under 3 names (measured). fb966cc then made them pass.



Without a fork, `zig build --fetch` still downloads the turboexl3 pin (the `b.dependency("turboexl3", ...)` call in
build.zig `build`). At ecc4a1c, scripts/build.sh ran the MLX-free tests that way and then failed in the ReleaseFast
build, with 7 errors on the c7ce66c pin (measured, `~/lingp/logs/portcs-cs1.log`). It starts that build only after the
tests pass (scripts/build.sh, whose ReleaseFast build follows the test step after `&&`).

Our gates run both before we serve any dylib. Run the GPU byte tests after a kernel or forward-path change, and the
restore check after a cache-layout change. They need turboexl3's Ling branch, and some of them need a pack. The GPU byte
tests run on an M5-class Mac. `<dir>` is the serving mlx-serve build's lib/mlx/lib, which holds libmlxc, libmlx and
mlx.metallib. From plugins/turboexl3_ling:

    MLX_SERVE_LING_GPU_TESTS=1 ../../.zig-toolchain/zig build gpu-test -Dmlx-lib=<dir> --fork=<turboexl3 clone>

Without the variable, every GPU byte test skips. At bd00b1a the GPU byte tests passed 52 of 52, with 653 PASS lines and
no FAIL line (measured, dylib 01ca9769). At fb966cc they passed 53 of 53, with 653 PASS lines and no FAIL line
(measured, dylib 4856f028).

The restore check finds out whether a prefix that mlx-serve restores from its SSD tier holds the bits that the prompt
pass wrote. The SSD tier is `--prefix-cache-disk`, which step 6 does not set. It covers the trunk's MLA cache, the MTP
head's latent cache and every checkpoint's KDA states. It runs in two parts, because one binary cannot hold both the
plugin's copy of the MLX bindings and mlx-serve's. Part A writes the plugin's cache rows and states to a file. Part B
commits them through mlx-serve's own cache code and restores them from the SSD tier at every checkpoint. It then
compares them with part A's, bit for bit.

1. Part A (src/ling_restore_test.zig) builds with the GPU byte tests.
   - Build it with `../../.zig-toolchain/zig build gpu-test-build -Dmlx-lib=<dir> --fork=<turboexl3 clone>`.
   - Run `MLX_SERVE_LING_GPU_TESTS=1 MLX_SERVE_LING_PACK=<pack> MLX_SERVE_LING_RESTORE_OUT=<file>
     zig-out/tests/ling_restore_gpu`.
   - It runs two prompts through the plugin's MLA layers: 16,700 tokens for the flush, and their first 9,000 for the
     decline spill.
   - It writes every CacheOps append and every checkpoint's states to the file, 8,628,529,825 B at 4c13ec7 (measured,
     dylib 583523d4). The file's metadata names the cache's SSD namespace as config.zig serves it.
   - Part A holds the whole file in memory until it writes the file.
2. Part B (tests/ling_restore_host_test.zig) runs in a clone of the serving mlx-serve tree, in that tree's own unit-test
   build.
   - Copy tests/ling_restore_host_test.zig into the clone's src/.
   - Make it the only test of src/tests.zig. Keep that test block's two llama engine imports (engines/llama/ffi.zig and
     bridge.zig). The linked llama shim calls `mlx_llama_log`, which bridge.zig exports.
   - In the clone, run `.zig-toolchain/zig build test-build -Dtest-filter='turboexl3_ling restore bytes'`. The filter
     keeps the imported files' own tests out.
   - Run the binary that this step installs, `zig-out/tests/test`, with `MLX_SERVE_LING_RESTORE_IN=<file>`. The SSD
     tiers' files go in the test's temporary folders, one entry at a time. To keep them, also set
     `MLX_SERVE_LING_RESTORE_DISK=<empty folder>`.
   - Part B commits the shorter prompt through mlx-serve's decline spill, with the RAM tier at its floor of 1 byte, as
     served. It commits the longer one through a flush. It restores at every checkpoint and compares every restored row
     and state with part A's, bit for bit. It also damages its own copies on purpose in five places: four planted errors
     and one flipped bit in an SSM state. Each comparison must then fail.
   - Part B also checks that a tier at the namespace of an earlier build's layout never restores the served entry.
   - The last line, `TURBOEXL3_LING_RESTORE_BYTES B DONE`, ends in `BYTES_OK` when every check passed. With a clone of
     66aad949, it passed 24 checks over 7 restores at 4c13ec7 (dylib 583523d4). Its 24 checks passed again at 1dffdcd,
     bd00b1a and fb966cc, in the gates of dylibs 63668c19, 01ca9769 and 4856f028 (measured).

The decline spill writes at most 4 GiB in each commit: the checkpoints first, then 1,024-token chunks of the cache at
296,064 bytes for each token. So a 16,700-token prompt would stop at 13,312 tokens (calculated), short of its last
checkpoints. The spill gets the 9,000-token prompt, which it holds whole. Part B's LIMIT line states the limit with the
entry's own sizes. On 66aad949 the decline spill also writes the MTP head's rows (ddalcu/mlx-serve#1150). Part B
compared them after every restore, the spill's too: its DONE line counts 36,001,152 head words, as many as the trunk's K
words.

Three GPU byte tests also run on the pack's own tensors when you set `MLX_SERVE_LING_PACK=<pack>`. Two check the W_hats
of the shared expert (layers 4 and 64) and of the trunk's attention and dense-MLP projections (layers 0, 4 and 7). A
W_hat is an EXL3 weight decoded to f16. The third, EMBED_ROWS' test, reads the pack's embedding rows and compares them
with the resident table's take. Only the tests read these variables.

## Parity

A change that alters the text of plain decoding must pass a parity test. It compares two builds, with and without the
change, against exllamav3's logits (below). On 2026-11-08, on dylib 7b1843f1, MLA_FLASH met the test's four statistics
and missed a fifth (measured), which we then ruled no longer decides. Each later change kept the text 09571f6a or
changed no forward pass's output, which is weaker evidence than a new parity run.

In this section, A is the base and B is the candidate, not the arms of "Measured". KL is the Kullback-Leibler divergence
of an arm's next-token distribution from the oracle's. Top-1 agreement is the share of positions where the arm's most
likely token is the oracle's. Top-20 overlap is the share of the oracle's 20 most likely tokens that are also among the
arm's 20. B passes when its four oracle statistics meet these bars:

- its median KL is at most 1.25 times A's, plus 0.0002;
- its mean KL over positions 0-49 is at most A's plus 0.005;
- its top-1 agreement is at most 1 point below A's;
- its top-20 overlap is at most 1 point below A's.

The fifth statistic is the median KL of B from A, against 0.00286, the median KL between the oracle's two runs. It shows
how far the change moved the output, not how close B is to the oracle. We report it, but since 2026-11-08, the day
MLA_FLASH met the four and missed it, it does not decide (our ruling). The results, measured on our M5 Max on 2026-11-08
with dylib 7b1843f1:

| Statistic | Bar for B | A | MLA_FLASH as B | 2.10bpw as B |
|---|---|---|---|---|
| Median KL from the oracle | at most 0.00561 | 0.00433 | 0.00443 | 0.05926 |
| Mean KL, positions 0-49 | at most 0.0169 | 0.0119 | 0.0125 | 0.1693 |
| Top-1 agreement | at least 88.52% | 89.52% | 89.52% | 79.18% |
| Top-20 overlap | at least 85.63% | 86.63% | 86.18% | 68.60% |
| Median KL of B from A | at most 0.00286 | Not measured (A is the reference) | 0.00399 | 0.06068 |

The 2.10bpw pack has no oracle run of its own, so its row compares it with the 2.60bpw oracle. It missed all five, which
shows that the bar fails a lower-bit pack. Arm A's second serve (A2) decoded A's text word for word, so no margin grew.

The parity request is one greedy `/v1/completions` call on a 1,146-token prompt. It asks for `echo`, the top 256
log-probabilities at each prompt position, and 64 generated tokens. Its 1,146 tokens are under the rule-7 limit
("Memory"). The prompt is one user message: an explain-this-code ask and the first 4,000 characters of glm53-eval's
`workloads/coding.json`, in the pack's own chat template. The oracle is exllamav3's full logits on the same token ids,
from the 2.60bpw download. The parity report does not record which machine ran exllamav3. exllamav3 ran some MoE layers
on the CPU, which changes the rounding, so we ran it twice. The bar uses the CPU-40 run (40 MoE layers on the CPU). The
CPU-52 run (52 on the CPU) gives the oracle's own spread.

Each test also serves arm A a second time (A2). If A2's text differs from A's, each margin becomes the larger of the
margin and twice the A-to-A2 difference in that statistic. The fifth statistic's bar becomes the larger of 0.00286 and
twice the median KL of A2 from A.

Some changes touch only decode's kernels, such as the MLA attention of a one-row call. The prompt pass does not run
those kernels. So their parity test makes every prompt position a one-row forward pass: both arms serve with
`--prefill-chunk 1` and `TURBOEXL3_LING_LAYER_MAJOR=0`. That value of the option turns the layer-major pass off, so
mlx-serve runs its chunk loop, and each prompt position takes the one-row decode path. The load log then shows
`LAYER_MAJOR off`. No served configuration sets the option to 0.

Our ledger's rows S39 to S55 record the changes since the parity run, except three. The strip, our removal of unused
code on 2026-11-09, the port and the port's follow-ups have no row. Each had A/B tests of its own. Every A/B of these
changes decoded 09571f6a, as did the runs of 01ca9769 and 4856f028 ("Measured"). Seven rows also record a GPU byte test
that found no differing word: PF_DFIN, KDA_REC_SHAPE, PF_F16_PAIRS, PF_MOE_TAIL, EMBED_ROWS, KDA_GATE_EPI and
KDA_NORM_PREP. MLA_LATENT_STAGE and MLA_LATENT_NAX change only the MTP head's drafts. The rows of SHARED_LAYER_WHAT,
LAYER_WHAT, MLA_ROPE_ONCE, ROUTER_ROWS and the re-fit record the same text and no byte test. The change to rule 7's room
and the test-only commits change no forward pass's output, so they had no A/B. RQ (S55) changes only which experts are
resident, and its A/B decoded 09571f6a, with no byte test.

The changes from the re-fit on came after the earlier context ladder's build, c8d54a83. On 01ca9769, the final ladder's
greedy runs decoded that ladder's text at all six prompt sizes, 24 of 24 runs ("Measured"). 4856f028's 32K check decoded
its text at 31,652 tokens, 919cf0fb.

## Measured

This section gives every run behind the README's numbers. SPEC1_BUSY's A/B test ran the served dylib, 4856f028, as arm
A. Those are its own standard-test runs, which the README quotes. RQ's A/B test measured the same code earlier, as arm
B, in dylib 89be02b1 with `TURBOEXL3_LING_RQ=1`. fb966cc builds that code in without the option. 4856f028's 32K check
gives its runs at 31,652 tokens. The other tests below measured 01ca9769 and earlier builds, before RQ.

### SPEC1_BUSY's A/B test: the served dylib's own runs (2026-11-11)

Our standard A/B test of SPEC1_BUSY compared the served dylib, 4856f028, as arm A, with dylib 23731721, as arm B. Arm B
ran SPEC1_BUSY as the option `TURBOEXL3_LING_SPEC1_BUSY=1`. With it, the read pool's second speculative thread may start
an unclaimed chunk beside one demand read, as its first thread does. The test ran on 2026-11-11, from 08:26:20 to
09:01:19Z, with the runs and settings of "How we measured". All 8 runs decoded the same text, 09571f6a. Every B log
printed 1 SPEC1_BUSY engagement line, and no A log printed one. All numbers are measured.

| Run | Arm | Seed | Decode tok/s | Prefill tok/s | TTFT s | Wall s | MTP accepted / drafted | Server's peak footprint GB (`time -l`) |
|---|---|---|---|---|---|---|---|---|
| 1 | A | 1234 | 25.70 | 698.5 | 26.33 | 66.28 | 464 / 752 = 61.7% | 108.5 |
| 2 | B | 1234 | 25.90 | 700.3 | 26.26 | 65.87 | 452 / 737 = 61.3% | 108.4 |
| 3 | B | 4321 | 25.40 | 695.9 | 26.42 | 66.84 | 444 / 756 = 58.7% | 108.5 |
| 4 | A | 4321 | 25.90 | 702.4 | 26.18 | 65.80 | 447 / 732 = 61.1% | 108.3 |
| 5 | B | 1234 | 25.80 | 699.5 | 26.29 | 66.03 | 464 / 756 = 61.4% | 108.4 |
| 6 | A | 1234 | 25.70 | 699.3 | 26.30 | 66.20 | 449 / 749 = 59.9% | 108.5 |
| 7 | A | 4321 | 26.00 | 699.9 | 26.27 | 65.79 | 450 / 734 = 61.3% | 108.5 |
| 8 | B | 4321 | 25.90 | 698.5 | 26.33 | 66.03 | 446 / 732 = 60.9% | 108.5 |
| A, mean of 4 | | | 25.83 | 700.0 | 26.27 | 66.02 | 1,810 / 2,967 = 61.0% | 108.5 |
| B, mean of 4 | | | 25.75 | 698.6 | 26.33 | 66.19 | 1,806 / 2,981 = 60.6% | 108.5 |

A's and B's unrounded means were 25.825 and 25.750 tok/s decode, and 700.025 and 698.550 tok/s prefill. Their TTFT means
were 26.270 and 26.325 s, their wall times 66.0175 and 66.1925 s, and their peak footprints both 108.450 GB. B's decode
was 0.29% lower and its prefill 0.21% lower. In runs 1 to 4 and 5 to 8, B's decode changed by −0.58% and 0.00%, and its
prefill by −0.34% and −0.09% (all calculated). We rejected SPEC1_BUSY under rule 6 of our decision rules.

Free RAM fell to 0.676 to 1.759 GB a run in A and to 0.424 to 1.498 GB in B. Each run had 34 to 172 pageouts, and swap
stayed at 0.127 GB in every run. A's runs opened with 214 prompt slot rows, and RQ gave each A request 237 decode slot
rows, with O at 1,222 MiB. Each A request line named 36 of the 41 paths.

### RQ's A/B test (2026-11-11)

Our standard A/B test of RQ compared 01ca9769, as arm A, with 89be02b1, as arm B, which ran RQ as the option
`TURBOEXL3_LING_RQ=1`. It ran on 2026-11-11, and we decided it at 07:11:10Z. The runs and settings were as in "How we
measured". All 8 runs decoded the same text, 09571f6a. Every B log printed 3 RQ engagement lines, and no A log printed
one. All numbers are measured.

| Run | Arm | Seed | Decode tok/s | Prefill tok/s | TTFT s | Wall s | MTP accepted / drafted | Server's peak footprint GB (`time -l`) |
|---|---|---|---|---|---|---|---|---|
| 1 | A | 1234 | 25.80 | 700.0 | 26.27 | 66.07 | 446 / 734 = 60.8% | 108.5 |
| 2 | B | 1234 | 26.00 | 701.2 | 26.23 | 65.73 | 450 / 736 = 61.1% | 108.4 |
| 3 | B | 4321 | 26.10 | 700.5 | 26.25 | 65.68 | 450 / 734 = 61.3% | 108.5 |
| 4 | A | 4321 | 25.90 | 698.6 | 26.32 | 65.93 | 450 / 735 = 61.2% | 108.5 |
| 5 | B | 1234 | 26.00 | 700.4 | 26.26 | 65.76 | 448 / 727 = 61.6% | 108.5 |
| 6 | A | 1234 | 25.90 | 702.4 | 26.18 | 65.88 | 434 / 657 = 66.1% | 108.5 |
| 7 | A | 4321 | 25.70 | 700.3 | 26.26 | 66.21 | 425 / 644 = 66.0% | 108.5 |
| 8 | B | 4321 | 25.90 | 701.9 | 26.20 | 65.88 | 452 / 738 = 61.2% | 108.5 |
| A, mean of 4 | | | 25.83 | 700.3 | 26.26 | 66.02 | 1,755 / 2,770 = 63.4% | 108.5 |
| B, mean of 4 | | | 26.00 | 701.0 | 26.24 | 65.76 | 1,800 / 2,935 = 61.3% | 108.5 |

A's and B's unrounded means were 25.825 and 26.000 tok/s decode, and 700.325 and 701.000 tok/s prefill. Their TTFT means
were 26.258 and 26.235 s, their wall times 66.0225 and 65.7625 s, and their peak footprints 108.500 and 108.475 GB. B's
decode was 0.68% higher and its prefill 0.10% higher. In runs 1 to 4 and 5 to 8, B's decode was 0.77% and 0.58% higher,
and its prefill 0.22% higher and 0.03% lower. B's TTFT was 0.09% and its wall time 0.39% lower (all calculated). Rule 3
of our decision rules kept RQ.

mlx-serve's MTP regime gate picks the drafting pattern from measured times, so runs of the same text can draft different
tokens. No A run drafted the same tokens as a B run. Free RAM fell to 0.994 to 1.725 GB a run in A and to 0.774 to 1.404
GB in B. Each run had 12 to 125 pageouts, and swap stayed at 0.127 GB in every run. Both arms opened with 214 prompt
slot rows. A held 236 decode slot rows, and RQ gave B's requests 237. MLX's peak through the prompt pass was 94,727 to
94,857 MiB in A and 94,693 to 94,856 MiB in B.

### The served dylib at 31,652 tokens (its 32K check)

The 32K check of 4856f028 ran the ladder's greedy 31,652-token prompt with the ladder's procedure. A warm-up ran first.
Then 4 new servers ran with seeds 1234, 1234, 4321 and 4321, each after the chip cooled to 40 C, at max_tokens 1,024.
All numbers are measured.

| Run | Seed | Decode tok/s | Prefill tok/s | TTFT s | Wall s | MTP accepted / drafted | Peak footprint GB |
|---|---|---|---|---|---|---|---|
| 1 | 1234 | 23.70 | 616.5 | 51.36 | 94.65 | 438 / 645 = 67.9% | 114.6 |
| 2 | 1234 | 23.80 | 614.7 | 51.51 | 94.72 | 435 / 650 = 66.9% | 114.8 |
| 3 | 4321 | 23.50 | 615.8 | 51.42 | 95.06 | 457 / 680 = 67.2% | 114.8 |
| 4 | 4321 | 23.70 | 614.4 | 51.54 | 94.83 | 439 / 654 = 67.1% | 114.7 |
| Mean of 4 | | 23.68 | 615.4 | 51.46 | 94.82 | 1,769 / 2,629 = 67.3% | 114.7 |

Against the ladder's means at that size on 01ca9769, decode was 2.0% lower and prefill 0.2% higher. We calculated these
from the unrounded means, 23.675 and 615.35 tok/s. The two ran hours apart, not interleaved as our A/B tests are, so
they give no speed result for RQ. We infer that decode's 221 slot rows, against 236, cause the lower decode.

### 01ca9769, before RQ

Our main standard A/B test of 01ca9769, of 2026-11-10, ran it as arm A. We leave out arm B, PRED_F16, which we rejected.
Each run sent an 18,380-token coding prompt to a new server on our M5 Max and decoded 1,024 tokens greedily (below). All
numbers are measured.

| Dylib | Decode tok/s | Prefill tok/s | TTFT s | Wall s | MTP accepted / drafted | Peak footprint GB |
|---|---|---|---|---|---|---|
| 01ca9769, mean of 4 | 25.78 | 700.4 | 26.26 | 66.09 | 1,783 / 2,877 = 62.0% | 108.5 |

All 4 runs decoded the same text, 09571f6a. Free RAM fell to 0.25 to 1.35 GB a run, and swap stayed flat. Peak
footprint, the server process's peak memory (`time -l`), is larger than MLX's peak, which "Memory" compares with the
plan's ceiling. Earlier standard A/B tests measured the re-fit, EMBED_ROWS, KDA_GATE_EPI and KDA_NORM_PREP, each against
the build before it, and each kept text 09571f6a (below).

Our context ladder ran one coding task at six prompt sizes on 01ca9769, on 2026-11-10 and 2026-11-11 (below). The table
gives its greedy means, all measured.

| Prompt tokens | Decode tok/s | Prefill tok/s | TTFT s | Wall s | MTP acceptance | Peak footprint GB |
|---|---|---|---|---|---|---|
| 1,120 | 31.58 | 114.0 | 9.83 | 42.35 | 70.0% | 99.1 |
| 2,042 | 32.30 | 190.9 | 10.71 | 42.50 | 69.9% | 99.4 |
| 4,094 | 28.78 | 364.1 | 11.25 | 46.98 | 69.6% | 100.5 |
| 8,185 | 26.65 | 659.1 | 12.43 | 50.95 | 70.4% | 103.4 |
| 16,377 | 26.53 | 708.7 | 23.12 | 61.81 | 65.8% | 109.3 |
| 31,652 | 24.15 | 614.3 | 51.55 | 94.03 | 67.5% | 116.0 |

Up to 8,185 tokens, TTFT stayed between 9.83 and 12.43 s, so prefill tok/s rose with the prompt's size. Decode fell from
about 32 tok/s at the two smallest sizes to 24.15 tok/s at 31,652 tokens. Greedy decoding gave the same text in all 4
runs at each size. In the ladder, free RAM fell as low as 0.06 GB, and swap stayed flat (measured).

### How we measured, and 01ca9769's runs

The prompt is the coding prompt of glm53-eval, our benchmark harness. The runs used seeds 1234 and 4321, and greedy
decoding, so the seed has no effect. The 8 runs of an A/B go A B B A, then B A A B, each after the chip cooled to 40 C.
They ran on mlx-serve main 66aad949, with a 112 GiB GPU wired limit, the 2.60bpw pack and step 6's options. Before the
runs, each arm's dylib served one warm-up run, which the means leave out. We did not record the macOS version or the
SSD. Decode and prefill come from mlx-serve's own lines, and TTFT and wall time from the client. The peak is the server
process's peak memory footprint (`time -l`). The MTP hit rate is the MTP head's accepted over drafted tokens, from
mlx-serve's `[spec-stats]` lines. Prompt lookup, mlx-serve's drafting from matches in the prompt, also drafts tokens.

In the main test, of 2026-11-10, arm B was PRED_F16, dylib 97c37406 with `TURBOEXL3_LING_PRED_F16=1`. Its read-ahead
predictor reads the pack's f16 router weight, in place of a 272 MiB bf16 copy. We rejected it under rule 6 of our
decision rules. Arm A ran on 2026-11-10, in the test's window from 21:22:11 to 21:54:34Z. Prompt lookup drafted 210 to
216 tokens in each run, and the trunk accepted 201 to 204 of them. Swap stayed at 0.127 GB, and each run had 44 to 121
pageouts. The runs held 214 and 236 slot rows, and MLX's peak through the prompt pass was 94,693 to 94,856 MiB. Each
request line named 35 of the 40 paths ("Logs: the run-path lines"). Arm A's runs, all measured:

| Run | Seed | Decode tok/s | Prefill tok/s | TTFT s | Wall s | MTP accepted / drafted | Server's peak footprint GB (`time -l`) |
|---|---|---|---|---|---|---|---|
| 1 | 1234 | 26.00 | 698.8 | 26.31 | 65.77 | 450 / 734 = 61.3% | 108.5 |
| 4 | 4321 | 25.60 | 701.9 | 26.20 | 66.27 | 431 / 673 = 64.0% | 108.5 |
| 6 | 1234 | 25.90 | 701.5 | 26.22 | 65.83 | 452 / 737 = 61.3% | 108.5 |
| 7 | 4321 | 25.60 | 699.5 | 26.29 | 66.49 | 450 / 733 = 61.4% | 108.3 |

A later standard A/B test, SPEC0_IDLE's, also ran 01ca9769 as arm A, on 2026-11-11 from 04:25:10 to 04:58:03Z. Its arm
B, dylib 965ef461 with `TURBOEXL3_LING_SPEC0_IDLE=1`, holds the first speculative read thread to the second one's idle
rule. We rejected it under rule 6. Arm A decoded 09571f6a in all 4 runs. Free RAM fell to 0.25 to 0.82 GB a run, with
184 to 256 pageouts. Arm A's runs in that test, all measured:

| Run | Seed | Decode tok/s | Prefill tok/s | TTFT s | Wall s | MTP accepted / drafted | Server's peak footprint GB (`time -l`) |
|---|---|---|---|---|---|---|---|
| 1 | 1234 | 25.90 | 698.4 | 26.33 | 66.00 | 466 / 752 = 62.0% | 108.4 |
| 4 | 4321 | 25.90 | 698.7 | 26.32 | 66.06 | 447 / 733 = 61.0% | 108.4 |
| 6 | 1234 | 25.90 | 699.7 | 26.28 | 65.88 | 451 / 738 = 61.1% | 108.3 |
| 7 | 4321 | 25.70 | 697.9 | 26.35 | 66.37 | 463 / 766 = 60.4% | 108.4 |
| Mean of 4 | | 25.85 | 698.7 | 26.32 | 66.08 | 1,827 / 2,989 = 61.1% | 108.4 |

### KDA_NORM_PREP's A/B test

Our standard A/B test of KDA_NORM_PREP compared 63668c19, which we served before 01ca9769, as arm A with 7f053407 as arm
B. B ran KDA_NORM_PREP as an option, `TURBOEXL3_LING_KDA_NORM_PREP=1`. The runs and settings were as above. All numbers
are measured.

| Arm | Decode tok/s | Prefill tok/s | TTFT s | Wall s | MTP accepted / drafted | Server's peak footprint GB (`time -l`) |
|---|---|---|---|---|---|---|
| A, mean of 4 | 25.80 | 696.9 | 26.39 | 66.19 | 1,778 / 2,877 = 61.8% | 108.4 |
| B, mean of 4 | 25.83 | 700.3 | 26.26 | 66.04 | 1,788 / 2,958 = 60.4% | 108.4 |

Both arms decoded the same text, 09571f6a, in all 8 runs. B's prefill was 0.49% higher and its decode 0.10% higher,
calculated from the unrounded means. Prompt lookup drafted 198 to 216 tokens in every run, and the trunk accepted 187 to
205 of them. Free RAM fell to 0.25 to 2.14 GB a run. Swap stayed flat at 0.127 GB in every run, and each run had 7 to
100 pageouts. The runs:

| Run | Arm | Seed | Decode tok/s | Prefill tok/s | TTFT s | Wall s | MTP accepted / drafted | Server's peak footprint GB (`time -l`) |
|---|---|---|---|---|---|---|---|---|
| 1 | A | 1234 | 25.50 | 699.2 | 26.30 | 66.50 | 450 / 737 = 61.1% | 108.5 |
| 2 | B | 1234 | 25.70 | 701.4 | 26.22 | 66.23 | 446 / 746 = 59.8% | 108.4 |
| 3 | B | 4321 | 25.60 | 700.6 | 26.25 | 66.40 | 446 / 750 = 59.5% | 108.4 |
| 4 | A | 4321 | 25.80 | 697.0 | 26.39 | 66.14 | 445 / 739 = 60.2% | 108.4 |
| 5 | B | 1234 | 26.10 | 699.7 | 26.28 | 65.62 | 450 / 734 = 61.3% | 108.5 |
| 6 | A | 1234 | 25.90 | 696.9 | 26.39 | 66.07 | 431 / 667 = 64.6% | 108.4 |
| 7 | A | 4321 | 26.00 | 694.5 | 26.48 | 66.03 | 452 / 734 = 61.6% | 108.4 |
| 8 | B | 4321 | 25.90 | 699.5 | 26.30 | 65.92 | 446 / 728 = 61.3% | 108.3 |

B's TTFT was 0.48% and its wall time 0.22% lower than A's (calculated). In runs 1 to 4 and in runs 5 to 8, B's prefill
was higher and its TTFT and wall time lower. B's decode equaled A's in runs 1 to 4 and was 0.19% higher in runs 5 to 8.
Every B run's prefill, 699.5 to 701.4 tok/s, was above every A run's, 694.5 to 699.2. KDA_NORM_PREP changes only the
prompt pass, and the arms' MTP drafts differed, so we read the decode difference as noise from the drafts. mlx-serve's
MTP regime gate picks the drafting pattern from measured times, so runs of the same text can draft different tokens. No
A run drafted the same tokens as a B run. Both arms held 214 and 236 slot rows. MLX's peak through the prompt pass was
94,693 to 94,856 MiB in each arm.

Rule 3 of our decision rules keeps a change when prefill or decode is faster by more than the 0.25% wash band, and the
other is no worse than −1% (our ledger). KDA_NORM_PREP met it with prefill +0.49% and decode +0.10%.

### KDA_GATE_EPI's A/B test

Our standard A/B test of KDA_GATE_EPI compared dylib 73db9cb6 (commit 859cd93) as arm B with 7e446929 (commit 1a16e40)
as arm A. Only B ran KDA_GATE_EPI, as an option, `TURBOEXL3_LING_KDA_GATE_EPI=1`. Neither arm had the drafter decline
(d73e465). The runs and settings were as above. All numbers are measured.

| Arm | Decode tok/s | Prefill tok/s | TTFT s | Wall s | MTP accepted / drafted | Server's peak footprint GB (`time -l`) |
|---|---|---|---|---|---|---|
| A, mean of 4 | 25.93 | 695.9 | 26.42 | 66.06 | 1,805 / 2,959 = 61.0% | 108.4 |
| B, mean of 4 | 26.05 | 697.7 | 26.36 | 65.80 | 1,759 / 2,791 = 63.0% | 108.4 |

Both arms decoded the same text, 09571f6a, in all 8 runs. B's decode was 0.48% higher and its prefill 0.26% higher
(calculated). KDA_GATE_EPI changes only the prompt pass, and the arms' MTP drafts differed, so we read the decode
difference as noise from the drafts. Prompt lookup drafted 195 to 219 tokens in every run, and the trunk accepted 186 to
208 of them.

Free RAM fell to 1.24 to 2.16 GB a run. Memory pressure stayed at 1, normal, in every run, and swap stayed flat at 0.127
GB. Each run had 5 to 113 pageouts. The runs:

| Run | Arm | Seed | Decode tok/s | Prefill tok/s | TTFT s | Wall s | MTP accepted / drafted | Server's peak footprint GB (`time -l`) |
|---|---|---|---|---|---|---|---|---|
| 1 | A | 1234 | 25.70 | 697.8 | 26.35 | 66.32 | 454 / 781 = 58.1% | 108.4 |
| 2 | B | 1234 | 26.10 | 697.5 | 26.37 | 65.67 | 450 / 734 = 61.3% | 108.4 |
| 3 | B | 4321 | 25.90 | 697.8 | 26.35 | 65.97 | 429 / 653 = 65.7% | 108.4 |
| 4 | A | 4321 | 26.00 | 696.2 | 26.41 | 65.93 | 436 / 678 = 64.3% | 108.4 |
| 5 | B | 1234 | 26.00 | 698.3 | 26.34 | 65.89 | 430 / 670 = 64.2% | 108.4 |
| 6 | A | 1234 | 26.00 | 697.2 | 26.38 | 65.89 | 467 / 755 = 61.9% | 108.4 |
| 7 | A | 4321 | 26.00 | 692.5 | 26.55 | 66.09 | 448 / 745 = 60.1% | 108.4 |
| 8 | B | 4321 | 26.20 | 697.2 | 26.38 | 65.66 | 450 / 734 = 61.3% | 108.4 |

B's TTFT was 0.24% and its wall time 0.39% lower than A's (calculated). Runs 1 to 4 and runs 5 to 8 each gave the same
signs as the means. mlx-serve's MTP regime gate picks the drafting pattern from measured times, so runs of the same text
can draft different tokens. No A run drafted the same tokens as a B run. Both arms held 214 and 236 slot rows. MLX's
peak through the prompt pass was 94,693 to 94,857 MiB in A and 94,693 to 94,727 MiB in B.

### The re-fit's A/B test

Before EMBED_ROWS, our standard A/B test of the re-fit compared commit 0ebec77 (dylib fdd4e0d8) as arm B with the build
before it as arm A. That build was 0e1f5c0 (dylib c8d54a83), the earlier context ladder's build. Both ran turboexl3
c2783f5 on mlx-serve main 66aad949, with the embedding table resident. B differs from A only in MEM_REFIT's pass part,
989 MiB in place of 3,606 MiB. That gives B's bank 8 more prompt slot rows and lowers the refusal's limit from 2,941 to
2,426 tokens. The runs and settings were as above. All numbers are measured.

| Run | Arm | Seed | Decode tok/s | Prefill tok/s | TTFT s | Wall s | MTP accepted / drafted | Server's peak footprint GB (`time -l`) |
|---|---|---|---|---|---|---|---|---|
| 1 | A | 1234 | 25.60 | 693.7 | 26.51 | 66.64 | 426 / 650 = 65.5% | 105.6 |
| 2 | B | 1234 | 26.00 | 697.6 | 26.36 | 65.93 | 447 / 715 = 62.5% | 108.3 |
| 3 | B | 4321 | 25.70 | 697.4 | 26.37 | 66.38 | 429 / 656 = 65.4% | 108.4 |
| 4 | A | 4321 | 25.50 | 695.4 | 26.45 | 66.68 | 429 / 651 = 65.9% | 105.6 |
| 5 | B | 1234 | 25.60 | 696.3 | 26.41 | 66.51 | 430 / 660 = 65.2% | 108.4 |
| 6 | A | 1234 | 25.60 | 695.4 | 26.45 | 66.53 | 430 / 657 = 65.4% | 105.6 |
| 7 | A | 4321 | 25.90 | 695.5 | 26.44 | 66.06 | 447 / 715 = 62.5% | 105.6 |
| 8 | B | 4321 | 25.50 | 693.2 | 26.53 | 66.85 | 444 / 718 = 61.8% | 108.2 |
| B, mean of 4 | | | 25.70 | 696.1 | 26.42 | 66.42 | 1,750 / 2,749 = 63.7% | 108.3 |
| A, mean of 4 | | | 25.65 | 695.0 | 26.46 | 66.48 | 1,732 / 2,673 = 64.8% | 105.6 |

Both arms decoded 09571f6a in all 8 runs. Run 2 (B) and run 7 (A) accepted the same 447 of 715 drafted tokens, in 373
verify passes each. They took 105.74 and 105.87 ms for each verify pass. Prompt lookup drafted 213 to 222 tokens in each
run, and the trunk accepted 201 to 205 of them. Each run had 18 to 229 pageouts, and swap stayed flat at 0.136 GB. Free
RAM fell to 0.07 to 1.06 GB a run. Memory pressure stayed at 1, normal, at every reading of our gate check
(abcheck.txt).

B's runs held 210 and 233 slot rows, and A's runs 202 and 233. MLX's peak through the prompt pass was 94,636 to 94,800
MiB in B and 92,070 to 92,233 MiB in A. B's peak footprint is 2.7 GB above A's, about the 8 more slot rows of
336,427,008 B each (2.69 GB, calculated).

### An early build

The same workload on an M5 Max with 128 GiB, 2026-11-06, on mlx-serve 26.11.1 with `--ctx-size 20480`. That build of the
plugin (its mlx-serve copy at bc6c5f2c) came before the bill in src/bill.zig and before every optimization of "The
served stack". It loaded no MTP head, so it drafted by prompt lookup only (its `[spec-stats]` lines show `mode=pld`).
The bank then had 228 / 250 slot rows (2.60) and 285-286 / 312-313 (2.10). The Mac's free RAM fell to 0.15-2.6 GB, with
109 to 360 pageouts in each run. All numbers are measured.

| Pack | Decode tok/s | Prefill tok/s | TTFT s | Wall s | PLD hit % | Peak footprint GB |
|---|---|---|---|---|---|---|
| 2.60bpw (4 runs) | 13.30 | 276.4 | 66.52 | 143.46 | 49.4 | 118.3 |
| 2.10bpw (4 runs) | 13.98 | 306.9 | 59.92 | 133.32 | 42.4 | 118.3 |

The two packs decode different text, so this is a record per pack, not a same-text comparison. The `[stream-stats]`
lines give the bank's hit rates and the SSD reads:

| Pack | Prompt pass hit | Prompt pass reads | Decode hit |
|---|---|---|---|
| 2.60bpw | 45.7% | 389 GB at 13.0 GB/s | 93.3% |
| 2.10bpw | 56.1% | 252 GB at 12.6 GB/s | 97.1% |

### The context ladder

The ladder (`~/lingp/bench/ladder.sh`) ran from 2026-11-10 at 21:58:34Z to 2026-11-11 at 01:52:59Z. It held the GPU lock
and the exclusive build lock for the whole series. So no other GPU job or build of ours could start during it. It
measured 01ca9769 (commit bd00b1a), before RQ, with turboexl3 4bf2df8 on mlx-serve main 66aad949. It used the 2.60bpw
pack, a 112 GiB GPU wired limit and step 6's options, and it set no plugin option (its greedy.env and card.env). Its
prompts are coding.json's task after CPython 3.14.4 code, cut to six sizes. Ling's own tokenizer counted them (the
ladder's `manifest.json`). At the smallest size the task alone is 1,120 tokens, over the 1,024-token target. Each size
ran greedy, and then with the model card's settings: temperature 1.0, top_p 0.95 and top_k 20.

For each size and mode, one warm-up ran first, which the means leave out. Then 4 new servers ran, with seeds 1234, 1234,
4321 and 4321, each after the chip cooled to 40 C, at max_tokens 1,024. At 31,652 tokens that leaves 92 tokens of the
32,768-token context (calculated). All 48 measured runs finished. No size failed, and the memory guard stopped no run.
The guard stops a run at critical memory pressure or at more than 1 GB of swap growth (memguard.sh's kill mode, in our
results folder `ladder-01ca9769/scripts/`). The memory pressure level stayed at 1, normal, and swap stayed flat at 0.127
GB in every run. The card settings gave a different text in each run.

We have not broken the times down. A call holds at most 18,431 rows (plugin.zig `layer_major_rows` and `callRows`). So
each routed layer runs its experts in one call at 16,377 tokens, and in two at 31,652. Each call reads the experts it
needs.

The numbers in the two run tables are measured. Decode comes before prefill, as in "Measured". The peak is the server
process's peak memory footprint in GB (10^9 bytes), as the parser of our standard A/B test reports it
(~/glm53/tools/parse_run.py). Each mean row averages its 4 runs, and its MTP column adds their accepted and drafted
tokens. The text column gives each run's text hash.

The first table gives the greedy runs.

| Prompt tokens | Run | Seed | Decode tok/s | Prefill tok/s | TTFT s | Wall s | MTP accepted / drafted | Peak footprint GB | Text |
|---|---|---|---|---|---|---|---|---|---|
| 1,120 | 1 | 1234 | 32.00 | 114.1 | 9.82 | 41.88 | 462 / 644 = 71.7% | 99.1 | 45d479f2 |
| 1,120 | 2 | 1234 | 32.30 | 114.0 | 9.83 | 41.57 | 473 / 670 = 70.6% | 99.1 | 45d479f2 |
| 1,120 | 3 | 4321 | 30.60 | 113.9 | 9.84 | 43.42 | 441 / 678 = 65.0% | 99.1 | 45d479f2 |
| 1,120 | 4 | 4321 | 31.40 | 113.9 | 9.84 | 42.52 | 439 / 600 = 73.2% | 99.1 | 45d479f2 |
| 1,120 | mean of 4 | | 31.58 | 114.0 | 9.83 | 42.35 | 1,815 / 2,592 = 70.0% | 99.1 | same in all 4 |
| 2,042 | 1 | 1234 | 32.20 | 190.5 | 10.73 | 42.65 | 437 / 636 = 68.7% | 99.4 | 452c1992 |
| 2,042 | 2 | 1234 | 32.20 | 191.3 | 10.68 | 42.54 | 444 / 639 = 69.5% | 99.4 | 452c1992 |
| 2,042 | 3 | 4321 | 32.40 | 190.8 | 10.71 | 42.39 | 439 / 629 = 69.8% | 99.4 | 452c1992 |
| 2,042 | 4 | 4321 | 32.40 | 190.8 | 10.71 | 42.42 | 442 / 618 = 71.5% | 99.4 | 452c1992 |
| 2,042 | mean of 4 | | 32.30 | 190.9 | 10.71 | 42.50 | 1,762 / 2,522 = 69.9% | 99.4 | same in all 4 |
| 4,094 | 1 | 1234 | 28.90 | 364.3 | 11.25 | 46.84 | 605 / 853 = 70.9% | 100.5 | 99e00e60 |
| 4,094 | 2 | 1234 | 28.60 | 363.8 | 11.26 | 47.21 | 612 / 898 = 68.2% | 100.5 | 99e00e60 |
| 4,094 | 3 | 4321 | 28.60 | 363.4 | 11.27 | 47.25 | 613 / 901 = 68.0% | 100.5 | 99e00e60 |
| 4,094 | 4 | 4321 | 29.00 | 364.8 | 11.23 | 46.61 | 608 / 851 = 71.4% | 100.5 | 99e00e60 |
| 4,094 | mean of 4 | | 28.78 | 364.1 | 11.25 | 46.98 | 2,438 / 3,503 = 69.6% | 100.5 | same in all 4 |
| 8,185 | 1 | 1234 | 26.50 | 658.5 | 12.44 | 51.20 | 570 / 808 = 70.5% | 103.4 | c5866685 |
| 8,185 | 2 | 1234 | 27.00 | 659.0 | 12.43 | 50.44 | 585 / 841 = 69.6% | 103.4 | c5866685 |
| 8,185 | 3 | 4321 | 26.30 | 658.7 | 12.44 | 51.46 | 547 / 743 = 73.6% | 103.4 | c5866685 |
| 8,185 | 4 | 4321 | 26.80 | 660.3 | 12.41 | 50.69 | 581 / 851 = 68.3% | 103.4 | c5866685 |
| 8,185 | mean of 4 | | 26.65 | 659.1 | 12.43 | 50.95 | 2,283 / 3,243 = 70.4% | 103.4 | same in all 4 |
| 16,377 | 1 | 1234 | 26.40 | 709.3 | 23.10 | 62.01 | 460 / 721 = 63.8% | 109.3 | 84d457cd |
| 16,377 | 2 | 1234 | 26.70 | 707.2 | 23.17 | 61.57 | 460 / 679 = 67.7% | 109.3 | 84d457cd |
| 16,377 | 3 | 4321 | 26.40 | 708.2 | 23.14 | 61.98 | 458 / 697 = 65.7% | 109.3 | 84d457cd |
| 16,377 | 4 | 4321 | 26.60 | 710.0 | 23.08 | 61.66 | 465 / 706 = 65.9% | 109.3 | 84d457cd |
| 16,377 | mean of 4 | | 26.53 | 708.7 | 23.12 | 61.81 | 1,843 / 2,803 = 65.8% | 109.3 | same in all 4 |
| 31,652 | 1 | 1234 | 23.70 | 615.3 | 51.47 | 94.73 | 456 / 682 = 66.9% | 116.0 | 919cf0fb |
| 31,652 | 2 | 1234 | 24.20 | 613.8 | 51.59 | 93.95 | 432 / 637 = 67.8% | 116.0 | 919cf0fb |
| 31,652 | 3 | 4321 | 24.10 | 614.6 | 51.52 | 94.06 | 454 / 673 = 67.5% | 116.0 | 919cf0fb |
| 31,652 | 4 | 4321 | 24.60 | 613.4 | 51.62 | 93.39 | 435 / 641 = 67.9% | 115.9 | 919cf0fb |
| 31,652 | mean of 4 | | 24.15 | 614.3 | 51.55 | 94.03 | 1,777 / 2,633 = 67.5% | 116.0 | same in all 4 |

The second table gives the runs with the card settings.

| Prompt tokens | Run | Seed | Decode tok/s | Prefill tok/s | TTFT s | Wall s | MTP accepted / drafted | Peak footprint GB | Text |
|---|---|---|---|---|---|---|---|---|---|
| 1,120 | 1 | 1234 | 32.40 | 113.8 | 9.85 | 41.52 | 476 / 679 = 70.1% | 99.1 | 8b564406 |
| 1,120 | 2 | 1234 | 31.20 | 113.9 | 9.84 | 42.77 | 530 / 772 = 68.7% | 99.1 | 86b7410b |
| 1,120 | 3 | 4321 | 31.60 | 113.9 | 9.84 | 42.27 | 469 / 754 = 62.2% | 99.1 | 96d7e007 |
| 1,120 | 4 | 4321 | 30.60 | 114.0 | 9.83 | 43.37 | 526 / 800 = 65.8% | 99.1 | ae482e4f |
| 1,120 | mean of 4 | | 31.45 | 113.9 | 9.84 | 42.48 | 2,001 / 3,005 = 66.6% | 99.1 | 4 different |
| 2,042 | 1 | 1234 | 26.60 | 190.3 | 10.74 | 49.35 | 533 / 980 = 54.4% | 99.5 | 55539ea7 |
| 2,042 | 2 | 1234 | 26.60 | 190.5 | 10.73 | 49.27 | 506 / 918 = 55.1% | 99.4 | 63ecd9f3 |
| 2,042 | 3 | 4321 | 25.50 | 190.1 | 10.75 | 50.92 | 516 / 889 = 58.0% | 99.4 | b668ad08 |
| 2,042 | 4 | 4321 | 25.80 | 190.4 | 10.73 | 50.55 | 514 / 898 = 57.2% | 99.5 | 4987cd02 |
| 2,042 | mean of 4 | | 26.13 | 190.3 | 10.74 | 50.02 | 2,069 / 3,685 = 56.1% | 99.5 | 4 different |
| 4,094 | 1 | 1234 | 24.20 | 357.6 | 11.46 | 53.92 | 485 / 875 = 55.4% | 100.5 | 955c61d1 |
| 4,094 | 2 | 1234 | 25.50 | 363.1 | 11.28 | 51.56 | 534 / 972 = 54.9% | 100.5 | 1a0b27be |
| 4,094 | 3 | 4321 | 27.80 | 363.1 | 11.28 | 48.27 | 469 / 815 = 57.5% | 100.5 | bdc5a356 |
| 4,094 | 4 | 4321 | 24.70 | 364.3 | 11.25 | 52.75 | 528 / 1022 = 51.7% | 100.5 | f9673327 |
| 4,094 | mean of 4 | | 25.55 | 362.0 | 11.32 | 51.63 | 2,016 / 3,684 = 54.7% | 100.5 | 4 different |
| 8,185 | 1 | 1234 | 23.80 | 658.0 | 12.45 | 55.62 | 444 / 631 = 70.4% | 103.5 | ef6cc6b3 |
| 8,185 | 2 | 1234 | 24.60 | 659.2 | 12.43 | 54.15 | 518 / 822 = 63.0% | 103.5 | 71e612e4 |
| 8,185 | 3 | 4321 | 27.00 | 659.7 | 12.42 | 50.47 | 529 / 721 = 73.4% | 103.5 | 0926a4de |
| 8,185 | 4 | 4321 | 26.80 | 658.8 | 12.43 | 50.72 | 540 / 703 = 76.8% | 103.5 | cde6c5a3 |
| 8,185 | mean of 4 | | 25.55 | 658.9 | 12.43 | 52.74 | 2,031 / 2,877 = 70.6% | 103.5 | 4 different |
| 16,377 | 1 | 1234 | 22.50 | 710.0 | 23.08 | 68.79 | 489 / 789 = 62.0% | 109.4 | 9c23edd1 |
| 16,377 | 2 | 1234 | 22.50 | 706.8 | 23.18 | 68.81 | 487 / 810 = 60.1% | 109.4 | e4fb743b |
| 16,377 | 3 | 4321 | 25.00 | 709.4 | 23.10 | 64.17 | 529 / 711 = 74.4% | 109.4 | b5995805 |
| 16,377 | 4 | 4321 | 24.40 | 708.0 | 23.15 | 65.17 | 531 / 788 = 67.4% | 109.4 | 4ef240e8 |
| 16,377 | mean of 4 | | 23.60 | 708.6 | 23.13 | 66.74 | 2,036 / 3,098 = 65.7% | 109.4 | 4 different |
| 31,652 | 1 | 1234 | 19.70 | 612.1 | 51.73 | 103.70 | 521 / 927 = 56.2% | 116.0 | 98d01385 |
| 31,652 | 2 | 1234 | 20.00 | 613.8 | 51.59 | 102.93 | 498 / 881 = 56.5% | 115.6 | 8e7bb50d |
| 31,652 | 3 | 4321 | 22.50 | 613.0 | 51.66 | 97.39 | 532 / 761 = 69.9% | 116.0 | 7d9d1207 |
| 31,652 | 4 | 4321 | 22.30 | 612.7 | 51.68 | 97.78 | 524 / 721 = 72.7% | 116.0 | 77fec13b |
| 31,652 | mean of 4 | | 21.13 | 612.9 | 51.67 | 100.45 | 2,075 / 3,290 = 63.1% | 115.9 | 4 different |

An earlier ladder ran the same prompts, byte for byte, on dylib c8d54a83 (commit 0e1f5c0). Its bank held 202 prompt and
233 decode slot rows. These changes came between that build and the served one, in this order:

- the re-fit (0ebec77), which gave the bank 8 more prompt slot rows and moved the prompt log-probability limit from
  2,941 to 2,426 tokens;
- the room change (165d8ca), which moved the limit to 3,317 tokens;
- EMBED_ROWS (1a16e40), after which the bank held 214 prompt and 236 decode slot rows;
- KDA_GATE_EPI (4c13ec7), served together with the DFlash capture refusal (d73e465);
- the test-only commits (1dffdcd);
- KDA_NORM_PREP (bd00b1a).

The re-fit, the room change, the DFlash refusal and the test-only commits change no forward pass's output. The earlier
ladder ran on 2026-11-10 from 00:25:08 to 04:14:36Z, in series with the final one, not interleaved as our A/B tests are.
So the table below records the two builds, and it gives no speed result for any one change. Every greedy run of the
final ladder decoded the earlier ladder's text at its size and seed, 24 of 24 runs. The card runs decoded a different
text in each run, so their means compare different texts. The means are measured, and each change is calculated from the
unrounded means.

| Mode | Prompt tokens | Decode tok/s, final | Decode tok/s, earlier | Decode change | Prefill tok/s, final | Prefill tok/s, earlier | Prefill change | Peak footprint GB, final | Peak footprint GB, earlier |
|---|---|---|---|---|---|---|---|---|---|
| Greedy | 1,120 | 31.58 | 31.00 | +1.85% | 114.0 | 114.7 | −0.63% | 99.1 | 98.9 |
| Greedy | 2,042 | 32.30 | 32.03 | +0.86% | 190.9 | 191.2 | −0.20% | 99.4 | 99.4 |
| Greedy | 4,094 | 28.78 | 28.63 | +0.52% | 364.1 | 366.4 | −0.62% | 100.5 | 100.9 |
| Greedy | 8,185 | 26.65 | 26.50 | +0.57% | 659.1 | 647.3 | +1.82% | 103.4 | 103.7 |
| Greedy | 16,377 | 26.53 | 26.68 | −0.56% | 708.7 | 705.8 | +0.41% | 109.3 | 109.6 |
| Greedy | 31,652 | 24.15 | 23.43 | +3.09% | 614.3 | 613.2 | +0.18% | 116.0 | 115.9 |
| Card settings | 1,120 | 31.45 | 30.75 | +2.28% | 113.9 | 114.6 | −0.59% | 99.1 | 99.0 |
| Card settings | 2,042 | 26.13 | 25.75 | +1.46% | 190.3 | 190.6 | −0.14% | 99.5 | 99.6 |
| Card settings | 4,094 | 25.55 | 24.85 | +2.82% | 362.0 | 365.9 | −1.05% | 100.5 | 100.8 |
| Card settings | 8,185 | 25.55 | 25.15 | +1.59% | 658.9 | 646.7 | +1.90% | 103.5 | 103.8 |
| Card settings | 16,377 | 23.60 | 23.35 | +1.07% | 708.6 | 704.8 | +0.54% | 109.4 | 109.7 |
| Card settings | 31,652 | 21.13 | 21.28 | −0.71% | 612.9 | 612.4 | +0.08% | 115.9 | 115.9 |

Greedy decode was higher at five of the six sizes. In both modes, prefill was lower at the three smallest sizes and
higher at the three largest.

The memory curve sets MLX's peak against the bill at 01ca9769. The `[mem]` line at the end of a prompt pass gives MLX's
peak since the server started, so it never covers the request's decode. At the four sizes up to 8,185 tokens it reads
91,165 MiB in every run. The value does not change with the prompt's size, so we take it to be the warm-up's decode
phase. The line then masks the request's own prompt-pass peak, which was at most that. The earlier ladder's line masked
16,377 tokens too, at 91,431 MiB. In the final ladder, the peak at 16,377 tokens is the request's own, 92,885 to 92,887
MiB. That is 3,128 to 3,130 MiB under its terms of 96,015 MiB. At 31,652 tokens the peak is 101,429 to 101,430 MiB,
against 100,442 MiB of terms. Those terms are 14,734 MiB for the trunk, 68,660 for the bank's prompt slot rows and 1,132
for its prompt windows. The KV of the 32,000 tokens that mlx-serve allocated takes 9,070, `prompt_work` 2,048 and the
layer-major pass 4,797. The 987 to 988 MiB left is within MEM_REFIT's pass part, 989 MiB, as in the 32K checks. The peak
is 426 to 427 MiB under the plan's ceiling (calculated). At that size the peak footprint reached 115.6 to 116.0 GB, and
the lowest "Pages free" reading 0.065 GB, at pressure level 1. So the curve's MLX peaks are prompt-pass peaks. At 31,652
tokens, decode then passed 32,000 tokens, and the KV grew past the bill's term ("Memory").

| Prompt tokens | MLX peak MiB, from the `[mem]` line | This request's terms without MEM_REFIT, MiB | Planned total MiB | Peak footprint GB, highest | Lowest "Pages free" reading, GB | Pageouts in each run | Pressure level, highest |
|---|---|---|---|---|---|---|---|
| 1,120 | 91,165 (masked) | 91,734 | 93,428 | 99.1 | 3.315 | 0 | 1 |
| 2,042 | 91,165 (masked) | 91,952 | 93,690 | 99.5 | 2.299 | 0 | 1 |
| 4,094 | 91,165 (masked) | 92,532 | 94,271 | 100.5 | 0.974 | 0 | 1 |
| 8,185 | 91,165 (masked) | 93,693 | 95,431 | 103.5 | 0.061 | 6 to 110 | 1 |
| 16,377 | 92,885 to 92,887 | 96,015 | 97,753 | 109.4 | 0.056 | 118 to 265 | 1 |
| 31,652 | 101,429 to 101,430 | 100,442 | 102,083 | 116.0 | 0.065 | 238 to 389 | 1 |

Each row covers both modes' 8 runs at that size. MLX's peak, the planned total, the footprint, the free reading and the
pressure come from report.md's memory curve, and the pageouts from results.tsv. The harness reads vm_stat's free pages
and its pageouts every 5 s (the `mem()` function of run1-mlxserve.sh, in our results folder `ladder-01ca9769/scripts/`).
The terms column comes from memfit v3, our peak-fit script, which counts the bank's prompt windows
(`memfit-windows.txt`). The planned total, report.md's "planned bill" column, is a sum (calculated). It adds the 14,734
MiB in use when the bank opened and the bank line's decode budget of 76,329 MiB. It also adds this request's KV, at
297,216 B for each prompt token, and `prompt_work`'s 2,048 MiB. The decode budget is the prompt budget plus the
layer-major term and all of MEM_REFIT (src/bill.zig `budget`). The bank line rounds each budget down to whole MiB. So
its decode budget, less the prompt budget and the layer-major term, reads 1,534 MiB against MEM_REFIT's 1,533. The
planned total thus includes MEM_REFIT's 544 MiB server part, which MLX does not hold. Compare MLX's peak with the plan's
ceiling, 101,856 MiB, not with this total.

## Gaps for a merge

Before a merge, we must fix the first six items below, and a reviewer must rule on the last five.

- build.zig.zon pins turboexl3 main (c7ce66c), which has no Ling kernels, so the ReleaseFast build fails (measured). The
  fix is to publish turboexl3's Ling kernels before the merge and pin 4bf2df8.
- No one else can make a pack until we publish our converter's Ling script (README, "Requirements"). The fix is to
  publish it before the merge.
- CI's kit, built from the repo's MLX_SERVE_REF (mlx-serve bc6c5f2c), is too old for the plugin (config.zig `kitGap` and
  the comptime check that calls it). The fix is the 26.11.3 release kit, or the repo's MLX_SERVE_REF moved to 66aad949
  or later.
- Under AGENTS.md's rule 7, each bill term must be a measured upper bound. The bill's load-time KV term holds 32,768
  tokens. On 01ca9769 a KV grow step could go past it by up to 2,104 MiB ("The KV past the bill's term"). RQ, served
  since 08:08:00Z on 2026-11-11, bills each request's KV at the capacity that mlx-serve allocates, growth included. It
  refuses by name a prompt forward that would grow mlx-serve's KV past 32,000 tokens ("RQ"). Two long requests at once
  are not tested: RQ counts another live request's KV as it is, not its growth. The fix is a test of two long requests
  at once.
- mlx-serve takes a `--ctx-size` above 32,768 as given, but the bill's load-time KV term holds 32,768 tokens. RQ refuses
  a prompt whose KV would pass 32,000 tokens at any `--ctx-size`. But its decode rows never go under the base, and we
  have not measured decode past 32,768 tokens. The fix refuses a larger value at load, or measures it.
- A 4- or 8-bit KV, from `--kv-quant`, model-settings.json or a request's `kv_quant`, gets no refusal. mlx-serve never
  tells the plugin the scheme at load, so the plugin cannot refuse it there ("The quantized KV"). Each MLA layer then
  reads a dense copy of its KV that no bill term holds, about 1.0 GiB a call at 32,768 tokens. The quantized cache takes
  about 53% (8-bit) or 28% (4-bit) of the KV term (all calculated). We have not measured whether these stay within the
  term, or the text that a lossy KV decodes. Since fb966cc the plugin logs one warning a load, but it refuses nothing.
  The fix is an mlx-serve change, not yet made, that refuses the scheme by name at load and before mlx-serve creates a
  request's cache.
- RQ's stage 1 keeps the prompt pass at the bank's 214-row base. Near 32,000 tokens its plan holds 211 slot rows, so the
  bank stays 3 rows over the plan, and RQ logs `RQ over plan`. MLX's peak stayed 426 MiB under the plan's ceiling at
  31,652 tokens on 4856f028, and 538 MiB under it at 32,000 tokens on 89be02b1 (measured). A reviewer must rule on this
  case, or ask for a streamer that can shrink the prompt pass's rows under its base.
- Our patches and cut-down MLX kernels need a ruling under rule 9, on edits to third-party code (rules table below).
- Four other rule 7 choices need a ruling (rules table below).
- Parity's four-statistic bar needs a ruling, or we must serve without MLA_FLASH ("Parity").
- The parity test needs a ruling in place of the pull-request template's greedy diff against `--no-plugins`, because
  mlx-serve's own Ling model cannot read EXL3.

A reviewer must also rule on four smaller points in the rules table below. They are step 4's manual install, our staged
kit, the copied parse's checks and exllamav3's tensor names.

### AGENTS.md's rules, one by one

| Rule | Status | Evidence | For the reviewer |
|---|---|---|---|
| 1. Every plugin compiles | not met | build.zig.zon pins turboexl3 main c7ce66c, and its comment names an older branch (ling, d5f9989). The ReleaseFast build fails on that pin (measured; "Tests"). CI's kit from MLX_SERVE_REF bc6c5f2c has no span fields, so CI's `zig build test` stops at config.zig's comptime kit check (`kitGap`) before the ReleaseFast build (the CI workflow's steps "Pinned toolchain and kit" and "Every plugin compiles and passes its tests" in .github/workflows/ci.yml; scripts/build.sh). | The pull request stays a draft until the pin and CI's kit move ("Gaps for a merge"). |
| 2. Compile from source | met, except for scripts/install.sh | No prebuilt dylib: step 4 builds the source. scripts/install.sh does not fit yet, because it uses the mlx-serve on your PATH and builds without `--fork`. | Accept step 4's manual install until the pin moves and 26.11.3 is on PATH. |
| 3. Pinned toolchain only | met, with a staged kit | scripts/setup.sh stages the pinned Zig. The kit comes from `MLX_SERVE_KIT`, as AGENTS.md allows while the release has no kit asset. Our kit, built from 66aad949, names itself 26.11.1 (step 3). | Accept a kit staged from 66aad949 in place of the 26.11.3 release kit. The pull request must say that the kit is staged. |
| 4. One plugin per PR | met | Outside the plugin's folder, the branch adds only the catalog row in README.md, the plugins.json entry and docs/assets/icons/turboexl3_ling.svg (`git diff --stat cfc3704`). | A fix that moves the repo's MLX_SERVE_REF ("Gaps for a merge") changes a shared file. |
| 5. Narrow claim, MLX-free parse | met | The plugin claims by marker, exllamav3's `quantization_config` (config.zig `claims`). Its claims and parse are MLX-free (config.zig's file comment). | None. |
| 6. Refuse by name | partly met; not met for a quantized KV | Every refusal is named (README, "Logs"). The plugin bounds-checks `num_hidden_layers` (1 to 512) and the clamp lists (config.zig `checkPack`). The copied built-in parse reads the other fields (config.zig `parse`). A 4- or 8-bit KV scheme gets no refusal, and the request runs. Since fb966cc the plugin logs one warning a load. mlx-serve never tells the plugin the scheme at load ("The quantized KV"). | Decide whether the copied parse's checks are enough. A quantized KV needs a host change, not yet made, that refuses the scheme by name at load ("Gaps for a merge"). |
| 7. Bill what you allocate | met for one request at a time; two long requests at once not tested; stage 1 and four decisions need a ruling | The bill's load-time KV term holds 32,768 tokens, and on 01ca9769 a KV grow step could go past it by up to 2,104 MiB (calculated; "The KV past the bill's term"). RQ bills each request's KV at the capacity that mlx-serve allocates, growth included, and refuses by name a prompt forward that would grow it past 32,000 tokens ("RQ"). Stage 1 keeps the prompt pass at the 214-row base: near 32,000 tokens the plan holds 211 slot rows, and MLX's peak stayed 426 MiB under the plan's ceiling at 31,652 tokens on 4856f028 and 538 MiB under it at 32,000 tokens on 89be02b1 (measured). Two long requests at once are not tested: RQ counts another live request's KV as it is, not its growth. mlx-serve also takes a `--ctx-size` above 32,768 as given (66aad949 src/engines/mlx/admission/context.zig `getEffectiveContextLength`), and decode past 32,768 tokens is not measured. The KV term and the layer-major term are calculated, and the 4 GiB reserve is a setting. `prompt_work` bounds the 1.2 to 2.0 GiB measured at 4,096-token chunks (src/bill.zig `prompt_work`). MEM_REFIT is fitted to measured peaks. Since 0ebec77 it is fitted to 13 runs at 31,652 tokens. The 32K checks of 01ca9769 and 4856f028 peaked 426 MiB under the plan's ceiling there ("Memory"). In decode the bank also takes MEM_REFIT's 544 MiB server part (plugin.zig `openBank`). Our A/Bs served it ("Measured"). The bill holds no term for the prompt log-probability step, the plugin's own 2 B for each entry included, and the refusal replaces that term (src/bill.zig `headRowsJoined`). | A test of two long requests at once must pass first. Rule on stage 1's case near 32,000 tokens, or ask for a streamer that can shrink the prompt pass's rows under its base. A larger `--ctx-size` needs a refusal at load, or a measured run. Then accept the four decisions, or ask for their fixes: a measured term for each part of the bill, and mlx-serve's row-sliced step from our issue draft ("Gaps for a merge"). |
| 8. Tests for behaviour that can regress | met | The MLX-free tests, the GPU byte tests and the restore check ("Tests"). Rule 7's tests failed before their code (`~/lingp/logs/portgpu-r7red.log`, `~/lingp/logs/portgpu-r7cred.log`). The re-fit's tests failed before its constant (`~/lingp/logs/portgpu-r8cared.log`), and the room's tests before 165d8ca's code (`~/lingp/logs/portgpu-r7ered.log`). EMBED_ROWS', KDA_GATE_EPI's, KDA_NORM_PREP's and the DFlash refusal's tests failed before their code too (`~/lingp/logs/portgpu-m1hred.log`; our port notes). RQ's built-in tests failed at 44ba83b, before fb966cc (our results folder `gates-4856f028/portgpu-rqhred.log`). scripts/test_reach.py names any test block that no test binary reaches, and the gates check each binary's test count against it ("Tests"). | None. |
| 9. Third-party code unchanged | not met as written | The copy is generated and untracked (.gitignore's `src/mlx_serve/` line), and prepare.sh applies the patches right after the sync script (scripts/prepare.sh). So a reviewer may count them as the sync step's edits. They are still separate patch files: four, editing 14 copied mlx-serve files, with 2,883 lines added and 148 removed at fb966cc, and 2,843 and 148 at bd00b1a (calculated). The five MLX kernels, overlay/mlx_serve/kernels/mla/ling_verify_mla_*.metal, are edited copies committed in overlay/ (NOTICE's MLX section). | Accept both kinds of edit, or ask us to move them into sync scripts ("Gaps for a merge"). |
| 10. Numbers are measured | met | We mark each number's kind where the text does not make it clear, and "Receipts" gives its record. Decode comes before prefill, and "Smaller gaps" lists what we have not measured. The measured records are on our Mac and are not published ("Receipts"). | Note that the records are not published, so you cannot check the measured numbers against them. |
| 11. Leave the user's mlx-serve alone | met | Step 4 installs under a throwaway HOME, and each of our runs used one server. | None. |
| 12. Public text is short and plain | met for the README's main text | The README's main text is 2,399 words by `wc -w`, counted over the lines outside `<details>` blocks and before "## Terms", with commands and tables included, within the 2,400-word budget. The whole README is 6,792 words, and this file 33,209. The evidence is in this file, such as "Receipts" and the per-run tables. The pull-request body is not written yet. | Read the pull-request body when it is open. |
| 13. Open the PR only when asked | not applicable | The rule governs the agent that opens the pull request. | None. |
| "Create a plugin": weight names and config keys follow mlx-lm | not met | The pack keeps exllamav3's tensor names (`.trellis`, `.suh`, `.svh`) and puts the routed experts in experts.bin, so other engines need a new reader for it. | Accept exllamav3's names, or ask for a renaming step in the converter. |
| PR steps: branch `add-<name>` | not met yet | The work is on `turboexl3_ling-rq`. | A fix is to push it as `add-turboexl3_ling` when the pull request opens. |
| PR template: `scripts/lint.sh` and `scripts/gen-index.sh` | met at ecc4a1c; run again at the PR head before the PR opens | lint printed "lint: ok", and gen-index left plugins.json unchanged (measured, `~/lingp/logs/portcs-cs1.log`). | None. |
| PR template: `scripts/check.sh` | met by its exception | `"fixture": "none"` with a `fixture_reason` (plugin.json's `fixture` and `fixture_reason`). So `scripts/check.sh` needs `--model` ("Smaller gaps"). | None. |

### Smaller gaps

- Without `--ctx-size`, mlx-serve picks at most 32,768 tokens, and it can pick fewer. Its auto-sizer prices a token at
  368,640 B, its own count for the MLA cache, against the plugin's 297,216 B (both calculated). A fix needs the
  auto-sizer to read the plugin's bill.
- Without `--prefill-chunk`, mlx-serve sizes the chunk from the memory left after the load, so the text can depend on
  the bank's size. A fix is for the plugin to cut the prompt at a fixed width itself. The prefix cache's snapshots then
  need a new check.
- We have not measured a Mac with less memory, the macOS default GPU wired limit, or the 2.10bpw pack on this build. Nor
  have we measured a long output after a shorter prompt or `--prefix-cache-disk`. A prompt over 32,000 tokens is refused
  ("RQ"). The ladder stopped 92 tokens short of the 32,768-token context. RQ's safety test filled it on 89be02b1, with a
  32,000-token prompt and 768 decoded tokens, but it measured no speed. Each one needs a measured run before the README
  can claim it.
- config.zig declares no batched decode, and the layer-major pass refuses a prompt from more than one slot. So mlx-serve
  interleaves a second request's forward passes with the first's and never batches them (66aad949
  src/host/http/server.zig `serve`, its concurrency line). Serving several requests at once needs batched decode in the
  plugin.
- The plugin serves one Mac. Serving on a Thunderbolt pair needs the plugin to support mlx-serve's pair mode.
- mlx-serve keeps no prefix between requests (README, "What it serves"), about 26 s for each turn at 18,380 tokens. Its
  RAM tier gets no memory beside the bank (`mem-cap=0.0 MB`), and step 6 sets no SSD tier. A fix needs a measured
  `--prefix-cache-disk` setting, or RAM for the prefix cache.
- plugin.json asks for mlx-serve 26.11.3 or later (`requires_mlx_serve`). A server built at 66aad949 reports 26.11.1, so
  the app's plugin catalog finds that the plugin does not run on it (66aad949
  app/Sources/MLXServe/Services/PluginCatalog.swift `PluginCatalog.runs(on:requirement:)`). A server of the 26.11.3
  release passes the check.
- The plugin has no conformance fixture (`"fixture": "none"` in plugin.json). So `scripts/check.sh` needs `--model`,
  then runs only the plugin's own MLX-free tests. A fixture needs a small streamed EXL3 pack.
- The no-spans refusal tells the reader to update mlx-serve to a release with whole-prompt spans, and no release has
  them yet. A fix changes that text once 26.11.3 is out.
- mlx-serve keys top log-probabilities by token string (66aad949 src/host/api/logprobs.zig `formatCompletionsLogprobs`),
  so the partial-UTF-8 byte tokens collide. A fix needs mlx-serve to key them by token id.
- Prompt log-probabilities stop at the limit in "Memory". A longer request gets only HTTP 500. Our issue draft for
  mlx-serve, not filed, proposes two mlx-serve changes. One runs the whole-prompt step in slices of 256 rows. The other
  gives the client a forward pass's `too_large` reason as a named HTTP 400.
- A release needs the parity scripts, the oracle's logits and glm53-eval to be published. It then needs one serve with
  mlx-serve's 26.11.3 release, which can hold commits after 66aad949.
- This item is an optional follow-up, not a fix that we owe. The spans' bf16 copy, 2 B of the 32 B counted for each row
  and vocabulary entry, stays alive through mlx-serve's step ("Memory"). If `spanMajorOr` evaluated the join before it
  returns, the copy would go before the step. The change touches memory on the echo path only, and no served bit. At 30
  B for each entry, bd00b1a's formula gives a limit of 3,525 tokens rather than 3,317, about 6% more (calculated). We
  have not built it.
- RQ's O counts two idle items that mlx-serve keeps alive into decode ("RQ"). They are its copies of the prompt's KDA
  checkpoints and the prompt's last hidden rows. A fix is an mlx-serve change. It frees the checkpoint copies when the
  prefix cache keeps nothing, and it keeps its own copy of the last hidden row. Decode would then get 2 more slot rows
  at 18,380 tokens and 4 more at 32,000 (calculated). We have not made it.

## Receipts

Measured numbers come from runs on our M5 Max with 128 GiB and a 112 GiB GPU wired limit. Their records are on that Mac,
under `~/lingp`, and are not published. They are these:

- the records of the served dylib, 4856f028, in our results folder: its gates and the red at 44ba83b (`gates-4856f028`)
  and its 32K check (`ladder-rqv-4856f028`), and RQ's safety test (`ladder-rqsafe-89be02b1`) and A/B test
  (`smoke-rq-89be02b1`) on 89be02b1;
- the served dylib's standard-test runs, as arm A of SPEC1_BUSY's A/B test, in our results folder
  (`smoke-spec1-23731721`);
- the A/Bs' smoke.txt, summary.txt and warm-up records: 01ca9769's main test (`~/lingp/abba/smoke-m2-97c37406`, arm A),
  KDA_NORM_PREP's (`~/lingp/abba/smoke-l5-7f053407`), KDA_GATE_EPI's (`~/lingp/abba/smoke-l6-73db9cb6`), EMBED_ROWS'
  (`~/lingp/abba/smoke-m1-38f84c44`), and the re-fit's with its abcheck.txt (`~/lingp/abba/smoke-portrc-fdd4e0d8`);
- the 32K checks of 01ca9769, 583523d4 and 7e446929 in `~/lingp/ladder-l5v`, `~/lingp/ladder-l6v` and
  `~/lingp/ladder-m1v`: report.txt, memfit.txt, verdict.txt and results.tsv, and memfit-prewarm.txt for the first two;
- the refusal test's records, all three runs, and the third run's script, r7e-echo.py;
- portgates-l5h.log, portgates-dt.log, portgates-l5.log, portgates-l6h.log, portgates-l6.log, portgates-m1h.log,
  m1hpack.log, portgates-r7c.log, portgates-r8c.log, portgates-r7e.log and portcs-cs1.log;
- portgpu-m1hred.log, portgpu-r7red.log, portgpu-r7cred.log, portgpu-r8cared.log and portgpu-r7ered.log;
- the parity report, with a copy in our results folder (`parity-w2/report.md`);
- each pack's receipt, LING-REPACK-RECEIPT.json, in its folder under `~/models`, and the download and pack sizes, with
  copies in our results folder (`pack-records`);
- our hand-off for KDA_NORM_PREP: turboexl3 4bf2df8's gates, and its section 5, the memory reading;
- the final ladder's report.md, results.tsv, memfit-windows.txt, manifest.json, progress.txt, greedy.env and card.env in
  `~/lingp/ladder-01ca9769`, and its comparison, compare-c8d54a83.md, and charts in our results folder
  `ladder-01ca9769`;
- the earlier ladder's report.md, memfit-windows.txt and results.tsv in `~/lingp/ladder-c8d54a83`, and memfit.txt, the
  record behind the earlier re-fit;
- the earlier re-fit's check in `~/lingp/ladder-rf`: report.txt, memfit.txt and results.tsv;
- the re-fit's own check in `~/lingp/ladder-rc`: verdict.txt, memfit.txt, memfit-prewarm.txt, report.txt and
  results.tsv.

Calculated numbers come from the code. Set numbers are settings, such as `--ctx-size 32768`, or the model's sizes, such
as 512 experts. The settings come from the code and step 6, and the model's sizes from its config.json
(tests/fixtures/config-2.60bpw.json).

| Number | Kind | Receipt |
|---|---|---|
| 01ca9769's main standard A/B test: 01ca9769 as arm A, runs 1, 4, 6 and 7; means of 25.78 tok/s decode (25.775 unrounded), 700.4 tok/s prefill (700.425), TTFT 26.26 s (26.255), wall 66.09 s, MTP 1,783 / 2,877 = 62.0%, peak footprint 108.5 GB (108.450) | measured, 2026-11-10, 21:22:11 to 21:54:34Z | `~/lingp/abba/smoke-m2-97c37406/smoke.txt` and `summary.txt`, arm A's rows; our port notes, for the window, not published |
| 01ca9769's runs in that test: the same text, 09571f6a, in all 4; each request line naming 35 of the 40 paths; prompt lookup's drafts, 210 to 216 tokens for each run with 201 to 204 accepted | measured, 2026-11-10 | that test's `smoke.txt`, its table, run-path section and trajectory table (lookup R/D/A) |
| 01ca9769's runs in that test: the Mac's free RAM down to 0.246 to 1.345 GB a run; 44 to 121 pageouts in each run; swap flat at 0.127 GB; 214 and 236 slot rows; MLX's peak through the prompt pass of 94,693 to 94,856 MiB; peak footprints of 108.3 to 108.5 GB | measured, 2026-11-10 | that test's `summary.txt`, its per-run free-RAM lines; `smoke.txt`, its pageout and memory lines; `memory-1.tsv`, `memory-4.tsv`, `memory-6.tsv` and `memory-7.tsv` |
| SPEC0_IDLE's A/B: 01ca9769 as arm A, runs 1, 4, 6 and 7; means of 25.85 tok/s decode (25.850 unrounded), 698.7 tok/s prefill (698.675), TTFT 26.32 s (26.320), wall 66.08 s (66.078), MTP 1,827 / 2,989 = 61.1%, peak footprint 108.4 GB (108.375); text 09571f6a in all 4 runs; free RAM down to 0.254 to 0.815 GB, and 184 to 256 pageouts, a run | measured, 2026-11-11, 04:25:10 to 04:58:03Z | `~/lingp/abba/smoke-spec0-965ef461/smoke.txt` and `summary.txt`, arm A's rows |
| Arm B of SPEC0_IDLE's A/B: dylib 965ef461 with `TURBOEXL3_LING_SPEC0_IDLE=1`, the first speculative read thread held to the second one's idle rule; rejected under rule 6 | set, and decided by us on 2026-11-11 after the test | that test's `B.env`; our ledger, CANDIDATES.md R22, not published |
| Arm B of 01ca9769's main test: PRED_F16, dylib 97c37406 with `TURBOEXL3_LING_PRED_F16=1`, which drops the predictor's 272 MiB bf16 router copy; rejected under rule 6 | set, and decided by us on 2026-11-10 after the test | that test's `B.env`; our ledger, CANDIDATES.md R20, not published |
| KDA_NORM_PREP's A/B, the tables of its "Measured" block: A (63668c19) at 25.80 tok/s decode (25.800 unrounded) and 696.9 tok/s prefill, B (7f053407) at 25.83 (25.825) and 700.3, means of 4 runs | measured, 2026-11-10, 19:15:54 to 19:46:09Z | `~/lingp/abba/smoke-l5-7f053407/smoke.txt` and `summary.txt` |
| KDA_NORM_PREP's A/B: B's means against A's, prefill +0.49%, decode +0.10%, TTFT −0.48% and wall time −0.22%; in runs 1 to 4 and 5 to 8, prefill +0.42% and +0.56%, decode +0.00% and +0.19% | calculated from the runs | `smoke.txt`, its table of means |
| KDA_NORM_PREP's A/B: the same text, 09571f6a, in all 8 runs; one `KDA_NORM_PREP engaged: S=4096` line in each B log, and none in A | measured, 2026-11-10 | `smoke.txt`, its text and engagement lines; its `serve-<run>-<arm>.log` files |
| KDA_NORM_PREP's A/B: a request's line names 34 of the 39 paths in the A runs and 35 of the 40 in the B runs | measured, 2026-11-10 | `smoke.txt`, its run-path section |
| KDA_NORM_PREP's A/B: prompt lookup's drafts, 198 to 216 tokens for each run with 187 to 205 accepted; MTP 1,778 / 2,877 in A and 1,788 / 2,958 in B | measured, 2026-11-10 | `smoke.txt`, its trajectory table and its MTP lines |
| KDA_NORM_PREP's A/B: the Mac's free RAM down to 0.248 to 2.137 GB a run; 7 to 100 pageouts in each run; swap flat at 0.127 GB | measured, 2026-11-10 | `summary.txt`, its per-run free-RAM lines; `smoke.txt`, its pageout lines; `memory-1.tsv` to `memory-8.tsv` |
| KDA_NORM_PREP's A/B: both arms at 214 and 236 slot rows, MLX's peak through the prompt pass of 94,693 to 94,856 MiB in each arm, and peak footprints of 108.3 to 108.5 GB | measured, 2026-11-10 | `smoke.txt`, its memory lines and its table |
| KDA_NORM_PREP kept by rule 3 of our decision rules | decided by us, 2026-11-10, after the A/B | our ledger (CANDIDATES.md S54 and its decision rules), not published |
| Dylib 7f053407 (4,765,880 B), built at b0079f1: MLX-free tests 104 of 104; GPU byte tests 59 of 59 with 653 PASS lines and no FAIL line; restore check part B 24 checks, BYTES_OK; KDA_NORM_PREP's byte test 8 lines, none failed, 1 red check, 41,603,072 words | measured, 2026-11-10 | `~/lingp/logs/portgates-l5.log`; our port notes, L5's chain, not published |
| turboexl3 4bf2df8 (branch ling-l5-normprep, on aa0ce54) declares `kdaNormGatePrep`, `takes` and `shapeOk` (src/models/ling/kdanorm.zig), and aa0ce54 has none of them; no branch on GitHub has either commit | read from the code, 2026-11-10 | our turboexl3 clone |
| turboexl3's own byte test of the launch: no differing word at 33 to 16,384 rows; its refused line and its red lines passed | measured, 2026-11-10, by 18:20:51Z | our hand-off for KDA_NORM_PREP, not published; turboexl3 4bf2df8 src/models/ling/README.md, its `kdanorm_test.zig` row |
| KDA_NORM_PREP never writes flat, 16 KiB for each row: 64 MiB for each layer at a 4,096-row span and 30.7 MiB at 1,965; 315 launches and 37.9 GB of traffic less for each standard prompt; no bill term | calculated from the code | our hand-off for KDA_NORM_PREP, section 5, not published; src/ling_kda_norm_prep.zig |
| bd00b1a's counts: 104 MLX-free tests, 52 GPU byte tests and 26 tests in the restore check's part A, 0 test blocks unreached | calculated by scripts/test_reach.py's model, 2026-11-10, at 45ab5ba, and again at bd00b1a in 01ca9769's gates | `python3 scripts/test_reach.py`; `~/lingp/logs/portgates-l5h.log`, its REACH line |
| scripts/mlx_serve.sha256 at 2fac1c06 at bd00b1a (turboexl3_ling-l5) | calculated by scripts/sync-mlx-serve.sh | commits 16541d1 and a6ea7ee |
| KDA_GATE_EPI's A/B, an earlier A/B in "Measured": its tables (decode 26.05 tok/s and prefill 697.7 tok/s for B's mean) | measured, 2026-11-10 | `~/lingp/abba/smoke-l6-73db9cb6/smoke.txt` and `summary.txt` |
| Each arm's one warm-up run in each A/B, which the means leave out | measured, 2026-11-10 | `prewarm-A.txt` and `prewarm-B.txt` in each A/B's folder |
| KDA_GATE_EPI's A/B: prompt lookup's drafts, 195 to 219 tokens for each run with 186 to 208 accepted | measured, 2026-11-10 | `smoke.txt`, its trajectory table (lookup R/D/A) |
| In each A/B, no A run with the same drafted tokens as a B run | measured, 2026-11-10 | `smoke.txt`, its trajectory table and its line on identical-trajectory pairs |
| KDA_GATE_EPI's A/B: B's means against A's: decode +0.48%, prefill +0.26%, TTFT −0.24% and wall time −0.39%, with the same signs in runs 1 to 4 and 5 to 8 | calculated from the runs | `smoke.txt`, its table of means |
| KDA_GATE_EPI's A/B: the Mac's free RAM down to 1.242 to 2.159 GB a run | measured, 2026-11-10 | `summary.txt`, its per-run free-RAM lines |
| KDA_GATE_EPI's A/B: 5 to 113 pageouts in each run | measured, 2026-11-10 | `smoke.txt`, its pageout lines |
| KDA_GATE_EPI's A/B: swap flat at 0.127 GB, and memory pressure 1, in every run | measured, 2026-11-10 | `memory-1.tsv` to `memory-8.tsv` in that A/B's folder (swap); our port notes, L6's A/B (the memory guard's pressure), not published |
| KDA_GATE_EPI's A/B: both arms at 214 and 236 slot rows, MLX's peak through the prompt pass of 94,693 to 94,857 MiB in A and 94,693 to 94,727 MiB in B, and peak footprints of 108.4 GB | measured, 2026-11-10 | `smoke.txt`, its memory lines and its table |
| KDA_GATE_EPI's A/B: a request's line in the B runs names 34 of the 39 paths | measured, 2026-11-10 | `smoke.txt`, its run-path section |
| KDA_GATE_EPI's timing before its A/B: 0.987 to 0.993 times the time of the kernels it replaces, about 0.09 s less a prompt | measured by turboexl3's `TURBOEXL3_LING_TIME` arm, a microbenchmark | our port notes, L6, not published |
| Dylib 73db9cb6's gates: MLX-free tests 76 of 76, GPU byte tests 50 of 50 with 645 PASS lines, restore check part B 24 checks BYTES_OK; KDA_GATE_EPI's byte test 25 lines, none failed, 12 red checks, 937,988,992 words | measured, 2026-11-10, at 859cd93 | `~/lingp/logs/portgates-l6.log` |
| Dylib 01ca9769 (4,765,800 B), built at bd00b1a with turboexl3 4bf2df8 and a kit from mlx-serve 66aad949, served from 20:59:54Z on 2026-11-10 to 08:08:00Z on 2026-11-11 | measured, 2026-11-10 and 2026-11-11 | `~/lingp/logs/portgates-l5h.log`, its START line (turboexl3 head 4bf2df8, host 66aad949) and RELEASEFAST_END line; our port notes, tk-l5hgates (the tree, bd00b1a) and L5's 32K check (the move to 01ca9769), not published; `lane-decisions.md`, 08:08:00Z, for the end |
| 01ca9769's gates: the red at 32f7c02 RED-OK, 2 error records under 1 name; MLX-free tests 104 of 104 in 22 of 22 steps; GPU byte tests 52 of 52 with 653 PASS lines and no FAIL line; restore check part A rc=0, and part B 24 checks over 7 restores, BYTES_OK; test_reach.py and its self-test passing, and its totals agreeing for all three binaries, 104, 52 and 26; KDA_NORM_PREP's byte test 8 lines, none failed, 1 red check, 41,603,072 words, BYTES_OK | measured, 2026-11-10, 20:33:57 to 20:40:29Z at bd00b1a, and the red from 20:07:40 to 20:07:55Z | `~/lingp/logs/portgates-l5h.log`; our port notes, tk-l5hcpu (the red) and L5's 32K check (the byte test's DONE line), not published |
| 01ca9769's load: 14,734 MiB in use, room 84,733 MiB, 69,998 and 76,329 MiB for the slots, 214 and 236 slot rows, the limit of 3,317 tokens | measured, at every run of its 32K check | `~/lingp/ladder-l5v/report.txt`, its load lines |
| 01ca9769's 32K check: 1,024 decoded tokens in each of its 5 runs, so prompt and output passed 32,000 tokens; free RAM down to 0.188 to 1.099 GB, and 347 to 639 pageouts, a run | measured, 2026-11-10; the crossing calculated: 31,652 + 1,024 tokens | `~/lingp/ladder-l5v/results.tsv` (`out_tokens`, `free_min_gb`, `pageouts`) |
| 01ca9769's 32K check: a warm-up and 4 greedy runs at 31,652 tokens; MLX's peak 101,430 MiB and the remainder 988 MiB in all 5 runs, against terms of 100,442 MiB, as in 583523d4's check; peak footprints of 115.97 to 115.98 GB; pressure normal, swap flat at 0.127 GB, no stop; text 919cf0fb in all 5 runs, as in the checks of 583523d4 and 7e446929 | measured, 2026-11-10, 20:40:29 to 20:59:54Z | `~/lingp/ladder-l5v` (`report.txt`, `memfit.txt`, `memfit-prewarm.txt`, `verdict.txt`, `results.tsv`), copied to our results folder as `ladder-l5v-01ca9769`; `~/lingp/ladder-l6v/memfit.txt` and `results.tsv`; `~/lingp/ladder-m1v/results.tsv`; our port notes, L5's 32K check, for the times |
| Dylib 63668c19 (4,763,176 B), built at 1dffdcd with turboexl3 aa0ce54, served from 18:19:14 to 20:59:54Z; its 15 Mach-O sections the same bytes as 583523d4's, and only metadata different | measured, 2026-11-10 | `~/lingp/logs/portgates-dt.log`; our port notes, tk-dtgates and L5's 32K check (the move to 01ca9769), not published |
| 63668c19's gates: MLX-free tests 100 of 100 in 22 of 22 steps, as test_reach.py counts them; GPU byte tests 50 of 50 with 645 PASS lines and no FAIL line; restore check part B 24 checks, BYTES_OK; test_reach.py and its self-test passing | measured, 2026-11-10, 18:12:51 to 18:19:14Z, at 1dffdcd | `~/lingp/logs/portgates-dt.log`; our port notes, tk-dtgates |
| Dylib 583523d4 (4,763,176 B), built at 4c13ec7 with turboexl3 aa0ce54, served from 17:44:49 to 18:19:14Z | measured, 2026-11-10 | `~/lingp/logs/portgates-l6h.log`, its RELEASEFAST_END line; our port notes, L6 KEEP |
| 583523d4's gates: MLX-free tests 100 of 100 in 22 of 22 steps; GPU byte tests 50 of 50 with 645 PASS lines and no FAIL line; restore check part A's file of 8,628,529,825 B, and part B 24 checks over 7 restores with 36,001,152 head words, BYTES_OK; EMBED_ROWS' byte test 16 lines and KDA_GATE_EPI's 25 lines, none failed, with 12 red checks and BYTES_OK | measured, 2026-11-10, at 4c13ec7 | `~/lingp/logs/portgates-l6h.log`; the byte tests' DONE lines in our port notes, L6 KEEP |
| 583523d4's 32K check: a warm-up and 4 greedy runs at 31,652 tokens; MLX's peak 101,429 to 101,430 MiB; remainders of 987 to 988 MiB; peak footprints of 115.97 to 115.98 GB; pressure normal, swap flat at 0.127 GB, no stop; text 919cf0fb in all 5 runs, as in 7e446929's check | measured, 2026-11-10, 17:27:51 to 17:44:49Z | `~/lingp/ladder-l6v` (`report.txt`, `memfit.txt`, `memfit-prewarm.txt`, `verdict.txt`, `results.tsv`); `~/lingp/ladder-m1v/results.tsv` |
| The A/B of EMBED_ROWS as the option `TURBOEXL3_LING_EMBED_ROWS=1`: the means, the same text, and 214 and 236 slot rows in arm B | measured, 2026-11-10; arm A dylib 090d2b0f (165d8ca), arm B dylib 38f84c44 (2773ee6) | `~/lingp/abba/smoke-m1-38f84c44/smoke.txt` |
| Dylib 7e446929 (1a16e40): MLX-free tests 73 of 73, GPU byte tests 48 of 48, restore check part B BYTES_OK; EMBED_ROWS' byte test with the pack, 23 lines, none failed, no differing word | measured, 2026-11-10 | `~/lingp/logs/portgates-m1h.log` and `~/lingp/logs/m1hpack.log` |
| 7e446929's 32K check: a remainder of 988 MiB in all 5 runs | measured, 2026-11-10, 15:07:13 to 15:23:50Z | `~/lingp/ladder-m1v` (`memfit.txt`, `verdict.txt`); our port notes, M1 KEEP, for the warm-up |
| EMBED_ROWS served from 15:23:50Z, KDA_GATE_EPI from 17:44:49Z and KDA_NORM_PREP from 20:59:54Z | measured | our port notes, M1 KEEP, L6's 32K check and L5's 32K check, not published |
| Two text changes in the served stack: the exact verify, 744ebd6b to 5fb1329b, the text of plain decoding; and MLA_FLASH, 5fb1329b to 09571f6a | measured, 2026-11-06 to 2026-11-08 | our ledger, CANDIDATES.md rows S4 and S38, not published; the early build's `summary.txt` (744ebd6b, 2.60bpw) |
| Every change kept after the parity run, the port, EMBED_ROWS, KDA_GATE_EPI and KDA_NORM_PREP included, decoded 09571f6a | measured, at each A/B | our ledger, CANDIDATES.md S39 to S49, S51, S52 and S54, not published; each A/B's `smoke.txt` |
| The strip, the port and the port's follow-ups: no ledger row, and the same text, 09571f6a, in each of their A/Bs | measured, 2026-11-09 and 2026-11-10 | our ledger, CANDIDATES.md: in its Stack section, the Strip paragraph (the strip's four A/Bs) and the Port A/B paragraph (the port's A/B, smoke Q and smoke Q8), not published |
| Five kept by a ruling although at least one measure was slower: COMBINE, decode 14.40 to 14.30 tok/s; MTP drafting, prefill 295.874 to 290.501 tok/s; LAYER_MAJOR, decode 21.899 to 20.750 tok/s, a mixed keep whose follow-up LM_DECODE_ROWS took decode from 20.928 to 21.723 tok/s; ROUTER_FUSED, prefill 460.05 to 459.84 and decode 22.400 to 22.375 tok/s; MEM_REFIT, a calibration, prefill 547.475 to 544.325 and decode 23.175 to 23.100 tok/s | measured; the rulings ours, LAYER_MAJOR's under rule 5's mixed keep | our ledger, CANDIDATES.md rows S3, S13, S16, S17, S21, S37 and S49 (the re-fit under MEM_REFIT's ruling), and its decision rules 3 to 6, not published |
| Of ledger rows S39 to S54: a byte test with no differing word in S41, S45, S46, S48, S51, S52 and S54; draft-only changes in S40 and S47; the same text and no byte test in S39, S42, S43, S44 and S49; no forward-pass change and no A/B in S50 and S53 | read from the ledger, 2026-11-10 | our ledger, CANDIDATES.md rows S39 to S54, not published |
| The MTP head's 512 experts, resident in GPU memory | measured, the load log | the third run's `serve.log` line 21 |
| The bank's decode slots, 236 a layer at bd00b1a, from decode's first forward pass to the next prompt | set by the code | the phase switch in plugin.zig `forward` at bd00b1a; overlay/mlx_serve/sdk/weights/streamed/ling_source.zig `Source.setPhase` at bd00b1a |
| The re-fit's A/B table (decode 25.70 tok/s, prefill 696.1 tok/s for B; 25.65 and 695.0 for A) | measured, 2026-11-10 | `~/lingp/abba/smoke-portrc-fdd4e0d8/smoke.txt` and `summary.txt` |
| The re-fit's A/B: prompt lookup's 213 to 222 drafted tokens in each run with 201 to 205 accepted; runs 2 and 7 at 373 verify passes each, 105.74 and 105.87 ms for each verify pass | measured, 2026-11-10 | that A/B's `smoke.txt`, its trajectory table |
| The re-fit's A/B: free RAM down to 0.07 to 1.056 GB a run; 18 to 229 pageouts for each run; swap flat at 0.136 GB, and memory pressure 1 at every reading | measured, 2026-11-10 | that A/B's `summary.txt`, `smoke.txt` lines 24-26 and `abcheck.txt` |
| The port's A/B and the re-fit's A/B: the same text, 09571f6a, from a byte-identical generated copy | measured, 2026-11-09 and 2026-11-10 | `~/lingp/abba/smoke-portb-c003fc7c/smoke.txt` and the re-fit A/B's `smoke.txt`; our port notes, decision 1, not published |
| At c8d54a83: 66,153 and 75,101 MiB for the slots, room 82,116 MiB, 202 and 233 slot rows. At 0ebec77 and 165d8ca: 68,770 and 75,101 MiB, room 84,733 MiB, 210 and 233 slots. In both: 15,962 MiB in use, a slot row of 336,427,008 B (320.8 MiB) | measured, at every run of the re-fit's A/B, the re-fit's check and the refusal test's second and third runs | the re-fit A/B's `smoke.txt`, its memory lines; `~/lingp/ladder-rc/report.txt`; `~/lingp/ladder-r7/ladder-r7echo/serve.log` lines 26-38; `~/lingp/ladder-r7e/ladder-r7eecho/serve.log` lines 26-37 |
| The re-fit's A/B: MLX's peak through the prompt pass, 94,636 to 94,800 MiB in B and 92,070 to 92,233 MiB in A; peak footprints of 108.2 to 108.4 GB in B and 105.6 GB in A | measured, 2026-11-10 | that A/B's `smoke.txt`, its memory lines and its table |
| The re-fit A/B's B remainders of −2,484 to −2,320 MiB at 18,380 tokens | calculated by memfit v3 from B's measured peaks | that A/B's `abcheck.txt` |
| The re-fit's A/B: B's peak footprint 2.7 GB above A's, and 2.69 GB for 8 slot rows | calculated: 108.325 - 105.6 GB, and 8 x 336,427,008 B | that A/B's `summary.txt`; the bank lines |
| The refusal test, first run (limit 4,172 tokens, 31.6 B for each entry, peak footprint 115.1 GB, and the rest of its column) | measured, 2026-11-09, at 9700aa6 | `~/lingp/ladder-r7/ladder-r7echo.234907/` (`verdict.txt`, `r7echo.json`), `~/lingp/logs/r7echo.log`, src/bill.zig's comment on `echo_entry_bytes` |
| The refusal test, second run (limit 2,941 tokens, 30.5 B for each entry, 5,412 MiB under the plan's ceiling of 101,856 MiB, peak footprint 110.5 GB, and the rest of its column) | measured, 2026-11-09, at 0e1f5c0 | `~/lingp/ladder-r7/ladder-r7echo/` (`verdict.txt`, `r7echo.json`, `serve.log`), `~/lingp/logs/r7echo.log` |
| The refusal test, third run (limit 3,317 tokens, 29.8 B for each entry, 1,359 MiB under the plan's ceiling of 101,856 MiB, peak footprint 114.6 GB, and the rest of its column) | measured, 2026-11-10, 13:22:03 to 13:22:29Z, at 165d8ca (dylib 090d2b0f) | `~/lingp/ladder-r7e/ladder-r7eecho/` (`verdict.txt`, `r7echo.json`, `serve.log`, `memory.tsv`, `pressure.tsv`), `~/lingp/logs/r7eecho.log`, and the request body at line 44 of its script, r7e-echo.py |
| The third run's port 11391 and step 6's options | set | its `arm.env` (`FLAGS`) and `serve.log` line 2 |
| The first run's 1,238 MiB over its plan's ceiling of 101,856 MiB | calculated from its measured peak and the plan | our port notes, not published |
| mlx-serve's two 3-bit copies, built after the bank opens | measured, the load log's order | `~/lingp/ladder-r7e/ladder-r7eecho/serve.log` lines 36-42 |
| MLX-free tests 67 of 67, GPU byte tests 43 of 43 with 604 PASS lines, restore check 24 checks over 7 restores, part A's file of 8,628,529,825 B, 36,001,152 head words | measured, at 165d8ca (dylib 090d2b0f) | `~/lingp/logs/portgates-r7e.log` |
| The same gates, with 66 of 66 MLX-free tests, at 0ebec77 (dylib fdd4e0d8) and at 0e1f5c0 | measured | `~/lingp/logs/portgates-r8c.log`, `~/lingp/logs/portgates-r7c.log` |
| Rule 7's tests failing before their code: 63 of 66 passed at feb6187, and again at 8f65671 | measured | `~/lingp/logs/portgpu-r7red.log`, `~/lingp/logs/portgpu-r7cred.log` |
| The re-fit's tests failing before its constant: 61 of 66 passed at 64719a0, failing the MEM_REFIT test, the spike test and rule 7's tests of the limit, the all-rows rows and the 16,384-token plan | measured | `~/lingp/logs/portgpu-r8cared.log` |
| The room's tests failing before 165d8ca's code: 63 of 67 passed at 8e06360, failing rule 7's tests of the limit, the all-rows rows and the 16,384-token plan, and the idle-room test | measured | `~/lingp/logs/portgpu-r7ered.log` |
| EMBED_ROWS', KDA_GATE_EPI's, KDA_NORM_PREP's and the DFlash refusal's tests failing before their code: 71 of 73 at 0afdc47, 73 of 76 at 8315619, 102 of 104 at 32f7c02, and 95 of 96 at d0d441c, each failing only the tests it was written to fail | measured | `~/lingp/logs/portgpu-m1hred.log`; our port notes (L6's red, L5's red at 32f7c02, the DFlash red), not published |
| 7 build errors on turboexl3 c7ce66c, after the MLX-free tests ran with no fork | measured, at ecc4a1c | `~/lingp/logs/portcs-cs1.log`; scripts/build.sh builds ReleaseFast only after the tests pass |
| The parity statistics, and the same 1,024 tokens with and without MTP drafting | measured, 2026-11-08, on dylib 7b1843f1 | `~/lingp/parity/w2/report.md`, copied to our results folder as `parity-w2/report.md`; the ruling on the fifth statistic is in our notes, not published; the on/off check ran with MLA_LATENT and MLA_FUSED on (our ledger: its parity result of 2026-11-08, and row R12) |
| The read pool's 8 worker threads in the server process | set | overlay/mlx_serve/sdk/weights/streamed/ling_source.zig `read_workers`; src/host_shims.zig `q3ld_mtl_signal` |
| mlx-serve's MTP draft loop, which calls the plugin's NextN head | set | plugin.zig `spec_hooks` |
| experts.bin 172.3 GB (172,268,453,888 B) | measured | the 2.60bpw pack's LING-REPACK-RECEIPT.json (`experts_bin`), copied to our results folder as `pack-records/LING-REPACK-RECEIPT-2.60bpw.json` |
| The download and pack sizes: at 2.60bpw 188,364,986,842 B and 188,221,814,270 B, with 55,595,155 B of the pack's receipts; at 2.10bpw 153,635,910,276 B and 153,492,941,070 B, with 55,546,809 B | measured on 2026-11-11, by `ls -l` of each folder's top level and of each pack's receipts folder | our results folder `pack-records/pack-sizes-2026-11-11.txt` |
| About 377 GB (2.60bpw) and 307 GB (2.10bpw) of disk | calculated: a download, its pack and the pack's receipts, 376,642,396,267 B (376.6 GB) and 307,184,398,155 B (307.2 GB) | `pack-records/pack-sizes-2026-11-11.txt` |
| The converter script's sha256, a135306e79949e13 | calculated: `shasum -a 256` of the script in our converter clone | the clone, not published; the pack's receipt records no script hash (`pack-records/LING-REPACK-RECEIPT-2.60bpw.json`) |
| macOS 26.2 or later, and Xcode 26.2 or later with its Metal toolchain | set by mlx-serve | mlx-serve 66aad949 docs/building.md, its Prerequisites |
| 2.59 bits per weight in the 2.60bpw config | set: the pack's own | tests/fixtures/config-2.60bpw.json (`quantization_config.bits`) |
| MTP drafting up to depth 3 | set: `--mtp-max-depth` is a maximum | mlx-serve 66aad949 src/main.zig `main`, its parse of `--mtp-max-depth` into `mtp_depth_max` |
| plugin.json's `>=26.11.3`, against 26.11.1 from a 66aad949 server | set | plugin.json's `requires_mlx_serve`; the third run's `serve.log` line 4; 66aad949 app/Sources/MLXServe/Services/PluginCatalog.swift `PluginCatalog.runs(on:requirement:)` |
| K6 for the output head | set: the pack's own | the config's `head_bits` (tests/fixtures/config-2.60bpw.json) |
| The ladder's six prompt sizes, 1,120 to 31,652 tokens, byte for byte the earlier ladder's prompts | measured: Ling's own token counts, and the same sha256 at every size | the final ladder's `manifest.json`; our port notes, the final ladder's prompts (the sha256 check), not published |
| The ladder's settings: step 6's options and no plugin option; greedy, and temperature 1.0, top_p 0.95, top_k 20; seeds 1234, 1234, 4321, 4321; max_tokens 1,024; the memory guard's limits | set | the final ladder's `greedy.env` and `card.env` (`FLAGS`, `SAMPLING`, `SERVER_ENV`); its scripts as they ran, in our results folder `ladder-01ca9769/scripts/`: ladder.sh lines 2-11 (the procedure, max_tokens included), 20 (the card's sampling) and 27-37 (the sizes, the warm-up and the seeds), and memguard.sh lines 12-13 and 62-75 (the guard's limits) |
| The final ladder's window, 2026-11-10 21:58:34Z to 2026-11-11 01:52:59Z, under one hold of the GPU lock and the exclusive build lock, and its end at rc=0 | measured | our port notes, the final benchmark and the final ladder (tk-ladder), not published; the ladder's `progress.txt` for each run's end; ladderhold.sh lines 2-7 and 24-30, and run1-mlxserve.sh line 24, for the hold |
| The earlier ladder's window, 2026-11-10 00:25:08 to 04:14:36Z, under the same hold | measured | `~/lingp/logs/next.log` line 250, the hold of the GPU lock and the exclusive build lock; `~/lingp/logs/chain.log` line 370, "END tk-ladder rc=0" |
| 92 tokens left at 31,652 prompt tokens | calculated: 32,768 - 31,652 - 1,024 | the ladder's sizes and settings above |
| Port 11234 | set by mlx-serve | mlx-serve 66aad949 src/main.zig `main`, its `port` |
| K 2.5 in every routed layer of the 2.60bpw pack, K 2 in the 2.10bpw pack | set: each pack's own, as the converter recorded it | each pack's LING-REPACK-RECEIPT.json (`k_by_layer`), copied to our results folder `pack-records` |
| The early build's table, bank lines, hit rates and pageouts | measured, 2026-11-06 | our results folder `ling-260-vs-210-d65ed111` (`summary.txt`, `serve-*.log`), not published |
| The context ladder's 48 runs, their means and the memory curve | measured, 2026-11-10 to 2026-11-11 | `~/lingp/ladder-01ca9769/report.md` and `memfit-windows.txt`; these pages round exact halves up, so four means differ from report.md's in the last digit |
| The ladder's lowest "Pages free" readings and its pageouts a run; pressure 1 and swap flat at 0.127 GB in all 48 runs, and no run stopped | measured, 2026-11-10 to 2026-11-11 | `~/lingp/ladder-01ca9769/results.tsv` (`free_min_gb`, `pageouts`), from each run's memory.tsv, which run1-mlxserve.sh lines 101-128 write every 5 s (`MEM=5`, ladder.sh lines 32 and 37); in our results folder `ladder-01ca9769/runs/<tag>/`, memory.tsv (`swap_gb` 0.127 in all 2,426 rows) and pressure.tsv (level 1 and swap 0.127 GB in all 3,881 readings, every 3 s, memguard.sh line 65); progress.txt's 48 "done" lines, which ladder.sh lines 39-45 write only for a run that memguard did not kill |
| The memory curve's planned total, 93,428 to 102,083 MiB | calculated: 14,734 MiB in use and the 76,329 MiB decode budget from the bank line, the KV of the request's prompt tokens at 297,216 B a token, and 2,048 MiB for `prompt_work` | `~/lingp/ladder-01ca9769/report.md`, its "planned bill" column, from ladder_report.py line 101 (in our results folder `ladder-01ca9769/scripts/`); 01ca9769's load lines; src/bill.zig `budget` |
| The bank line's decode budget less its prompt budget and the layer-major term, 1,534 MiB, against MEM_REFIT's 1,533 | calculated from the bank lines, which round each budget down to whole MiB | our results folder `ladder-c8d54a83/memrefit-billline-check.txt`: 1,534 MiB in all 60 serve logs of the final ladder, and 4,151 against 4,150 in the earlier ladder's |
| The memory curve's terms, 91,734 to 100,442 MiB; the remainders, −3,130 to −3,128 MiB at 16,377 tokens and 987 to 988 MiB at 31,652; 32 requests masked at 91,165 MiB, at the four sizes up to 8,185 tokens | calculated by memfit v3, which counts the bank's 5 prompt windows of 48 records (1,132 MiB) | `~/lingp/ladder-01ca9769/memfit-windows.txt`; the windows in src/bill.zig's MEM_REFIT test |
| The earlier ladder's 8 runs at 31,652 tokens on 202 slot rows: MLX's peak 98,807 MiB, 987 MiB past its terms of 97,820 MiB; its `[mem]` line masked the sizes up to 16,377 tokens, at 91,431 MiB | measured, 2026-11-10, and calculated by memfit v3 | `~/lingp/ladder-c8d54a83/report.md` and `memfit-windows.txt`; the fit in src/bill.zig's MEM_REFIT test, and the comment on `mem_refit_extra` at 186eae1 (the mask's condition) |
| The ladder's peak 426 to 427 MiB under the plan's ceiling at 31,652 tokens | calculated: 101,856 - 101,430 and 101,429 MiB | report.md's peaks and the plan's ceiling |
| The final ladder against the earlier one: the same prompts, 24 of 24 greedy runs with the earlier ladder's text at their size and seed, and the comparison table's means and changes | measured; the changes calculated from the unrounded means | our results folder `ladder-01ca9769` (`compare-c8d54a83.md`, from `ladder_compare.py`), not published; `~/lingp/ladder-c8d54a83/report.md` for the earlier means. compare-c8d54a83.md takes its changes from means rounded to 2 decimals, so some differ by 0.01 point |
| The changes between the two ladders' builds, in order: the re-fit (0ebec77), the room change (165d8ca), EMBED_ROWS (1a16e40), KDA_GATE_EPI (4c13ec7) with the DFlash capture refusal (d73e465), the test-only commits (1dffdcd) and KDA_NORM_PREP (bd00b1a) | measured: the order in which we served them | our port notes, the serving moves from 12:15:41Z to 20:59:54Z on 2026-11-10 and the DFlash refusal served from 583523d4, not published; this table's rows on the limits and the slot rows |
| The ladder's 8 charts: the six standard columns with the earlier ladder's means as a dashed line, the memory curve with the masked points flagged, and least free RAM and pageouts | drawn from the ladder's measured records | our results folder `ladder-01ca9769/charts/` (`ladder-01ca9769-<n>-<metric>.png`, n = 1 to 8), not published |
| The plan's ceiling as the refusal test's second and third runs record it, 101,855 MiB | calculated by the test script from whole-MiB load lines | the third run's r7e-echo.py, lines 56-64 |
| The 8 GiB margin: MLX's buffer cache, which mlx-serve caps at 8,192 MiB on this Mac, and other processes' wired GPU memory | set by mlx-serve | mlx-serve 66aad949 src/host/registry/memory_ceiling.zig `WIRED_LIMIT_MARGIN_BYTES` and the comment on `wiredLimitFloor` (the margin), and src/mlx/state.zig `mlxCacheLimitBytes` and `applyMlxCacheLimit` (the cap, RAM/16 up to 8 GiB); `~/lingp/ladder-l5v/report.txt` line 2, the cap's log line |
| The 32K check's peak footprints, 4 to 13 MiB past 108 GiB, the GPU wired limit less the reserve | calculated: 115,968,264,384 to 115,977,423,088 B, which is 110,596 to 110,605 MiB, less 110,592 MiB | `~/lingp/ladder-l5v/report.txt`, each run's footprint line |
| About 165 GB for the prompt log-probability step at 32,768 tokens | calculated: 32,768 x 157,184 x 32 B | src/bill.zig `echo_entry_bytes` (the 32 B); the vocabulary's 157,184 entries |
| The served code, fb966cc on turboexl3_ling-rq, from e44e00d on turboexl3_ling-l5: bd00b1a's code with one comment in src/bill.zig (186eae1), then RQ and the KV-quant warning (1875c04 to fb966cc) | read from git | `git log --stat bd00b1a..186eae1`; `git log --stat e44e00d..fb966cc` |
| The plan's ceiling, 101,856 MiB | calculated: 114,688 MiB less 8,192, 4,096 and 544 | the load line's room plus the bill's reserves (`top` in src/bill.zig's MEM_REFIT test) |
| The earlier re-fit's check: dylib 901a8e4a, 206 slot rows, 100,091 to 100,092 MiB in a warm-up and 4 runs, 988 to 989 MiB left over | measured, 2026-11-10 | `~/lingp/ladder-rf/memfit.txt` (memfit v3, the 4 runs) and `report.txt` (the warm-up's 100,092 MiB); the fit in src/bill.zig's MEM_REFIT test |
| The re-fit's own check: dylib fdd4e0d8, 210 slot rows, 101,373 MiB in a warm-up and 4 runs at 31,652 tokens, 987 MiB left over, peak footprints of 115.93 to 115.94 GB | measured, 2026-11-10 | `~/lingp/ladder-rc` (`memfit.txt`, `memfit-prewarm.txt`, `report.txt`, `verdict.txt`) |
| The re-fit's check 483 MiB under the plan's ceiling | calculated: 101,856 - 101,373 MiB | the check's peak and the plan's ceiling |
| MEM_REFIT's 1,533 MiB since 0ebec77 (989 and 544) | calculated from peaks measured on 2026-11-10 (989) and 2026-11-08 (544) | src/bill.zig `mem_refit_extra`, `mem_refit_pass` and `mem_refit_server`; tested by its MEM_REFIT test |
| The re-fit's room of 84,733 MiB, 210 prompt and 233 decode slot rows, and 75,101 MiB for decode | measured at 0ebec77 and 165d8ca; calculated from c8d54a83's bank line | the load lines above; src/bill.zig's MEM_REFIT test |
| Decode's 6,331 MiB more for the bank, and up to 102,400 MiB for MLX in decode | calculated: 76,329 - 69,998 MiB from 01ca9769's load lines, and 75,101 - 68,770 MiB at 0ebec77; 114,688 less 8,192 and 4,096 MiB | plugin.zig `promptOnlyBytes` and `openBank`; src/bill.zig `budget` |
| The streamer's windows and staging, 1,245 MiB; without them, 218 prompt slot rows at 01ca9769 and 214 at 0ebec77 | calculated | src/bill.zig's EMBED_ROWS test, and the windows and staging in its MEM_REFIT test |
| The remainder's rise of 0.25 to 0.26 MiB for each token from 18,380 to 31,652 tokens at 0ebec77, and of 0.27 MiB from 16,377 to 31,652 tokens at 01ca9769 | calculated: (987 + 2,320 to 2,484) / (31,652 - 18,380), and (988 + 3,130) / (31,652 - 16,377) | the re-fit A/B's `abcheck.txt` and the re-fit check's `memfit.txt`; the final ladder's memfit v3 remainders |
| 3,316 tokens at fb966cc; 3,317 at 165d8ca and at bd00b1a (2,426 at 0ebec77, 2,941 at 0e1f5c0) | calculated at load, and printed in each load line | src/bill.zig `echoRowsMax`, and `echo_plan` and its load line in plugin.zig `openBank`, at fb966cc and at bd00b1a; tested by src/bill.zig's tests `rule 7 at the served plan ...` (fb966cc) and `rule 7 at today's plan ...` (bd00b1a); the load lines of 01ca9769 and 4856f028 |
| The idle room, 7,933,640,704 B: `prompt_work`, MEM_REFIT's 989 MiB and the layer-major term's 5,030,789,120 B, less the rerank head's 281,673,728 B | calculated | src/bill.zig `echoIdle`, `rerankHead`, `ling_layer_major` and `echo_today`; tested by its tests `rule 7 at the served plan ...` and `rule 7: the plan's idle room ...` |
| 15,914 MiB of room at 165d8ca's limit (16,686,949,120 B at 3,317 tokens); for a refused 3,318-token request, a need of 15,916 MiB (16,689,168,384 B) against 15,914 MiB of room (16,686,651,904 B), which the refusal prints as 15916 and 15913 MiB, cutting off the fractions | calculated, and printed in the third run's refusal | src/bill.zig's test `rule 7 at today's plan ...` and `echoRefusal` at bd00b1a; the third run's `serve.log` line 106 |
| 32 B for each entry | set: the measured 31.6 B rounded up | src/bill.zig `echo_entry_bytes` |
| 22 B for each entry | calculated: mlx-serve's own account | mlx-serve 66aad949 src/engines/mlx/sampling/logprobs.zig `rowLogprobs` and the comment on src/engines/mlx/prefill_interleave.zig `PROMPT_LOGPROBS_CHUNK` |
| 26 B for each entry | calculated: MLX's six 4-byte arrays and the bf16 logits | MLX c69303b0 mlx/backend/metal/sort.cpp `multi_block_sort` and `ArgPartition::eval_gpu`; src/bill.zig's comment on `echo_entry_bytes` |
| The plugin's 2 B for each entry | calculated: the spans' bf16 copy | the logits' join in plugin.zig `spanMajorOr`, MLX c69303b0 mlx/backend/metal/eval.cpp `eval`, src/bill.zig's comment on `echo_entry_bytes` |
| 1.8 to 3.6 B not explained | calculated: 29.8, 30.5 and 31.6 B less 28 B | the refusal test's three runs; src/bill.zig's comment on `echo_entry_bytes` gives 2.5 to 3.6 B from the first two |
| About 269 MiB (281,673,728 B) for each 3-bit copy; one stays resident, because the rerank head drops the draft-only copy | calculated: 157,184 x 4,096 weights at 3 bits, with gs64 bf16 scales and biases (src/bill.zig `rerankHead`) | mlx-serve 66aad949 src/sdk/spec/mtp/mtp.zig: the comment on `MtpModel.rerank_coarse`, `MtpModel.buildDraftHead`, `MtpModel.maybeBuildDraftRerank` (which drops the draft-only copy) and `buildRerankCoarse`, called from src/engines/mlx/plugin/head.zig `load`; the readout mlx-serve borrows, the comment on src/ling_mtp.zig `q4_borrowed` |
| 1,577 and 1,578 rows; 26,065 tokens at `TURBOEXL3_LING_LAYER_MAJOR=0` | calculated | src/bill.zig's test `rule 7: an all-rows prompt forward of 1,577 rows or fewer ...` and src/bill.zig's formula at bd00b1a, with mlx-serve 66aad949 src/engines/mlx/generate/prefill_chunk.zig `nextChunkEnd` and src/sdk/bases/run.zig `TAIL_MERGE_MAX` |
| 2,394 tokens for a 16,384-token plan, and 1,800 for a room of the KV alone, at fb966cc; 2,403 and 1,828 at bd00b1a | calculated | src/bill.zig's test `rule 7: the limit follows the plan's context and room ...` at fb966cc and at bd00b1a |
| 3,525 tokens at 30 B for each entry | calculated: 17,672,814,592 / (157,184 x 30 + 297,216) | src/bill.zig's formula at bd00b1a (`echoRowsMax`) at 30 B |
| 256 rows an echo chunk | set by mlx-serve | mlx-serve 66aad949 src/engines/mlx/prefill_interleave.zig `PROMPT_LOGPROBS_CHUNK` and `promptLogprobsWidthCap` |
| 297,216 B of KV a token (296,064 and 1,152) | calculated | src/bill.zig `ropeOnceLayer` and `headLatentKv`; tested by its test `the rope-once cache: 2,268 MiB less KV ...` |
| 368,640 B for each token | calculated: mlx-serve's count | mlx-serve 66aad949 src/sdk/config/model.zig `ModelConfig.kvBytesPerToken` |
| MEM_REFIT's 4,150 MiB at c8d54a83 (3,606 and 544) | calculated from peaks measured on 2026-11-08 | `extra_c` in src/bill.zig's MEM_REFIT test |
| The bill's terms for a 32,768-token context | calculated | src/bill.zig, from `context` to `budget` |
| 4,797.7 MiB (5,030,789,120 B) for the layer-major pass; the load line prints it as 4797 MiB | calculated | plugin.zig `layerMajorBytes` and `layer_major_checkpoints`; src/bill.zig `ling_layer_major` and its test `rule 7 at the served plan ...` |
| 16,384 and 18,431 rows a call | set, and calculated | plugin.zig `layer_major_rows`, `layer_major_tail_rows` and `callRows` |
| 5,120 rows for PF_DFIN | calculated | src/ling_switches.zig `dfinCallTakes` and `dfin_call_min` |
| 883 MiB (925,499,392 B) for MTP_HEAD_Q4: 345 MiB in place of 1,228 MiB; the load line prints 882, its fraction cut off | calculated, and printed at load | src/bill.zig's test `the 4-bit draft readout ...`; `~/lingp/ladder-r7e/ladder-r7eecho/serve.log` line 22 |
| 13,312 tokens for the decline spill | calculated | the SPILL item in tests/ling_restore_host_test.zig's file comment |
| 96 GiB, the macOS default GPU wired limit on this Mac | calculated: 75% of 128 GiB | mlx-serve 66aad949 src/host/registry/memory_ceiling.zig `wiredLimitFloor` |
| The patches' 14 files at bd00b1a, 2,843 lines added and 148 removed | calculated | `git apply --numstat` on each file in patches/ at bd00b1a |
| 426 MiB under the plan's ceiling in each run of 01ca9769's 32K check | calculated: 101,856 - 101,430 MiB | the check's peaks and the plan's ceiling |
| mlx-serve reserves no KV for a whole-prompt pass, and its chunk loop reserves the prompt's KV and one chunk more for a prompt longer than one chunk; the KV grows by a quarter of its size, at least 256 and at most 8,192 tokens, rounded up to 256 | read from the code | mlx-serve 66aad949 src/engines/mlx/generate/init.zig `initWithOptions` (its `final_start`, its chunk loop's reserve and its whole-prompt forward); src/engines/mlx/generate/prefill_chunk.zig `reservedPrefillTokens`; src/sdk/kv/cache.zig `KVCache.nextCapacity` |
| The first grow step past the bill's 32,768-token term, for every prompt from 32 to 32,767 tokens: 32,000 to 40,192 tokens, 2,104 MiB past the term, at token 32,001 after prompts of 16,160 to 16,415, 16,927 to 20,511 and 21,023 to 32,000 tokens, and in the prompt pass past 32,000; 73 or 145 MiB at token 26,369 or 26,625 after prompts of 16,416 to 16,926 and 20,768 to 21,022; none after prompts of 20,512 to 20,767, whose KV reaches 32,768 exactly; eleven smaller steps in decode for prompts under 16,160 tokens, 73 to 1,959 MiB past the term, at tokens 26,369 to 31,745; the KV after each prompt pass, which is memfit's at all six ladder sizes | calculated: the growth rule replayed over the whole-prompt pass's spans, then one token at a time; each excess at 297,216 B a token | mlx-serve 66aad949 src/sdk/kv/cache.zig `KVCache.nextCapacity`; src/engines/mlx/generate/prefill_chunk.zig `nextChunkEnd` (which joins a last part under 512 tokens to the span before it) and `wholePromptPlan` (spans every 4,096 tokens up to 31 tokens before the prompt's end, then at its end); src/sdk/bases/run.zig `TAIL_MERGE_MAX`; the final ladder's memfit-windows.txt |
| A `--ctx-size` above 32,768 taken as given, while the bill's KV term stays at 32,768 tokens | read from the code | mlx-serve 66aad949 src/engines/mlx/admission/context.zig `getEffectiveContextLength`; config.zig `served`, whose `context_cap` line caps only the auto-sized context; src/bill.zig `context`, and `echo_plan` in plugin.zig `openBank` |
| The KV past 32,000 tokens after a prompt of 31,652 tokens: 40,192 tokens, 7,424 tokens and 2,104 MiB past the bill's 32,768-token term; 1.05 GB copied for each MLA layer as it grows | calculated: 32,000 + 8,000 rounded up to 256 tokens; (40,192 - 32,768) x 297,216 B; 32,000 x 32,896 B | mlx-serve 66aad949 src/sdk/kv/cache.zig `KVCache.nextCapacity`, `KVCache.updateDense` and `KVCache.growQuantBuf`; src/bill.zig `ropeOnceLayer` and `headLatentKv` |
| The final ladder's 8 runs at 31,652 tokens past 32,000 tokens in decode: the GPU's memory in use from 108.1 to 108.3 GB to 114.0 to 114.6 GB within 20 s, for one or two 5-second samples, then 109.2 to 110.6 GB; a rise of 2.0 to 2.3 GB from decode's first sample to its last; free RAM in decode down to 1.04 to 1.36 GB; pressure 1 and swap 0.127 GB | measured, 2026-11-10 and 2026-11-11; the 349th token calculated: 32,001 - 31,652; the 2.4 GB calculated: 8,192 x 297,216 B; the KV's growth as the rise's cause inferred from the timing | our results folder `ladder-01ca9769/runs/`, the eight `ladder-greedy-31744-<n>` and `ladder-card-31744-<n>` folders: memory.tsv (`mlx_gb`, ioreg's in-use system memory, and `free_gb`) and pressure.tsv; run1-mlxserve.sh's `mem()` for the columns |
| MLX's buffer cache, 8,126 to 8,190 MiB as each prompt pass of 01ca9769's 32K check ended | measured, 2026-11-10 | `~/lingp/ladder-l5v/report.txt`, its `[mem] a prompt pass ends` lines |
| The embedding table's 1,228 MiB (1,287,651,328 B), 3.83 slot rows, which gives 4 more prompt and 3 more decode slot rows | calculated | src/bill.zig `embedRowsFreed` and its EMBED_ROWS test |
| KDA_GATE_EPI's 315 launches, and 64 MiB of f's output a layer and 4,096-row span, at the standard prompt | calculated: 63 KDA layers x 5 wide spans; 4,096 rows x 8,192 channels x 2 B | the prompt's 18,380 tokens in 4,096-row spans and a 31-token tail |
| RQ on turboexl3_ling-rq, made from e44e00d: tests 1875c04 and code 5d0f17f, then the KV-quant warning's tests 04aedcb and code f6f3415, then RQ built in, tests 44ba83b and code fb966cc | read from git | `git log e44e00d..fb966cc` on turboexl3_ling-rq |
| RQ's dylib 89be02b1 with `TURBOEXL3_LING_RQ=1`: its gates (MLX-free tests 146 of 146, GPU byte tests 53 of 53, restore check part B 24 checks); its safety test, SAFE-PASS (1,120 tokens, then 32,000, then a 32,198-token prompt refused by name in 0.018 s; no kill, pressure level 1, swap flat at 0.127 GB); its A/B test against 01ca9769, kept under rule 3 (decode 25.825 to 26.000 tok/s, prefill 700.325 to 701.000 tok/s, text 09571f6a in all 8 runs); at the safety test's 32,000-token prompt, one `RQ over plan` line (the plan holds 211 rows, under the base 214, so the bank stays at its base), MLX's prompt-pass peak 101,318 MiB, 538 MiB under the plan's ceiling, and O at 1,733 MiB in that request's decode plan | measured, 2026-11-11: the gates 06:25:36 to 06:32:06Z, the A/B decided at 07:11:10Z; the 538 MiB calculated | `~/lingp/logs/portgates-rq.log`; `~/lingp/ladder-rqsafe-89be02b1/report.txt`; `~/lingp/abba/smoke-rq-89be02b1/smoke.txt`; our results folder `ladder-rqsafe-89be02b1/report.txt` lines 29, 32 and 35; our port notes and ledger (`lane-decisions.md`), not published |
| RQ's refusal: a prompt forward whose KV capacity, by mlx-serve's schedule, would pass 32,000 tokens; its slot table, which counts another live request's KV as it is | read from the code | src/ling_rq.zig `prompt_capacity_max`, `promptRefusal` and `SlotTable` (the other request's KV in `SlotTable.touch`), at f6f3415 and at fb966cc |
| mlx-serve's buffer-cache clear every 256 decoded tokens; the 8 runs' high samples from about the 349th to about the 512th decoded token, and the card runs' dip near the 256th | read from the code; the token windows estimated from each sample's time and its run's decode speed | mlx-serve 66aad949 src/engines/mlx/generate/generator.zig `Generator.advanceStep` and its comment, and prefill_chunk.zig `CACHE_CLEAR_INTERVAL`; our results folder `ladder-01ca9769/runs/`, each `ladder-*-31744-*` folder's memory.tsv (`mlx_gb`) and run.json (`decode_tok_s`) |
| The 32K check's footprint reported, not tested | read from the record | `~/lingp/ladder-l5v/verdict.txt`: "reported, not a hold" |
| A request's KV scheme: its own `kv_quant`, then model-settings.json's, then `--kv-quant`, which is dense bf16 by default; no scheme reaches this plugin at load | read from the code | mlx-serve 66aad949 src/engines/mlx/scheduler.zig `Scheduler.submit`, src/engines/mlx/slot.zig `Slot.init`, src/host/registry/model_settings_apply.zig `applyModelSettings` and src/host/gpu_thread/params.zig `LoadParams.kv_quant_config`; src/engines/mlx/plugin/driver.zig `loadSettings` and `streamRoomAt`; config.zig `served`, which sets no `owns_decode_state` |
| A quantized cache's dense K and V at each append, which the plugin's MLA layers read: about 1.0 GiB for each MLA layer and call at 32,768 tokens; the quantized cache itself, about 53% (8-bit) or 28% (4-bit) of the bf16 KV term | read from the code; the sizes calculated: 32,768 x 16,384 x 2 B for V's rows, in the dtype that the cache was fed, and (1 or 0.5 B + 4 B / 64) / 2 B a value; not measured | mlx-serve 66aad949 src/sdk/kv/cache.zig `KVCache.updateAffine` and src/sdk/blocks/attention/kv_quant.zig `KVQuantConfig.affine` (groups of 64) and `dequantizeAffine`; patches/ling-mla.patch, the rope-once append; src/ling_mla_rope1.zig `views` |
| Layer 7, the first MLA layer, where the plugin first sees the scheme: in the load's warm-up for `--kv-quant` or model-settings.json, whose failure mlx-serve only logs, and partway through a request's first forward pass for its own `kv_quant` | read from the code | mlx-serve 66aad949 src/engines/mlx/load.zig `doLoadOnInferenceThread` (its `kv_quant_config` on the model's own cache, and its warm-up, "Warmup failed ... continuing"); src/main.zig `main`, its `warmup_eager` (the warm-up on by default); src/engines/mlx/dispatch.zig `forwardWith` (a forward goes to the plugin); the config's `layer_group_size`, 8 (tests/fixtures/config-2.60bpw.json); an MLA layer is one whose index plus 1 is a multiple of 8 (mlx-serve bc6c5f2c src/plugins/glm5_next/config.zig `Config.isLinearLayer`, which the generated copy keeps) |
| mlx-serve's rule for drafting paths: a path that refuses a KV scheme is gated at load, or implemented | read from the docs | mlx-serve 66aad949 src/engines/mlx/AGENTS.md, its spec-decode rule "A spec path that refuses a KV scheme must be gated at LOAD, or implemented" |
| A DFlash drafter declined at load | set by the code | config.zig `served`; mlx-serve 66aad949 src/engines/mlx/load.zig `doLoadOnInferenceThread`, its `drafter_dir` |
| The served dylib 4856f028 (4,784,760 B), built at fb966cc with turboexl3 4bf2df8 and a kit from mlx-serve 66aad949, served from 08:08:00Z on 2026-11-11 | measured, 2026-11-11 | our results folder `gates-4856f028/portgates-rqh.log`, its START and RELEASEFAST_END lines; `lane-decisions.md`, 08:08:00Z |
| 4856f028's gates: MLX-free tests 144 of 144 in 24 of 24 steps; GPU byte tests 53 of 53 with 653 PASS lines and no FAIL line; restore check part B 24 checks over 7 restores, 0 failed, BYTES_OK; test_reach.py green, its model 144 MLX-free tests and 84 GPU tests (53, and 31 in part A), 0 unreached, each binary's count agreeing; carries.sh rc=0 | measured, 2026-11-11, 07:41:41 to 07:48:23Z | our results folder `gates-4856f028/portgates-rqh.log` |
| The red at 44ba83b: 138 of 144 MLX-free tests passed, 6 error records under 3 names, RED-OK | measured, 2026-11-11 | our results folder `gates-4856f028/portgpu-rqhred.log` |
| 4856f028's 32K check: a warm-up and 4 greedy runs at 31,652 tokens; MLX's peak 101,430 MiB and the remainder 988 MiB in every run, against terms of 100,442 MiB; peak footprints of 114.62 to 114.80 GB, the highest 1.178 GB under the highest of 01ca9769's 32K check, 115.98 GB; 221 decode slot rows; free RAM down to 0.059 to 1.681 GB and 253 to 408 pageouts a run; pressure 1, swap flat at 0.127 GB, no stop; text 919cf0fb in all 5 runs; each run's speeds, wall time and MTP counts | measured, 2026-11-11, 07:48:23 to 08:08:00Z; the 1.178 GB calculated, 115,977,423,088 B less 114,799,162,856 B. The check script's own line, −1.144 GB, compares with the earlier ladder's runs on c8d54a83, 115.94 GB. 01ca9769's final ladder printed 116.0, 116.0, 116.0 and 115.9 GB at that size, in 0.1 GB steps | our results folder `ladder-rqv-4856f028`: `verdict.txt`, `memfit.txt`, `report.txt` (each run's footprint line, and the script's `footprint max` line), `results.tsv` (`free_min_gb`, `pageouts`), `rqv.env`, and each run's `serve.log`, `client.json` (`wall_s`) and `cool.txt`; the MTP counts from each `serve.log`'s last `[spec-stats] mode=mtp` line; 01ca9769's footprints in `ladder-l5v-01ca9769/report.txt`; `peak_gb` in `ladder-c8d54a83/results.tsv` and `ladder-01ca9769/results.tsv` |
| 4856f028's 32K check footprints, 1,111 to 1,286 MiB short of 108 GiB, the GPU wired limit less the reserve | calculated: 114,615,285,200 to 114,799,162,856 B, which is 109,306 to 109,481 MiB, against 110,592 MiB | `ladder-rqv-4856f028/report.txt`, each run's footprint line |
| 4856f028 against the ladder at 31,652 tokens: decode 2.0% lower and prefill 0.2% higher | calculated from the unrounded means: 23.675 against 24.150 tok/s, and 615.35 against 614.275 tok/s | `ladder-rqv-4856f028/results.tsv`; the ladder's run table in "The context ladder" |
| RQ's A/B test: the runs, the means, and B against A | measured, 2026-11-11, decided at 07:11:10Z; the changes calculated | our results folder `smoke-rq-89be02b1`: `smoke.txt` (its run table, its table of means and blocks, and its MTP, engagement, trajectory, pageout and memory lines), `summary.txt` (free RAM) and `memory-1.tsv` to `memory-8.tsv` (swap); `lane-decisions.md`, 07:11:10Z |
| SPEC1_BUSY's A/B test: 4856f028 as arm A and 23731721 as arm B; the runs, the means, B against A, free RAM, pageouts, swap, the 214 and 237 slot rows, O at 1,222 MiB and the 36 of 41 paths | measured, 2026-11-11, 08:26:20 to 09:01:19Z; the changes calculated; rejected by us under rule 6 | our results folder `smoke-spec1-23731721`: `smoke.txt`, `summary.txt` (free RAM), `memory-1.tsv` to `memory-8.tsv` (swap), `A.env` and `B.env`, and the A runs' `serve-<n>-A.log` (the load, RQ and request lines); `lane-decisions.md`, its SPEC1_BUSY row; our port notes (`port-step2-manifest.md`), for the window, not published |
| RQ's plans: the KV capacities, O and the decode slot rows at 1,120, 18,380, 31,652 and 32,000 prompt tokens, and `RQ over plan` at 31,652 and 32,000 | measured, 2026-11-11 | the `RQ` lines in `ladder-rqsafe-89be02b1/report.txt` (lines 21-32), in `smoke-rq-89be02b1/serve-2-B.log`, `serve-3-B.log`, `serve-5-B.log` and `serve-8-B.log`, in `smoke-spec1-23731721/serve-<n>-A.log` for runs 1, 4, 6 and 7, and in each `ladder-rqv-4856f028/ladder-rqv-31744-<n>/serve.log` |
| O's parts: the rerank head 268.63 MiB, the KDA state 134.86 MiB, 1, 5 and 8 checkpoint copies of 134.86 MiB, and 8 KiB a prompt token of hidden rows; their sums, 547.09, 1,221.38 and 1,732.36 MiB, against the logged 548, 1,222 and 1,733; 2 and 4 more decode slot rows with the idle items freed | calculated from the code and the logs, 2026-11-11 | our port notes, `port-step2-manifest.md`, its O root-cause entry, not published |
| RQ's mechanism: the schedule's replay, the plans, the 4,096-token headroom, stage 1, the refusal past 32,000 tokens and the slot table of 8 | read from the code | src/ling_rq.zig at fb966cc (`nextCapacity`, `replay`, `grownTo`, `kvBill`, `headroom`, `decodeEnd`, `rowsAt`, `planPrompt`, `planDecode`, `prompt_capacity_max`, `promptRefusal`, `SlotTable`, `max_slots`); plugin.zig `forward`, `phaseTo`, `rqPlan`, `rqLine`, `rqAtForward` and `rqForward` |
| RQ's safety test: 1,120 tokens, then 32,000 with 768 decoded tokens, then 32,198 refused by name in 0.018 s with HTTP 500; no kill, pressure level 1, swap flat at 0.127 GB; MLX's prompt-pass peak 101,318 MiB at 32,000 tokens | measured, 2026-11-11, 06:32:07 to 06:34:35Z, on 89be02b1 | our results folder `ladder-rqsafe-89be02b1/report.txt`; our port notes, `port-step2-manifest.md`, for the times |
| The KV_SCHEME_DENSE_COPIES warning, once a load, and its 1,028 MiB a layer and call at 32,768 tokens | read from the code; the size calculated: 32,768 x (64 x 256 + 64) x 2 B | src/ling_kv_scheme.zig `warning_fmt`, `warning` and `atView`, and its tests, at fb966cc |
| 3,328 tokens of KV for a whole prompt of 3,316 or 3,317 tokens, and the room of 16,683,679,744 B there | calculated | src/ling_rq.zig `wholePromptCapacity`, and src/bill.zig's test `rule 7 at the served plan ...`, at fb966cc |
| The patches at fb966cc: 14 files, 2,883 lines added and 148 removed | calculated | `git apply --numstat` on each file in patches/ at fb966cc |
| 41 `Path` names at fb966cc, RQ the last; a request line naming 36 of the 41 on 4856f028 and on 89be02b1 | read from the code; measured | src/ling_ran.zig `Path` at fb966cc; each `ladder-rqv-4856f028` run's `serve.log`; `smoke-spec1-23731721/serve-<n>-A.log`; `smoke-rq-89be02b1/serve-2-B.log` |
| scripts/mlx_serve.sha256 at 26b576c4 at fb966cc | calculated by scripts/sync-mlx-serve.sh | the file at fb966cc |

## Terms

| Term | Meaning here |
|---|---|
| 32K check | our memory check of a dylib: a warm-up and 4 greedy runs at 31,652 prompt tokens, each on a new server ("Memory") |
| A/B, arm | a comparison of two builds on the same workload; each build is an arm |
| A2 | a parity test's second serve of the base, A, which measures the run-to-run noise ("Parity") |
| Bank | the expert cache: the GPU memory slots that hold the routed experts each layer has read from the SSD |
| Bill | the memory that the plugin asks mlx-serve to hold back from the bank. In the prompt pass it holds the KV, `prompt_work`, the layer-major term, the reserve and MEM_REFIT; in decode, the KV, decode's working set and the reserve (src/bill.zig) |
| bpw | bits per weight; the 2.60bpw branch's config records 2.59 |
| Byte test | a GPU test that a path we keep as bit-exact gives the same bits as the path it replaces |
| Call | one routed layer's run of its experts over a set of rows: up to 18,431 prompt rows in the layer-major pass, or a decode step's rows |
| Checkpoint | the KDA layers' state at a prompt position, which the prefix cache stores |
| Chunk loop | mlx-serve's prompt pass in forward passes of `--prefill-chunk` tokens |
| Context ladder | our runs of one coding task at six prompt sizes, 1,120 to 31,652 tokens, greedy and with the card's settings, on 01ca9769; an earlier ladder ran on dylib c8d54a83 ("Measured") |
| Decline spill, flush | mlx-serve's two ways to write a prompt's cache to its SSD tier |
| Download | a branch of turboderp/Ling-3.1-flash-exl3 as Hugging Face serves it |
| Dylib `<hash>` | a build of the plugin, named by the first 8 hex digits of its dylib's sha256 (the load line prints 12); not a commit id |
| Exact verify | a verify pass that computes each row as a one-row decode does (`Caps.exact_drafted`) |
| EXL3, exllamav3 | exllamav3 is turboderp's quantization library; EXL3 is its trellis format |
| Forward pass | one run of a set of tokens through the model |
| Gates | our build and test run of a dylib on the Mac before we serve it: the MLX-free tests, test_reach.py, the ReleaseFast build, the GPU byte tests and the restore check ("Tests") |
| glm53-eval | our benchmark harness, which is not in this repo |
| Host, engine | mlx-serve, as the plugin's log lines name it |
| Idle room | the part of the bill that nothing holds during mlx-serve's prompt log-probability step: the KV past the prompt and the idle terms, less mlx-serve's resident rerank head ("Memory") |
| Idle terms | the bill's terms that a prompt pass alone uses: `prompt_work`, MEM_REFIT's pass part and the layer-major term |
| K, mul1 | K is one EXL3 tensor's bits per weight; mul1 is the codebook, whose multiplier each projection stores and the plugin checks |
| KDA, MLA | Ling's two attention kinds: Kimi Delta Attention (linear) and multi-head latent attention (9 layers) |
| Kit | the mlx-serve plugin kit the plugin compiles against (`KIT_VERSION`, or the folder `MLX_SERVE_KIT` names) |
| KV | the attention cache that grows with each token: here the MLA layers' rows and the MTP head's latent, 297,216 B for each token |
| Last-row flag | `Caps.prompt_last_row` (ddalcu/mlx-serve#1133): mlx-serve asks a prompt for its last row of logits only |
| Launch | one GPU kernel dispatch |
| Layer-major pass | the plugin's prompt pass: the whole prompt, one layer at a time, over the chunk loop's spans |
| MEM_REFIT | the bill's term fitted to measured prompt-pass peaks: a pass part and the server's own 544 MiB (src/bill.zig) |
| MLX-free tests | the plugin's `zig build test`: no GPU, no MLX and no pack |
| MoE | mixture of experts: a block whose router sends each token to a few of its expert FFNs |
| MTP head | the pack's NextN layer, which drafts tokens (multi-token prediction) for the verify pass |
| NAX | the matrix units of M5-class GPUs |
| Norm-gate | mlx-serve's launch that writes a KDA layer's gated norm at prompt rows: the RMS norm of the recurrence's output times the sigmoid of the gate projection's output |
| Number kinds | measured in our runs; calculated from the code; set, a setting or the model's size; or fitted to measured peaks (MEM_REFIT). "Receipts" gives each number's record. "Inferred" marks a cause that no record states |
| o_proj | a KDA layer's output projection |
| Oracle | exllamav3's own logits for the parity prompt |
| Output head | the trunk's last projection, from the hidden state to the vocabulary's logits |
| Outside plugin | a dylib that mlx-serve loads at run time, built from this repo |
| Overlay, patches | overlay/ holds this plugin's own files at mlx-serve paths; patches/ holds the four edits to the copied files |
| Pack | the converter's output, the folder that mlx-serve serves: `experts.bin`, its manifests and the other tensors |
| Parity test | a test that a change keeps the plugin within set margins of its distance from the oracle ("Parity") |
| Plan | the bill's layout of the GPU memory for its 32,768-token context; RQ makes one for each request ("RQ") |
| Plan's ceiling | the most memory that the prompt-pass bill lets MLX hold: the 112 GiB limit minus the 8 GiB margin, the 4 GiB reserve and MEM_REFIT's 544 MiB server part, which makes 101,856 MiB ("Memory") |
| Port | our move of the plugin's kit and server to mlx-serve main 66aad949; the generated model copy stayed byte-identical |
| Prefix cache | mlx-serve's cache of earlier prompts' state, in RAM (the RAM tier) and on the SSD (the SSD tier) |
| Prompt log-probabilities | an `echo` request with `logprobs`: the log-probability of every prompt token |
| Prompt lookup (PLD) | drafting by copying tokens that follow a match in the prompt |
| Prompt pass | the forward passes over a request's prompt (prefill) |
| Refusal test | our served test of the rule-7 refusal, run twice on 2026-11-09 and once on 2026-11-10 ("Memory") |
| Release kit | the kit that an mlx-serve release publishes beside it (`mlx-serve-plugin-kit-<version>.tar.gz`) |
| Remainder | MLX's peak minus the trunk, the bank's prompt slot rows and read windows, the KV that mlx-serve allocated, `prompt_work` and the layer-major term; MEM_REFIT's pass part holds it ("Memory") |
| Read-ahead predictor | the scores that choose which expert records the streamer reads before the next routed layer needs them: that layer's gate on the current rows. They never pick a routed expert (src/ling_predictor.zig) |
| Read window | the streamer's scratch for one batch of expert reads, 48 records of MLX memory; the prompt pass holds 5, decode 1 ("Memory") |
| Rerank head | mlx-serve's resident 3-bit copy of the 4-bit draft readout, 281,673,728 B; the log calls it the "coarse 3-bit/gs64 head" |
| Reserve | the bill's 4 GiB for the rest of the Mac (`free_ram`) |
| Restore check | our test that a prefix restored from mlx-serve's SSD tier holds the bits that the prompt pass wrote ("Tests") |
| Room | the GPU memory that mlx-serve gives the plugin for the trunk and the bank: the 112 GiB limit minus the 8 GiB margin and the prompt-pass bill, 84,733 MiB ("Memory") |
| Routed experts | an MoE block's 512 expert FFNs, of which the router picks 8 for each token |
| Row | one token position of a forward pass; the bank's slot rows are another thing ("Slot row") |
| Rule 7 | AGENTS.md's rule 7, "Bill what you allocate"; the code names the prompt-log-probability refusal and its tests after it |
| RQ | the served plan of each request's KV and decode slot rows, built in at fb966cc; stage 1 keeps the prompt pass at the base ("RQ") |
| RQ's terms | the base: the 214 slot rows that the bank opens with for the prompt pass; O: the other MLX memory that RQ measures at a plan; a trim: a new decode plan when decode passes the planned KV; stage 1: RQ as served, which never takes the prompt pass under the base ("RQ") |
| Run-path line | `[turboexl3_ling] served paths ran ...`: the optimizations that ran, since load or for one request |
| Served stack | every optimization in the served dylib; our decision rules kept most after an A/B test, and our ledger records five kept by a ruling ("The served stack") |
| Slot | one expert's place in the bank, for one layer |
| Slot row | in the bank's log lines, one slot in each of the 68 routed layers: 336,427,008 B |
| Span, span fields | a span is one chunk that the chunk loop runs, or would run, as mlx-serve names it to the plugin; the span fields carry them (ddalcu/mlx-serve#1126) |
| SSD tier | mlx-serve's prefix cache on disk (`--prefix-cache-disk`) |
| Standard A/B test | our speed test: glm53-eval's coding workload in 8 runs, A B B A then B A A B ("Measured") |
| Streamer | the expert streamer, a patched copy of mlx-serve's (sdk/weights/streamed): it reads routed experts from the SSD into the bank |
| Trunk | every weight except the routed experts and the MTP head; it includes the embedding table and the output head (README, "How it works") |
| TTFT | time to the first token, measured by the client |
| turboexl3 | the library of Metal kernels for EXL3 weights that runs the plugin's EXL3 math |
| Units | GB is 10^9 bytes, and GiB and MiB are 2^30 and 2^20 bytes |
| Verify pass | the trunk's check of a round of drafted tokens |
| W_hat | an EXL3 weight decoded to f16 |
