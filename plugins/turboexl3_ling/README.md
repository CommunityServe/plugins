# turboexl3_ling

This is the pre-review README of turboexl3_ling 0.1.0, which we serve on one Mac and have not released. We, its
maintainers (davidtai), wrote it on 2026-11-10 and 2026-11-11, and times are in UTC. The source is
`plugins/turboexl3_ling` on our unpushed branch `turboexl3_ling-rq` of CommunityServe/plugins. Since 08:08:00Z on
2026-11-11 we serve dylib 4856f028, built from commit fb966cc with turboexl3 4bf2df8 and a kit from mlx-serve 66aad949.
We name a dylib, and a decoded text such as 09571f6a, by the first 8 hex digits of its sha256.
[EVIDENCE.md](EVIDENCE.md) holds this page's evidence and names the record of each number.

## Summary

turboexl3_ling is an outside plugin that lets mlx-serve run inclusionAI's Ling-3.1-flash in EXL3, turboderp's trellis
format. We convert turboderp's 2.60bpw (bits per weight) weights into a pack, the folder that mlx-serve serves. Its
routed experts take 172.3 GB (measured). That is over the Mac's memory, so the plugin streams them from the SSD. On an
M5 Max with 128 GiB, the served dylib averaged 25.83 tok/s decode and 700.0 tok/s prefill on an 18,380-token coding
prompt ("Measured"). With our serve command, mlx-serve keeps no prefix between requests, so each chat turn runs its
whole prompt pass (prefill) again. You cannot yet build the plugin or make a pack, because three parts are not published
("Requirements"). The plugin plans its memory for each request's attention cache (KV), and it refuses a prompt of more
than 32,000 tokens. We have not tested two long requests at once, and a 4- or 8-bit KV gets only a warning ("Memory").

## What it serves

The plugin serves text only, on one Mac, with a 32,768-token context and prompts of up to 32,000 tokens. It has no
batched decode, so mlx-serve interleaves a second request's forward passes with the first's. At 18,380 tokens, each
turn's prompt pass takes about 26 s.

We have served this build only on the 2.60bpw pack. It also accepts the 2.10bpw pack. Our parity test found it further
than 2.60bpw from the output of exllamav3, turboderp's library.

We have not published our run records, benchmark harness or parity scripts, so you cannot repeat our measurements yet.

See [EVIDENCE.md, "Parity"](EVIDENCE.md#parity) and ["Smaller gaps"](EVIDENCE.md#smaller-gaps).

## How it works

The trunk is every weight except the routed experts and the MTP (multi-token prediction) head. The plugin keeps the
trunk in GPU memory, except its embedding table, whose rows the CPU reads from the pack. It streams the routed experts
from the SSD.

Ling-3.1-flash has 72 layers with two kinds of attention: KDA (Kimi Delta Attention, a linear attention) and MLA
(multi-head latent attention, 9 layers). From layer 4 on, each layer ends in a mixture-of-experts (MoE) block, whose
router picks 8 of 512 routed experts for each token.

The MTP head drafts tokens, and the trunk checks a round of them in one verify pass. The plugin's exact verify computes
each verify row as a one-row decode does. So in greedy decoding, MTP drafts do not change the text.

The plugin's streamer reads the routed experts of layers 4 to 71 from the SSD into a GPU cache, the bank. mlx-serve runs
the HTTP server, the tokenizer, sampling, the prefix cache and the MTP draft loop, and the plugin runs each forward
pass. turboexl3 (github.com/davidtai/turboexl3), a library of Metal kernels for EXL3 weights, runs every EXL3
projection and routed expert call.

mlx-serve's chunk loop runs a prompt in forward passes of `--prefill-chunk` tokens. This plugin runs the whole prompt
pass (prefill) in one forward pass, one layer at a time. We call it the layer-major pass. Each routed layer then reads
each expert it needs once for up to 18,431 prompt tokens, not once for each 4,096-token chunk.

Most of the plugin's optimizations change mlx-serve's Ling model and expert streamer, and a plugin cannot change
mlx-serve itself. So the plugin compiles its own copy of both, with four patches and five cut-down MLX kernels.

See [EVIDENCE.md, "The copy of mlx-serve's code"](EVIDENCE.md#the-copy-of-mlx-serves-code) and
["The served stack"](EVIDENCE.md#the-served-stack).

## Memory

The plugin divides its GPU memory between the trunk, the bank and a bill. The bill is the memory that the plugin asks
mlx-serve to hold back from the bank. At load, the prompt pass's bill has five terms:

| Term | MiB | Kind |
|---|---|---|
| The KV for 32,768 tokens | 9,288 | calculated |
| `prompt_work`, the pass's working set | 2,048 | set |
| The layer-major pass's extra buffers | 4,797.7 | calculated |
| `free_ram`, the reserve for the rest of the Mac | 4,096 | set |
| MEM_REFIT, what the other terms miss: 989 for the pass and 544 for the server | 1,533 | fitted to measured peaks |
| Total | 21,763 | calculated |

mlx-serve takes its 8 GiB margin and this bill from the GPU wired limit of 112 GiB. That leaves the plugin 84,733 MiB,
the room (calculated). The trunk and the MTP head held 14,734 MiB of the room (measured). The streamer's read windows
and staging take 1,245 MiB of it (calculated). The bank gets the rest.

The bank holds experts in slots. A slot row, 320.8 MiB, is one slot in each of the 68 routed layers. In the prompt pass
the bank holds 214 slot rows, 214 of each layer's 512 experts (measured). This is the bank's base. The streamer reads
any other expert from the SSD when a token needs it. Decode's bill leaves out the layer-major term and MEM_REFIT, and
decode keeps 1 of the prompt pass's 5 read windows. So at load the bank's decode budget holds 236 slot rows.

RQ, the plugin's memory plan for each request, sets decode's slot rows. mlx-serve grows a KV by a quarter of its size,
between 256 and 8,192 tokens, to a multiple of 256. Before decode, RQ replays that growth. It bills the KV that
mlx-serve will allocate, for the prompt and 4,096 more tokens, in place of the bill's KV term. It also counts O, the
other MLX memory that it measures then. Decode gets the slot rows that are left: 258 at 1,120 tokens on 89be02b1, 237 at
18,380 and 221 at 31,652 (measured). If decode passes the planned KV, RQ plans again.

The ceiling, 101,856 MiB, is the most that MLX may hold in the prompt pass (calculated). It is the 112 GiB limit less
the 8 GiB margin, the 4 GiB reserve and MEM_REFIT's 544 MiB server part. Before we serve a dylib with changed code, our
32K check sends it a warm-up and 4 greedy requests of 31,652 tokens. In the served dylib's check, MLX's peak stayed 426
MiB under the ceiling, and memory pressure stayed normal (measured).

Three open points remain:

- RQ keeps the prompt pass at the base. Near 32,000 tokens RQ's prompt plan holds only 211 slot rows, so RQ logs
  `RQ over plan` and keeps 214. MLX's peak stayed 538 MiB under the ceiling at 32,000 tokens on dylib 89be02b1
  (measured).
- We have not tested two long requests at once. RQ counts another live request's KV as it is, not its growth.
- A 4- or 8-bit KV gets no refusal, because mlx-serve never tells the plugin the KV scheme at load. Each MLA layer then
  reads a dense copy of its KV that no bill term holds, 1,028 MiB a call at 32,768 tokens (calculated).

See [EVIDENCE.md, "Memory"](EVIDENCE.md#memory).

## Requirements

We have served the plugin only on an M5 Max with 128 GiB (137.4 GB). Step 6's 112 GiB GPU wired limit is for that Mac.
It needs an M5-class GPU, because turboexl3's prompt kernels run on NAX, its matrix units. You also need about 377 GB of
disk, macOS 26.2 or later and some tools (block below). Close other large apps, because free RAM fell to about 0.06 GB
in our long-prompt runs.

Three unpublished parts stop you today, so you can do only steps 1 and 3.

- turboexl3's Ling kernels (step 4) are on its branch ling-l5-normprep at 4bf2df8, which is not on GitHub yet.
- Our converter's Ling script (step 2) is scripts/ling_exl3_to_streaming.py. Its repo,
  github.com/OpenSourceWTF/qwen38-exl3-mlx-serve-converter, does not have it yet.
- This plugin's pull request, which step 4 fetches, is not open yet.

<details>
<summary>Other apps, disk, tools, the SSD and turboexl3's Ling branch</summary>

- On a Mac with less memory, which we have not measured, the load refuses the pack by name (`StreamedRowsDoNotFit`) if
  no slot row fits.
- Close other large apps, and run one large server at a time (AGENTS.md rule 11).
- Put the download and the pack on one volume, because the converter hard-links the small files. Together they take
  about 377 GB at 2.60bpw and 307 GB at 2.10bpw (calculated).
- The plugin reads experts from the SSD during each request, so the drive's read speed affects the plugin's speed. We
  did not record our Mac's SSD. The early build's prompt pass read 12.6 to 13.0 GB/s
  ([EVIDENCE.md, "An early build"](EVIDENCE.md#an-early-build)).
- The steps need network access, git, the Hugging Face CLI (`hf`) and Python 3. Step 3 also needs Homebrew, and Xcode
  26.2 or later with its Metal toolchain (`xcodebuild -downloadComponent MetalToolchain`).
- For the 2.10bpw branch, pass `--revision 2.10bpw` in step 1 and `--rev 2.10bpw` in step 2. We have not served that
  pack on this build ("What it serves").
- turboexl3's Ling kernels for this branch and the served dylib are on its branch ling-l5-normprep at 4bf2df8. Its four
  commits on aa0ce54, da0b1b1 to 4bf2df8, add KDA_NORM_PREP's launch (24c47fa) and its tests. aa0ce54 (branch
  ling-l6-kdagate) adds KDA_GATE_EPI's launch to ling-strip2's c2783f5.

</details>

## Install and serve

Each `<...>` placeholder that names a file or folder is an absolute path.

1. Download a branch of turboderp's repo (https://huggingface.co/turboderp/Ling-3.1-flash-exl3):

       hf download turboderp/Ling-3.1-flash-exl3 --revision 2.60bpw --local-dir <download>

2. Convert the download into a pack with the converter's Ling script (block below; not yet possible, "Requirements").

   <details>
   <summary>Step 2's commands</summary>

   Run these from the converter's clone:

       python3 scripts/ling_exl3_to_streaming.py plan --src <download> --rev 2.60bpw --plan plan.json
       python3 scripts/ling_exl3_to_streaming.py run --src <download> --out <pack> --plan plan.json
       python3 scripts/ling_exl3_to_streaming.py finalize --src <download> --out <pack> --plan plan.json

   In a good run, `run` prints `ALL_SHARDS_PASS <n>/<n>` last, and `finalize` prints `FINALIZED <pack>: ...`.

   </details>

3. Build mlx-serve and its kit at 66aad949 (block below), because no mlx-serve release has this plugin's kit fields
   yet. `<server>` is then `<mlx-serve clone>/zig-out/bin/mlx-serve`.

   <details>
   <summary>Step 3's commands, other kits and servers, and our kit's name</summary>

       git clone --recurse-submodules https://github.com/ddalcu/mlx-serve && cd mlx-serve
       git checkout 66aad949 && git submodule update --init
       brew bundle install --file=Brewfile
       ./scripts/fetch-zig.sh && export PATH="$PWD/.zig-toolchain:$PATH"
       ./scripts/fetch-llama.sh && ./scripts/build-mlx.sh
       .zig-toolchain/zig build -Doptimize=ReleaseFast
       .zig-toolchain/zig build kit -Dversion=26.11.3
       export MLX_SERVE_KIT=$PWD/zig-out/kit/mlx-serve-plugin-kit-26.11.3

   If you do step 4 in a new shell, run the `export MLX_SERVE_KIT=...` line there first.

   Since the port, our move of the plugin to mlx-serve main 66aad949, we have built, tested and served it only there.
   scripts/prepare.sh fetches bc6c5f2c itself, for the plugin's copy of mlx-serve's Ling model.

   - The kit must have the span fields (ddalcu/mlx-serve#1126) and the last-row flag (ddalcu/mlx-serve#1133). Without
     them the plugin does not build (config.zig `kitGap` and the comptime check that calls it, at fb966cc). The kit at
     66aad949 has both, and also ddalcu/mlx-serve#1131's interfaces.
   - The branch contains CommunityServe main cfc3704, so it builds against those interfaces.
   - The span fields are `Forward.prompt_spans`, `Forward.ssm_checkpoints` and `CacheOps.checkpoint`. They carry a
     prompt's spans and its checkpoints, the KDA layers' states at prompt positions that mlx-serve's prefix cache keeps.
   - The last-row flag is `Caps.prompt_last_row`. It lets mlx-serve ask a prompt forward pass for its last row of logits
     only.
   - The 26.11.1 and 26.11.2 releases came before the span fields, so their kits do not build this plugin.
   - Without `MLX_SERVE_KIT`, scripts/setup.sh looks for the 26.11.3 release kit, finds none, and stops
     (scripts/fetch-kit.sh at fb966cc). With `MLX_SERVE_KIT_FROM_SOURCE=1`, it builds a kit from the repo's
     MLX_SERVE_REF, bc6c5f2c. That kit has no span fields, so config.zig stops the build.
   - `KIT_VERSION` names the kit 26.11.3, which mlx-serve has not released. Step 3's `-Dversion=26.11.3` names the kit
     the same way. We built our own kit at 66aad949 without it. So its folder name and its build.zig.zon say 26.11.1,
     mlx-serve's default at that commit (app/Info.plist). The version only names the kit (mlx-serve's build/kit.zig),
     and the server reports 26.11.1 too.
   - With another server, three checks can refuse by name ("Logs"). mlx-serve checks the plugin's MLX, and the plugin
     checks the host table at load and the span fields at each prompt.

   </details>

4. Build and install the plugin (block below; not yet possible, "Requirements"). Use a new, empty folder as
   `<test home>`. Use the same HOME in steps 5, 6 and 7. Your own `~/.mlx-serve` then stays unchanged.

   <details>
   <summary>Step 4's commands, and what they do</summary>

   `<number>` is the pull request's number. No pull request is open yet ("Requirements"):

       cd <work folder>
       git clone https://github.com/davidtai/turboexl3 <turboexl3 clone>
       git -C <turboexl3 clone> checkout 4bf2df8
       git clone https://github.com/CommunityServe/plugins && cd plugins
       git fetch origin pull/<number>/head:turboexl3_ling && git checkout turboexl3_ling
       scripts/setup.sh
       cd plugins/turboexl3_ling
       bash scripts/prepare.sh
       ../../.zig-toolchain/zig build --fetch
       ../../.zig-toolchain/zig build test --fork=<turboexl3 clone>
       ../../.zig-toolchain/zig build -Doptimize=ReleaseFast --fork=<turboexl3 clone>
       HOME=<test home> <server> plugin install zig-out/lib/libturboexl3_ling.dylib --yes

   - `cd <work folder>` keeps the clones out of step 3's mlx-serve clone.
   - `git fetch origin pull/<number>/head:...` checks out this plugin's pull request by its number, until it merges.
   - The plugin loads only in an mlx-serve built with its kit's MLX. So step 4 installs it with step 3's `<server>`, not
     an mlx-serve on your PATH. Restart the server after each reinstall.
   - scripts/setup.sh copies the kit that `MLX_SERVE_KIT` names into `.kit/kit`. It keeps any kit labelled 26.11.3,
     so if you run step 4 again with another kit, run `rm -rf .kit` before scripts/setup.sh.
   - `--fork` makes Zig build against your turboexl3 clone, not the pin in build.zig.zon.
   - scripts/install.sh uses the mlx-serve on your PATH and builds without `--fork`, so it does not fit this plugin yet.
   - `zig build test` runs the plugin's MLX-free tests, which need no GPU, MLX or pack.

   </details>

5. Turn plugins on under that HOME:

       mkdir -p <test home>/.mlx-serve
       echo '{"disabled": [], "enabled": true}' > <test home>/.mlx-serve/plugins.json

6. Set the GPU wired limit to 112 GiB. Then serve with the same `<server>` and HOME:

       sudo sysctl iogpu.wired_limit_mb=114688
       HOME=<test home> <server> --serve --model <pack> --ctx-size 32768 --prefill-chunk 4096 --mtp-max-depth 3

   Run the `sysctl` line again after each reboot, which resets the limit to 96 GiB (calculated). Keep
   `--ctx-size 32768`. Keep `--prefill-chunk 4096`. The layer-major pass still uses these chunks, and their width can
   change the decoded text (block below). Leave `--kv-quant` off. Set no `kv_quant` in model-settings.json or in a
   request ("Safety limits").

   <details>
   <summary>What step 6's options do</summary>

   Every speed and memory run in this README and in EVIDENCE.md used step 6's options, except the early build's runs of
   2026-11-06.

   - `--ctx-size 32768` is the context that the bill plans for. The plugin accepts a larger value, but its bill still
     holds 32,768 tokens of KV at load. We have not measured decode past 32,768 tokens ("Safety limits").
   - `--kv-quant` is off by default. mlx-serve never tells this plugin a request's KV scheme, so the plugin cannot
     refuse a 4- or 8-bit KV. It logs one `KV_SCHEME_DENSE_COPIES` warning instead (read from the code). The plugin's
     MLA layers would then read a dense copy of the KV that no bill term holds. A 4- or 8-bit KV is also lossy, so the
     decoded text can differ from our runs (not measured). The code is in
     [EVIDENCE.md, "The quantized KV"](EVIDENCE.md#the-quantized-kv).
   - `--prefill-chunk 4096` fixes the chunk width. Ling's prompt pass gives different bits at different widths.
   - `--mtp-max-depth 3` lets the MTP head draft up to 3 tokens in each round. The plugin's default is 1. "How it works"
     says why drafting does not change the greedy text.
   - Leave the plugin's one option, `TURBOEXL3_LING_LAYER_MAJOR`, at 1, its default, which keeps the layer-major pass
     on. Only our decode-only parity test sets it to 0.
   - [EVIDENCE.md, "Smaller gaps"](EVIDENCE.md#smaller-gaps) says what happens without `--ctx-size` or
     `--prefill-chunk`.
   - Ling's thinking is on by default. To turn it off for one chat request, send `"enable_thinking": false`.

   </details>

7. Check that the log has `[plugin] turboexl3_ling serves bailing_hybrid` at load (block below). If the plugin refuses
   the pack, "Logs" gives the cause.

   <details>
   <summary>A healthy load of the served dylib</summary>

   The server prints its log and writes it to `<test home>/.mlx-serve/logs/mlx-serve-<port>.log`. Once the plugin serves
   the pack, `curl -s http://127.0.0.1:<port>/v1/models` shows `"plugin":"turboexl3_ling"`.

   The plugin's lines below are from the served dylib, 4856f028, in the first run of its 32K check, a 31,652-token
   prompt. mlx-serve's own lines are from our refusal test's third run, at 165d8ca (dylib 090d2b0f), with
   `--port 11391`. Both loads served the 2.60bpw pack with step 6's command on an M5 Max with 128 GiB and a 112 GiB GPU
   wired limit. Each group is in the order it prints, with other lines left out and the HOME path shortened.

   4856f028's plugin lines at load:

       [turboexl3_ling] [bank-bill] the plugin holds 14734 MiB (MLX live 14734, cached 0); the box reads 33334 MiB in use
       [turboexl3_ling] LM_DECODE_ROWS: the decode rows take back the layer-major pass's 4797 MiB
       [turboexl3_ling] [bank-bill] room 84733 MiB, in use 14734 MiB: prompt 69998 MiB, decode 76329 MiB for the slot rows
       [turboexl3_ling] streamed experts: 214 prompt / 236 decode slot rows per layer of 512 (336427008 B a row of every layer; ...)
       [turboexl3_ling] prompt log-probabilities: whole prompts up to 3316 tokens (the engine's step 32 B for each of a row's 157184 vocabulary entries, KV 297216 B a token, in the plan's room); a longer one is refused by name

   4856f028's RQ lines for the 31,652-token request:

       [turboexl3_ling] RQ engaged: slot=1 P=31652 capacity=32000 rows=214
       [turboexl3_ling] RQ prompt: slot=1 P=31652 r=0 capacity=32000 head=31744 KV=9873 MiB O=269 MiB rows 214 -> 214 free=0.8 ms alloc=0.0 ms
       [turboexl3_ling] RQ over plan: slot=1 P=31652: with O at 269 MiB the plan holds 211 rows, under the base 214, so the bank stays at its base
       [turboexl3_ling] RQ decode: slot=1 P=31652 r=0 capacity=40192 head=47872 KV=12404 MiB O=1730 MiB rows 214 -> 221 free=25.5 ms alloc=55.9 ms

   mlx-serve's lines at 165d8ca:

       Logging to <HOME>/.mlx-serve/logs/mlx-serve-11391.log (rotates at 32 MB)
       [plugins] loaded turboexl3_ling (model, sha256 090d2b0fa12f) from <HOME>/.mlx-serve/plugins/turboexl3_ling.dylib
       [plugin] turboexl3_ling serves bailing_hybrid (72 layers)
       [mtp] draft-only lm_head requantized to 3-bit/gs64
       [mtp] draft rerank: coarse 3-bit/gs64 head built (157184 rows); greedy drafts re-rank through the trunk head
       Hot prefix cache: ENABLED (capacity=32, mem-cap=0.0 MB) [hybrid: SSM checkpoints]
       Server listening on http://127.0.0.1:11391
       [admission] needed=0 MB available=24450 MB reclaimable=0 MB width=4096 verdict=admit

   The served dylib also prints EMBED_ROWS' load line ("Logs"), which the 32K check's report does not keep.

   - The `streamed experts:` line gives the bank's opening slot rows, 214 for the prompt pass and 236 for decode. RQ
     then plans each request's own decode rows ("Memory").
   - The room, 84,733 MiB, is the GPU limit minus the bill and a margin. Other apps do not change it, but they take the
     Mac's free RAM.
   - The `LM_DECODE_ROWS` line prints unless `TURBOEXL3_LING_LAYER_MAJOR=0`.
   - The load's warm-up also prints `RQ engaged` and `RQ prompt` lines for slot 4294967295. That is normal.
   - At a 31,652-token prompt, `RQ over plan` is normal: RQ keeps the bank's 214-row base ("Memory").
   - The two `[mtp]` lines are mlx-serve's 3-bit copies of the plugin's 4-bit draft readout. mlx-serve drops the first
     copy when it builds the second, the rerank head
     ([EVIDENCE.md, "The optimizations, part by part"](EVIDENCE.md#the-optimizations-part-by-part)). In the second
     line, "coarse 3-bit/gs64 head" is the rerank head, and "the trunk head" is the 4-bit draft readout.
   - `mem-cap=0.0 MB` means that mlx-serve keeps no prefix in RAM between requests, because the bank uses that memory.
     With step 6's command mlx-serve has no SSD tier either, so it keeps no prefix at all ("What it serves"). With
     `--prefix-cache-disk <size>`, each kept prefix goes through the decline spill
     ([EVIDENCE.md, "Tests"](EVIDENCE.md#tests)). We have not measured that setting.
   - The `Server listening on` line gives the port. The default port is 11234.
   - The first request's `[admission] ... width=` line shows 4096. A smaller width still serves correctly, but the text
     can differ from ours. The line prints at the first request after a load. After that it prints only for a request
     that needs over 90% of the memory available (mlx-serve 66aad949 src/engines/mlx/admission/admit.zig
     `admissionLogLevel`).
   - After your first chat request, its `served paths ran (request, prompt)` line names LAYER_MAJOR and
     LAST_ROW_LOGITS.

   </details>

## Measured

On 2026-11-11, our standard test ran the served dylib, 4856f028, 4 times. Each run sent an 18,380-token coding prompt to
a new server on our M5 Max and decoded 1,024 tokens greedily. We measured all numbers.

| Dylib | Decode tok/s | Prefill tok/s | TTFT s | Wall s | MTP acceptance | Peak footprint GB |
|---|---|---|---|---|---|---|
| 4856f028, mean of 4 | 25.83 | 700.0 | 26.27 | 66.02 | 1,810 / 2,967 = 61.0% | 108.5 |

All 4 runs decoded the same text, 09571f6a. Peak footprint, the server's peak memory (`time -l`, in GB of 10^9 bytes),
is larger than MLX's peak ("Memory"). The served dylib also passed our build and test gates
([EVIDENCE.md, "Tests"](EVIDENCE.md#tests)).

On 2026-11-10 and 2026-11-11, our context ladder ran one coding task at six prompt sizes on 01ca9769, before RQ. RQ
changes decode's slot rows with the prompt's size, so the served dylib's decode can differ from this table. The last row
is the served dylib's own 32K check, on 2026-11-11. Each row gives the greedy means of 4 runs of 1,024 decoded tokens,
all measured.

| Prompt tokens | Decode tok/s | Prefill tok/s | TTFT s | Wall s | MTP acceptance | Peak footprint GB |
|---|---|---|---|---|---|---|
| 1,120 | 31.58 | 114.0 | 9.83 | 42.35 | 70.0% | 99.1 |
| 2,042 | 32.30 | 190.9 | 10.71 | 42.50 | 69.9% | 99.4 |
| 4,094 | 28.78 | 364.1 | 11.25 | 46.98 | 69.6% | 100.5 |
| 8,185 | 26.65 | 659.1 | 12.43 | 50.95 | 70.4% | 103.4 |
| 16,377 | 26.53 | 708.7 | 23.12 | 61.81 | 65.8% | 109.3 |
| 31,652 | 24.15 | 614.3 | 51.55 | 94.03 | 67.5% | 116.0 |
| 31,652 on 4856f028 | 23.68 | 615.4 | 51.46 | 94.82 | 67.3% | 114.7 |

Greedy decoding gave the same text in all 4 runs at each size, and 4856f028 decoded 01ca9769's text. At 31,652 tokens,
4856f028 holds 221 decode slot rows, not 236 ("Memory"). We infer that this causes its 2.0% slower decode (not
A/B-tested).

See [EVIDENCE.md, "Measured"](EVIDENCE.md#measured).

## Safety limits

The served dylib refuses two kinds of request before any work. The log names the cause, and the client gets HTTP 500.
A prompt of more than 32,000 tokens would grow its KV to 40,192 tokens inside the prompt pass, past the bill's
32,768-token KV term. Prompt log-probabilities need about 165 GB at 32,768 tokens (calculated), so the plugin serves
them only for short prompts.

| Request or setting | What the served dylib does |
|---|---|
| A prompt of more than 32,000 tokens | It refuses the request. It refused a 32,198-token prompt in 0.018 s (measured on 89be02b1) |
| Prompt log-probabilities (`echo` with `logprobs`) for a whole prompt over 3,316 tokens | It refuses the request |
| A drafter model, from `--drafter <dir>` or the pack's folder | mlx-serve declines the drafter at load, names it in the log and serves without it (read from the code). The MTP head is not a drafter model, and it still drafts |
| A 4- or 8-bit KV, from `--kv-quant`, model-settings.json or a request's `kv_quant` | No refusal. After each load, its first use logs one `KV_SCHEME_DENSE_COPIES` warning (read from the code). Do not use one ("Memory") |
| A `--ctx-size` above 32,768 | mlx-serve takes it as given, but we have not measured decode past 32,768 tokens. Keep `--ctx-size 32768` |
| Two long requests at once | No refusal, and not tested. Send one long request at a time. The length limit is not measured ("Memory") |

See [EVIDENCE.md, "RQ"](EVIDENCE.md#rq-the-banks-rows-for-each-request) and
["Prompt log-probabilities"](EVIDENCE.md#prompt-log-probabilities) for the refusals, and
["EMBED_ROWS"](EVIDENCE.md#embed_rows) for the drafter.

## Gaps

| Gap | What a fix needs |
|---|---|
| The three unpublished parts in "Requirements" | Publish them before a merge, and pin turboexl3 4bf2df8 |
| No refusal of a 4- or 8-bit KV | An mlx-serve change, not yet made, that refuses the scheme by name at load |
| A `--ctx-size` above 32,768 is not refused | A refusal at load, or a measured run past 32,768 tokens |
| We have not tested two long requests at once | A test of two long requests at once |
| The prompt pass stays at the base near 32,000 tokens | A streamer that can shrink the prompt pass's rows under the base |
| No prefix kept between requests | A measured `--prefix-cache-disk` setting, or RAM for the prefix cache |

See [EVIDENCE.md, "Gaps for a merge"](EVIDENCE.md#gaps-for-a-merge).

## Logs

Each refusal names its cause in the log (block below).

<details>
<summary>Every refusal and its fix, and the other log lines</summary>

Each request prints two run-path lines. `served paths ran (request, prompt)` names the paths of its prompt pass, and
`(request, decode)` repeats them and adds those of its first decode round.

The first table gives the common refusals and their fixes. The second gives every refusal by the step that prints it.

| Line or reason | Cause | Do this |
|---|---|---|
| no `[plugin] turboexl3_ling serves` line | plugins are off, or installed under another HOME | do step 5 under the serving HOME, then restart |
| `Exl3PackNotRepacked` | the folder is the download | run step 2 |
| `StreamedRowsDoNotFit` | no bank slot fits in the GPU memory | raise the GPU wired limit |
| `HostTooOld`, `this host names no chunk-loop spans`, or `plugin was built against another MLX` | the server is not the kit's mlx-serve | serve with step 3's server |
| `a prompt whose KV passes 32000 tokens` | a prompt of more than 32,000 tokens | send a shorter prompt |
| `prompt log-probabilities for a <n>-row prompt forward ...` | prompt log-probabilities over the load line's limit | send a shorter prompt |

At parse, the plugin prints `[turboexl3_ling] the pack is refused: <reason>`. mlx-serve then prints
`[plugin] bailing_hybrid: turboexl3_ling: <reason>; no other path serves it` (mlx-serve 66aad949
src/engines/mlx/plugin/config.zig `bindServed`). At load, mlx-serve prints `[plugin] turboexl3_ling: load failed:
<message>`. At a prompt, it prints `[plugin] turboexl3_ling: forward failed: <reason>`, and the client gets HTTP 500
"generation failed". A refusal not listed here is an internal error. Report it to the maintainers with the server log.

| When | Reason or message | Cause |
|---|---|---|
| Start | `plugin was built against another MLX` (mlx-serve's line) | the server's MLX is not the kit's: serve with the server from step 3 |
| Start | `changed since it was installed (hash differs from its receipt)` (mlx-serve's line) | the dylib changed after its install: install it again, then restart the server |
| Parse | `Exl3PackNotRepacked` | the folder has no `experts.bin` or expert manifests: it is the download, so run step 2 |
| Parse | `Exl3CodebookUnsupported`, `Exl3QuantConfigNotAnObject`, `LingClampListLength`, `LingClampValue`, `LingLayerCount` | a config that this plugin does not serve: convert turboderp's 2.60bpw or 2.10bpw branch with step 2 |
| Parse | `turboexl3_ling: no Thunderbolt pair (a streamed bank serves one Mac)`, with no line from the plugin | a two-Mac serve |
| Load | `turboexl3_ling: HostTooOld`, or `the server's host table is older than this plugin's kit` | the server is older than the kit |
| Load | `turboexl3_ling: an M5-class GPU (NAX) is needed (turboexl3's prefill kernels run on its matrix units)` | another GPU |
| Load | `StreamedRowsDoNotFit`, after the plugin's `the streamed bank does not open: StreamedRowsDoNotFit` | no bank slot fits in the GPU memory |
| Load | the plugin's `the pack's shards do not open: ...` or `the MTP head does not load: ...` | a damaged or partial pack: run step 2 again |
| Load | `LingNoRoutedLayers`, `LingDenseLayerPastFirst`, or `<name>.mul1 is 0x...` | a pack that is not this model's: convert turboderp's 2.60bpw or 2.10bpw branch with step 2 |
| Load | `LingEmbedRowsTiedHead`, `LingEmbedRowsMissing`, `LingEmbedRowsNotBf16`, `LingEmbedRowsShape`, or `turboexl3_ling: EMBED_ROWS needs the pack's own EXL3 head`, after a `[turboexl3_ling] EMBED_ROWS` line | a pack whose output head or embedding table this plugin does not serve ([EVIDENCE.md, "EMBED_ROWS"](EVIDENCE.md#embed_rows)): convert turboderp's 2.60bpw or 2.10bpw branch with step 2 |
| Load | `LingEmbedRowsIndex`, `LingEmbedRowsOpen`, `LingEmbedRowsHeader` or `LingEmbedRowsTruncated`, after a `[turboexl3_ling] EMBED_ROWS` line | a damaged or partial pack: run step 2 again |
| Load | `the drafter at <dir> needs the plugin's trunk_weight and layer capture: served without it` (mlx-serve's line) | a drafter, given with `--drafter` or found in the pack's folder, which the plugin does not serve ("Safety limits") |
| Prompt | `this host names no chunk-loop spans ...` | the kit has the span fields, and the server does not |
| Prompt or decode | `LingEmbedRowsRead` or `LingEmbedRowsTokenPastTable` | a read of the table's shard failed (the file was cut short, or its drive failed or was disconnected), or a token id is outside the table |
| Prompt | `turboexl3_ling: a prompt whose KV passes 32000 tokens: a <n>-row prompt forward from position <p> grows mlx-serve's KV to <c> tokens ...` | a prompt whose KV would pass 32,000 tokens ("Safety limits") |
| Prompt | `turboexl3_ling: prompt log-probabilities for a <n>-row prompt forward from position <p> need <m> MiB, past the <r> MiB this plan leaves ...` | prompt log-probabilities past the limit ("Safety limits") |
| Prompt | `turboexl3_ling: one slot's prompt at a time`, or `... a layer-major prompt carries no media or captures` | a forward pass that carries more than one slot's prompt, images, or a hidden-state capture (`capture_layers`) (plugin.zig `spanMajorOr`, or `promptCalls` without spans, at fb966cc). mlx-serve interleaves a second request's forward passes with the first's ("What it serves"), and we have not seen the first case. Send text only |
| Prompt or decode | `turboexl3_ling: the bank has no RQ state`, `LingBankWithoutRq`, or `a head forward before the bank opened (no RQ hook)` | a forward pass before the bank opened, which a served load never makes: report it to the maintainers |

The `(prompt)` and `(decode)` lines name every path run since load. A request starts at every prompt forward pass that
does not continue the previous one (src/ling_ran.zig `Boundary`). A pass that starts where the previous one ended, with
no decode between them, continues it. So the warm-up at load makes requests of its own, and its paths never appear in a
real request's line. The served dylib carries 41 path names, EMBED_ROWS, KDA_GATE_EPI, KDA_NORM_PREP and RQ among them,
and 01ca9769 carried 40. This port removed HEAD_ONE_BUFFER, which only a kit without the last-row flag ran.
[EVIDENCE.md, "Logs: the run-path lines"](EVIDENCE.md#logs-the-run-path-lines) says which paths a request names.

RQ prints a line at each re-plan of a request. `RQ engaged` comes first, then `RQ prompt` before the prompt pass and `RQ
decode` before decode. `RQ trim` shows a new plan when decode passes the planned KV. `RQ over plan` shows that a plan
holds fewer slot rows than the base, so the bank keeps the base. `RQ interleaved` shows that a request's forward passes
resumed after another request's, once a request. After each load, the first use of a 4- or 8-bit KV prints one
`KV_SCHEME_DENSE_COPIES: ...` warning.

EMBED_ROWS also prints two lines of its own. At load, `[turboexl3_ling] EMBED_ROWS: the embedding table ... is not
resident` shows that it runs. `... the embedding table the trunk lends was evaluated` means that MLX holds the table
outside the bill. If it prints, report it to the maintainers with the server log. The first call that takes
KDA_GATE_EPI's or KDA_NORM_PREP's launch prints `[turboexl3_ling] <name> engaged: S=<rows>`.

The log lines say "host" and "engine" for mlx-serve. The load's warm-up also prints `LAYER_MAJOR in calls of 16384
rows: the engine names no spans, so not the chunk loop's bytes`. That line is normal: the warm-up's short prompts come
with no spans. A request's prompt line prints when its first decode round starts, and its decode line after that
round. A `(prompt)` or `(decode)` line prints each time a path runs for the first time since load.

Other lines show the names in capitals that are not `Path` entries.

- turboexl3's `[turboexl3] ... engaged` lines show BANKED_T4, MLA_LATENT_STAGE, ROUTER_ROWS and ONE_ROW_TABLE.
- `[turboexl3_ling] PF_DFIN calls: ...` counts the routed calls since load by how each one ended: in the fused
  finish, or in separate launches. It prints after each prompt forward pass, every 64 decode steps, and when the next
  prompt starts.
- A load line shows LM_DECODE_ROWS. KDA_GATE_FUSED, KDA_PREFILL_FUSED and KDA_PREWORK print `[kda] ... engaged`
  lines. VERIFY_MLA_ROWS prints `[turboexl3_ling] exact verify MLA rows: ...` only when a verify falls back to the
  expanded key. On the served rope-once cache, its rows print `[turboexl3_ling] MLA_ROPE_ONCE: ... decode or verify
  rows ...` instead. KDA_DECODE_FUSED and VERIFY_KDA_ROWS print a line only when their launch runs without the epilogue.
- `strings zig-out/lib/libturboexl3_ling.dylib | grep 'served path names'` lists the `Path` names, 41 at fb966cc.
- `[turboexl3_ling] [mem] ...` gives MLX's memory at the load's steps and when a prompt pass ends.
- The expert streamer prints `[stream-stats]` lines.

</details>

## Terms

| Term | Meaning here |
|---|---|
| 32K check | our memory check of a dylib: a warm-up and 4 greedy runs at 31,652 prompt tokens, each on a new server ("Memory") |
| A/B, arm | a comparison of two builds on the same workload; each build is an arm |
| Bank, base | the bank is the expert cache: the GPU memory slots that hold the routed experts each layer has read from the SSD. Its base is the 214 slot rows that the prompt pass holds |
| Bill | the memory that the plugin asks mlx-serve to hold back from the bank. At load, in the prompt pass, it holds the KV, `prompt_work`, the layer-major term, the reserve and MEM_REFIT; in decode, the KV, decode's working set and the reserve. RQ then bills each request's own KV |
| bpw | bits per weight; the 2.60bpw branch's config records 2.59 |
| Ceiling | the plan's ceiling: the most memory that the prompt-pass bill lets MLX hold, 101,856 MiB ("Memory") |
| Chunk loop | mlx-serve's prompt pass in forward passes of `--prefill-chunk` tokens |
| Context ladder | our runs of one coding task at six prompt sizes, 1,120 to 31,652 tokens, greedy and with the card's settings, on 01ca9769 ("Measured") |
| Download | a branch of turboderp/Ling-3.1-flash-exl3 as Hugging Face serves it |
| Dylib `<hash>` | a build of the plugin, named by the first 8 hex digits of its dylib's sha256 (the load line prints 12); not a commit id |
| Exact verify | a verify pass that computes each row as a one-row decode does (`Caps.exact_drafted`) |
| EXL3, exllamav3 | exllamav3 is turboderp's quantization library; EXL3 is its trellis format |
| Forward pass | one run of a set of tokens through the model |
| Gates | our build and test run of a dylib on the Mac before we serve it: the MLX-free tests, test_reach.py, the ReleaseFast build, the GPU byte tests and the restore check |
| Host, engine | mlx-serve, as the plugin's log lines name it |
| KDA, MLA | Ling's two attention kinds: Kimi Delta Attention (linear) and multi-head latent attention (9 layers) |
| Kit | the mlx-serve plugin kit the plugin compiles against (`KIT_VERSION`, or the folder `MLX_SERVE_KIT` names) |
| KV | the attention cache that grows with each token: here the MLA layers' rows and the MTP head's latent, 297,216 B for each token |
| Last-row flag | `Caps.prompt_last_row` (ddalcu/mlx-serve#1133): mlx-serve asks a prompt for its last row of logits only |
| Layer-major pass | the plugin's prompt pass: the whole prompt, one layer at a time, over the chunk loop's spans |
| MEM_REFIT | the bill's term fitted to measured prompt-pass peaks: a pass part and the server's own 544 MiB |
| MLX-free tests | the plugin's `zig build test`: no GPU, no MLX and no pack |
| MoE | mixture of experts: a block whose router sends each token to a few of its expert FFNs |
| MTP head | the pack's NextN layer, which drafts tokens (multi-token prediction) for the verify pass |
| NAX | the matrix units of M5-class GPUs |
| Number kinds | measured in our runs; calculated from the code; set, a setting or the model's size; or fitted to measured peaks (MEM_REFIT). "Inferred" marks a cause that no record states. EVIDENCE.md's "Receipts" gives each number's record |
| O | the other MLX memory that RQ measures at a re-plan, beside the request's KV and the plugin's own memory |
| Outside plugin | a dylib that mlx-serve loads at run time, built from this repo |
| Pack | the converter's output, the folder that mlx-serve serves: `experts.bin`, its manifests and the other tensors |
| Parity test | a test that a change keeps the plugin within set margins of its distance from exllamav3's own logits |
| Peak footprint | the server process's peak memory (`time -l`), which is larger than MLX's peak |
| Plan | the bill's layout of the GPU memory; RQ makes one for each request |
| Port | our move of the plugin's kit and server to mlx-serve main 66aad949 |
| Prefix cache | mlx-serve's cache of earlier prompts' state, in RAM (the RAM tier) and on the SSD (the SSD tier) |
| Prompt log-probabilities | an `echo` request with `logprobs`: the log-probability of every prompt token |
| Prompt pass | the forward passes over a request's prompt (prefill) |
| Read window, staging | the streamer's scratch: a read window holds one batch of expert reads, 48 records of MLX memory (the prompt pass holds 5, decode 1); the staging is the read pool's host memory |
| Refusal test | our served test of the prompt log-probability refusal, run twice on 2026-11-09 and once on 2026-11-10 |
| Rerank head | mlx-serve's resident 3-bit copy of the 4-bit draft readout; the log calls it the "coarse 3-bit/gs64 head" |
| Reserve | the bill's 4 GiB for the rest of the Mac (`free_ram`) |
| Room | the GPU memory that mlx-serve gives the plugin for the trunk and the bank: the 112 GiB limit minus the 8 GiB margin and the prompt-pass bill, 84,733 MiB ("Memory") |
| Routed experts | an MoE block's 512 expert FFNs, of which the router picks 8 for each token |
| Row | one token position of a forward pass; the bank's slot rows are another thing ("Slot row") |
| RQ | the plugin's memory plan for each request: it bills the request's KV, sets its decode slot rows and keeps the prompt pass at the base ("Memory") |
| Slot, slot row | a slot is one expert's place in the bank, for one layer; a slot row is one slot in each of the 68 routed layers, 336,427,008 B |
| Span, span fields | a span is one chunk that the chunk loop runs, or would run, as mlx-serve names it to the plugin; the span fields carry them (ddalcu/mlx-serve#1126) |
| SSD tier | mlx-serve's prefix cache on disk (`--prefix-cache-disk`) |
| Standard test | our speed test, an A/B test: glm53-eval's coding workload in 8 runs, A B B A then B A A B, 4 for each build ("Measured") |
| Streamer | the expert streamer, a patched copy of mlx-serve's: it reads routed experts from the SSD into the bank |
| Trunk | every weight except the routed experts and the MTP head; it includes the embedding table and the output head |
| TTFT | time to the first token, measured by the client |
| turboexl3 | the library of Metal kernels for EXL3 weights that runs the plugin's EXL3 math |
| Units | GB is 10^9 bytes, and GiB and MiB are 2^30 and 2^20 bytes |
| Verify pass | the trunk's check of a round of drafted tokens |
