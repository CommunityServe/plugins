# turboexl3_ling

This README describes turboexl3_ling 0.1.0, written on 2026-11-12 by its maintainers (davidtai; "we"). Times are in UTC.
It describes dylib 87046444, the build that we serve on one Mac ("Build", at the end).
[EVIDENCE.md](EVIDENCE.md) gives the record of each number.

## Summary

turboexl3_ling is an outside plugin for mlx-serve: a dylib that mlx-serve loads at start, built from this repo. It lets
mlx-serve run inclusionAI's Ling-3.1-flash in EXL3. EXL3 is a quantized weight format, made by exllamav3, the library of
the developer turboderp. A converter repacks turboderp's 2.60bpw (bits per weight) weights into a pack, the folder that
mlx-serve serves. The pack is 188.2 GB, more than the Mac's memory, so the plugin streams the routed experts from the SSD.
On an M5 Max with 128 GiB, the served build averaged 700.9 tok/s prefill and 25.90 tok/s decode on an 18,380-token coding
prompt ("Measured"). You set the context, the longest prompt plus its output, with `--ctx-size`. We measured prompts
from 1,058 to 134,615 tokens: decode fell from 27.8 to 8.6 tok/s as the prompt grew. The plugin plans its memory for
each request, and it refuses with a named error a request that its plan cannot hold ("Refusals"). We have not tested
two long requests at once.

## What it serves

The plugin serves text only, on one Mac with an M5-class GPU. The model's window is 262,144 tokens. We measured prompts
up to 134,615 tokens, at `--ctx-size 150999`. A larger `--ctx-size` takes more memory for the KV and leaves less memory
for the expert cache in the prompt pass ("Memory"). We have not measured a larger one. If the context leaves no memory
for the expert cache, the load refuses the pack with a named error.

The plugin has no batched decode, so mlx-serve interleaves a second request's forward passes with the first's. With the
serve command in step 6, mlx-serve's prefix cache gets no memory (`mem-cap=0.0 MB` in the log), so mlx-serve keeps no prefix. Each
chat turn then runs its whole prompt pass (prefill) again.

We have served this build only on the 2.60bpw pack. It also accepts the 2.10bpw pack. In our parity test, that pack's output
differed more from exllamav3's than the 2.60bpw pack's output did ([EVIDENCE.md, "Parity"](EVIDENCE.md#parity)).

## Requirements

We have served the plugin only on an M5 Max with 128 GiB, on macOS 26.6. Step 5's 112 GiB GPU wired limit is for that
Mac. We have not tested other memory sizes. The plugin needs an M5-class GPU, because turboexl3's prompt kernels run on the M5 GPU's matrix units (NAX). On a Mac with
less memory, which we have not measured, the load refuses the pack with a named error (`LingContextLeavesNoRoom` or
`StreamedRowsDoNotFit`) if no slot row fits.

You also need:

- about 377 GB of disk: 188.4 GB for the download and 188.2 GB for the pack. The pack hard-links the download's small
  files, so both must be on one volume;
- network access, git, the Hugging Face CLI (`hf`) and Python 3;
- Homebrew, and Xcode with its Metal toolchain (`xcodebuild -downloadComponent MetalToolchain`), to build mlx-serve.

Close other large apps, and run one large server at a time ([AGENTS.md](../../AGENTS.md) rule 11). The Mac's free RAM
fell to about 0.06 GB in our runs at short prompts, while the bank was at its largest. Memory pressure stayed normal.

The plugin needs mlx-serve's span fields (ddalcu/mlx-serve#1126) and last-row flag (ddalcu/mlx-serve#1133) in its kit.
No mlx-serve release has the kit that `KIT_VERSION` names (26.11.3) yet. So step 3 builds mlx-serve 9a173773 and its kit
from source, and you point `MLX_SERVE_KIT` at that kit.

## How it works

The trunk is every weight except the routed experts and the MTP (multi-token prediction) head. The plugin keeps the
trunk in GPU memory, except its embedding table, whose rows the CPU reads from the pack.

Ling-3.1-flash has 72 layers with two kinds of attention: KDA (Kimi Delta Attention, a linear attention) and MLA
(multi-head latent attention, 9 layers). From layer 4 on, each layer ends in a mixture-of-experts (MoE) block. Its router
picks 8 of 512 routed experts for each token.

The plugin's streamer reads the routed experts of layers 4 to 71 from the SSD into a GPU cache, the bank. mlx-serve runs
the HTTP server, the tokenizer, sampling, the prefix cache and the MTP draft loop. The plugin runs each forward pass.
turboexl3 (github.com/davidtai/turboexl3), a library of Metal kernels for EXL3 weights, runs every EXL3 projection and
routed expert call.

The MTP head drafts tokens, and the trunk checks a round of them in one verify pass. The plugin's exact verify computes
each verify row as a one-token decode does. So in greedy decoding, MTP drafts do not change the text.

mlx-serve's chunk loop runs a prompt in forward passes of `--prefill-chunk` tokens, the chunks. This plugin runs the
prompt one layer at a time over all its tokens. We call this the layer-major pass. A routed layer runs the prompt in
calls of up to about 16,384 tokens. It reads each expert it needs once for each call, not once for each 4,096-token
chunk.

A prompt of more than 32,768 tokens runs in segments. Each segment goes through every layer before the next one starts.
A segment ends at a routed call's end and holds at most 32,768 tokens. Only a last group of under 8,192 tokens can make
it larger: that group joins the segment before it. The segments cap the pass's working memory: 33,591 tokens ran in 2 segments, and 134,615
tokens in 4 (measured). The pass gives the same bits in segments as in one piece (a GPU byte test at 52,000 tokens,
[EVIDENCE.md, "Gates"](EVIDENCE.md#gates)).

Most of the plugin's optimizations change mlx-serve's Ling model and expert streamer, and a plugin cannot change
mlx-serve itself. So the plugin compiles its own copy of both from mlx-serve bc6c5f2c, with four patches and five
cut-down MLX kernels. `plugins/turboexl3_ling/scripts/prepare.sh` makes the copy at build time.

## Memory

The plugin divides the GPU memory between the trunk, the bank and a bill. The bill is the memory that the plugin
reserves for the KV, the prompt pass and free RAM. The bank cannot use it during the prompt pass. mlx-serve takes its 8 GiB margin and the
bill from the GPU wired limit, 112 GiB on our Mac (step 5). What is left is the room. The trunk, the MTP head and MLX's
other buffers at load hold 14,734 MiB of the room (measured). The bank gets the rest: first the streamer's read buffers, then whole slot rows. A
slot row is one expert slot in each of the 68 routed layers, 320.8 MiB.

At load, the bill holds four terms for the prompt pass (src/bill.zig, plugin.zig `prefillAt`):

- the KV that mlx-serve allocates for the longest whole prompt up to `--ctx-size` (the KV is the attention cache of the
  9 MLA layers and the MTP head, 297,216 B a token);
- W, the most working memory that the layer-major pass needs for any prompt up to the context. A formula fitted to
  measured peaks gives W, and W includes a fixed 2 GiB part, `prompt_work`;
- 4 GiB of free RAM for the rest of the Mac;
- 544 MiB for the server's own memory past MLX's counters.

So a larger `--ctx-size` leaves fewer slot rows for the prompt pass, even for a short prompt, and the prompt pass then
reads more experts from the SSD. Decode's rows come from the request plan below, which bills the request's own KV, so
`--ctx-size` does not set them (read from the code). Our sweep raised the prompt and the context together, so we did not
measure the effect of `--ctx-size` alone.

Decode needs less of the bill than the prompt pass. The part of W that only the prompt pass uses, and the 544 MiB server
term, go back to the bank. So decode has more slot rows than the prompt pass. The load prints the bill and the rows ("Logs"). This table gives the
load's numbers at the 18K test's context and at two of the sweep's. The [memory table in "Measured"](#memory-at-each-size)
has all eight sweep contexts.

| `--ctx-size` | KV billed, tokens | W MiB | Room MiB | Prompt slot rows | Decode slot rows at load, before RQ |
|---|---|---|---|---|---|
| 32,768 | 32,768 | 5,993 | 86,574 | 220 | 236 |
| 49,975 | 50,176 | 7,177 | 80,455 | 200 | 221 |
| 150,999 | 151,040 | 13,920 | 45,123 | 90 | 132 |

The request plan (RQ) then sets the bank's rows for each request. Before the prompt pass and before decode, RQ plans the
KV that mlx-serve will allocate for that request and gives the bank the rows that are left. If the memory cannot hold a
plan, RQ refuses the request with a named error ("Refusals"). No run of our sweep was refused.

<details>
<summary>How RQ plans a request</summary>

mlx-serve reserves no KV for a whole prompt. It grows each KV by a quarter of its size, by 256 to 8,192 tokens, to a
multiple of 256. RQ replays that growth to find the KV that mlx-serve will allocate. It also counts the other MLX memory
in use at that time.

- Before the prompt pass, RQ makes the prompt plan. The prompt pass keeps the load's prompt rows, the base.
- Before decode, RQ makes the decode plan. It bills the KV for the prompt and 4,096 more tokens, up to the context, with
  the MTP head's cache and the other MLX memory in use. So it can give decode more or fewer rows than the load did: 258
  rows at 1,058 tokens, where the load gave 247, and 115 at 134,615, where the load gave 132.
- When decode passes the planned KV, RQ trims the decode rows to a new plan. In our sweep, 16 of 64 runs trimmed once.

A plan that leaves fewer rows than the base is an over-plan. RQ refuses a prompt plan under the base before any work. A
decode or trim plan up to 12 rows under the base keeps the base with a warning, because the 4 GiB of free RAM covers 12
rows. A plan more than 12 rows under the base is refused. No run of our sweep made an over-plan.

</details>

We checked the bill at 33,591, 67,310 and 134,615 tokens, with a memory guard that stops the server at critical memory
pressure or 1 GB of new swap. The measured W stayed 2,310 to 3,855 MiB under the billed W, memory pressure stayed
normal, and the guard stopped no run.

See [EVIDENCE.md, "The served build"](EVIDENCE.md#the-served-build-87046444).

## Install and serve

Each `<...>` placeholder that names a file or folder is an absolute path. Start steps 2, 3 and 4 in one work folder,
`<work>`, so that no clone lands inside another.

1. Download the 2.60bpw branch of turboderp's repo:

       hf download turboderp/Ling-3.1-flash-exl3 --revision 2.60bpw --local-dir <download>

2. Convert the download into a pack with the converter's Ling script. The converter repo holds converters for several
   models. Its Ling script copies bytes and checks each record. It does not quantize.

       cd <work> && git clone https://github.com/OpenSourceWTF/qwen38-exl3-mlx-serve-converter
       cd qwen38-exl3-mlx-serve-converter
       python3 scripts/ling_exl3_to_streaming.py plan --src <download> --rev 2.60bpw --plan plan.json
       python3 scripts/ling_exl3_to_streaming.py run --src <download> --out <pack> --plan plan.json
       python3 scripts/ling_exl3_to_streaming.py finalize --src <download> --out <pack> --plan plan.json

   In a good run, `run` prints `ALL_SHARDS_PASS <n>/<n>` last, and `finalize` prints `FINALIZED <pack>: ...`.

3. Build mlx-serve 9a173773 and its plugin kit. `<server>` is then `<mlx-serve clone>/zig-out/bin/mlx-serve`.

       cd <work> && git clone --recurse-submodules https://github.com/ddalcu/mlx-serve && cd mlx-serve
       git checkout 9a173773649218b866089dbe9851b2ba594f34f9 && git submodule update --init
       brew bundle install --file=Brewfile
       ./scripts/fetch-zig.sh && export PATH="$PWD/.zig-toolchain:$PATH"
       ./scripts/fetch-llama.sh && ./scripts/build-mlx.sh
       .zig-toolchain/zig build -Doptimize=ReleaseFast
       .zig-toolchain/zig build kit
       export MLX_SERVE_KIT=$PWD/zig-out/kit/mlx-serve-plugin-kit-26.11.1

   The kit's folder says 26.11.1 because mlx-serve 9a173773 names its version so. Run the `export` line again in any
   new shell before step 4.

4. Build and install the plugin under a new, empty HOME, `<test home>`. Your own `~/.mlx-serve` then stays unchanged.

       cd <work> && git clone https://github.com/CommunityServe/plugins && cd plugins
       # until this plugin's pull request merges:
       git fetch https://github.com/davidtai/plugins add-turboexl3_ling && git checkout FETCH_HEAD
       PATH=<mlx-serve clone>/zig-out/bin:$PATH scripts/install.sh turboexl3_ling --home <test home>

   `scripts/install.sh` builds the plugin from source (`scripts/build.sh`: `prepare.sh`, the MLX-free tests, then the
   ReleaseFast dylib). It installs the dylib with step 3's server and turns plugins on under that HOME. The first build
   downloads turboexl3 at the pinned commit and mlx-serve's source at bc6c5f2c. If `.kit` holds another kit from an
   earlier build, run `rm -rf .kit` first, because `scripts/setup.sh` keeps a staged kit.

5. Set the GPU wired limit to 112 GiB. A reboot resets it, so run it again after each reboot:

       sudo sysctl iogpu.wired_limit_mb=114688

6. Serve with step 3's server and the same HOME:

       HOME=<test home> <server> --serve --model <pack> --ctx-size <context> --prefill-chunk 4096 --mtp-max-depth 3

   Set `<context>` to your longest prompt plus your largest `max_tokens`. Our sweep used the prompt plus 16,384, and our
   18K test used 32,768. Keep `--prefill-chunk 4096` and leave `--kv-quant` off (block below).

7. Check that the log has `[plugin] turboexl3_ling serves bailing_hybrid` at load. If the plugin refuses the pack, "Logs"
   gives the cause. The log's `Server listening on` line gives the port. `curl -s http://127.0.0.1:<port>/v1/models`
   then shows `"plugin":"turboexl3_ling"`.

<details>
<summary>What step 6's options do</summary>

- `--ctx-size` sets the context that the bill and RQ plan for. Without it, mlx-serve sizes the context itself, and the
  plugin caps that context at 32,768 tokens (config.zig).
- `--prefill-chunk 4096` fixes the chunk width. The layer-major pass still runs mlx-serve's spans of that width, and
  Ling's prompt pass gives different bits at different widths, so another width can change the decoded text.
- `--mtp-max-depth 3` lets the MTP head draft up to 3 tokens in each round. "How it works" says why drafting does not
  change the greedy text.
- `--kv-quant` is off by default. mlx-serve never tells this plugin a request's KV scheme, so the plugin cannot refuse a
  4- or 8-bit KV. It logs one `KV_SCHEME_DENSE_COPIES` warning instead. Each MLA layer would then read a dense copy of
  its KV that no bill term holds ([EVIDENCE.md, "The quantized KV"](EVIDENCE.md#the-quantized-kv)).
- Ling's thinking is on by default. To turn it off for one chat request, send `"enable_thinking": false`.
- Every run in "Measured" used these options, with no plugin environment variables.

</details>

## Measured

We measured all numbers in this section on our M5 Max with 128 GiB, with the served dylib 87046444 on the 2.60bpw pack.
Each run used a new server and a wait for the chip to cool to 40 C before each measured request. The sweep ran one
unmeasured warm-up for each size and mode first. Prefill and decode are the server's own rates: prompt tokens over its
prompt time, and completion tokens over its decode time. TTFT (time to first token) and wall time are measured at the
client. MTP acceptance is accepted draft tokens over drafted tokens. Peak footprint is the server's peak memory from
`time -l`, in GB of 10^9 bytes.

### The 18K coding test

On 2026-11-11, our standard speed test sent an 18,380-token coding prompt (glm53-eval's `workloads/coding.json`) and
decoded up to 1,024 tokens greedily, at `--ctx-size 32768`. It is an A/B test: 8 runs in the order A B B A then B A A B.
The served build was B. A was the build before it, f03b37e9, whose bill held a fixed 32,768-token context. The table
gives B's 4 runs by their place in that order, and both builds' means. EVIDENCE.md has A's runs.

| A/B run | Seed | Prefill tok/s | Decode tok/s | TTFT s | Wall s | MTP acceptance | Peak footprint GB |
|---|---|---|---|---|---|---|---|
| 2 | 1234 | 701.7 | 25.80 | 26.21 | 65.96 | 449 / 734 = 61.2% | 110.4 |
| 3 | 4321 | 699.4 | 26.00 | 26.29 | 65.82 | 446 / 726 = 61.4% | 110.4 |
| 5 | 1234 | 701.4 | 25.90 | 26.22 | 65.96 | 450 / 736 = 61.1% | 110.5 |
| 8 | 4321 | 701.2 | 25.90 | 26.23 | 65.87 | 451 / 738 = 61.1% | 110.4 |
| Mean of 87046444 (B) | | 700.9 | 25.90 | 26.24 | 65.90 | 1,796 / 2,934 = 61.2% | 110.4 |
| Mean of f03b37e9 (A) | | 700.8 | 25.75 | 26.25 | 66.13 | 1,819 / 2,970 = 61.2% | 108.4 |

All 8 runs decoded the same text, 09571f6a. B's peak is higher because its bank held 6 more slot rows, 220 against
A's 214 ([EVIDENCE.md, "The standard test"](EVIDENCE.md#the-standard-test)).

### The context sweep, 1K to 128K

From 2026-11-12 01:02:51Z to 12:45:02Z, our sweep ran one coding task at eight prompt sizes. The prompt is a code context
plus a task, built to each size. Each size ran with `--ctx-size` set to the prompt plus 16,384 tokens and `max_tokens`
16,384. Every run stopped on its own before that limit. The sweep ran two modes, greedy and card (below). Each mode ran 4 times at each size, with seeds 1234, 1234, 4321
and 4321. Greedy is temperature 0, where the seed has no effect. Card is the model card's settings: temperature 1.0, top_p 0.95 and top_k 20.

Prompt rows are the bank's slot rows in the prompt pass. Decode rows are the slot rows of RQ's decode plan, before any
trim. Each table row is the mean of 4 runs.

Greedy decoded the same text in all 4 runs at each size:

| Prompt tokens | Prefill tok/s | Decode tok/s | TTFT s | Wall s | MTP acceptance | Peak footprint GB | Completion tokens | Prompt rows | Decode rows |
|---|---|---|---|---|---|---|---|---|---|
| 1,058 | 109.3 | 27.78 | 9.680 | 202.9 | 63.2% | 108.3 | 5,373 | 235 | 258 |
| 2,093 | 201.2 | 27.80 | 10.41 | 144.5 | 66.1% | 107.7 | 3,727 | 234 | 256 |
| 4,181 | 372.2 | 27.15 | 11.24 | 157.9 | 65.8% | 108.8 | 3,981 | 230 | 255 |
| 8,397 | 679.6 | 25.45 | 12.37 | 164.3 | 64.6% | 109.5 | 3,865 | 229 | 250 |
| 16,800 | 620.9 | 22.30 | 27.07 | 358.2 | 66.5% | 112.7 | 7,379 | 219 | 242 |
| 33,591 | 609.4 | 19.18 | 55.15 | 315.5 | 68.7% | 114.1 | 4,997 | 200 | 219 |
| 67,310 | 516.3 | 14.05 | 130.4 | 561.3 | 74.9% | 112.1 | 6,053 | 161 | 182 |
| 134,615 | 380.2 | 8.575 | 354.2 | 832.1 | 70.4% | 110.6 | 4,080 | 90 | 115 |

<details>
<summary>The card settings' table</summary>

Card decoded a different text, of a different length, in each run:

| Prompt tokens | Prefill tok/s | Decode tok/s | TTFT s | Wall s | MTP acceptance | Peak footprint GB | Completion tokens | Prompt rows | Decode rows |
|---|---|---|---|---|---|---|---|---|---|
| 1,058 | 109.3 | 27.58 | 9.690 | 172.8 | 63.1% | 108.2 | 4,491 | 235 | 258 |
| 2,093 | 201.0 | 26.48 | 10.42 | 227.0 | 66.4% | 108.5 | 5,696 | 234 | 256 |
| 4,181 | 372.3 | 28.03 | 11.24 | 129.9 | 67.5% | 108.3 | 3,306 | 230 | 255 |
| 8,397 | 679.7 | 25.10 | 12.36 | 177.1 | 63.2% | 109.7 | 4,086 | 229 | 250 |
| 16,800 | 620.3 | 23.40 | 27.10 | 238.2 | 65.7% | 112.0 | 4,794 | 219 | 241 |
| 33,591 | 610.0 | 18.18 | 55.09 | 313.8 | 66.5% | 114.0 | 4,686 | 200 | 219 |
| 67,310 | 516.3 | 13.40 | 130.4 | 622.2 | 67.6% | 112.4 | 6,476 | 161 | 182 |
| 134,615 | 383.8 | 8.600 | 350.8 | 1,000 | 75.5% | 110.7 | 5,592 | 90 | 115 |

</details>

Decode falls with the prompt's size. Two causes act together, and we did not measure them apart: the attention reads a
longer KV, and the bank holds fewer rows, so more experts come from the SSD. Prefill peaks at 8,397 tokens and then
falls. We did not measure why short prompts prefill more slowly. The sweep's runs decoded 3,306 to 7,379 tokens on
average, where the 18K test decoded up to 1,024. So the sweep's decode means are not comparable with the 18K test's.

The sweep has no 256K point. Its prompt, 269,187 tokens, is past the model's 262,144-token window.

<details id="memory-at-each-size">
<summary>Memory at each size</summary>

The load's numbers come from its `C1:`, `C2:` and `streamed experts:` lines, and RQ's from its `RQ prompt` and
`RQ decode` lines. They are the same in all runs of a size. The MLX peak is the highest `[mem] a prompt pass ends` peak
of the size's 8 runs.

| Prompt tokens | `--ctx-size` | KV billed, tokens | W MiB | Room MiB | Rows at load (prompt / decode) | RQ decode rows | Trims (greedy / card) | MLX prompt-end peak MiB |
|---|---|---|---|---|---|---|---|---|
| 1,058 | 17,442 | 21,248 | 4,299 | 91,533 | 235 / 247 | 258 | 4 / 1 | 94,694 |
| 2,093 | 18,477 | 21,248 | 4,595 | 91,238 | 234 / 247 | 256 | 0 / 2 | 94,697 |
| 4,181 | 20,565 | 25,600 | 4,595 | 90,004 | 230 / 243 | 255 | 0 / 0 | 93,413 |
| 8,397 | 24,781 | 26,368 | 4,671 | 89,710 | 229 / 242 | 250 | 0 / 1 | 94,367 |
| 16,800 | 33,184 | 33,280 | 6,040 | 86,382 | 219 / 236 | 242 (card 241) | 4 / 2 | 94,364 |
| 33,591 | 49,975 | 50,176 | 7,177 | 80,455 | 200 / 221 | 219 | 0 / 0 | 94,395 |
| 67,310 | 83,694 | 83,712 | 10,273 | 67,854 | 161 / 191 | 182 | 0 / 1 | 93,023 |
| 134,615 | 150,999 | 151,040 | 13,920 | 45,123 | 90 / 132 | 115 | 0 / 1 | 94,520 |

</details>

Charts of every column, and every run of the sweep, are on the `pr-charts` branch of github.com/davidtai/plugins, in its
`turboexl3_ling` folder. See also
[EVIDENCE.md, "The served build"](EVIDENCE.md#the-served-build-87046444).

## Refusals

The plugin refuses a request that its plan cannot hold before it allocates for it. The log names the cause, and the
client gets an error. A whole prompt from position 0, up to the `--ctx-size` context, never meets the first or the
fourth refusal below.

| Request or setting | What the plugin does |
|---|---|
| A prompt forward whose KV would grow past the KV that the bill holds, such as a later turn on a kept prefix (only with a prefix cache turned on) | Refuses it (`a prompt whose KV passes the plan's`) |
| A prompt plan under the bank's base | Refuses it (`RQ over plan`) before any work |
| A decode or trim plan more than 12 rows under the base | Refuses it (`RQ over plan`), after the prompt pass |
| An MLA call of 2 to 16 tokens, not an exact verify, that ends past 87,381 cached tokens | Refuses it (`a call that would rebuild the rope-once key past the bill's point`) |
| Prompt log-probabilities (`echo` with `logprobs`) past the load line's limit | Refuses it. The limit grows with `--ctx-size`: 1,974 tokens at `--ctx-size 17442`, 11,008 at 150,999 |
| A `ctx_size` setting that is not a whole number from 0 to 2^24 | Refuses the load (`LingCtxSizeUnreadable`) |
| A context whose bill leaves no slot row | Refuses the load (`LingContextLeavesNoRoom` or `StreamedRowsDoNotFit`) |
| A 4- or 8-bit KV | No refusal: one `KV_SCHEME_DENSE_COPIES` warning. Do not use one |
| Two long requests at once | No refusal, and not tested. RQ counts another live request's KV as it is now, not its growth |

The MLA call refusal exists because such a call rebuilds 24 KiB of MLA key for each cached token. Past 87,381 tokens,
that key is larger than `prompt_work`, the 2 GiB of the bill that it would use. A 2- to 16-token follow-up turn on a
kept prefix past 87,381 tokens is an example.

## Gaps

| Gap | What a fix needs |
|---|---|
| No mlx-serve release has this plugin's kit | An mlx-serve release with the 26.11.3 kit; then step 3 is not needed |
| mlx-serve reserves no KV for a whole prompt, so each KV growth copies the KV, and RQ trims decode's rows | An mlx-serve change that reserves the request's KV at its first append |
| No refusal of a 4- or 8-bit KV | An mlx-serve change that tells the plugin the KV scheme at load |
| Two long requests at once are not tested | A test of two long requests at once |
| No prefix is kept between requests | A measured `--prefix-cache-disk` setting, or RAM for the prefix cache |
| Decode slows at long prompts: 8.575 tok/s at 134,615 tokens | Fewer bytes for the KV, such as an 8-bit KV that the plugin bills |
| Short prompts prefill slowly: 109.3 tok/s and a 9.68 s TTFT at 1,058 tokens | A measured cause |
| The 2.10bpw pack is not served on this build | A served run, and a limit on how far a lower-bit pack's output may differ from exllamav3's |

See [EVIDENCE.md, "Gaps for a merge"](EVIDENCE.md#gaps-for-a-merge).

## Logs

Each refusal names its cause in the log (block below).

<details>
<summary>A healthy load, every refusal and its fix, and the other log lines</summary>

The server prints its log and writes it to `<test home>/.mlx-serve/logs/mlx-serve-<port>.log`. These lines are from
the sweep's first greedy run at 134,615 tokens (`--ctx-size 150999`, `--port 11391`), in the order they print, with
other lines left out and long lines cut at `...`:

    [plugins] loaded turboexl3_ling (model, sha256 8704644450a2) from <HOME>/.mlx-serve/plugins/turboexl3_ling.dylib
    [turboexl3_ling] C1: the bill holds a 150999-token context: KV for 151040 tokens (...) and 37 checkpoints; room 86574 MiB at the default 32768-token reserve, 45123 MiB at this one
    [turboexl3_ling] C2: the layer-major pass's working set the bill holds at the 150999-token context: 13920 MiB (...)
    [turboexl3_ling] [bank-bill] room 45123 MiB, in use 14734 MiB: prompt 30389 MiB, decode 42805 MiB for the slot rows
    [turboexl3_ling] streamed experts: 90 prompt / 132 decode slot rows per layer of 512 (...)
    [turboexl3_ling] RQ: whole prompts up to the 150999-token context, mlx-serve's KV for each at most 151040 tokens (...); a prompt forward whose KV would pass 151040 tokens is refused by name
    [turboexl3_ling] prompt log-probabilities: whole prompts up to 11008 tokens (...); a longer one is refused by name
    [plugin] turboexl3_ling serves bailing_hybrid (72 layers)
    Hot prefix cache: ENABLED (capacity=32, mem-cap=0.0 MB) [hybrid: SSM checkpoints]
    Server listening on http://127.0.0.1:11391
    [turboexl3_ling] RQ prompt plan (C5): slot=1 P=134615: KV 38167 MiB at capacity 134656, the pass 13266 MiB, ...: room for 107 rows, base 90
    info: [turboexl3_ling] LM_SEGMENTS engaged: a run of 134615 prompt rows from position 0 ran in 4 segments of at most 32768 rows (a last group under 8192 rows joined to the segment before it), ...
    [turboexl3_ling] MTP_HISTORY_CHUNKS engaged: the MTP head's history of 134614 rows (...) ran in 33 forwards of at most 4096 rows, ...
    [turboexl3_ling] RQ decode: slot=1 P=134615 r=0 capacity=142848 head=146688 KV=44718 MiB O=3614 MiB rows 90 -> 115 ...

- `C1`, `C2` and `streamed experts` give the bill and the rows ("Memory"). `slot=<n>` in RQ's lines is mlx-serve's
  request slot, not a slot row. The load's warm-up also prints RQ lines for
  slot 4294967295. That is normal.
- `mem-cap=0.0 MB` means that mlx-serve keeps no prefix in RAM between requests.
  With step 6's command it has no SSD tier either.
- `LM_SEGMENTS` and `MTP_HISTORY_CHUNKS` print only for a prompt of more than 32,768 tokens. The MTP head's history runs
  in forwards of at most 4,096 rows there.
- `RQ trim` shows a new plan when decode passes the planned KV, and `RQ over plan` a plan under the base ("Memory").
- Each request prints `served paths ran (request, prompt)` and `(request, decode)` lines, which name the paths that its
  prompt pass and its first decode round ran.

At parse, the plugin prints `[turboexl3_ling] the pack is refused: <reason>`. At load, mlx-serve prints `[plugin]
turboexl3_ling: load failed: <message>`. At a request, the plugin prints the reason with `log.err` and returns it to
mlx-serve.

| When | Reason or message | Cause and fix |
|---|---|---|
| Start | `plugin was built against another MLX` (mlx-serve's line) | The server is not step 3's: serve with it |
| Start | `changed since it was installed (hash differs from its receipt)` (mlx-serve's line) | Install the dylib again, then restart the server |
| Parse | `Exl3PackNotRepacked` | The folder is the download: run step 2 |
| Parse | `Exl3CodebookUnsupported`, `Exl3QuantConfigNotAnObject`, `LingClampListLength`, `LingClampValue`, `LingLayerCount` | A config that this plugin does not serve: convert turboderp's 2.60bpw or 2.10bpw branch with step 2 |
| Parse | `turboexl3_ling: no Thunderbolt pair (a streamed bank serves one Mac)` | A two-Mac serve: serve on one Mac |
| Load | `the server's host table is older than this plugin's kit` | The server is older than the kit: serve with step 3's server |
| Load | `turboexl3_ling: an M5-class GPU (NAX) is needed ...` | Another GPU |
| Load | `LingCtxSizeUnreadable` | A `ctx_size` that is not a whole number from 0 to 2^24 |
| Load | `LingContextLeavesNoRoom` or `StreamedRowsDoNotFit` | No slot row fits: lower `--ctx-size`, or raise the GPU wired limit |
| Load | `the pack's shards do not open: ...` or `the MTP head does not load: ...` | A damaged or partial pack: run step 2 again |
| Load | `LingNoRoutedLayers`, `LingDenseLayerPastFirst`, `LingEmbedRowsTiedHead` | A pack that is not this model's: convert turboderp's branch with step 2 |
| Prompt | `this host names no chunk-loop spans ...` | The kit has the span fields and the server does not: serve with step 3's server |
| Prompt | `turboexl3_ling: a prompt whose KV passes the plan's <n> tokens: ...` | A prompt forward that grows the KV past the bill's: send a shorter prompt, or serve with a larger `--ctx-size` |
| Prompt or decode | `turboexl3_ling: RQ over plan: ...` | A prompt plan under the base, or a decode or trim plan more than 12 rows under it. Other MLX memory, such as a second request's KV, causes it: send one request at a time. A larger `--ctx-size` lowers the base |
| Prompt | `turboexl3_ling: a call that would rebuild the rope-once key past the bill's point: ...` | An MLA call of 2 to 16 tokens past 87,381 cached tokens: start a new request |
| Prompt | `turboexl3_ling: prompt log-probabilities for a <n>-row prompt forward ...` | Prompt log-probabilities past the load line's limit: send a shorter prompt |
| Prompt | `turboexl3_ling: one slot's prompt at a time`, or `... a layer-major prompt carries no media or captures` | More than one slot's prompt, an image, or a hidden-state capture in one forward pass: send text only |
| Prompt or decode | `LingEmbedRowsRead` or `LingEmbedRowsTokenPastTable` | A failed read of the pack's embedding shard, or a token id outside the table |

A refusal not listed here is an internal error. Report it to the maintainers with the server log.

</details>

## Build

Since 2026-11-12 01:02:51Z we serve dylib 87046444. We built it from this folder's source with turboexl3 4bf2df8, on
mlx-serve 9a173773 and that commit's plugin kit. This folder pins turboexl3 to the same commit by URL. We name a dylib,
and a decoded text such as 09571f6a, by the first 8 hex digits of its sha256.

## Terms

| Term | Meaning here |
|---|---|
| A/B, arm | a comparison of two builds on the same workload; each build is an arm |
| Bank, base | the bank is the GPU cache of routed experts that the streamer fills from the SSD; its base is the load's prompt slot rows, which the prompt pass keeps |
| Bill | the memory that the plugin asks mlx-serve to keep free of the bank ("Memory") |
| bpw | bits per weight |
| Card | the model card's sampling settings: temperature 1.0, top_p 0.95, top_k 20 |
| Dylib `<hash>` | a build of the plugin, named by the first 8 hex digits of its dylib's sha256; not a commit id |
| Exact verify | a verify pass that computes each row as a one-row decode does |
| EXL3, exllamav3 | exllamav3 is turboderp's quantization library; EXL3 is its trellis format |
| Forward pass | one run of a set of tokens through the model |
| KDA, MLA | Ling's two attention kinds: Kimi Delta Attention (linear) and multi-head latent attention (9 layers) |
| Kit | the mlx-serve plugin kit that the plugin compiles against |
| KV | the attention cache that grows with each token, 297,216 B a token here |
| Layer-major pass | the plugin's prompt pass: all of a prompt's tokens through one layer, then the next, in segments past 32,768 tokens |
| MLX-free tests | the plugin's `zig build test`: no GPU, no MLX and no pack |
| MoE | mixture of experts: a block whose router sends each token to a few of its expert FFNs |
| MTP head | the pack's NextN layer, which drafts tokens for the verify pass |
| NAX | the matrix units of M5-class GPUs |
| O | the other MLX memory that RQ measures at a plan, beside the request's KV |
| Over-plan | an RQ plan that leaves fewer slot rows than the base |
| Pack | the converter's output, the folder that mlx-serve serves |
| Prompt pass | the forward passes over a request's prompt (prefill) |
| Room | the GPU memory that mlx-serve gives the plugin for the trunk and the bank: the wired limit less mlx-serve's 8 GiB margin and the bill |
| RQ | the plugin's memory plan for each request ("Memory") |
| Slot row | one expert slot in each of the 68 routed layers, 336,427,008 B |
| Span | one chunk that mlx-serve's chunk loop runs, or would run, as mlx-serve names it to the plugin |
| Trunk | every weight except the routed experts and the MTP head |
| TTFT | time to the first token, measured by the client |
| Units | GB is 10^9 bytes; GiB and MiB are 2^30 and 2^20 bytes |
| W | the layer-major pass's working set, which the bill holds ("Memory") |
