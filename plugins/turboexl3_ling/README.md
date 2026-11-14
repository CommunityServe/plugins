# turboexl3_ling

Updated 2026-11-14. This README describes turboexl3_ling 0.2.0, a working quant plugin that we tested on one Mac. It
came into this repository through the pull request that added it. We measured its code as commit 79baca4 on our
branch, and the code here is the same. "We" are its maintainers (davidtai).

## Summary

mlx-serve is a local LLM server for Apple Silicon Macs. turboexl3_ling lets it serve inclusionAI's Ling-3.1-flash from
turboderp's EXL3 weights. EXL3 is the quantized format of exllamav3, turboderp's library. mlx-serve's own Ling model
(`bailing_hybrid`) runs the forward pass. The plugin supplies the EXL3 format, turboexl3's Metal kernels for it, and a
bank that streams the routed experts from the SSD. A converter repacks turboderp's 2.60bpw (bits per weight) branch
into a 188.2 GB pack, more than the Mac's memory.

The plugin needs mlx-serve's main branch at 7b13376d or later. No mlx-serve release has the plugin interface that it
needs yet, so you build mlx-serve from source ("Requirements").

Our standard test is an 18,380-token coding prompt with a 1,024-token greedy reply ("Measured"). Prefill is the rate at
which the server reads the prompt. Decode is the rate at which it writes the reply. We ran the test on an M5 Max with
128 GB and macOS 26.6, on 2026-11-13 and 2026-11-14. Each value is the mean of 4 runs.

| Plugin | mlx-serve build | Prefill tok/s | Decode tok/s |
|---|---|---|---|
| 0.2.0 | main 7b13376d | Not measured | Not measured |
| 0.2.0 | 56460793 (#1192 before its merge) | 113.1 | 11.28 |
| 0.2.0 | 37159073 (#1192 and #1195) | 112.9 | 12.68 |
| 0.2.0 | 3f8c8a57 (#1192, #1195 and #1205) | 339.7 | 11.64 |
| 0.1.0, the old whole-model plugin | 9a173773 | 701.5 | 25.85 |

#1192 is merged into main. #1195 and #1205 are open mlx-serve pull requests. #1195 raises decode, and #1205 raises
prefill. The plugin works without them.

Every speed of 0.2.0 in this README used our `bank-fit.json`, a memory calibration that we have not published. A slot
row is one expert of each routed layer, held in GPU memory. With our fit, the bank held 189 to 192 slot rows on
56460793. Without a fit, mlx-serve plans the bank from its bills, its own upper-bound estimates of memory use. That plan
gave 162 slot rows on the same build. We did not measure the speed without a fit. Decode may be lower then, because the
bank reads more experts from the SSD.

With greedy sampling, the text with MTP (multi-token prediction) drafts equals the text of plain decoding. Plain
decoding makes one token at a time, with no drafts.

## Why a quant plugin

Version 0.1.0, in the closed CommunityServe/plugins#50, was a whole-model plugin. It compiled its own copy of
mlx-serve's Ling model and expert streamer. Version 0.2.0 is a quant plugin. It adds a weight format and its kernels,
and mlx-serve runs the model. It holds no copy of mlx-serve's code.

We accept a slower plugin for that reason. With #1205, prefill reaches 339.7 tok/s, about half of 0.1.0's. Decode is
about half of 0.1.0's on every build that we measured. "Measured" gives the causes that we found.

## What runs where

The trunk is every weight except the routed experts and the MTP layer. Ling-3.1-flash has 72 main layers (0-71) and one
MTP layer (72). The MTP layer drafts tokens, and the main model then checks them in a verify pass. Most main layers use
KDA (Kimi Delta Attention, a linear attention), and 9 use MLA (multi-head latent attention). From layer 4 on, each
layer's router picks 8 of 512 routed experts for each token.

The plugin's bank holds routed experts in GPU memory, in slot rows. A slot row holds one expert of each of the 68
routed layers, 336,427,008 B without the records' padding. When a token needs an expert that is not in the bank, the
bank reads it from the SSD.

| Part | Who runs it |
|---|---|
| The trunk's EXL3 linears: KDA and MLA projections, dense layers 0-3, shared experts, the output head (`lm_head`) | this plugin, on turboexl3's kernels |
| The routed experts of layers 4-71 | this plugin's bank, filled from the pack's `experts.bin` over the plugin kit's expert stream |
| The routed experts of the MTP layer | this plugin, held in GPU memory from load |
| The output head that scores MTP drafts | mlx-serve, on a 4-bit copy of the head that it makes at load. The verify pass uses the full head. |
| Ling-3.1's SwiGLU clamp after the SiLU | this plugin for the routed experts (layers 65-71); mlx-serve for the shared experts (layers 64-71). The MTP layer has no clamp. |
| Attention, routing, norms, sampling, the MTP loop | mlx-serve's `bailing_hybrid` |

The plugin serves (claims) a pack only when its `config.json` has `model_type` `bailing_hybrid` and an exllamav3
`quantization_config`. mlx-serve serves all other Ling packs itself. The plugin builds against turboexl3 0a376d11,
pinned in `build.zig.zon`.

## Requirements

- mlx-serve built from main at 7b13376d or later. The plugin needs plugin-interface members that ddalcu/mlx-serve#1192
  adds. #1192 merged into main on 2026-11-13 (squash commit 23590f50), after the latest release, v26.11.2.
  `plugin.json` asks for 26.11.3 or later, the version that this repository's `KIT_VERSION` names. That version is not
  released yet.
- A Mac with an M5-class GPU. turboexl3's prompt kernels run on the M5 GPU's matrix units (NAX). On another GPU, the
  load refuses with `an M5-class GPU (NAX) is needed`. We ran macOS 26.6.
- Enough memory. We measured only an M5 Max with 128 GB, and we did not test a Mac with less memory. If no slot row
  fits, the load stops with `StreamedRowsDoNotFit`.
- About 188.3 GB of disk for the pack. To make the pack yourself, you need about 377 GB on one volume instead: 188.2 GB
  for the pack and 188.4 GB for turboderp's download. The converter hard-links the download's small files, so both
  must be on that volume. You can delete the download after `finalize`.
- git, Python 3, the Hugging Face CLI (`hf`), Homebrew, and Xcode with its Metal toolchain.

Two open mlx-serve pull requests raise the speed. #1195 raises decode, and #1205 raises prefill. "Measured" gives
their numbers.

## Get the pack

We published the streaming pack on Hugging Face, in the 2.60bpw branch of
https://huggingface.co/OpensourceWTF/Ling-3.1-flash-streaming-repack-exl3. Download it with this command:

    hf download OpensourceWTF/Ling-3.1-flash-streaming-repack-exl3 --revision 2.60bpw --local-dir <pack>

You can also make the same pack yourself from turboderp's pack. The converter copies bytes and checks each record. It
does not quantize.

1. Download the 2.60bpw branch:

       hf download turboderp/Ling-3.1-flash-exl3 --revision 2.60bpw --local-dir <download>

2. Clone https://github.com/OpenSourceWTF/mlx-serve-streaming-pack-converters. The Ling script is in its `ling31`
   folder. In the clone, run these commands. `plan` reads the branch's file records from Hugging Face.

       python3 ling31/scripts/ling_exl3_to_streaming.py plan --src <download> --rev 2.60bpw --plan plan.json
       python3 ling31/scripts/ling_exl3_to_streaming.py run --src <download> --out <pack> --plan plan.json
       python3 ling31/scripts/ling_exl3_to_streaming.py finalize --src <download> --out <pack> --plan plan.json

   A good `run` prints `ALL_SHARDS_PASS <n>/<n>` last. A good `finalize` prints `FINALIZED <pack>: ...`.

The pack keeps turboderp's `config.json` and moves the routed experts of layers 4-71 into `experts.bin`.

## Build and install

1. Build mlx-serve from main at 7b13376d, and its plugin kit:

       git clone --recurse-submodules https://github.com/ddalcu/mlx-serve && cd mlx-serve
       git checkout 7b13376d && git submodule update --init
       brew bundle install --file=Brewfile
       ./scripts/fetch-zig.sh && ./scripts/fetch-llama.sh && ./scripts/build-mlx.sh
       .zig-toolchain/zig build -Doptimize=ReleaseFast && .zig-toolchain/zig build kit
       KIT=$(ls -d $PWD/zig-out/kit/mlx-serve-plugin-kit-*/) && export MLX_SERVE_KIT=${KIT%/}
       export PATH=$PWD/zig-out/bin:$PATH

   To add #1195 and #1205, replace the second line with the line below. The head of #1205 (48c886ee) holds main
   7b13376d, #1195 and #1205. We measured #1205's commits as build 3f8c8a57. We did not measure the speed of 48c886ee.

       git fetch origin pull/1205/head && git checkout 48c886ee && git submodule update --init

2. In the same shell, go back to the parent folder. Then clone this repository and install the plugin:

       cd ..
       git clone https://github.com/CommunityServe/plugins && cd plugins
       scripts/install.sh turboexl3_ling --home <test home>

   `scripts/install.sh` compiles the plugin, runs its unit tests (which need no GPU), and calls
   `mlx-serve plugin install`. It builds against the kit that `MLX_SERVE_KIT` names, because the 26.11.3 kit download
   does not exist yet. With `--home`, the script installs under a new HOME and turns plugins on there. Without
   `--home`, set `"enabled": true` in `.mlx-serve/plugins.json` in your home folder. If `.kit` holds a kit from an
   earlier build, delete `.kit` first, because `scripts/setup.sh` keeps a staged kit.

## Serve

These values are for a 128 GB Mac. On another size, leave out `sysctl` and `MLX_SERVE_GPU_CEILING_MB`, and mlx-serve
plans to the Mac's own GPU limit. We did not test this. On our Mac, the server held 23,654 MiB before the bank opened.
Each slot row adds 336,427,008 B. The server listens on port 11234 (`/v1/chat/completions`).

    sudo sysctl iogpu.wired_limit_mb=114688
    HOME=<test home> MLX_SERVE_GPU_CEILING_MB=114688 MLX_SERVE_CACHE_LIMIT=2147483648 \
      mlx-serve --serve --model <pack> --ctx-size 32768 --prefill-chunk 4096 --mtp-max-depth 3

- `iogpu.wired_limit_mb` lets the GPU wire 112 GiB until a reboot.
- `MLX_SERVE_GPU_CEILING_MB` sets the ceiling, the GPU memory that mlx-serve plans to.
- `MLX_SERVE_CACHE_LIMIT` caps MLX's buffer cache at 2 GiB, given in bytes. Keep it. Our fit and our tests used it.
  "The cache limit" in "Measured" gives its cost.
- `--ctx-size` is the longest prompt plus its reply.
- `--prefill-chunk 4096` is an upper limit. mlx-serve narrows the chunk to 1,536 tokens for an 18K prompt.
- `--mtp-max-depth 3` lets the MTP layer draft up to 3 tokens a round.
- Leave out `HOME=` if you installed without `--home`.

At load, the log shows `[plugins] loaded turboexl3_ling (quant, ...)` and
`[turboexl3_ling] streamed experts: <P> prompt / <D> decode slot rows per layer`. P and D are the experts that the bank
holds for each routed layer in the prompt pass and in decode.

### The bank's memory, and bank-fit.json

The bank's slot rows take the memory that the rest of the server leaves. mlx-serve gives the bank a budget for each
phase (the prompt pass, then decode). Without a fit, it sets the budget from its bills, and our Mac got 162 slot rows
on 56460793. With a `bank-fit.json` in the pack's folder, it sets the budget from measured memory use, and the same
build got 189 to 192 slot rows. Our fitter, which makes the file from one run of the standard test, is not public yet.
This is a known gap. Until we publish the fitter, you cannot make your own fit, and mlx-serve plans the bank from its
bills.

<details>
<summary>How mlx-serve sets the budget, and when a fit applies</summary>

- From the bills only. A bill is mlx-serve's upper-bound estimate of a phase's memory. Each phase's budget is the
  ceiling, less its bills, less the Mac's non-file memory in use when the bank opens. Non-file memory is memory in use
  that is not file cache. This way leaves memory unused.
- From a fit. A `bank-fit.json` in the pack's folder holds each phase's measured memory use from one run of our
  standard test. Each phase's budget is the ceiling, less that measured use, less the bills' growth from that run to
  `--ctx-size`, less 1 GiB. The budget also loses any growth in the memory outside the server since that run.

A fit applies only under three conditions. Its format is v2. Its `ceiling_bytes` equals the server's ceiling in bytes.
For `MLX_SERVE_GPU_CEILING_MB=114688`, that is 120,259,084,288. Its `cell_tokens` is 17,408, as mlx-serve's is.
mlx-serve's Ling model loads through its GLM-5.3 loader, so the load log has a `glm5_next:` line about the fit. That
line says `the bank fits by bank-fit.json`, or it says `bank-fit.json not applied` or `does not read` with the reason.
Our fit is for one M5 Max with 128 GB at that ceiling.

</details>

## Measured

We ran our standard test on an M5 Max with 128 GB and macOS 26.6. The prompt is 18,380 tokens: CPython's `shlex.py`
and a coding task. The prompt file is not public yet. The reply is up to 1,024 tokens, greedy. Each arm ran 4 times, in
the order A B B A, then B A A B, on seeds 1234 and 4321. Each run used a new server. Before each run, we waited for the
chip to cool to 40 C. Prefill and decode are the server's own rates. The client measured TTFT (time to first token) and
wall time. "Terms" defines the other columns. "The builds" below says what each mlx-serve build holds.

<details>
<summary>The builds</summary>

| Build | What it holds | Where it is |
|---|---|---|
| 9a173773 | mlx-serve before #1192, for the old plugin | mlx-serve main's history |
| 56460793 | An earlier head of #1192, with the fix to how mlx-serve reads memory use when the bank opens | #1192's branch on davidtai/mlx-serve |
| 37159073 | 56460793 with #1195's two commits | Not published |
| cbb7f16c | #1195's two commits on #1192's later head f4ccce40 | #1195's branch on davidtai/mlx-serve |
| 3f8c8a57 | cbb7f16c with #1205's commits | #1205's history |

</details>

The two pull requests changed the speed as below. In each pair, both builds gave the same text.

| Pull request (head) | What it changes | Build without it | Build with it | Prefill tok/s, without / with | Decode tok/s, without / with |
|---|---|---|---|---|---|
| #1195 (7769248e) | Runs an exact verify's KDA and MLA rows in one launch | 56460793 | 37159073 (two separate tests, not one A/B) | 113.1 / 112.9 | 11.28 / 12.68 |
| #1205 (48c886ee) | Runs the prompt of a streaming pack as one whole-prompt pass | cbb7f16c | 3f8c8a57, in one A/B | 112.9 / 339.7 | 11.96 / 11.64 |

In an exact verify, each row of the verify pass gives the same bytes as plain decoding.

### Against the old plugin

We ran two tests of this plugin (B), each against the old plugin (A). A is turboexl3_ling 0.1.0 (dylib 87046444) on
9a173773. Both B arms are plugin 79baca4, served with the command in "Serve" and our `bank-fit.json`.

| Test, arm (mean of 4 runs) | Prefill tok/s | Decode tok/s | TTFT s | Wall s | MTP acceptance | Peak footprint GB | Slot rows (prompt / decode) |
|---|---|---|---|---|---|---|---|
| 1, A | 701.9 | 25.95 | 26.20 | 65.82 | 1,790 / 2,949 = 60.7% | 110.4 | 220 / 236 |
| 1, B (56460793) | 113.1 | 11.28 | 162.5 | 253.7 | 2,048 / 3,106 = 65.9% | 98.23 | 191-192 |
| 2, A | 701.5 | 25.85 | 26.22 | 65.93 | 1,806 / 2,978 = 60.6% | 110.5 | 220 / 236 |
| 2, B (37159073) | 112.9 | 12.68 | 162.8 | 244.0 | 2,115 / 3,525 = 60.0% | 98.00 | 191 |

Slot rows differ between loads because each load fits its bank when the bank opens. In every run, A decoded the text
09571f6a and B decoded 9632d6ec, so the decode rates compare different texts. We did not analyze why A's text differs.
The old plugin is a different implementation. Each B build also gave 9632d6ec in 2 runs of plain decoding
(`--no-mtp --no-pld`).

<details>
<summary>Every run, and an earlier test</summary>

Test 1 (B: mlx-serve 56460793, plugin 79baca4). The Mac's non-file memory peaked at 104.7 to 108.4 GB in B's prompts
and 104.4 to 104.9 GB in its decodes.

| Run | Arm | Seed | Prefill tok/s | Decode tok/s | TTFT s | Wall s | MTP acceptance | Peak footprint GB |
|---|---|---|---|---|---|---|---|---|
| 1 | A | 1234 | 701.0 | 26.00 | 26.23 | 65.75 | 448 / 734 = 61.0% | 110.3 |
| 2 | B | 1234 | 113.3 | 11.40 | 162.23 | 252.59 | 519 / 757 = 68.6% | 98.0 |
| 3 | B | 4321 | 113.1 | 11.20 | 162.53 | 254.56 | 502 / 720 = 69.7% | 98.3 |
| 4 | A | 4321 | 701.5 | 26.00 | 26.21 | 65.77 | 450 / 741 = 60.7% | 110.5 |
| 5 | B | 1234 | 113.0 | 11.10 | 162.65 | 255.35 | 510 / 844 = 60.4% | 98.3 |
| 6 | A | 1234 | 700.5 | 26.00 | 26.25 | 65.82 | 448 / 732 = 61.2% | 110.4 |
| 7 | A | 4321 | 704.7 | 25.80 | 26.10 | 65.94 | 444 / 742 = 59.8% | 110.5 |
| 8 | B | 4321 | 113.1 | 11.40 | 162.46 | 252.37 | 517 / 785 = 65.9% | 98.3 |

Test 2 (B: mlx-serve 37159073, plugin 79baca4).

| Run | Arm | Seed | Prefill tok/s | Decode tok/s | TTFT s | Wall s | MTP acceptance | Peak footprint GB |
|---|---|---|---|---|---|---|---|---|
| 1 | A | 1234 | 701.0 | 26.00 | 26.23 | 65.73 | 467 / 754 = 61.9% | 110.5 |
| 2 | B | 1234 | 112.6 | 12.70 | 163.18 | 244.09 | 531 / 939 = 56.5% | 98.0 |
| 3 | B | 4321 | 113.0 | 12.40 | 162.63 | 245.71 | 520 / 800 = 65.0% | 98.0 |
| 4 | A | 4321 | 701.8 | 26.00 | 26.21 | 65.70 | 451 / 736 = 61.3% | 110.4 |
| 5 | B | 1234 | 113.4 | 12.80 | 162.15 | 242.28 | 533 / 962 = 55.4% | 98.0 |
| 6 | A | 1234 | 700.2 | 25.70 | 26.26 | 66.22 | 444 / 746 = 59.5% | 110.5 |
| 7 | A | 4321 | 702.9 | 25.70 | 26.16 | 66.06 | 444 / 742 = 59.8% | 110.4 |
| 8 | B | 4321 | 112.5 | 12.80 | 163.40 | 243.88 | 531 / 824 = 64.4% | 98.0 |

An earlier test ran plugin af989ed (one commit before 79baca4) on mlx-serve a76e0ccc, an earlier head of #1192. That
build did not have the fix to how mlx-serve reads memory use when the bank opens. The test used the fit of that time
and no cache limit. B held 186 to 187 slot rows.

| Arm (mean of 4 runs) | Prefill tok/s | Decode tok/s | TTFT s | Wall s | MTP acceptance | Peak footprint GB |
|---|---|---|---|---|---|---|
| A | 701.0 | 25.63 | 26.23 | 66.29 | 62.6% | 110.4 |
| B | 116.0 | 10.95 | 158.4 | 252.2 | 65.5% | 98.95 |

| Run | Arm | Seed | Prefill tok/s | Decode tok/s | TTFT s | Wall s | MTP acceptance | Peak footprint GB |
|---|---|---|---|---|---|---|---|---|
| 1 | A | 1234 | 703.0 | 26.00 | 26.16 | 65.64 | 450 / 734 = 61.3% | 110.4 |
| 2 | B | 1234 | 116.4 | 11.00 | 157.96 | 251.68 | 503 / 791 = 63.6% | 99.1 |
| 3 | B | 4321 | 115.7 | 10.90 | 158.85 | 253.13 | 516 / 755 = 68.3% | 98.8 |
| 4 | A | 4321 | 699.4 | 25.70 | 26.29 | 66.16 | 445 / 710 = 62.7% | 110.5 |
| 5 | B | 1234 | 116.3 | 10.80 | 158.03 | 252.67 | 522 / 801 = 65.2% | 98.8 |
| 6 | A | 1234 | 700.1 | 25.20 | 26.27 | 67.04 | 461 / 760 = 60.7% | 110.3 |
| 7 | A | 4321 | 701.6 | 25.60 | 26.21 | 66.31 | 428 / 646 = 66.3% | 110.2 |
| 8 | B | 4321 | 115.7 | 11.10 | 158.84 | 251.46 | 522 / 802 = 65.1% | 99.1 |

</details>

B's prefill is slow without #1205. mlx-serve then runs the 18,380-token prompt in 14 chunks. Ten chunks have 1,536
tokens, two have 512, one has 1,965 and one has 31. Each chunk reads most experts again, so each prompt read 1.51 TB from the SSD.

B's decode is slower for two causes. B held 191 slot rows and A held 236 in decode, so B read more experts. A also had
trunk and MTP optimizations that B lacks. We did not measure the two causes apart.

### With #1205's whole-prompt pass

On 2026-11-14 we ran the standard test with plugin 79baca4 on two mlx-serve builds. A is cbb7f16c, which runs the
prompt in chunks. B is 3f8c8a57, which is A with #1205's commits. Each arm had its own `bank-fit.json` and the cache
limit. Both arms gave the text 9632d6ec on both seeds.

| Arm (mean of 4 runs) | Prefill tok/s | Decode tok/s | TTFT s | Wall s | MTP acceptance | Peak footprint GB | Slot rows | Prompt SSD reads TB | Prompt read wait s |
|---|---|---|---|---|---|---|---|---|---|
| A, cbb7f16c | 112.9 | 11.96 | 162.8 | 248.7 | 55.5% | 96.15 | 185-186 | 1.514 | 114.2 |
| B, 3f8c8a57 | 339.7 | 11.64 | 54.12 | 142.3 | 63.1% | 95.65 | 184-185 | 0.197 | 11.4 |

#1205 raises prefill. Decode stays at about 11.6 to 12.0 tok/s. In B, each routed call reads each expert that it needs
at most once. #1205 also reads ahead the experts that the router predicts. Of those read-ahead records, 2,455 became
hits. The other 7,251 records, about 36 GB, were reads that no call used. So B's prompt read 30,019 missed experts and
9,706 read-ahead records, each of 4,947,456 B.

A's decode here (11.96 tok/s) is lower than test 2's B (12.68 tok/s), although both builds hold #1195's commits.
cbb7f16c sits on #1192's later head f4ccce40, and its newer fit gave 185-186 slot rows against 191. We did not isolate
which change moved decode.

<details>
<summary>Each run</summary>

| Arm | Prefill tok/s, runs in order | Decode tok/s, runs in order |
|---|---|---|
| A, cbb7f16c | 112.78, 112.85, 112.97, 112.96 | 11.76, 12.10, 12.01, 11.96 |
| B, 3f8c8a57 | 341.83, 340.87, 342.04, 334.07 | 11.62, 11.87, 11.63, 11.46 |

</details>

### The cache limit

We keep `MLX_SERVE_CACHE_LIMIT=2147483648`. In an A/B without it, prefill was 2.7% faster, but the Mac paged memory
out. With the limit, the context sweep's least free RAM still fell below 1 GB at 1K, 2K and 16K (0.66 to 0.99 GB). The
sweep did not record pageouts.

<details>
<summary>The cache-limit A/B</summary>

On 2026-11-13 we ran the standard test on mlx-serve cbb7f16c with plugin 79baca4 and our fit, with and without
`MLX_SERVE_CACHE_LIMIT=2147483648`. Both arms gave the same text. Without the limit, prefill was faster and decode was
slower. The Mac's free memory also fell below 1 GB, and the Mac paged memory out. The prompt's SSD read wait was the
same in both arms.

| Arm (mean of 4 runs) | Prefill tok/s | Decode tok/s | Wall s | Prompt time outside SSD reads s | Mac's non-file memory peak GB, prompt | Same, decode | Least free RAM GB | Pageouts per run |
|---|---|---|---|---|---|---|---|---|
| With the 2 GiB limit | 112.88 | 12.13 | 247.7 | 48.3 | Not recorded | 104.5 | 1.8-4.1 | 0, 0, 0, 0 |
| Without a limit | 115.95 | 11.80 | 245.7 | 44.0 | Up to 110.75 | 105.6-105.8 | 0.23-0.91 | 198, 2, 6, 1 |

</details>

### Context sweep, 1K to 32K

From 2026-11-13 to 2026-11-14 we ran a context sweep on cbb7f16c, which runs the prompt in chunks. Up to 16K, decode
stayed between 11.1 and 11.9 tok/s, and at 32K it fell to 6.55 to 7.58 tok/s. Prefill peaked at 4K. We did not analyze
why for either result. 64K and 128K will run later as one sweep on the final stack, because #1205 replaces the chunks.

<details>
<summary>The sweep's tables, every sweep run, and the sweep's memory</summary>

The sweep used plugin 79baca4 and the standard test's environment: the ceiling, the cache limit and the pack's
`bank-fit.json`. Each size has its own coding prompt, and the prompts are not public. Each run set `--ctx-size` to the
prompt plus 16,384 tokens and `max_tokens` to 16,384. Each size ran 4 times greedy and 4 times at the model card's
sampling settings, on seeds 1234 and 4321. Every run ended on a stop token.

Greedy, mean of 4 runs. All 4 greedy runs of a size gave one text.

| Size | Prompt tokens | `--ctx-size` | Prefill tok/s | Decode tok/s | TTFT s | Wall s | MTP acceptance | Peak footprint GB | Reply tokens |
|---|---|---|---|---|---|---|---|---|---|
| 1K | 1,058 | 17,442 | 77.50 | 11.850 | 13.66 | 507.44 | 61.5% | 99.95 | 5,854 |
| 2K | 2,093 | 18,477 | 128.07 | 11.525 | 16.35 | 319.71 | 52.8% | 99.47 | 3,489 |
| 4K | 4,181 | 20,565 | 199.93 | 11.500 | 20.92 | 261.26 | 55.9% | 98.70 | 2,758 |
| 8K | 8,397 | 24,781 | 132.60 | 11.300 | 63.34 | 536.59 | 58.6% | 98.90 | 5,348 |
| 16K | 16,800 | 33,184 | 102.55 | 11.400 | 163.84 | 886.85 | 60.5% | 96.50 | 8,240 |
| 32K | 33,591 | 49,975 | 54.75 | 6.550 | 613.62 | 2,393.76 | 60.8% | 103.40 | 11,198 |

At the model card's sampling settings, mean of 4 runs. Each run gave its own text.

| Size | Prompt tokens | Prefill tok/s | Decode tok/s | TTFT s | Wall s | MTP acceptance | Peak footprint GB | Reply tokens |
|---|---|---|---|---|---|---|---|---|
| 1K | 1,058 | 77.50 | 11.700 | 13.66 | 502.26 | 56.8% | 99.85 | 5,685 |
| 2K | 2,093 | 127.72 | 11.825 | 16.39 | 512.30 | 57.0% | 100.22 | 5,844 |
| 4K | 4,181 | 199.60 | 11.900 | 20.95 | 366.62 | 54.6% | 99.42 | 4,132 |
| 8K | 8,397 | 132.65 | 11.725 | 63.31 | 503.12 | 60.8% | 99.05 | 5,121 |
| 16K | 16,800 | 102.67 | 11.100 | 163.63 | 505.64 | 57.7% | 94.10 | 3,801 |
| 32K | 33,591 | 54.62 | 7.575 | 614.96 | 1,292.97 | 63.5% | 86.85 | 4,204 |

At 32K, decode fell to 5.2 to 8.8 tok/s in the greedy runs and 5.1 to 9.8 tok/s in the card runs.

Greedy runs. The prompt tokens equal the sweep's calculated prompt length at every size. Tokens per round is the
sweep record's value.

| Size | Run | Seed | Prompt tokens | Prefill tok/s | Decode tok/s | TTFT s | Wall s | MTP acceptance | Tokens per round | Peak footprint GB | Reply tokens | Finish | Text |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1K | 1 | 1234 | 1,058 | 77.6 | 11.80 | 13.64 | 510.03 | 3,549 / 5,827 = 60.9% | 2.68 | 100.0 | 5,854 | stop | 23ec229d |
| 1K | 2 | 1234 | 1,058 | 77.6 | 11.90 | 13.64 | 506.50 | 3,458 / 5,498 = 62.9% | 2.58 | 99.7 | 5,854 | stop | 23ec229d |
| 1K | 3 | 4321 | 1,058 | 77.6 | 11.80 | 13.64 | 509.30 | 3,236 / 5,335 = 60.7% | 2.59 | 100.0 | 5,854 | stop | 23ec229d |
| 1K | 4 | 4321 | 1,058 | 77.2 | 11.90 | 13.70 | 503.91 | 3,201 / 5,188 = 61.7% | 2.58 | 100.1 | 5,854 | stop | 23ec229d |
| 2K | 1 | 1234 | 2,093 | 127.9 | 11.40 | 16.37 | 323.06 | 2,056 / 4,030 = 51.0% | 2.48 | 99.2 | 3,489 | stop | 1b0a9457 |
| 2K | 2 | 1234 | 2,093 | 128.1 | 11.50 | 16.35 | 321.06 | 2,018 / 3,564 = 56.6% | 2.39 | 99.5 | 3,489 | stop | 1b0a9457 |
| 2K | 3 | 4321 | 2,093 | 127.9 | 11.40 | 16.37 | 321.95 | 2,053 / 4,077 = 50.4% | 2.49 | 99.6 | 3,489 | stop | 1b0a9457 |
| 2K | 4 | 4321 | 2,093 | 128.4 | 11.80 | 16.31 | 312.76 | 2,015 / 3,751 = 53.7% | 2.44 | 99.6 | 3,489 | stop | 1b0a9457 |
| 4K | 1 | 1234 | 4,181 | 200.2 | 11.60 | 20.89 | 259.65 | 1,515 / 2,709 = 55.9% | 2.66 | 98.5 | 2,758 | stop | f63e30b4 |
| 4K | 2 | 1234 | 4,181 | 199.2 | 11.60 | 21.00 | 259.61 | 1,500 / 2,700 = 55.6% | 2.65 | 98.9 | 2,758 | stop | f63e30b4 |
| 4K | 3 | 4321 | 4,181 | 200.5 | 11.50 | 20.86 | 259.79 | 1,555 / 2,757 = 56.4% | 2.68 | 98.5 | 2,758 | stop | f63e30b4 |
| 4K | 4 | 4321 | 4,181 | 199.8 | 11.30 | 20.93 | 265.98 | 1,495 / 2,691 = 55.6% | 2.65 | 98.9 | 2,758 | stop | f63e30b4 |
| 8K | 1 | 1234 | 8,397 | 132.2 | 11.30 | 63.52 | 537.38 | 3,301 / 5,817 = 56.7% | 2.62 | 98.9 | 5,348 | stop | 7579168d |
| 8K | 2 | 1234 | 8,397 | 132.8 | 11.10 | 63.25 | 546.05 | 2,971 / 4,947 = 60.1% | 2.50 | 98.9 | 5,348 | stop | 7579168d |
| 8K | 3 | 4321 | 8,397 | 132.8 | 11.20 | 63.26 | 539.75 | 3,011 / 5,176 = 58.2% | 2.56 | 98.9 | 5,348 | stop | 7579168d |
| 8K | 4 | 4321 | 8,397 | 132.6 | 11.60 | 63.34 | 523.18 | 3,110 / 5,192 = 59.9% | 2.55 | 98.9 | 5,348 | stop | 7579168d |
| 16K | 1 | 1234 | 16,800 | 102.9 | 11.40 | 163.30 | 884.35 | 4,290 / 6,634 = 64.7% | 2.55 | 96.6 | 8,240 | stop | 137ff738 |
| 16K | 2 | 1234 | 16,800 | 102.5 | 11.10 | 163.99 | 908.08 | 4,205 / 6,823 = 61.6% | 2.51 | 96.2 | 8,240 | stop | 137ff738 |
| 16K | 3 | 4321 | 16,800 | 102.6 | 11.80 | 163.76 | 861.88 | 3,945 / 6,954 = 56.7% | 2.58 | 96.9 | 8,240 | stop | 137ff738 |
| 16K | 4 | 4321 | 16,800 | 102.2 | 11.30 | 164.33 | 893.10 | 4,213 / 7,133 = 59.1% | 2.59 | 96.3 | 8,240 | stop | 137ff738 |
| 32K | 1 | 1234 | 33,591 | 54.5 | 6.10 | 615.95 | 2,464.17 | 2,418 / 4,236 = 57.1% | 2.45 | 103.1 | 11,198 | stop | 83ebe546 |
| 32K | 2 | 1234 | 33,591 | 54.9 | 8.80 | 612.32 | 1,888.89 | 5,520 / 9,157 = 60.3% | 2.43 | 105.6 | 11,198 | stop | 83ebe546 |
| 32K | 3 | 4321 | 33,591 | 54.7 | 5.20 | 614.54 | 2,761.13 | 1,221 / 1,850 = 66.0% | 2.55 | 102.1 | 11,198 | stop | 83ebe546 |
| 32K | 4 | 4321 | 33,591 | 54.9 | 6.10 | 611.69 | 2,460.83 | 2,484 / 3,917 = 63.4% | 2.40 | 102.8 | 11,198 | stop | 83ebe546 |

Runs at the model card's sampling settings.

| Size | Run | Seed | Prompt tokens | Prefill tok/s | Decode tok/s | TTFT s | Wall s | MTP acceptance | Tokens per round | Peak footprint GB | Reply tokens | Finish | Text |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1K | 1 | 1234 | 1,058 | 77.6 | 12.00 | 13.64 | 435.56 | 2,640 / 4,807 = 54.9% | 2.38 | 99.4 | 5,055 | stop | 9d286915 |
| 1K | 2 | 1234 | 1,058 | 77.5 | 11.60 | 13.66 | 267.84 | 1,671 / 2,768 = 60.4% | 2.57 | 98.3 | 2,948 | stop | cf7bee9c |
| 1K | 3 | 4321 | 1,058 | 77.3 | 11.20 | 13.69 | 694.40 | 4,254 / 7,479 = 56.9% | 2.60 | 100.7 | 7,590 | stop | 5dd6b732 |
| 1K | 4 | 4321 | 1,058 | 77.6 | 12.00 | 13.64 | 611.25 | 4,008 / 7,069 = 56.7% | 2.55 | 101.0 | 7,147 | stop | 40569684 |
| 2K | 1 | 1234 | 2,093 | 127.4 | 11.90 | 16.43 | 340.89 | 2,166 / 4,072 = 53.2% | 2.38 | 99.4 | 3,863 | stop | c6568112 |
| 2K | 2 | 1234 | 2,093 | 128.0 | 12.00 | 16.36 | 380.62 | 2,370 / 4,568 = 51.9% | 2.46 | 99.6 | 4,369 | stop | 4d63f7b1 |
| 2K | 3 | 4321 | 2,093 | 127.7 | 12.00 | 16.39 | 642.85 | 4,287 / 7,464 = 57.4% | 2.57 | 100.8 | 7,541 | stop | 8e4deb72 |
| 2K | 4 | 4321 | 2,093 | 127.8 | 11.40 | 16.38 | 684.84 | 4,268 / 6,860 = 62.2% | 2.45 | 101.1 | 7,602 | stop | ebb38541 |
| 4K | 1 | 1234 | 4,181 | 198.9 | 11.50 | 21.03 | 300.94 | 1,790 / 3,477 = 51.5% | 2.53 | 99.0 | 3,211 | stop | 403c0793 |
| 4K | 2 | 1234 | 4,181 | 200.1 | 11.80 | 20.90 | 302.22 | 1,884 / 3,483 = 54.1% | 2.61 | 98.6 | 3,317 | stop | 36ff1066 |
| 4K | 3 | 4321 | 4,181 | 199.8 | 12.00 | 20.94 | 442.02 | 2,793 / 4,984 = 56.0% | 2.56 | 99.9 | 5,057 | stop | ac6c97ee |
| 4K | 4 | 4321 | 4,181 | 199.6 | 12.30 | 20.95 | 421.31 | 2,688 / 4,820 = 55.8% | 2.59 | 100.2 | 4,943 | stop | c789425a |
| 8K | 1 | 1234 | 8,397 | 133.0 | 11.70 | 63.15 | 351.42 | 2,098 / 3,534 = 59.4% | 2.67 | 98.9 | 3,364 | stop | 1dd18a2f |
| 8K | 2 | 1234 | 8,397 | 132.6 | 11.90 | 63.32 | 363.38 | 1,961 / 3,083 = 63.6% | 2.62 | 98.9 | 3,562 | stop | e91e0bb8 |
| 8K | 3 | 4321 | 8,397 | 132.9 | 11.40 | 63.21 | 795.32 | 4,931 / 7,851 = 62.8% | 2.54 | 99.5 | 8,351 | stop | 625cadd4 |
| 8K | 4 | 4321 | 8,397 | 132.1 | 11.90 | 63.56 | 502.38 | 2,674 / 4,711 = 56.8% | 2.56 | 98.9 | 5,206 | stop | 32ee85ea |
| 16K | 1 | 1234 | 16,800 | 102.9 | 11.30 | 163.33 | 274.58 | 763 / 1,113 = 68.6% | 2.60 | 93.1 | 1,260 | stop | 94cbd308 |
| 16K | 2 | 1234 | 16,800 | 102.1 | 11.00 | 164.52 | 419.93 | 1,547 / 2,545 = 60.8% | 2.52 | 93.3 | 2,807 | stop | d568f01f |
| 16K | 3 | 4321 | 16,800 | 102.8 | 11.60 | 163.37 | 715.31 | 3,429 / 6,434 = 53.3% | 2.47 | 96.8 | 6,413 | stop | 4315fd59 |
| 16K | 4 | 4321 | 16,800 | 102.9 | 10.50 | 163.30 | 612.74 | 2,453 / 4,095 = 59.9% | 2.39 | 93.2 | 4,723 | stop | 302a8f10 |
| 32K | 1 | 1234 | 33,591 | 54.9 | 6.50 | 611.55 | 1,738.18 | 2,306 / 3,576 = 64.5% | 2.40 | 87.7 | 7,318 | stop | 0c7ba5c8 |
| 32K | 2 | 1234 | 33,591 | 54.4 | 5.10 | 617.98 | 1,848.78 | 388 / 720 = 53.9% | 2.27 | 86.2 | 6,244 | stop | a77d7059 |
| 32K | 3 | 4321 | 33,591 | 54.6 | 8.90 | 614.80 | 864.62 | 1,253 / 2,033 = 61.6% | 2.51 | 87.3 | 2,232 | stop | 37cac747 |
| 32K | 4 | 4321 | 33,591 | 54.6 | 9.80 | 615.52 | 720.31 | 623 / 870 = 71.6% | 2.59 | 86.2 | 1,024 | stop | 2fcf4052 |

The sweep's memory, each mode's 4 runs at each size. Pressure is the macOS memory pressure level, and 1 is normal. The
sweep's record has no values for the bank's slot rows or the budget at each size, and no pageout count.

| Mode | Size | Prompt tokens | Trims / over budget | Peak footprint GB (max) | Least free RAM GB (min) | Pressure (max) |
|---|---|---|---|---|---|---|
| greedy | 1K | 1,058 | 0 / 0 | 100.1 | 0.664 | 1 |
| greedy | 2K | 2,093 | 0 / 0 | 99.6 | 0.701 | 1 |
| greedy | 4K | 4,181 | 0 / 0 | 98.9 | 2.515 | 1 |
| greedy | 8K | 8,397 | 0 / 0 | 98.9 | 2.975 | 1 |
| greedy | 16K | 16,800 | 0 / 0 | 96.9 | 0.871 | 1 |
| greedy | 32K | 33,591 | 0 / 0 | 105.6 | 3.752 | 1 |
| card | 1K | 1,058 | 0 / 0 | 101.0 | 0.993 | 1 |
| card | 2K | 2,093 | 0 / 0 | 101.1 | 0.819 | 1 |
| card | 4K | 4,181 | 0 / 0 | 100.2 | 1.246 | 1 |
| card | 8K | 8,397 | 0 / 0 | 99.5 | 1.564 | 1 |
| card | 16K | 16,800 | 0 / 0 | 96.8 | 2.143 | 1 |
| card | 32K | 33,591 | 0 / 0 | 87.7 | 15.538 | 1 |

</details>

We did not measure the standard test on main 7b13376d alone, or its speed without a `bank-fit.json`. We did not
measure the speed of #1205's head 48c886ee. We did not serve the 2.10bpw pack, 64K or 128K contexts, or two requests at
once. We did not test any Mac other than ours.

## Known limits

- Prefill is slow without #1205, which is open, not merged. mlx-serve caps one MLA layer's attention scores at 4 GiB.
  For Ling's 64 heads, that cap narrows the prompt chunk to 1,536 tokens at 18,380 tokens, and to 1,024 near 32,768.
  Each chunk reads most experts again ("Measured"). The old plugin ran each layer over the whole prompt, as #1205 does.
- A long reply near the context limit can exceed the budget. mlx-serve reserves KV cache for the prompt plus one prompt
  chunk of reply (1,536 tokens for an 18K prompt). A longer reply makes mlx-serve grow the cache, and during that copy
  the old and new buffers exist together. Without a clamp, the overrun at `--ctx-size 32768` would be about 5.5 GB,
  calculated from the code. #1192's KV-grow clamp (its commit f4ccce40, in main through the squash 23590f50) stops a
  grow at the context plus the widest verify. By calculation, that clamp lowers the overrun to about 2.7 GB. We did not
  measure either number or what happens past the budget.
- The context window is 262,144 tokens (256K). The pack's `config.json` sets it. The prompt and the reply must fit in
  it together.
- The fitter is not public ("The bank's memory, and bank-fit.json").

<details>
<summary>Refusals and log lines</summary>

| When | Message | Cause |
|---|---|---|
| Parse or load | `Exl3CodebookUnsupported` | A codebook other than mul1 |
| Load | `Exl3PackNotRepacked` | A folder that lacks `experts.bin`, `expert-manifest-v2.json` or `expert-manifest.json`. Run the converter. |
| Load | `LingBankDir`, `LingBankManifest`, `LingManifestJson`, `LingManifestFormat`, `LingManifestLayer`, `LingManifestSegment`, `LingBankSidecar`, `LingBankSidecarShort`, `LingBankLayers`, `LingBankGeometry` | The bank cannot open the pack's folder, or `expert-manifest-v2.json` or `experts.bin` is damaged, is short, or does not match the config. Run the converter again. |
| Parse or load | `LingClampListLength`, `LingClampValue`, `LingConfigUnreadable` | A config that is not Ling-3.1-flash's |
| Load | `LingClampDisagrees` | mlx-serve's clamp for a layer differs from the config's |
| Load | `turboexl3_ling: an M5-class GPU (NAX) is needed ...` | Another GPU |
| Load | `turboexl3_ling: the server's host table is older than this plugin's kit` | An mlx-serve whose host table is older than the plugin kit's |
| Load | `StreamedRowsDoNotFit` | No slot row fits. Lower `--ctx-size` or raise the ceiling. |
| Prompt | `LingPromptUnderBase` | The GPU memory left under the ceiling during the prompt pass is less than one slot row. |
| Decode | `LingDecodeUnderBase` | At decode, the memory left under the ceiling, less 4 GiB kept free, holds more than 12 slot rows fewer than the prompt pass held. Within 12 slot rows, the plugin logs a warning, and the bank keeps the prompt pass's slot rows. |

Each request prints `[stream-stats]` lines with the bank's hits and bytes read in each phase.

</details>

## Credits

- inclusionAI made Ling-3.1-flash: huggingface.co/inclusionAI/Ling-3.1-flash.
- turboderp made exllamav3 (MIT), the EXL3 format, and the EXL3 weights that the pack holds:
  huggingface.co/turboderp/Ling-3.1-flash-exl3.
- mlx-serve (David Dalcu, MIT) has the `bailing_hybrid` model, the plugin kit and its expert stream.
- MLX is Apple's (MIT). turboexl3 is github.com/davidtai/turboexl3 (MIT).

## Terms

| Term | Meaning |
|---|---|
| Prefill, decode | Prefill is the rate at which the server reads the prompt. Decode is the rate at which it writes the reply. |
| Bank, slot row | The bank is the GPU memory that holds routed experts read from the SSD. A slot row holds one expert of each of the 68 routed layers. |
| Budget | The memory that mlx-serve gives the bank in one phase (the prompt pass or decode) |
| Bill | mlx-serve's upper-bound estimate of a phase's memory |
| Fit, `bank-fit.json` | A pack's measured memory use per phase, which replaces the bills |
| Ceiling | The GPU memory that mlx-serve plans to, set by `MLX_SERVE_GPU_CEILING_MB` |
| Non-file memory | Memory in use that is not file cache: wired + anonymous - purgeable + compressor |
| Trunk | Every weight except the routed experts and the MTP layer |
| KDA, MLA | Ling's two attention kinds: Kimi Delta Attention (linear) and multi-head latent attention |
| MTP layer | Multi-token prediction: the extra layer that drafts tokens for the verify pass |
| Plain decoding | Decoding one token at a time, with no drafts (`--no-mtp --no-pld`). PLD is prompt-lookup decoding, mlx-serve's other source of drafts. |
| Exact verify | A verify pass in which each row gives the same bytes as plain decoding |
| Whole-prompt pass | A prompt pass that runs each layer over the whole prompt before the next layer. mlx-serve#1205 adds it for streaming packs, and its commits call it span-major. |
| MTP acceptance | Accepted draft tokens over drafted tokens |
| Peak footprint | The server's peak memory, in GB |
| Text | The first 8 hex digits of the reply's sha256 |
| GB, GiB | GB is 10^9 bytes. GiB is 2^30 bytes. |
