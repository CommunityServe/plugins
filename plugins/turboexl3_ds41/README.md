# turboexl3_ds41

Serves DeepSeek-V4.1-Flash's exllamav3 (EXL3) packs, such as the streaming repack
(https://huggingface.co/OpensourceWTF/DeepSeek-V4.1-Flash-1017-streaming-repack-exl3/tree/2.81bpw), as a quant plugin.
mlx-serve's own `deepseek_v41` serves the pack: its forward, DSpark, the prefix cache and the bill. This plugin claims
the pack's `quantization_config` (`quant_method` exl3), runs the weights the pack stores in it on turboexl3's Metal
kernels, and streams the routed experts from the repack's `experts.bin` (a4k layout included):

- every trellis linear of the trunk, the LM head and the DSpark stages (`matmul`: turboexl3's `linear`; a prompt's
  rows on its fused matmul + finish, pg2), the grouped `wo_a` as one weight (`src/woa_join.zig` at prompt rows);
- the shared expert at a verify's rows in one launch (`swiglu` with no routes, `src/shared_swiglu.zig`);
- the DSpark stages' experts, each stage's per-expert linears stacked into one bank at load (`swiglu`: turboexl3's
  `moeClamped`, the pack's `swiglu_limit`);
- the streamed bank (`bank_rows` / `open_bank`, an `abi.Bank` the trunk reads as its routed experts): the EXL3 lane
  over the kit's expert stream (`kit.streamed.expert`: slot pools, residency policy, the read pool, whose C the kit's
  `streamSources` compiles in), as this plugin's own source (module `mlx_stream`, `src/stream.zig`: `src/streamed/`
  the EXL3 bank, its source and waves over its own `sdk`; `src/exl3/` the prompt lane's kernels; `src/bank/`
  DeepSeek-V4.1's schedule over them), its decode waves on turboexl3's grouped banked decode, any layer at
  exllamav3's half-integer rates on turboexl3's MoE. MLX's internals the stream needs (events, buffer aliases, the
  launch cache, the command-buffer split) come through the host table (`src/host_shims.zig`).

The bank's prompt lane needs an M5-class GPU (NAX; elsewhere the bank is refused by name). Only the mul1 codebook is
served (an mcg pack is refused by name). Images are not served on these packs yet (the vision tower is EXL3 too). The
bank reads `TURBOEXL3_DS41_FRAC`, `TURBOEXL3_DS41_RESIDENT_SEED`, `TURBOEXL3_DS41_EXPERT_USAGE_DIR` (its options: the
app's Models ▸ Plugins page sets them) and `MLX_MAX_MB_PER_BUFFER` at its open; the old `MLX_SERVE_DSV41_*` names are
not read (a warning names any that is set).

`TURBOEXL3_DS41_PF_INT8` (an option, read at the format's open; not exact; `0`, the default: the exact prefill) puts
turboexl3's int8 prefill GEMMs on: `1` the routed experts turboexl3 serves at prompt rows (the layers at a half-integer
rate, the DSpark stages' banks, and the K 2 layers' prompt calls, which move off the streamer's DIG-X lane onto
turboexl3 under this option; decode stays on the banked decode, K 3 layers on DIG-X), int8 activations at a scale each
128-block; `3` = 1 plus the attention projections at prompt rows (wq_a, wkv and the compressors', wq_b, wo_a with
the joined run's groups, wo_b, the indexer's wq_b) at one scale a row; `4` = 3 plus every trunk linear (the shared
expert, DSpark's main_proj). `2` (one scale a row on the routed experts) is refused by name: hidden 5120 is past the 32
blocks that scheme holds. Any other value refuses the load by name. The int8 weights are rebuilt at the centered 509.5.
Decode and verify rows stay exact. A line at load names the level; turboexl3 logs an engage line where each int8 path
first runs (the joined wo_a's at its warm run, at load). turboexl3's src/models/deepseek_v41/deepseek_int8_test.zig
holds each path to the exact one at these shapes.

Measured at level 4 against the default (exact) prefill, on an M5 Max (128 GB) with turboderp's 2.25bpw streaming pack
on stock mlx-serve 9a173773, a 17,639-token coding prompt and 1,024 tokens, 4 runs an arm (ABBA, BAAB): prefill
1,618 against 1,463 tok/s (+10.6%) and decode 42.0 against 43.6 tok/s (-3.8%; DSpark accepts fewer drafts) with
greedy sampling; prefill 1,625 against 1,454 tok/s (+11.7%) and decode 38.8 against 41.4 tok/s (-6.2%) with the
card's sampling. Quality over 32 held-out documents (code, prose, chat; about 16K tokens each): perplexity +0.11% on
prompt rows, KL(exact || level 4) 0.0080 nats against 0.0061 for the exact attention path against the default flash
attention. HumanEval (164 problems, greedy, thinking off): 158 of 164 at level 4 and at the default, with one flip
each way. Levels 1 and 3 are not measured on this model; the other packs are not measured.

    scripts/build.sh turboexl3_ds41                    # from the repo root: tests, then ReleaseFast
    scripts/install.sh turboexl3_ds41 --home ~/tmp/h   # under a throwaway HOME

Needs an mlx-serve whose quant table has `open_bank` (`sdk/quant/outside.zig`). The log names the plugin at load:
`deepseek_v41: lm head on turboexl3_ds41, as shipped`, `... weights on turboexl3_ds41` and `deepseek_v41: streamed
experts, N prompt / M decode slot rows per layer`.
