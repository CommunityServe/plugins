# glm53f

GLM-5.3-Flash's runner for the turboexl3 EXL3 streaming repacks
([OpensourceWTF/GLM-5.3-Flash-streaming-repack-exl3](https://huggingface.co/OpensourceWTF/GLM-5.3-Flash-streaming-repack-exl3)),
with the speed changes of mlx-serve #1196.

- The KDA-hybrid trunk (loader, KDA layer, slots, table glue) is the plugin kit's (`kit.bases.kda_hybrid`).
- This plugin holds GLM's own forward: the hyper-connections, the sparse attention and the MTP head.
- It fills the trunk's hooks with the speed changes: the NAX router (`Hooks.router`) and the KDA stages
  (`KdaConsts.stages`).
- The pack's EXL3 linears and its routed-expert bank belong to the **turboexl3_glm53f** quant plugin, which must be
  installed too. This plugin reaches it through the host (`Host.quant_claimant`).

Its parse asks for the engine's own glm5_next config (`Parsed.engine_config`), so the server renames the pack's
weights, sizes its bank and bills its prompt and decode as it does for mlx-serve's own glm5_next. A server without
that field loads no pack through this plugin.

It claims a pack by its markers: a `glm5_next` config with exllamav3's `quantization_config`, an `expert_streaming`
block of family `turboexl3`, and the pack's `experts.bin`. mlx-serve's own glm5_next keeps every other GLM pack.
Text only: image rows are refused by name.

## Switches

Each switch is a plugin option, `GLM53F_<name>`. A value is on unless it is empty or starts with `0`. The defaults are
#1196's.

| Option | Default | What it does | Exact? |
|---|---|---|---|
| `KDA_VERIFY` | on | Runs a KDA verify's prework, bounded gate and norm-gate in one launch each | the chain's bytes |
| `KDA_EXPSIG` | on | With `KDA_VERIFY`: makes exp(A_log) inside the gate's launch | the same bits |
| `KDA_LAZY_SEQ` | on | Makes a verify's per-row states in a launch that MLX runs only when a rollback reads them | the same bits |
| `KDA_PROMPT_PROLOGUE` | on | Runs a prompt's conv, SiLU, q/k norms and scales in one launch | the chain's bytes |
| `KDA_PROMPT_NORMGATE` | on | Runs a prompt's norm-gate epilogue in one launch | the chain's bytes |
| `KDA_PROMPT_PERCORE` | off | Runs the prompt recurrence at two threadgroups a core | rounding-class |
| `KDA_PROMPT_CHUNK` | off | Runs the prompt recurrence as the chunkwise delta rule | rounding-class |
| `DSA_PROMPT_INDEXER` | on | Runs the prompt indexer's scores, top-k and expansion in one launch, and writes the output in one copy | the same tokens and bytes |
| `DSA_PROMPT_ONEP` | off | Meets the prompt's sparse attention values with P as one fp16 piece | rounding-class |
| `DSA_TAILS` | on | Reads a window's indexer tail rows first, so each cache append reuses its buffer | the same rows |
| `HEAD_LAST_ROW` | on | Runs the MTP head's routed experts for the last row only | that row's bytes |
| `PROMPT_ASYNC_EVAL` | on | Sends a prompt's cadence evals to the GPU without the host waiting | the same arrays |
| `ROUTER_NAX` | on | Routes 2 to 4 rows on MLX's split-K NAX chains over the stored router | the stock route's bytes |
| `ROUTER_ONE1` | on | With `ROUTER_NAX`: runs a one-row router call in one launch | the stock route's bytes |
| `WHOLE_PROMPT` | off | Takes the whole prompt in one forward (J5); runs the last chunk and the held-back tail (17 to 31 rows) layer by layer, their routed experts in one bank call | each span's bytes (not yet A/B'd) |

If `KDA_PROMPT_CHUNK` or `KDA_PROMPT_PERCORE` is set and its route cannot serve this GPU or the pack's heads, the
load is refused by name. When a route runs, its first use logs an `engaged` line.

## Build and test

    scripts/build.sh glm53f
    cd plugins/glm53f && ../../.zig-toolchain/zig build mlx-free-test        # MLX-free: claim, parse, options, plans
    ../../.zig-toolchain/zig build gpu-test -Dmlx-lib=<the server's lib dir>  # the speed changes' bytes tests, Metal

The GPU tests compare each speed change with the stock arm it replaces. The exact switches must match bit for bit;
the rounding-class ones must stay within their bound.

Two switches have no GPU test: `HEAD_LAST_ROW` and `PROMPT_ASYNC_EVAL`. The standard A/B's same-greedy-text check
covers them.

## Known gaps

- **The bill is glm5_next's**, so the published bank fits stay valid. Two copies are not billed:
  - the NAX router's stored copy (`GLM53F_ROUTER_NAX`): 288 x 4096 bf16, 2.25 MiB a routed layer;
  - the trunk's f32 router table, as in glm5_next.

  The measured bank fits include both, because they were measured on #1196's stack.
- **The whole-prompt pass is off.** `WHOLE_PROMPT` (src/prompt.zig) needs a server with mlx-serve's E3 and E7
  (main e285ab55). With it off, the prompt runs in the engine's chunks, as glm5_next's does, so #1196's split-prompt
  gain (prefill 804.9 to 887.4 tok/s in g53pfs1) is not here. Its A/B decides the default. It joins only with the
  prefix cache on (with it off, the last span is one row). It has no GPU test: the kit ships no harness that loads a
  pack and its caches, so the A/B's same greedy text (switch off against on) is its gate.
- **No GPU tests** for `HEAD_LAST_ROW` (its arm is the kit's `moe.mlpLastRow`, tested in mlx-serve, and it needs a
  format bank) and `PROMPT_ASYNC_EVAL` (the same arrays go to `mlx_async_eval`). The standard A/B's same-greedy-text
  check covers both.
- **Not measured on this plugin yet:** speed and greedy text. The standard A/B (16K coding cell, ABBA) against
  #1196's stack and against stock mlx-serve with turboexl3_glm53f decides both.
