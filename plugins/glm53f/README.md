# glm53f

GLM-5.3-Flash's runner for the turboexl3 EXL3 streaming repacks
([OpensourceWTF/GLM-5.3-Flash-streaming-repack-exl3](https://huggingface.co/OpensourceWTF/GLM-5.3-Flash-streaming-repack-exl3)),
with the speed changes first written in
[mlx-serve pull request 1196](https://github.com/ddalcu/mlx-serve/pull/1196). That pull request closed without a merge
on 2026-11-15, and its speed changes moved to this plugin.

- The KDA-hybrid trunk (loader, KDA layer, slots, table glue) is the plugin kit's (`kit.bases.kda_hybrid`).
- This plugin holds GLM's own forward: the hyper-connections, the sparse attention and the MTP head. The MTP head is a
  small extra layer that proposes (drafts) the next tokens; the model then checks the drafts in one pass.
- It fills the trunk's hooks with the speed changes: the NAX router (`Hooks.router`) and the KDA stages
  (`KdaConsts.stages`).
- The pack's EXL3 linears and its routed-expert bank belong to the **turboexl3_glm53f** quant plugin, which must be
  installed too. This plugin reaches it through the host (`Host.quant_claimant`).

Its parse asks for the engine's own glm5_next config (`Parsed.engine_config`), so the server renames the pack's
weights, sizes its bank and bills its prompt and decode as it does for the built-in glm5_next. A server without
that field loads no pack through this plugin.

It claims a pack by its markers: a `glm5_next` config with exllamav3's `quantization_config`, an `expert_streaming`
block of family `turboexl3`, and the pack's `experts.bin`. The built-in glm5_next keeps every other GLM pack.
The plugin serves text only. It refuses a request's image rows by name.


## Switches

Each switch is a plugin option named `GLM53F_<name>`. Set it as an environment variable of the mlx-serve process, or
on the app's plugin page. The plugin reads each switch once, so restart the server after a change.

To turn a switch off, write `0`. Any value that is not empty and does not start with `0` turns the switch on. So in an
environment variable, `off` and `false` turn the switch on. The app's plugin page accepts only `0` and `1`.

Every switch is on by default. Each one stayed because its standard A/B showed it faster. A standard A/B runs the
standard prompt (see Exactness) eight times, each on a fresh server: four runs with the switch off (arm A) and four
with it on (arm B). An exact switch also had to write the same text in both arms.

| Option | What it does | Class |
|---|---|---|
| `KDA_VERIFY` | Runs a KDA verify's prework, bounded gate and norm-gate in one launch each | exact |
| `KDA_EXPSIG` | With `KDA_VERIFY` on, computes exp(A_log) inside the gate's launch | exact |
| `KDA_LAZY_SEQ` | Makes a verify's per-row states in a launch that MLX runs only when a rollback reads them | exact |
| `KDA_PROMPT_PROLOGUE` | Runs a prompt's conv, SiLU, q/k norms and scales in one launch | exact |
| `KDA_PROMPT_NORMGATE` | Runs a prompt's norm-gate epilogue in one launch | exact |
| `KDA_PROMPT_PERCORE` | Runs the prompt recurrence on the kit's per-core kernel at up to two threadgroups a core | rounding-class |
| `KDA_PROMPT_CHUNK` | Runs the prompt recurrence as the chunkwise delta rule | rounding-class |
| `DSA_PROMPT_INDEXER` | Runs the prompt indexer's scores, top-k and expansion in one launch, and writes the output in one copy | exact |
| `DSA_PROMPT_ONEP` | Rounds the prompt's sparse attention weights (P) to one fp16 piece, not two, before it multiplies them by the values | rounding-class |
| `DSA_TAILS` | Reads a window's indexer tail rows first, so each cache append reuses its buffer | exact |
| `HEAD_LAST_ROW` | Runs the MTP head's routed experts for the last row only | exact |
| `PROMPT_ASYNC_EVAL` | Sends a prompt's periodic evaluations to the GPU without making the host wait | exact |
| `ROUTER_NAX` | Routes 2 to 4 rows on MLX's split-K NAX kernels over a stored copy of the router | exact |
| `ROUTER_ONE1` | With `ROUTER_NAX` on, runs a one-row router call in one launch | exact |
| `WHOLE_PROMPT` | Takes the whole prompt in one forward pass. It runs the last chunk and the held-back tail (17 to 31 rows) layer by layer, with each layer's routed experts in one bank call | exact |
| `PF_REUSE` | Keeps the buffers of a prompt's KDA runs in MLX's cache from a run's first layer to its last | exact |
| `DSA_HOLD` | Keeps a prompt DSA layer's q_abs and o_lat for the next DSA layer, which reuses their buffers | exact |
| `TRUNK_F16_FREE` | Frees the trunk's f16 matrices once their bf16 copies are evaluated, before the bank opens, so the bank's slot rows take those bytes (about 0.65 GB on sc_3.10) | exact |

Exactness, below, defines the classes and stock.

`PF_REUSE` raises MLX's cache limit by the bill's layer envelope for each row, and puts the limit back at the next DSA
layer. turboexl3_glm53f's `TURBOEXL3_GLM53F_PF_REUSE` is its routed-prefill half. `DSA_HOLD` trades memory for fewer
allocations: the two held arrays take 2 x 0.94 GB at a 16K prompt.

Below 64 GPU cores, a KDA layer's prompt recurrence takes the first route that can run. The order is the chunkwise
route (`KDA_PROMPT_CHUNK`), then the kit's per-core kernel at up to two threadgroups a core (`KDA_PROMPT_PERCORE`),
then the kit's stock per-token kernel. Where the kit's stock one-per-core kernel launches, which needs 64 cores or
more, that kernel takes it ("Which GPUs the three switches change", in Exactness, explains why). The KDA routes run
only on prompt spans of 17 rows or more at batch 1, and `DSA_PROMPT_ONEP` only on passes of 17 rows or more with more
than 8 sparse attention rows.

At load, at every core count, the plugin checks whether each of the two routes can run on this GPU at the pack's KDA
heads. When a route cannot run and you did not set its switch, the plugin turns the route off. It then logs an info
line that names the switch, says "on by default" and ends "the KDA prompt takes the next route".

You set a switch when its environment variable exists, whatever its value, or when the app's plugin page saved a value
for it. When you set a route's switch on, including to `1`, and the route cannot run, the plugin refuses the load. Its
error names the switch.

Each route logs one line at its first use, so the server log shows which route ran:

- `[kda] prompt chunk engaged` for `KDA_PROMPT_CHUNK`;
- `[kda] prompt percore engaged` for `KDA_PROMPT_PERCORE`;
- `[kda] per-core prefill recurrence engaged` for the kit's stock one-per-core kernel;
- `[glm5_next] DSA one-pass engaged` for `DSA_PROMPT_ONEP`.

## Exactness

Each switch belongs to one of three classes.

| Class | Meaning | Switches |
|---|---|---|
| Exact | Computes the same bytes as the stock code it replaces | 15 of glm53f's 18 switches (all but the three below). Every turboexl3_glm53f switch except `PF_INT8`, at the values that plugin accepts, with `PG2` at its default table |
| Rounding-class | Rounds differently from stock, uses no int8, and passed the NLL floor test | `KDA_PROMPT_CHUNK`, `KDA_PROMPT_PERCORE` and `DSA_PROMPT_ONEP` |
| Inexact | Uses lower-precision (int8) math | None in glm53f. turboexl3_glm53f's `TURBOEXL3_GLM53F_PF_INT8`, off by default |

A rounding-class switch adds values in another order, rounds an intermediate value once more, or uses Metal's fast
`exp` where stock uses the precise one. Stock means mlx-serve's built-in glm5_next forward, run with turboexl3_glm53f
at its defaults (`PF_INT8` at 0) for the pack's EXL3 kernels and expert bank. turboexl3_glm53f's exact switches change
no bytes, so stock is the same at any of their settings.

The default does not compute the same bytes as stock, because it runs rounding-class switches in the prompt pass. The
prompt pass fills the KDA state and the attention caches that decode reads. A tiny difference there can change which
token wins a near tie, and the text differs from that token on.

The text IDs below come from the standard prompt, `coding.json`. It is one coding request of 16,387 tokens, answered
greedily (the model always takes the most likely token) up to 1,024 tokens. A text ID is a short hash of the reply,
so two runs with the same ID wrote the same text.

### How to get each mode

Set the variables as environment variables of the mlx-serve process, or on the app's plugin page. A value saved on
the plugin page replaces the environment variable, so clear or set it on the plugin page too. Then restart the server.

| Mode | Settings | Greedy text ID (40-core M5 Max, sc_3.10bpw pack) |
|---|---|---|
| Default | None | `c3c34622230749d9` (note 1) |
| Stock bytes | `GLM53F_KDA_PROMPT_PERCORE=0`, `GLM53F_KDA_PROMPT_CHUNK=0` and `GLM53F_DSA_PROMPT_ONEP=0` (note 3) | `b1739c1005558f6f` (note 2) |
| Int8 prefill | The default, plus `TURBOEXL3_GLM53F_PF_INT8` at 1, 2, 3 or 4 | Differs from the default at levels 1, 3 and 4 (earlier combined build, 2026-11-10; not run on these plugins). Level 2: Not measured. See turboexl3_glm53f's README |

1. A run on 2026-11-18 measured this ID with the defaults of that day. Five exact switches, six variables, joined
   the default later. `PF_REUSE` is two variables in two plugins. The six are glm53f's `PF_REUSE` and `DSA_HOLD`, and
   turboexl3_glm53f's `PF_REUSE`, `HOST_ZERO`, `MOE_COMBINE1` and `ROUTE_VIEW`. Each wrote this text in all 8 runs of
   its own A/B. No run has checked the current default.
2. The built-in glm5_next, with turboexl3_glm53f, wrote this text in 8 of 8 runs on 2026-11-16. In the same run,
   glm53f with the three switches at 0 wrote it too.
3. Set all three. Each one alone changed the text in its A/B. With only `KDA_PROMPT_CHUNK` at 0, on a GPU of 32 to
   63 cores where its test launch passes, `KDA_PROMPT_PERCORE` takes the KDA layers.

For example, for stock bytes:

    GLM53F_KDA_PROMPT_PERCORE=0 GLM53F_KDA_PROMPT_CHUNK=0 GLM53F_DSA_PROMPT_ONEP=0 <your mlx-serve command>

The sc_3.10bpw pack is a branch of the HF repo. Another GPU can write another text.

<details><summary>The standard prompt and the text ID in detail</summary>

The standard prompt is `coding.json`: one user message of 16,387 tokens, `max_tokens` 1,024, temperature 0. A text ID
is the first 16 hex digits of a SHA-256 over the reply's reasoning text, a NUL byte, then its content text. The GLM
bench harness (`ladder.py`) computes it.

</details>

### Which GPUs the three switches change

The stock KDA prompt kernel depends on the GPU's core count. For GLM-5.3-Flash's KDA layers, the kit's stock
one-per-core kernel needs 64 cores. `KDA_PROMPT_PERCORE` runs the same kernel at up to two threadgroups a core, so it
needs 32. Below 64 cores, the stock route is the kit's per-token kernel. Where the kit's one-per-core kernel launches,
the trunk calls none of this plugin's KDA prompt code, so the four `KDA_PROMPT_*` switches do not run there.

| GPU cores | Stock KDA prompt kernel | `KDA_PROMPT_CHUNK` | `KDA_PROMPT_PERCORE` | Rounding-class switches the default runs |
|---|---|---|---|---|
| Fewer than 32 | The kit's per-token kernel | Runs where its load check passes | Fails its load check (no plan) | `KDA_PROMPT_CHUNK` (expected) and `DSA_PROMPT_ONEP` |
| 32 to 63, such as the 40-core M5 Max | The kit's per-token kernel | Runs where its load check passes | Runs where its load check passes and `KDA_PROMPT_CHUNK` is 0 or turned off | `KDA_PROMPT_CHUNK` and `DSA_PROMPT_ONEP` |
| 64 or more | The kit's stock one-per-core kernel | Not used where the kit's one-per-core kernel launches (the kit tries one test launch first) | Not used where the kit's one-per-core kernel launches (the kit tries one test launch first) | `DSA_PROMPT_ONEP` |

The plugin runs both load checks at every core count. So a switch you set can refuse the load (see Switches) even
where its route goes unused.

`DSA_PROMPT_ONEP` needs NAX. turboexl3_glm53f needs an M5-class (NAX) GPU, so every row assumes NAX. Only the 40-core
M5 Max was measured. The other rows come from the code.

<details><summary>What the three switches change</summary>

GLM-5.3-Flash's KDA layers have 64 value heads of 128 rows, so 64 x 128 = 8,192 value rows. One threadgroup takes at
most 128 rows. The kit's stock one-per-core kernel runs one threadgroup a core, so it needs 8,192 / 128 = 64 cores. At
two threadgroups a core, the same kernel needs 32 cores.

- `KDA_PROMPT_CHUNK` runs the KDA recurrence over 16-row chunks. It solves each chunk in f32, and it reads and writes
  the state once a chunk, not once a row. Its sums run in another order.
- `KDA_PROMPT_PERCORE` runs the kit's per-core kernel at up to two threadgroups a core (80 threadgroups of 104 rows on
  40 cores). Below 64 cores the stock route is the per-token kernel, and the per-core kernel sums the delta rule in
  another order and computes the gate's sigmoid with Metal's fast `exp`, so it is rounding-class. The plan changes
  only which threadgroup takes which rows, not a row's arithmetic. At 64 cores or more, its plan is the kit's
  one-per-core plan, which is the stock route.
- `DSA_PROMPT_ONEP` rounds P to fp16 once. The stock kernel keeps each weight of P as two fp16 pieces, a high piece
  and a low piece that holds the remainder (src/dsa_nax.zig). The switch keeps only one piece. It is rounding-class,
  not inexact, because it uses no int8 and it passed the NLL floor test.

`KDA_PROMPT_CHUNK`'s load check does not depend on the core count. It checks the KDA heads, the gate's bound and one
test launch. So below 32 cores the default is expected to run it, but no run has checked this.

</details>

### Evidence

Each rounding-class switch's GPU test runs the switch and the stock path on the same inputs. For each output, the
largest absolute difference must be at most 0.01 of the stock output's largest absolute value. The KDA tests check
the outputs and the state this way, and the conv window bit for bit.

The quality check is an NLL floor test. NLL (negative log-likelihood, in nats a token) measures how well the model
predicts each real next token. Lower is better. The test compares a switch's NLL shift from E, the exact stack, with
the shift of another rounding-class switch, the floor. A switch passes when its shift is lower than the floor's, or no
more than two standard errors of the floor's shift (2 SE) above it.

The floor is not an independent reference. `DSA_PROMPT_ONEP` and `KDA_PROMPT_PERCORE` are each other's floors, and
`DSA_PROMPT_ONEP` is `KDA_PROMPT_CHUNK`'s floor too. The main evidence is the absolute size. Every shift from E is
below 0.001 nats a token, and each switch also has the GPU bound test above.

<details><summary>The NLL floor test in detail</summary>

The test scored 8 code documents of 16,384 tokens each on three servers:

- E, the exact stack: PR 1196's glm5_next with its exact speed changes on and the three switches off;
- E plus the switch under test;
- the floor, E plus another rounding-class switch.

The test scores the prompt in 256-row chunks, not in the served one-pass forward. The run log computes the SE over
blocks of rows of the 8 documents.

</details>

| Switch | NLL shift, switch (nats) | NLL shift, floor (nats) | Floor | 2 SE | Prefill tok/s, off | Prefill tok/s, on |
|---|---|---|---|---|---|---|
| `KDA_PROMPT_CHUNK` | -0.00077 | -0.00037 | `DSA_PROMPT_ONEP` | 0.00055 | 923.5 | 1,074 |
| `KDA_PROMPT_PERCORE` | -0.00038 | -0.00037 | `DSA_PROMPT_ONEP` | 0.00055 | 921.8 | 1,055 |
| `DSA_PROMPT_ONEP` | -0.00037 | -0.00038 | `KDA_PROMPT_PERCORE` | 0.00056 | 922.6 | 933.4 |

Read the NLL shifts as no measured loss, not as a gain.

These numbers come from PR 1196's glm5_next, not this plugin, on 2026-11-13 and 2026-11-14, on an Apple M5 Max
(40-core GPU, 128 GB) with the sc_3.10bpw pack. Each prefill pair is one switch alone against E, with the standard
prompt, four runs an arm. Decode speed is not compared, because the two arms write different texts. The 1,203 to
1,206 tok/s at level 0 in turboexl3_glm53f's README come from another build, so do not compare them with these.

<details><summary>Run log entries</summary>

- Stock text ID (note 2): g53plugp4, arm A.
- NLL floor tests: kfl5chb, kfl5pcb and kfl5opb.
- Prefill A/Bs: g53kp5ab_ch, g53kp5ab_pc and g53kp5ab_op.

</details>

### Not measured as of 2026-11-19

- The current default as one configuration, with all five later exact switches on (note 1), and `TRUNK_F16_FREE`
  and turboexl3_glm53f's `SHARED_HOIST`.
- The stock text with the current switch set. The 2026-11-16 run predates the five later switches.
- The prefill gain of the three switches together, against all three at 0.
- The NLL floor test and the per-switch prefill A/Bs on this plugin.
- Any GPU other than the 40-core M5 Max.
- The 1K to 256K speed sweep on these two plugins.

## Build and test

    scripts/build.sh glm53f
    cd plugins/glm53f && ../../.zig-toolchain/zig build mlx-free-test        # MLX-free: claim, parse, options, plans
    ../../.zig-toolchain/zig build gpu-test -Dmlx-lib=<the server's lib dir>  # the speed changes' bytes tests, Metal

The GPU tests compare each speed change with the stock code it replaces. Nine exact switches must match bit for bit:
`KDA_VERIFY`, `KDA_EXPSIG`, `KDA_LAZY_SEQ`, `KDA_PROMPT_PROLOGUE`, `KDA_PROMPT_NORMGATE`, `DSA_PROMPT_INDEXER`,
`DSA_TAILS`, `ROUTER_NAX` and `ROUTER_ONE1`. The three rounding-class switches must stay within the 0.01 bound (see
Exactness).

Some GPU tests skip on some GPUs. The `KDA_PROMPT_PERCORE` and `KDA_PROMPT_CHUNK` tests skip where the kit's stock
one-per-core kernel launches (64 cores or more). The `KDA_PROMPT_PERCORE` test also skips below 32 cores. The
`KDA_PROMPT_CHUNK` test skips where its test launch fails, and the `DSA_PROMPT_ONEP` test skips without NAX.

The other six exact switches have no GPU test. Each wrote the same greedy text in both arms of its standard A/B
instead.

- `HEAD_LAST_ROW` runs the kit's `moe.mlpLastRow`, which mlx-serve tests. A test here would need a format bank.
- `PROMPT_ASYNC_EVAL` sends the same arrays to `mlx_async_eval`.
- `WHOLE_PROMPT` needs a pack and its caches, and the kit ships no harness that loads them.
- `PF_REUSE` and `DSA_HOLD` change only which buffers MLX reuses, not the values.
- `TRUNK_F16_FREE` changes only when the f16 bytes are freed; the trunk reads the same bf16 arrays.

## Known gaps

- **The bill is glm5_next's**, so the published bank fits stay valid. Two copies are not billed:
  - the NAX router's stored copy (`GLM53F_ROUTER_NAX`): 288 x 4096 bf16, 2.25 MiB a routed layer;
  - the trunk's f32 router table, as in glm5_next.

  The measured bank fits include both, because they were measured with PR 1196's speed changes on.
- **The whole-prompt pass** (`WHOLE_PROMPT`, src/prompt.zig) needs mlx-serve main at commit e285ab55 or later. It
  joins only with the prefix cache on. With the prefix cache off, the last span is one row.
- **What is not measured** is listed at the end of Exactness.
- **`TRUNK_F16_FREE` moves the bank's memory.** A bank fit (`bank-fit.json`) measured with it on gives the rows its
  freed bytes; a server that turns it off under that fit peaks about 0.65 GB higher. Measure the fit with the switch
  set as served. It needs a kit with `kda_hybrid` `Hooks.free_f16_sources` and `Source.release`; without the
  engine's `Source.release` the bytes stay until the load's end and the rows do not grow.
