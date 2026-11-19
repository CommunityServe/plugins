# turboexl3_glm53f

This quant plugin lets mlx-serve's built-in `glm5_next` serve GLM-5.3-Flash's exllamav3 (EXL3) packs in the streaming
repack of turboderp's self-calibrated branches (https://huggingface.co/turboderp/GLM-5.3-Flash-exl3). mlx-serve keeps
the model: its forward, MTP, the prefix cache and the bill. The plugin does only these jobs:

- It claims the pack's `quantization_config` (`quant_method` exl3) when the config's `expert_streaming` block has
  `family` `turboexl3` (config.zig). The plugin claims no other pack.
- It loads each trellis linear of the trunk (`load_tensor`). turboexl3's `linear` runs them (`matmul`).
- It reads the MTP layer's routed experts from `mtp-experts.bin` as one resident bank (`load_bank`, src/mtp_bank.zig).
  turboexl3's `moeClamped` runs them (`swiglu`).
- It streams the routed layers' experts from `experts.bin` (`bank_rows`, `open_bank`). The bank is the GLM-5.3-Flash
  EXL3 streamer in src/streamed/ and src/exl3/ (module `mlx_stream`). It runs over the kit's expert stream
  (`kit.streamed`) and its read pool. The trunk gives the bank each routed layer's router, and the bank uses it to
  predict the next reads.

## Needs

- mlx-serve main at commit 23590f50 (pull request 1192) or later, which lets `glm5_next` serve a pack that an outside
  quant plugin claims. Older builds refuse the pack's `expert_streaming` block by name.
- An M5-class GPU (NAX). On any other GPU, the plugin refuses the pack by name.
- The mul1 codebook. The plugin refuses any other codebook by name.

## Switches

The plugin reads 24 `TURBOEXL3_GLM53F_*` switches at load (src/options.zig). Set a switch as an environment variable
of the mlx-serve process, or on the app's plugin page. A value saved on the plugin page replaces the environment
variable, so clear or set it on the plugin page too. The kit exports the switches as the plugin's options, and the
server applies them when it starts, so restart the server after a change. Each default is the value of the served
configuration.

Write `0` or `1` for an on/off switch. Any other value, such as `off` or `false`, makes the plugin refuse the load.
This differs from glm53f, which treats such values as on. `PROMPT_KEEP` and `ADOPT_PIECES` need a seam in the kit's
stream, and `SHARED_HOIST` needs the bank ABI's `rows_hoist`. If the kit has no such seam, they default to off, and
the plugin refuses the bank if one is set.

`SHARED_HOIST` (on): the trunk's MoE builds a decode call's shared expert before the routed call and hands it to the
bank, which commits it behind the call's hit wave when the call misses, so the GPU runs it while the misses load. The
same ops run, and the trunk adds it as before.

The model's own speed changes (the KDA and DSA kernels, the fused router, the split prompt forward and others) are not
in this plugin. They are `GLM53F_*` switches of the glm53f runner plugin.

## Exactness

All of this plugin's switches are exact except `TURBOEXL3_GLM53F_PF_INT8`, which is off (0) by default. They are
exact at the values the plugin accepts, with `PG2` at its default table (note 1). An exact switch here computes the
same bytes as this plugin with that switch at 0. glm53f's README defines the three classes (exact, rounding-class and
inexact), defines stock, and explains why the default text differs from stock. This plugin has no rounding-class
switch, and its defaults need no change for stock bytes.

### How to get each mode

- For the default, set nothing.
- For stock bytes, follow "How to get each mode" in glm53f's README.
- For int8 prefill, set `TURBOEXL3_GLM53F_PF_INT8` to 1, 2, 3 or 4 (see the levels below).

The default stack is both plugins' switches at their defaults. Here, every exact switch is on except `DEMAND_HELP`,
and `PF_INT8` is at 0. Its KDA prompt routes depend on the GPU (glm53f's README, "Which GPUs the three switches
change").

| Switches (`TURBOEXL3_GLM53F_` + name) | Class | Default | Evidence |
|---|---|---|---|
| `PF_ROWS`, `DEC_BF16`, `MOE_COMBINE1`, `ROUTE_VIEW`, `HOST_ZERO` | exact | 1 | Fewer launches or copies for the same values. turboexl3's own tests check `PF_ROWS` and `DEC_BF16` against the path without them. Same greedy text in both arms of each switch's standard A/B |
| `PF_FEED`, `PF_ALIGN`, `PF_STAGE`, `PF_DRAIN_LAG`, `PF_REUSE`, `SHARED_HOIST` | exact | 1 (`SHARED_HOIST`: 0 on a kit without `rows_hoist`) | Schedules of the same arithmetic. turboexl3's tests check `PF_ALIGN`. Same greedy text in each switch's standard A/B |
| `BANKED`, `DEC`, `PF_OPTS`, `PG2` | exact (note 1) | `g4m3.t4.t4`, `g1`, `bk128`, `50115151111` | Schedules or tiles of the same arithmetic. turboexl3's tests check `BANKED` and `DEC`. Same greedy text in each switch's standard A/B |
| `LAYER_BILL`, `GROW_UNFILLED`, `PROMPT_KEEP`, `SPEC_PICK1`, `ADOPT_PIECES`, `SEED_DECODE`, `KEEP_WARM` | exact | 1 (`PROMPT_KEEP` and `ADOPT_PIECES`: 0 on a kit without their seam) | They change memory, reads and residency, or keep the GPU warm during the stream's waits (`KEEP_WARM`), not the math. Same greedy text in each switch's standard A/B |
| `DEMAND_HELP` | exact | 0 | Takes 0 or 2. At 2, decode's read pool serves demand reads first, and a waiting demand read helps finish its own record. Only the order of reads changes. Off because its A/B decoded slower (note 2) |
| `PF_INT8` | inexact | 0 | Int8 prefill GEMMs. See the levels below |

1. `BANKED`, `DEC` and `PF_OPTS` take only 0 or their default value; any other value refuses the load. `PG2` takes
   other tile-code tables, and no run has checked those for exactness.
2. `DEMAND_HELP`'s standard A/B decoded at 40.58 tok/s at 0 and 39.30 tok/s at 2 (arm means, four runs an arm). It ran
   on 2026-11-12 on an earlier combined build, on the 40-core M5 Max. Both arms wrote the same text.

At levels 1 to 4, some prefill GEMMs read the activations as int8. Each such GEMM decodes each weight's trellis to
int8 and sums the int8 products in int32. The levels are not cumulative. Levels 1 and 2 put the routed experts' GEMMs
on int8, with one scale per 128-value activation block at level 1 and one per row at level 2. Levels 3 and 4 are
level 1 plus trunk linears on int8: the two KDA projections at level 3, and all 11 at level 4. A value other than 0
to 4 refuses the load by name.

| Level | Routed experts' gate/up and down GEMMs | Trunk linears at prefill rows | Prefill tok/s, standard prompt | Prefill tok/s, 16K sweep point | Decode tok/s, 16K sweep point | HumanEval pass@1 |
|---|---|---|---|---|---|---|
| 0 (default) | exact | exact | 1,203 to 1,206 | 1,236 | 40.90 | 152 / 164 |
| 1 | int8, one scale per 128-value activation block | exact | 1,340 | 1,381 | 43.45 | 152 / 164 |
| 2 | int8, one scale per activation row | exact | Not measured | 1,423 | 43.00 | Not measured |
| 3 | as level 1 | KDA `qkv_proj` and `o_proj` on int8, one scale per row | 1,511 | 1,546 | 42.90 | 155 / 164 |
| 4 | as level 1 | all 11 on int8, one scale per row: the KDA, DSA and indexer projections, and the shared experts' and dense layers' MLPs | 1,603 | 1,669 | 40.85 | 152 / 164 |

All of these numbers come from an earlier combined build of turboexl3_glm53f, which also held the model's speed
changes before they moved to glm53f. Its level 0 ran glm53f's three rounding-class kernels and wrote text ID
`c3c346222307` on the standard prompt. That is the first 12 digits of the 2026-11-18 default run's ID. All runs used
an Apple M5 Max (40-core GPU, 128 GB) with the sc_3.10bpw pack, greedy. The about 922 tok/s in glm53f's README come
from another build, so do not compare them with level 0 here.

- Standard prompt (glm53f's README defines it), 2026-11-10: each level ran its own A/B against level 0, four runs an
  arm. Level 0 measured 1,203, 1,206 and 1,206 tok/s in the A/Bs of levels 1, 3 and 4.
- 16K sweep point: the 16K point of a 1K to 256K sweep on code prompts, 2026-11-12, two runs a level.
- HumanEval: 164 problems, thinking off, `max_tokens` 1,024, scored on each reply's last code block, one greedy run a
  level, 2026-11-11 and 2026-11-12. A gap of three problems in one run is too small to rank the levels.

No level changes the decode kernels, yet the decode column differs between levels. That column comes from the 16K
sweep point, which used another prompt, and this README records no texts for it. A different greedy text changes how
often the MTP head's drafts are accepted, so decode speed can differ with the same kernels. On the standard prompt,
levels 1, 3 and 4 each wrote a different text from level 0.

<details><summary>Run log entries</summary>

- Standard prompt A/Bs of levels 1, 3 and 4: g53i8s1, g53q8s1 and g53q8s4.
- `DEMAND_HELP` (note 2): g53g2h1.

</details>

### Not measured as of 2026-11-19

- `PF_INT8` at any level on these two plugins.
- Level 2 on the standard prompt, and level 2's HumanEval score.
- `PG2` tables other than the default.
- The items in glm53f's "Not measured" list, which cover this plugin's defaults too.

## Build and check

    scripts/build.sh turboexl3_glm53f                    # from the repo root: tests, then ReleaseFast
    scripts/install.sh turboexl3_glm53f --home ~/tmp/h   # under a throwaway HOME

`zig build test` runs the MLX-free tests: the claim and the pack check, the bank's row plan, the switches, and the
MTP bank's manifest. It also runs the tile gate, which compares turboexl3's tile decode with exllamav3's rule on real
GLM-5.3-Flash tiles at each routed rate (src/frac_test.zig, src/glm_tiles.json).

At load, the log names the plugin: `[plugins] loaded turboexl3_glm53f (quant, sha256 ...)`. It also prints the
bank's rows: `glm5_next: streamed experts, N prompt / M decode slot rows per layer (R B the widest row): ...`.
Each bank switch that is on prints its own line, and `bank bytes: prompt P B, decode D B` gives the bytes the rows
bill in each phase. With a server whose bank ABI has `meters`, each request's summary ends with two `[stream-stats]`
lines (prompt and decode): the stream's routes, hits, misses, reads and lookahead in that phase.
