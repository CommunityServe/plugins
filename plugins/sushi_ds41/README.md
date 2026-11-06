# sushi-ds41

DeepSeek-V4.1-Flash on sushi's own model code, as an outside plugin for mlx-serve. Claims only packs whose
`expert_quant.format` is `stq`; every other V4.1 pack keeps the built-in plugin.

    scripts/build.sh sushi_ds41             # from the repo root: tests, then ReleaseFast
    scripts/install.sh sushi_ds41           # compile and install into mlx-serve
    scripts/sync-sushi.sh <sushi checkout>  # refresh src/sushi/ from sushi
    scripts/sync-sushi.sh --check <sushi checkout> <tag>  # does a sushi release touch this slice? exit 0 = keep the pin
    zig build gpu-test -Dmlx-lib=<dir holding libmlxc.dylib>  # from this folder: the Metal tests (the matmul2d ones skip without an M5)
    zig build gpu-test -Dmlx-lib=<dir> -Ddsv41-tiny=<dir>    # plus the prefill pipeline test on sushi's tiny checkpoint

Needs an mlx-serve with the native drafter round (`NativeDrafter.step`, `verify`, `finish`), outside-plugin image hooks
(ddalcu/mlx-serve#1002), host-decoded media (`MediaInput`, ddalcu/mlx-serve#1003) and the array-based aux snapshot hooks
(ddalcu/mlx-serve#1000).

The slice is sushi ec2f5a29 (main, past v1.4.1; NOTICE). The numbers below were measured on the slice of sushi 3fc47d8a
and later ones. Against the cc1981df slice, ec2f5a29 measured prefill +1.5-2.4% (8k-64k) and decode +1.2-4.6% on an
M5 Ultra, output byte-identical (https://github.com/CommunityServe/plugins/pull/28). This repo's edits to the slice are `patches/index-nax.patch`,
`patches/prefill-pipeline.patch`, `patches/shared-expert-ane.patch` and `patches/stream-ablation.patch`, applied in that
order by `scripts/sync-sushi.sh`.

Measured on M4 Max 128 GB, Sushi-STQ-1.75bpw, greedy:

| | decode tok/s | prefill tok/s |
|---|---|---|
| serial, short prompt | 42 | 48 |
| DSpark, predictable text | 56 to 64 (greedy or temperature 0.6) | 48 |
| DSpark, free prose, temperature 0.6 | 40 | 48 |
| 3.6k prompt, cold | 26 | 366 |
| 3.6k prompt, repeated (prefix cache) | 43 | 3638 cached |

Measured on M5 Ultra 256 GB (every routed expert resident, 384 slots a layer), Sushi-STQ-1.75bpw, 1 stream, mean of 2
runs against sushi v1.4.0 serving the same pack:

| | plugin tok/s | sushi tok/s |
|---|---|---|
| prefill 8k / 32k / 64k | 1903 / 2074 / 2050 | 1804 / 1904 / 1871 |
| decode, short / after 32k / after 64k (DSpark) | 100.3 / 91.6 / 98.5 | 100.1 / 92.5 / 99.4 |

Text matches sushi (prompt 2 identical, prompt 1 identical for the first 239 characters, a known near-tie), with
DSpark on or off.

- DSpark: greedy and sampled requests (the engine accepts a sampled block; the plugin drafts and keeps what it accepts). `{"mtp": false}` in the model settings keeps the stages off.
- Prefix cache: a restore point at the prompt's last token, images included: it is keyed by its ids and a hash of the
  image rows they cover (`src/vision_key.zig`), so two images, whose placeholder ids are the same, never share one.
- Images: one frame, decoded by mlx-serve (`MediaInput`); the plugin lays it out as the release's processor does. The
  bf16 tower is requantized to affine 8-bit (group 128) at load, as sushi does.
- Token embedding: resident on the GPU and billed with the trunk, as sushi does since 3568d932.
- Memory: sushi runs each forward under an MLX transient cap that needs its patched MLX; mlx-serve's MLX has none, so
  the plugin keeps its own buffer-pool caps and bills the prefill pool in the expert room.
- Attention, decode index scan and prefill, from the sushi slice: the wide-span attention on the M5's matrix units and
  the decode index scan of the packed keys on them (sushi fafc69c8, byte for byte the scalar kernel's scores), each
  kept only if its load-time trial agrees with the SIMD kernel (`setAttnNax`, `setIndexNax`; other GPUs and a declined
  trial keep that kernel); the attention epilogue in the kernel's store, state-only CED chunks, the shared expert on
  the routed prefill launches, the hyper-connection mix tile and the decode fold of the hyper-connection expand.
- Index scores, prefill: sushi's matrix-unit scan covers only decode widths (16 rows or fewer) and the candidate
  gather, so on an M5 GPU a prompt's wide spans score their compressed blocks on the matrix units here
  (`src/index_nax.zig`, `patches/index-nax.patch`): 16 query rows share each staged 32-key tile, with the relu, head
  weights and head sum in registers. 3.3x on the scorer (0.70 against 2.32 ms a 256-row span at 24k blocks); the
  scores are within 1.2e-7 of `indexScoresTile`'s and 10 of 194 million top-512 picks differ on real 32k-64k prompts,
  so long-prompt text can differ from the simdgroup scorer after a few tokens.
- Expert prefetch: off, sushi's default since a7242135 (its `--ssd-prefetch` defaults to 0); the plugin has no switch.
- Prefill: the whole prompt in one host call (`whole_prompt_prefill`), chunked here at `prefill_width` as sushi does; a
  long prompt is one uncancellable call. A chunk that another chunk follows, both wider than the window, skips its CED
  decoder tail (`extendState`, from the slice since sushi 022b9222): it runs the layers below the split and the split
  layer's compressor, since the next chunk's tail overwrites the 128-row rings that tail would fill. Measured when it was
  this repo's patch: prefill 8k / 32k / 64k 2466 / 2758 / 2729 to 2562 / 3079 / 3063 tok/s on the M5 Ultra; text and
  logprobs byte-identical on 8k to 64k prompts, a merged tail, an image and a prefix-cache restore.
- Prefill pipeline: a prefill-width layer is submitted, not waited on, with up to 2 in flight
  (`SUSHI_DS41_PREFILL_PIPELINE`, 0 turns it off; `patches/prefill-pipeline.patch`, `pipeOn`). Same graphs, same bytes.
- Shared expert on the Neural Engine, prefill, off by default (`SUSHI_DS41_ANE_SHARED=1|2`, the number of ANE units;
  `src/ane_shared.zig`, `patches/shared-expert-ane.patch`): in each 2,048-row chunk, the layers below the
  CED split run the shared expert's SwiGLU as int8 weights and fp16 activations on the ANE while the GPU runs the
  routed experts. Lossy. The load self-check compares every covered layer against the GPU chain (worst `cos 0.999767`
  on the release pack), and if the ANE is missing or a program fails to build, the GPU path keeps serving
  (`ane shared: off (...)`). The programs are built and run by the host's Neural Engine runtime (`Host.ane`,
  mlx-serve's `lib/ane`); a host without it serves this plugin on the GPU. M5 Ultra, `=2`: cold prefill x1.037 at 25.6k and x1.042 at 60k, KL 0.0108 against the
  GPU path, peak footprint +3 GB, API KLD to the FP8 teacher 0.1849 (unchanged).
- Fewer routed experts at prefill, off by default (`SUSHI_DS41_PREFILL_TOP_K=<k>`, `routedK`): a forward wider than
  `narrow_rows` (16) routes each row to `k` experts instead of the pack's 6, renormalized over the `k`; decode and DSpark
  verify keep 6. M5 Ultra, release pack r2.0, `=5`: cold prefill x1.058 at 32k and x1.035 at 8k, TTFT -4 to -5% on
  76k-305k-token agent prompts, API KLD to the FP8 teacher 0.1589 against 0.1596 at 6; DSpark acceptance unchanged.
  Serving 5 everywhere costs KLD +5.6% (0.1686) for about 3% cheaper verify rounds at agent context depth.
- Directional ablation (`Caps.ablation`, mlx-serve's `--ablation-profiles`): a request's direction is removed from
  every write to the hyper-connection stream: the embedding (text and image rows, and the DSpark stages' block input),
  each attention output, each FFN output (in the fused decode kernel's expanded stream through its component formula)
  and each Engram value. Armed per call on `forward`, `NativeDrafter.step` and `verify`; a request without one runs
  the graphs unchanged. An armed request does not take the slice's decode fold of the hyper-connection expand, so its
  FFN output is ablated before the expand. The slice edit is `patches/stream-ablation.patch` (applied last).
- Calibration tap: `SUSHI_DS41_TAP_DIR=<dir>` writes each prompt's stream (the mean of the hc copies entering every
  layer and leaving the last, for the prefix pass's last 4 rows and the last token) as `tap-<seq>.npy` (f32
  `[layers + 1, rows, dim]`) plus a line in `index.jsonl`. Route speculation is off only for the passes it records.
- Not supported: Thunderbolt pair, other quant formats, snapshot serialization (no SSD tier).

`plugin check` needs a conformance fixture pack, which an STQ pack cannot provide small; the plugin is checked by serving.
