# sushi-ds41

DeepSeek-V4.1-Flash on sushi's own model code, as an outside plugin for mlx-serve. Claims only packs whose
`expert_quant.format` is `stq`; every other V4.1 pack keeps the built-in plugin.

    scripts/build.sh sushi_ds41             # from the repo root: tests, then ReleaseFast
    scripts/install.sh sushi_ds41           # compile and install into mlx-serve
    scripts/sync-sushi.sh <sushi checkout>  # refresh src/sushi/ from sushi
    scripts/sync-sushi.sh --check <sushi checkout> <tag>  # does a sushi release touch this slice? exit 0 = keep the pin
    zig build gpu-test -Dmlx-lib=<dir holding libmlxc.dylib>  # from this folder: the Metal tests (the matmul2d ones skip without an M5)

Needs an mlx-serve with the native drafter round (`NativeDrafter.step`, `verify`, `finish`), outside-plugin image hooks
(ddalcu/mlx-serve#1002), host-decoded media (`MediaInput`, ddalcu/mlx-serve#1003) and the array-based aux snapshot hooks
(ddalcu/mlx-serve#1000).

The numbers below were measured on the slice of sushi 3fc47d8a; the slice is now sushi cc1981df (NOTICE).

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

With the wide-span attention on the M5's matrix units (`src/attn_nax.zig`), same machine and pack, A/B against the
simdgroup kernel: prefill 8k 1906 to 2138 tok/s, 32k 2075 to 2368, 64k 2050 to 2343 (1.12x to 1.14x); KLD against the FP8 teacher
0.1849, against 0.1854 for the simdgroup kernel. Decode does not run it (spans of 16 rows or fewer).

- DSpark: greedy and sampled requests (the engine accepts a sampled block; the plugin drafts and keeps what it accepts). `{"mtp": false}` in the model settings keeps the stages off.
- Prefix cache: a restore point at the prompt's last token, images included: it is keyed by its ids and a hash of the
  image rows they cover (`src/vision_key.zig`), so two images, whose placeholder ids are the same, never share one.
- Images: one frame, decoded by mlx-serve (`MediaInput`); the plugin lays it out as the release's processor does. The
  bf16 tower is requantized to affine 8-bit (group 128) at load, as sushi does.
- Token embedding: an affine-quantized embedding stays in its shard and a forward reads its rows from SSD, as sushi does;
  the bytes it frees go to the expert cache.
- Memory: sushi runs each forward under an MLX transient cap that needs its patched MLX; mlx-serve's MLX has none, so
  the plugin keeps its own buffer-pool caps and bills the prefill pool in the expert room.
- Attention: on an M5 GPU (`applegpu_g17` or later, macOS 26.2 or later) a prompt's wide spans run on the matrix
  units through Metal 4 tensor ops (`src/attn_nax.zig`), within 3e-5 of the simdgroup kernel, with the picked
  compressed rows read packed from the store as sushi's kernel reads them; other GPUs keep that kernel. The edit to the
  slice that calls it is `patches/attn-nax.patch` (`fusedHeads`).
- Prefill: the whole prompt in one host call (`whole_prompt_prefill`), chunked here at `prefill_width` as sushi does; a
  long prompt is one uncancellable call. A chunk that another chunk follows skips its CED decoder tail
  (`patches/ced-state.patch`, `extendState`): it runs the layers below the split and the split layer's compressor, since
  the next chunk's tail overwrites the 128-row rings that tail would fill. Prefill 8k / 32k / 64k 2466 / 2758 / 2729 to
  2562 / 3079 / 3063 tok/s on the M5 Ultra; text and logprobs are byte-identical on 8k to 64k prompts, a merged tail, an
  image and a prefix-cache restore.
- Not supported: Thunderbolt pair, other quant formats, snapshot serialization (no SSD tier).

`plugin check` needs a conformance fixture pack, which an STQ pack cannot provide small; the plugin is checked by serving.
