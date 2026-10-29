# sushi-ds41

DeepSeek-V4.1-Flash on sushi's own model code, as an outside plugin for mlx-serve. Claims only packs whose
`expert_quant.format` is `stq`; every other V4.1 pack keeps the built-in plugin.

    scripts/build.sh sushi_ds41             # from the repo root: tests, then ReleaseFast
    scripts/install.sh sushi_ds41           # compile and install into mlx-serve
    scripts/sync-sushi.sh <sushi checkout>  # refresh src/sushi/ from sushi
    scripts/sync-sushi.sh --check <sushi checkout> <tag>  # does a sushi release touch this slice? exit 0 = keep the pin

Needs an mlx-serve with the native drafter round (`NativeDrafter.step`, `verify`, `finish`) and outside-plugin image hooks (ddalcu/mlx-serve#1002), images decoded by the host (#1003) and the aux hooks' import/export (#1000).

Measured on M4 Max 128 GB, Sushi-STQ-1.75bpw, greedy:

| | decode tok/s | prefill tok/s |
|---|---|---|
| serial, short prompt | 42 | 48 |
| DSpark, predictable text | 56 to 64 (greedy or temperature 0.6) | 48 |
| DSpark, free prose, temperature 0.6 | 40 | 48 |
| 3.6k prompt, cold | 26 | 366 |
| 3.6k prompt, repeated (prefix cache) | 43 | 3638 cached |

Text matches sushi (prompt 2 identical, prompt 1 identical for the first 239 characters, a known near-tie), with
DSpark on or off.

- DSpark: greedy and sampled requests (the engine accepts a sampled block; the plugin drafts and keeps what it accepts). `{"mtp": false}` in the model settings keeps the stages off.
- Prefix cache: a restore point at the prompt's last token, text only.
- Images: decoded by mlx-serve (JPEG, PNG, GIF, BMP, WebP), transparency composited onto white.
- Not supported: Thunderbolt pair, other quant formats, snapshot export and import (no SSD tier).

`plugin check` needs a conformance fixture pack, which an STQ pack cannot provide small; the plugin is checked by serving.
