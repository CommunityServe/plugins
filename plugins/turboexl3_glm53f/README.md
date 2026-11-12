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

- mlx-serve with the outside-quant seams in `glm5_next` (the core-outside-quant change, not yet upstream). Without
  them, `glm5_next` refuses the pack's `expert_streaming` block by name.
- An M5-class GPU (NAX). On any other GPU, the plugin refuses the pack by name.
- The mul1 codebook. The plugin refuses any other codebook by name.

## Switches

The plugin reads 19 `TURBOEXL3_GLM53F_*` switches at load (src/options.zig). The kit exports them as the plugin's
options, so the app's plugin page can set them. Each default is the value of the served configuration.
`PROMPT_KEEP` and `ADOPT_PIECES` need a seam in the kit's stream. If the kit has no such seam, they default to off,
and the plugin refuses the bank if one is set.

The model's own levers (the KDA and DSA kernels, the fused router, the split prompt forward and others) are not
in this plugin. Each is an mlx-serve change to `glm5_next`.

## Build and check

    scripts/build.sh turboexl3_glm53f                    # from the repo root: tests, then ReleaseFast
    scripts/install.sh turboexl3_glm53f --home ~/tmp/h   # under a throwaway HOME

`zig build test` runs the MLX-free tests: the claim and the pack check, the bank's row plan, the switches, and the
MTP bank's manifest. It also runs the tile gate, which compares turboexl3's tile decode with exllamav3's rule on real
GLM-5.3-Flash tiles at each routed rate (src/frac_test.zig, src/glm_tiles.json).

At load, the log names the plugin: `[plugins] loaded turboexl3_glm53f (quant, sha256 ...)`. It also prints the
bank's rows: `glm5_next: streamed experts, N prompt / M decode slot rows per layer (R B the widest row): ...`.
