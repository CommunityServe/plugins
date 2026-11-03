# hy_mt2_sherry

Serves the **Hy-MT2-1.8B 1.25-bit MLX** packs in mlx-serve without mlx-lm:

- `Sube126/Hy-MT2-1.8B-1.25Bit-MLX`
- `kuotient/Hy-MT2-1.8B-1.25Bit-MLX` (same weights)

The pack is `hunyuan_v1_dense` whose projections are llama.cpp **STQ1_0** ternary blocks (`qs` / `sign` / `scales`,
1.3125 bpw) and whose embedding is 6-bit affine, group 64. The built-in engine reads neither the layout nor the
`model_file: sherry_model.py` architecture, so this plugin ports the model and its two Metal kernels:

| Path | Kernel | What it does |
|---|---|---|
| 1 row | `stq1_gemv_v9c8` | one simdgroup per 8 output rows, codebook expanded to a `half4` constant table; reads the 1.31-bpw buffers once per token |
| 2–64 rows per slot (decode, verify, short prompts) | `stq1_batch_gemv_bcl4` | each packed weight word is loaded once and dotted against every row and slot (`BAT` 2–8, then 8-row chunks); attention stays per slot |
| >64 rows per slot (long prompt) | `stq1_dequant` | one-shot dense f16 tile per projection, then the stock GEMM; the residual is evaluated per layer so 32 layers' tiles never pile up |

The 64-row crossover is the pack's own measurement. PLD / speculative verify therefore runs through the batch kernel
too, with one causal attention pass per slot over its draft chain.

Everything else is the host: tokenizer, chat template, sampling, KV cache, prefix cache, continuous batching,
OpenAI / Anthropic / Ollama APIs.

## Use

```bash
scripts/install.sh hy_mt2_sherry            # build from source + install into ~/.mlx-serve
# plugins load only when ~/.mlx-serve/plugins.json says "enabled": true

mlx-serve --serve --model /path/to/Hy-MT2-1.8B-1.25Bit-MLX/snapshots/<rev>
curl http://localhost:11234/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "Hy-MT2-1.8B-1.25Bit-MLX",
  "messages": [{"role":"user","content":"Translate into English, output only the translation:\n\n今天天气真好。"}],
  "temperature": 0
}'
```

The log must say `[plugin] hy_mt2_sherry serves hunyuan_v1_dense`, and `/v1/models` shows `"plugin": "hy_mt2_sherry"`.

## Measured

Apple M5 Max, 128 GB, mlx-serve 26.11.1 (MLX 0.32.3), greedy, `--no-pld`, idle machine. Prefill prompts carry a unique
nonce so the prefix cache only matches the 12-token instruction.

| Scenario | Decode | Prefill |
|---|---:|---:|
| 1 stream, 74-token prompt | 355 tok/s (40 out) | 1.8–2.0k tok/s |
| 1 stream, 282-token prompt | 290–333 tok/s (128 out) | 4.5–5.5k tok/s |
| 1 stream, 646-token prompt | — | 6.3–7.6k tok/s |
| 2 slots, 272-token prompt | 183–209 each, **369–414 aggregate** | — |
| 4 slots, 272-token prompt | 146–157 each, **586–623 aggregate** | — |

Single-stream decode is ~3x the pack's own ~112 tok/s on M3; four slots aggregate to ~5.5x one M3 stream. (An earlier
run on the same Mac under heavy background load measured 130 / 183 / 258–312, so treat these as idle-machine numbers.)
PLD's verify rows take the batch kernel up to 64 rows per slot; the 64-row crossover itself is the pack's measurement.

## Check

```bash
scripts/build.sh hy_mt2_sherry      # 16 MLX-free tests + ReleaseFast
scripts/check.sh hy_mt2_sherry      # conformance suite on the built-in fixture pack
scripts/check.sh hy_mt2_sherry --model ~/.mlx-serve/models/Sube126/Hy-MT2-1.8B-1.25Bit-MLX
scripts/lint.sh
```

`src/conformance_fixture.zig` builds a 2-layer Hy-MT2 shape (hidden 2048, 16h/4kv, head_dim 128) with real STQ1_0
projections and a real 6-bit affine embedding, quantized from seeded float weights the tests keep as the reference.
`mlx-serve plugin check` loads it, runs a forward through both Metal kernels, and `mlx-serve plugin fixture` writes the
same pack to a directory you can serve with `--model`.

## Costs that were measured, not estimated

| Quantity | Value | How |
|---|---|---|
| One layer's dequant tiles (prefill transient) | **92.0 MB** | allocated the seven `[N, K]` f16 tiles standalone and read MLX's own active memory; the `bill` term is the same arithmetic |
| Concurrent prefill | 2 slots **3.0x**, 4 slots **6.0x** one stream | 225-token prompts, aggregate tok/s over the single-stream rate; per-layer `evalKeep` does not serialize slots |

## Not supported

- **Layer split** across Macs (`span != null` is refused).
- **Vision / audio / MoE**: the pack has none.
- **Other gemma/hunyuan dense packs**: only packs carrying `sherry_embed_bits` (or `model_file: sherry_model.py`) are
  claimed; the forward hard-codes the STQ1_0 projections, the 6-bit affine embedding, tied head and `use_qk_norm`.
- **The pack's full 262144-token window as a default**: KV is 64 KiB/token, so `context_cap` is 32768. A longer
  `--ctx-size` is refused by the host; raise the constant in `config.zig` and reinstall if you have the memory.
- **Speculative decoding**: PLD may run; its verify rows take the batch kernel up to 64 rows per slot (`Caps` marks
  no exactness contract and there is no native drafter).

## Loaded through the app

`mlx-serve pull Sube126/Hy-MT2-1.8B-1.25Bit-MLX` works, and a server started without `--model` discovers the pack and
marks it `"plugin": "hy_mt2_sherry"` in `/v1/models`; the first request loads it on demand. `--kv-quant 8` and
`--kv-quant 4` both serve this pack (the plugin hands the host's KV cache to every append, so quantization is the
host's own path).

One mlx-serve caveat that is not the plugin's: `mlx-serve --serve --model <org>/<repo>` (a short id rather than a
path) fails with `error: FileNotFound` for every pack on this build, the built-in ones included — pass the absolute
path or start `--serve` without `--model`.
