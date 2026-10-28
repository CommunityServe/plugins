#!/usr/bin/env bash
# Copies the DeepSeek-V4.1 slice of sushi into src/sushi/ unchanged. Usage: scripts/sync-sushi.sh <sushi checkout>
# src/sushi/{model,transformer,status,scheduler,test_models,tokenizer}.zig are ours (the cut edges) and never overwritten.
# scripts/sync-sushi.sh --check <sushi checkout> <ref>: does <ref> (a release tag or commit) change the slice since
# SUSHI_COMMIT? Exit 0 when the pin holds, 1 with the changed files when a sync is due.
set -euo pipefail
dst="$(cd "$(dirname "$0")/.." && pwd)/src/sushi"
files="dsv41_borrow dsv41_config dsv41_engram dsv41_kernels dsv41_model dsv41_owned dsv41_qat dsv41_scope dsv41_source dsv41_stream dsv41_vision vision_common dspark_rows dflash_policy spec_cost_gate mtp_depth_bounds
expert_bf16_kernels expert_io expert_quant expert_stream expert_usage io_util log mlx mlx_kernel_keys test_expect glm_stream_fixture"

if [ "${1:-}" = "--check" ]; then
  repo="${2:?sushi checkout}"; ref="${3:?release tag or commit}"
  pin="$(tr -d '[:space:]' < "$dst/SUSHI_COMMIT")"
  paths=(src/exl3 lib/stb_image.h lib/stb_image_impl.c src/fixtures/dsv41_engram_token_map.bin)
  for f in $files; do paths+=("src/$f.zig"); done
  changed="$(git -C "$repo" diff --name-only "$pin" "$ref" -- "${paths[@]}")"
  if [ -z "$changed" ]; then echo "pin holds: $ref changes nothing in the slice since ${pin:0:8}"; exit 0; fi
  echo "sync due: $ref changes the slice since ${pin:0:8}:"; echo "$changed" | sed 's/^/  /'; exit 1
fi

src="${1:?sushi checkout}/src"
for f in $files; do cp "$src/$f.zig" "$dst/$f.zig"; done
rm -rf "$dst/exl3"; mkdir "$dst/exl3"; cp "$src"/exl3/*.zig "$dst/exl3/"
mkdir -p "$dst/../stb"; cp "${1}/lib/stb_image.h" "${1}/lib/stb_image_impl.c" "$dst/../stb/"
mkdir -p "$dst/fixtures"; cp "$src/fixtures/dsv41_engram_token_map.bin" "$dst/fixtures/"
# sushi's bill multiplies `@min(rows, span_rows)`, which Zig narrows to u9: the product overflows (wraps in a release build).
cfg="$dst/dsv41_config.zig"
grep -q 'const span = @min(rows, span_rows);' "$cfg"
sed -i.bak 's/const span = @min(rows, span_rows);/const span: u64 = @min(rows, span_rows);/' "$cfg"; rm "$cfg.bak"
echo "$(git -C "${1}" rev-parse HEAD)" > "$dst/SUSHI_COMMIT"
