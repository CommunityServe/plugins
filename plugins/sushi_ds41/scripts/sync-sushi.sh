#!/usr/bin/env bash
# Copies the DeepSeek-V4.1 slice of sushi into src/sushi/ in sushi's own folders (src/dsv41/, src/glm5/, src/exl3/), then
# applies this repo's edits to it (patches/*.patch). Usage: scripts/sync-sushi.sh <sushi checkout>
# The cut edges are ours and never overwritten: src/sushi/{model,transformer,status,scheduler,test_models,tokenizer,
# dflash,model_load}.zig and src/sushi/dsv41/transient_peak.zig (the host's MLX has no transient-peak counters).
# scripts/sync-sushi.sh --check <sushi checkout> <ref>: does <ref> (a release tag or commit) change the slice since
# SUSHI_COMMIT? Exit 0 when the pin holds, 1 with the changed files when a sync is due.
set -euo pipefail
plugin="$(cd "$(dirname "$0")/.." && pwd -P)"
dst="$plugin/src/sushi"
dsv41="dsv41_borrow dsv41_config dsv41_embed dsv41_engram dsv41_kernels dsv41_model dsv41_owned dsv41_qat dsv41_scope dsv41_source dsv41_stream dsv41_vision dspark_rows expert_usage"
shared="vision_common dflash_policy spec_cost_gate mtp_depth_bounds expert_bf16_kernels expert_io expert_quant expert_stream io_util log mlx mlx_kernel_keys nax_gate test_expect"
cut="model transformer status scheduler test_models tokenizer dflash model_load"

slice_paths() {
  local p
  echo src/exl3; echo src/fixtures/dsv41_engram_token_map.bin; echo src/glm5/glm_stream_fixture.zig
  for p in $dsv41; do echo "src/dsv41/$p.zig"; done
  for p in $shared; do echo "src/$p.zig"; done
}

if [ "${1:-}" = "--check" ]; then
  repo="${2:?sushi checkout}"; ref="${3:?release tag or commit}"
  pin="$(tr -d '[:space:]' < "$dst/SUSHI_COMMIT")"
  paths=()
  while read -r p; do
    # A path the ref does not have would match nothing and read as "pin holds": the slice moved, so a sync is due.
    git -C "$repo" cat-file -e "$ref:$p" 2>/dev/null || { echo "sync due: $ref has no $p (the slice moved)"; exit 1; }
    paths+=("$p")
  done < <(slice_paths)
  changed="$(git -C "$repo" diff --name-only "$pin" "$ref" -- "${paths[@]}")"
  if [ -z "$changed" ]; then echo "pin holds: $ref changes nothing in the slice since ${pin:0:8}"; exit 0; fi
  echo "sync due: $ref changes the slice since ${pin:0:8}:"; echo "$changed" | sed 's/^/  /'; exit 1
fi

src="${1:?sushi checkout}/src"
# Everything but the cut edges goes first, so a file sushi dropped does not linger.
keep=()
for f in $cut; do keep+=(! -name "$f.zig"); done
find "$dst" -maxdepth 1 -type f -name '*.zig' "${keep[@]}" -delete
find "$dst/dsv41" -type f ! -name transient_peak.zig -delete
rm -rf "$dst/glm5" "$dst/exl3" "$dst/fixtures"
mkdir -p "$dst/glm5" "$dst/exl3" "$dst/fixtures"
for f in $dsv41; do cp "$src/dsv41/$f.zig" "$dst/dsv41/$f.zig"; done
for f in $shared; do cp "$src/$f.zig" "$dst/$f.zig"; done
cp "$src/glm5/glm_stream_fixture.zig" "$dst/glm5/"
cp "$src"/exl3/*.zig "$dst/exl3/"
cp "$src/fixtures/dsv41_engram_token_map.bin" "$dst/fixtures/"
# patches/*.patch, in name order (paths relative to the plugin). `git apply` is all or nothing: a patch that no longer
# applies to the new slice stops the sync with the files it names.
top="$(git -C "$plugin" rev-parse --show-toplevel)"
for p in "$plugin"/patches/*.patch; do
  [ -e "$p" ] || continue
  git -C "$top" apply --directory="${plugin#"$top"/}" "$p" || { echo "sync-sushi: $(basename "$p") does not apply; redo it on the new slice" >&2; exit 1; }
done
echo "$(git -C "${1}" rev-parse HEAD)" > "$dst/SUSHI_COMMIT"
