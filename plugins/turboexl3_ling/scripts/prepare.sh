#!/usr/bin/env bash
# Generates src/mlx_serve/ for a build: the slice of mlx-serve at the plugin's own MLX_SERVE_REF
# (scripts/sync-mlx-serve.sh) with overlay/mlx_serve/ over it, then patches/*.patch applied. build.sh runs it
# (plugin.json's "prepare"). It regenerates when the ref, the overlay, the patches or the scripts changed since the last
# run, and on every run it checks the tree against the hash in scripts/mlx_serve.sha256 (scripts/check-mlx-serve.sh), so
# only the reviewed tree builds.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
root="$(cd "$here/../.." && pwd)"
# The mlx-serve commit the copies come from: the plugin's own MLX_SERVE_REF, not the repo's. The repo's ref builds the
# kit while KIT_VERSION has no release kit (scripts/build-kit.sh, in .kit-src/); a bump of it does not move the copies.
ref="$(tr -d '[:space:]' < "$here/MLX_SERVE_REF")"
stamp="$ref $(cat "$here/scripts/sync-mlx-serve.sh" "$here/scripts/slice.txt" "$here/scripts/slice-stream.txt" "$here/scripts/cut.txt" | shasum | cut -c1-16) $(cd "$here" && find overlay patches -type f | LC_ALL=C sort | xargs shasum | shasum | cut -c1-16)"
dst="$here/src/mlx_serve"
if [ ! -f "$dst/.generated" ] || [ "$(cat "$dst/.generated")" != "$stamp" ]; then
  # A checkout of that commit in the plugin's folder (.mlx-serve-src/, git-ignored), made as scripts/fetch-source.sh
  # makes the repo's in .kit-src/, which this script leaves alone.
  src="$here/.mlx-serve-src"
  if [ "$(git -C "$src" rev-parse HEAD 2>/dev/null)" != "$ref" ]; then
    mkdir -p "$src"
    git -C "$src" init -q
    git -C "$src" remote remove origin 2>/dev/null || true
    git -C "$src" remote add origin https://github.com/ddalcu/mlx-serve
    git -C "$src" fetch -q --depth 1 origin "$ref" >&2
    git -C "$src" checkout -q --detach FETCH_HEAD >&2
  fi
  if [ "$(git -C "$src" rev-parse HEAD)" != "$ref" ] || [ -n "$(git -C "$src" status --porcelain -- src)" ]; then
    echo "prepare: $src is not a clean checkout of $ref: delete it and build again" >&2
    exit 1
  fi
  rm -rf "$dst"
  mkdir -p "$dst"
  cp -R "$here/overlay/mlx_serve/." "$dst/"
  MLX_SERVE_KIT="${MLX_SERVE_KIT:-$root/.kit/kit}" OVERLAY="$here/overlay/mlx_serve" bash "$here/scripts/sync-mlx-serve.sh" "$src" >&2
  cp -R "$here/overlay/mlx_serve/." "$dst/"
  # The edits to the copies (patches/*.patch, unified diffs against the generated tree): turboexl3.patch first, then
  # the others in name order, each made against the tree the ones before it wrote (a bare glob puts turboexl3.patch
  # after turboexl3-*.patch: '-' sorts before '.').
  base="$here/patches/turboexl3.patch"
  for p in "$base" "$here"/patches/*.patch; do
    [ -e "$p" ] || continue
    if [ "$p" = "$base" ]; then [ -z "${base_done:-}" ] || continue; base_done=1; fi
    patch -s -N -p1 -d "$dst" -i "$p" || { echo "prepare: $(basename "$p") does not apply to the generated tree" >&2; exit 1; }
  done
  find "$dst" \( -name '*.orig' -o -name '*.rej' \) -delete
  echo "$stamp" > "$dst/.generated"
fi
exec bash "$here/scripts/check-mlx-serve.sh"
