#!/usr/bin/env bash
# Generates src/mlx_serve/ for a build: the slice of mlx-serve at MLX_SERVE_REF (scripts/sync-mlx-serve.sh) with
# overlay/mlx_serve/ over it. build.sh runs it (plugin.json's "prepare"). It regenerates when the ref, the overlay, the
# scripts or build.zig (whose test filters the sync reads) changed since the last run, and on every run it checks the
# tree against the hash in scripts/mlx_serve.sha256, so only the reviewed tree builds.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
root="$(cd "$here/../.." && pwd)"
ref="$(tr -d '[:space:]' < "$root/MLX_SERVE_REF")"
stamp="$ref $(cat "$here/scripts/sync-mlx-serve.sh" "$here/scripts/slice.txt" "$here/scripts/cut.txt" "$here/build.zig" | shasum | cut -c1-16) $(cd "$here/overlay" && find . -type f | LC_ALL=C sort | xargs shasum | shasum | cut -c1-16)"
dst="$here/src/mlx_serve"
if [ ! -f "$dst/.generated" ] || [ "$(cat "$dst/.generated")" != "$stamp" ]; then
  src="$("$root/scripts/fetch-source.sh")"
  if [ "$(git -C "$src" rev-parse HEAD)" != "$ref" ] || [ -n "$(git -C "$src" status --porcelain -- src)" ]; then
    echo "prepare: $src is not a clean checkout of $ref: delete it and build again" >&2
    exit 1
  fi
  rm -rf "$dst"
  mkdir -p "$dst"
  cp -R "$here/overlay/mlx_serve/." "$dst/"
  MLX_SERVE_KIT="${MLX_SERVE_KIT:-$root/.kit/kit}" OVERLAY="$here/overlay/mlx_serve" bash "$here/scripts/sync-mlx-serve.sh" "$src" >&2
  cp -R "$here/overlay/mlx_serve/." "$dst/"
  echo "$stamp" > "$dst/.generated"
fi
got="$(cd "$dst" && find . -type f ! -name .generated | LC_ALL=C sort | xargs shasum -a 256 | shasum -a 256 | cut -d' ' -f1)"
want="$(tr -d '[:space:]' < "$here/scripts/mlx_serve.sha256")"
if [ "$got" != "$want" ]; then
  echo "prepare: src/mlx_serve hashes to $got, not the reviewed $want (scripts/mlx_serve.sha256): re-sync (delete" >&2
  echo "src/mlx_serve, run scripts/prepare.sh, review what it wrote) and commit the new hash" >&2
  exit 1
fi
