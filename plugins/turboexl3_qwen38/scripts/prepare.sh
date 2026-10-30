#!/usr/bin/env bash
# Generates src/mlx_serve/ for a build: the slice of mlx-serve at MLX_SERVE_REF (scripts/sync-mlx-serve.sh) with
# overlay/mlx_serve/ over it. build.sh runs it (plugin.json's "prepare"); it is a no-op while the ref, the overlay and the
# scripts are what the last run used.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
root="$(cd "$here/../.." && pwd)"
ref="$(tr -d '[:space:]' < "$root/MLX_SERVE_REF")"
stamp="$ref $(cat "$here/scripts/sync-mlx-serve.sh" "$here/scripts/slice.txt" "$here/scripts/cut.txt" | shasum | cut -c1-16) $(cd "$here/overlay" && find . -type f | sort | xargs shasum | shasum | cut -c1-16)"
dst="$here/src/mlx_serve"
if [ -f "$dst/.generated" ] && [ "$(cat "$dst/.generated")" = "$stamp" ]; then exit 0; fi
src="$("$root/scripts/fetch-source.sh")"
rm -rf "$dst"
mkdir -p "$dst"
cp -R "$here/overlay/mlx_serve/." "$dst/"
MLX_SERVE_KIT="${MLX_SERVE_KIT:-$root/.kit/kit}" OVERLAY="$here/overlay/mlx_serve" bash "$here/scripts/sync-mlx-serve.sh" "$src" >&2
cp -R "$here/overlay/mlx_serve/." "$dst/"
echo "$stamp" > "$dst/.generated"
