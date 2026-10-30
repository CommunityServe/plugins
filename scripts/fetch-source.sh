#!/usr/bin/env bash
# A checkout of mlx-serve at MLX_SERVE_REF in .kit-src/ (no submodules), for plugins that run a built-in's forward and
# generate their copy of it from that source. Idempotent; prints the checkout's path. build-kit.sh uses the same one.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REF="$(tr -d '[:space:]' < "$ROOT/MLX_SERVE_REF")"
SRC="$ROOT/.kit-src"
if [ "$(git -C "$SRC" rev-parse HEAD 2>/dev/null)" != "$REF" ]; then
  mkdir -p "$SRC"
  git -C "$SRC" init -q
  git -C "$SRC" remote remove origin 2>/dev/null || true
  git -C "$SRC" remote add origin https://github.com/ddalcu/mlx-serve
  git -C "$SRC" fetch -q --depth 1 origin "$REF" >&2
  git -C "$SRC" checkout -q --detach FETCH_HEAD >&2
fi
echo "$SRC"
