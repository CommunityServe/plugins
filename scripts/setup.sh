#!/usr/bin/env bash
# One command to a working tree: the pinned Zig, the pinned kit, and a report of what else is needed.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/fetch-zig.sh"
"$ROOT/scripts/fetch-kit.sh"
if command -v mlx-serve >/dev/null 2>&1; then
  echo "[setup] mlx-serve: $(command -v mlx-serve) (needed for check and install)"
else
  echo "[setup] mlx-serve is not on PATH: you can build plugins, but not run 'plugin check' or 'plugin install'"
fi
echo "[setup] ready: scripts/build.sh <plugin|--all>"
