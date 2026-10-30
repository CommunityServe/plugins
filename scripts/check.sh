#!/usr/bin/env bash
# `mlx-serve plugin check` on a built plugin: the conformance suite on its fixture, and with --model, does it claim
# and parse that pack. Usage: scripts/check.sh <name> [--model <pack dir>]
# A plugin whose plugin.json says "fixture": "none" has no conformance fixture; only --model is run for it.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
name="${1:?usage: scripts/check.sh <name> [--model <pack dir>]}"; shift
command -v mlx-serve >/dev/null || { echo "mlx-serve is not on PATH" >&2; exit 1; }
# `mlx-serve plugin` alone prints its usage and exits 1, so its output is read before grep (pipefail would fail the probe).
usage="$(mlx-serve plugin 2>&1 || true)"
grep -q "plugin install" <<<"$usage" || { echo "this mlx-serve ($(mlx-serve --version 2>&1 | tail -1)) has no 'plugin' command: use a release that ships the plugin kit" >&2; exit 1; }
dylib="$ROOT/plugins/$name/zig-out/lib/lib$name.dylib"
[ -f "$dylib" ] || { echo "build it first: scripts/build.sh $name" >&2; exit 1; }
if grep -q '"fixture": *"none"' "$ROOT/plugins/$name/plugin.json"; then
  [ "${1:-}" = "--model" ] || { echo "$name has no conformance fixture; pass --model <pack dir>" >&2; exit 1; }
  echo "[check] $name: no fixture, claim and parse only"
  # plugin check stops at the fixture, so claim and parse run through the plugin's own tests.
  step="$(sed -n 's/.*"test_step": *"\([^"]*\)".*/\1/p' "$ROOT/plugins/$name/plugin.json" | head -1)"
  (cd "$ROOT/plugins/$name" && "$ROOT/.zig-toolchain/zig" build "${step:-test}")
else
  mlx-serve plugin check "$dylib" "$@"
fi
