#!/usr/bin/env bash
# Build plugins into plugins/<name>/zig-out/lib/lib<name>.dylib. Usage: scripts/build.sh <name>... | --all
# Every plugin must compile (and pass its tests) at every commit on main: `--all` is the CI gate and the
# check an agent runs before it pushes.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/setup.sh" >/dev/null
ZIG="$ROOT/.zig-toolchain/zig"

names=()
if [ "${1:-}" = "--all" ]; then
  for d in "$ROOT"/plugins/*/; do names+=("$(basename "$d")"); done
else
  names=("$@")
fi
[ ${#names[@]} -gt 0 ] || { echo "usage: scripts/build.sh <plugin>... | --all" >&2; exit 2; }

failed=()
for n in "${names[@]}"; do
  dir="$ROOT/plugins/$n"
  [ -f "$dir/build.zig" ] || { echo "no such plugin: $n" >&2; exit 2; }
  echo "── $n"
  # The plugin's own MLX-free tests: the kit's `test` step, or the step plugin.json names as "test_step".
  step="$(sed -n 's/.*"test_step": *"\([^"]*\)".*/\1/p' "$dir/plugin.json" 2>/dev/null | head -1)"
  # A plugin that generates part of its source (plugin.json's "prepare": a script path in its folder) does it first.
  prepare="$(sed -n 's/.*"prepare": *"\([^"]*\)".*/\1/p' "$dir/plugin.json" 2>/dev/null | head -1)"
  if [ -n "$prepare" ] && ! (cd "$dir" && bash "$prepare"); then failed+=("$n"); continue; fi
  if (cd "$dir" && "$ZIG" build "${step:-test}" && "$ZIG" build -Doptimize=ReleaseFast); then
    echo "   built $dir/zig-out/lib/lib$n.dylib"
  else
    failed+=("$n")
  fi
done
if [ ${#failed[@]} -gt 0 ]; then echo "FAILED: ${failed[*]}" >&2; exit 1; fi
