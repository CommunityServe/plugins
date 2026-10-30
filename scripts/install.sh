#!/usr/bin/env bash
# Compile a plugin from source and install it into mlx-serve. Usage: scripts/install.sh <name> [--home <dir>]
# Plugins are always compiled here, from the source you can read; nothing downloads a prebuilt binary.
# --home installs under a throwaway HOME (plugins.json enabled) so a running app's ~/.mlx-serve is left alone.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
name="${1:?usage: scripts/install.sh <name> [--home <dir>]}"; shift
command -v mlx-serve >/dev/null || { echo "mlx-serve is not on PATH (install it from github.com/ddalcu/mlx-serve)" >&2; exit 1; }
# `mlx-serve plugin` alone prints its usage and exits 1, so its output is read before grep (pipefail would fail the probe).
usage="$(mlx-serve plugin 2>&1 || true)"
grep -q "plugin install" <<<"$usage" || { echo "this mlx-serve ($(mlx-serve --version 2>&1 | tail -1)) has no 'plugin' command: use a release that ships the plugin kit" >&2; exit 1; }
"$ROOT/scripts/build.sh" "$name"
dylib="$ROOT/plugins/$name/zig-out/lib/lib$name.dylib"
if [ "${1:-}" = "--home" ]; then
  home="${2:?--home needs a directory}"
  mkdir -p "$home/.mlx-serve"
  HOME="$home" mlx-serve plugin install "$dylib" --yes
  printf '{\n  "disabled" : [\n\n  ],\n  "enabled" : true\n}' > "$home/.mlx-serve/plugins.json"
  echo "[install] installed under HOME=$home; serve with: HOME=$home mlx-serve --model <pack dir>"
else
  mlx-serve plugin install "$dylib"
  echo "[install] plugins load only when ~/.mlx-serve/plugins.json says \"enabled\": true (Settings > Server in the app)."
  echo "[install] restart the server; the log shows '[plugin] $name serves <model_type>' for a pack it serves."
fi
