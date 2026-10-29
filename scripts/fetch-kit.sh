#!/usr/bin/env bash
# Stage the mlx-serve plugin kit named by KIT_VERSION at .kit/kit/. Idempotent.
# Source, in order: $MLX_SERVE_KIT (a staged kit folder, e.g. mlx-serve's zig-out/kit/...), the release asset
# mlx-serve-plugin-kit-<version>.tar.gz of ddalcu/mlx-serve, then, with MLX_SERVE_KIT_FROM_SOURCE=1, a kit built from
# mlx-serve at MLX_SERVE_REF (scripts/build-kit.sh; CI uses it while KIT_VERSION has no release). A plugin builds
# against exactly this kit: the kit's MLX pin must equal the server's.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="$(tr -d '[:space:]' < "$ROOT/KIT_VERSION")"
DEST="$ROOT/.kit/kit"

if [ -f "$DEST/.version" ] && [ "$(cat "$DEST/.version")" = "$VERSION" ]; then
  echo "[fetch-kit] kit $VERSION already staged"; exit 0
fi
rm -rf "$ROOT/.kit"; mkdir -p "$ROOT/.kit"

if [ -n "${MLX_SERVE_KIT:-}" ]; then
  [ -f "$MLX_SERVE_KIT/build.zig" ] || { echo "[fetch-kit] MLX_SERVE_KIT=$MLX_SERVE_KIT is not a kit folder" >&2; exit 1; }
  cp -R "$MLX_SERVE_KIT" "$DEST"
else
  URL="https://github.com/ddalcu/mlx-serve/releases/download/v$VERSION/mlx-serve-plugin-kit-$VERSION.tar.gz"
  TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
  if curl -fsSL "$URL" -o "$TMP/kit.tar.gz"; then
    mkdir "$TMP/x"; tar -xzf "$TMP/kit.tar.gz" -C "$TMP/x"
    mv "$TMP/x"/mlx-serve-plugin-kit-* "$DEST"
  elif [ "${MLX_SERVE_KIT_FROM_SOURCE:-0}" = 1 ]; then
    echo "[fetch-kit] no release asset for $VERSION; building the kit from mlx-serve $(cat "$ROOT/MLX_SERVE_REF")" >&2
    cp -R "$("$ROOT/scripts/build-kit.sh")" "$DEST"
  else
    echo "[fetch-kit] cannot download $URL" >&2
    echo "  Is v$VERSION released with its kit? Point MLX_SERVE_KIT at a staged kit (mlx-serve: zig build kit)," >&2
    echo "  or set MLX_SERVE_KIT_FROM_SOURCE=1 to build one from mlx-serve at MLX_SERVE_REF (scripts/build-kit.sh)." >&2
    exit 1
  fi
fi
echo "$VERSION" > "$DEST/.version"
echo "[fetch-kit] staged kit $VERSION at $DEST"
