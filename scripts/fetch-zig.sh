#!/usr/bin/env bash
# Stage the pinned Zig at .zig-toolchain/ (the same pin and path as mlx-serve). Idempotent.
set -euo pipefail

ZIG_VERSION="${ZIG_VERSION:-0.17.0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/.zig-toolchain"

[ "$(uname -s)" = Darwin ] || { echo "[fetch-zig] outside plugins are macOS only" >&2; exit 1; }
case "$(uname -m)" in arm64) ARCH=aarch64 ;; x86_64) ARCH=x86_64 ;; *) echo "[fetch-zig] unsupported CPU $(uname -m)" >&2; exit 1 ;; esac

if [ -x "$DEST/zig" ] && [ "$(cat "$DEST/.version" 2>/dev/null)" = "$ZIG_VERSION" ]; then
  echo "[fetch-zig] zig $ZIG_VERSION already staged"; exit 0
fi

NAME="zig-$ARCH-macos-$ZIG_VERSION"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
curl -fsSL "https://ziglang.org/download/$ZIG_VERSION/$NAME.tar.xz" -o "$TMP/zig.tar.xz"
tar -xJf "$TMP/zig.tar.xz" -C "$TMP"
rm -rf "$DEST"; mv "$TMP/$NAME" "$DEST"
echo "$ZIG_VERSION" > "$DEST/.version"
echo "[fetch-zig] staged zig $ZIG_VERSION at $DEST"
