#!/usr/bin/env bash
# Build the mlx-serve plugin kit from source at MLX_SERVE_REF, for a KIT_VERSION whose release has no kit asset yet.
# Same steps as mlx-serve's release workflow: the pinned mlx + mlx-c submodules built by its scripts/build-mlx.sh (NAX
# kernels in), then `zig build kit`. Needs Xcode with the Metal toolchain, cmake and webp (brew). Prints the kit folder.
# Idempotent: a checkout at the ref and a staged lib/mlx are reused (CI caches .kit-src/lib/mlx).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REF="$(tr -d '[:space:]' < "$ROOT/MLX_SERVE_REF")"
VERSION="$(tr -d '[:space:]' < "$ROOT/KIT_VERSION")"
SRC="$ROOT/.kit-src"

if [ "$(git -C "$SRC" rev-parse HEAD 2>/dev/null)" != "$REF" ]; then
  mkdir -p "$SRC"
  git -C "$SRC" init -q
  git -C "$SRC" remote remove origin 2>/dev/null || true
  git -C "$SRC" remote add origin https://github.com/ddalcu/mlx-serve
  git -C "$SRC" fetch -q --depth 1 origin "$REF"
  git -C "$SRC" checkout -q --detach FETCH_HEAD
fi
git -C "$SRC" submodule update -q --init --depth 1 lib/mlx-src lib/mlxc-src >&2

xcrun -sdk macosx metal --version >/dev/null 2>&1 || {
  echo "[build-kit] the Metal toolchain is missing: xcodebuild -downloadComponent MetalToolchain" >&2
  exit 1
}
cd "$SRC"
bash scripts/fetch-zig.sh >&2
if [ ! -f lib/mlx/lib/libmlxc.dylib ] || [ ! -f lib/mlx/lib/mlx.metallib ] || [ ! -f lib/mlx/.version ]; then
  bash scripts/build-mlx.sh >&2
fi
.zig-toolchain/zig build kit -Dversion="$VERSION" >&2
KIT="$SRC/zig-out/kit/mlx-serve-plugin-kit-$VERSION"
[ -f "$KIT/build.zig" ] || { echo "[build-kit] no kit at $KIT" >&2; exit 1; }
echo "$KIT"
