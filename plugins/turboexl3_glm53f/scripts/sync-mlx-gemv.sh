#!/usr/bin/env bash
# Copies MLX's gemv.h (mlx/backend/metal/kernels/gemv.h) into src/vendor/mlx/gemv.h, unchanged: the text
# TURBOEXL3_GLM53F_ROUTER_ONE1's launch compiles (src/fused_router.zig, NOTICE: MLX).
# Usage: scripts/sync-mlx-gemv.sh <MLX checkout at the plugin kit's MLX commit>
# The copy is pinned: the checkout must be at MLX_COMMIT and the file must hash to GEMV_SHA256. A kit with another MLX
# moves both pins here and in NOTICE together. The edits the launch needs are not made here or in the copy:
# src/fused_router.zig makes them at compile time (`gemv_edits`), each found exactly once, so a changed upstream text
# fails the build.
set -euo pipefail
MLX_COMMIT=c69303b00f5f61860436e832f23fa8c34b96f93a
GEMV_SHA256=4d72d35da3662554fe46c869691d188c407c56ffd62d3876a6df046ad014a551
here="$(cd "$(dirname "$0")/.." && pwd)"
repo="${1:?MLX checkout}"
src="$repo/mlx/backend/metal/kernels/gemv.h"
head="$(git -C "$repo" rev-parse HEAD)"
[ "$head" = "$MLX_COMMIT" ] || { echo "the MLX checkout is at $head, not $MLX_COMMIT" >&2; exit 1; }
[ -z "$(git -C "$repo" status --porcelain -- mlx/backend/metal/kernels/gemv.h)" ] || { echo "gemv.h has local changes in $repo" >&2; exit 1; }
got="$(shasum -a 256 "$src" | cut -d' ' -f1)"
[ "$got" = "$GEMV_SHA256" ] || { echo "gemv.h at $MLX_COMMIT hashes to $got, not $GEMV_SHA256" >&2; exit 1; }
mkdir -p "$here/src/vendor/mlx"
cp "$src" "$here/src/vendor/mlx/gemv.h"
echo "src/vendor/mlx/gemv.h: MLX $MLX_COMMIT, sha256 $GEMV_SHA256"
