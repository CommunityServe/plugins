#!/usr/bin/env bash
# Scaffold plugins/<name>/ from the kit's example, ready to build against this repo's kit.
# Usage: scripts/new-plugin.sh <name> [--media|--quant]      (name: [a-z0-9_], the dylib and the plugin's identity)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
name="${1:?usage: scripts/new-plugin.sh <name> [--media|--quant]}"
kind=model; ex=_example
case "${2:-}" in --media) kind=media; ex=_example_media ;; --quant) kind=quant; ex=_example_quant ;; "") ;; *) echo "unknown option ${2}" >&2; exit 2 ;; esac
[[ "$name" =~ ^[a-z0-9_]{1,64}$ ]] || { echo "name must be [a-z0-9_], 1 to 64 bytes" >&2; exit 2; }
dest="$ROOT/plugins/$name"
[ ! -e "$dest" ] || { echo "plugins/$name already exists" >&2; exit 1; }
"$ROOT/scripts/fetch-kit.sh" >/dev/null
src="$ROOT/.kit/kit/examples/$ex"
[ -d "$src" ] || { echo "the kit has no examples/$ex" >&2; exit 1; }

mkdir -p "$dest"
for f in build.zig plugin.zig config.zig; do sed "s/$ex/$name/g" "$src/$f" > "$dest/$f"; done
fingerprint="$(python3 -c "import zlib,secrets,sys; print('0x%016x' % ((zlib.crc32(sys.argv[1].encode()) << 32) | max(1, secrets.randbelow(0xfffffffe))))" "$name")"
cat > "$dest/build.zig.zon" <<Z
.{
    .name = .$name,
    .version = "0.0.0",
    .fingerprint = $fingerprint,
    .minimum_zig_version = "0.17.0",
    .dependencies = .{ .mlx_serve_kit = .{ .path = "../../.kit/kit" } },
    .paths = .{ "build.zig", "build.zig.zon", "plugin.zig", "config.zig" },
}
Z
cat > "$dest/plugin.json" <<J
{
  "name": "$name",
  "kind": "$kind",
  "version": "0.1.0",
  "summary": "One line: what this plugin serves.",
  "models": [],
  "requires_mlx_serve": ">=$(tr -d '[:space:]' < "$ROOT/KIT_VERSION")",
  "fixture": "kit",
  "maintainers": [],
  "license": "MIT"
}
J
cat > "$dest/README.md" <<R
# $name

One paragraph: what it serves, which packs it claims, what it does not support.

## Measured

| Model | Hardware | Decode tok/s | Prefill tok/s |
|---|---|---|---|

## Not supported
R
echo "[new-plugin] plugins/$name ($kind) from the kit's $ex"
echo "  next: edit plugins/$name/{config,plugin}.zig, then scripts/build.sh $name"
