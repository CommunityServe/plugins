#!/usr/bin/env bash
# Regenerate plugins.json from plugins/*/plugin.json. CI fails a PR whose plugins.json differs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 - "$ROOT" <<'P'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
kit = (root / "KIT_VERSION").read_text().strip()
plugins = []
for f in sorted((root / "plugins").glob("*/plugin.json")):
    p = json.loads(f.read_text())
    assert f.parent.name == p["name"], f"{f}: name must equal its folder"
    plugins.append({k: p[k] for k in ("name", "kind", "version", "summary", "models", "requires_mlx_serve", "fixture") if k in p})
(root / "plugins.json").write_text(json.dumps({"kit": kit, "plugins": plugins}, indent=2) + "\n")
print(f"plugins.json: {len(plugins)} plugin(s), kit {kit}")
P
