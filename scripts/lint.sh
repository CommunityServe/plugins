#!/usr/bin/env bash
# Repo rules a script can check; CI runs it first. Exit 1 lists every violation.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 - "$ROOT" <<'P'
import json, pathlib, re, subprocess, sys
root = pathlib.Path(sys.argv[1])
bad = []
kit = (root / "KIT_VERSION").read_text().strip()
if not re.fullmatch(r"\d+\.\d+\.\d+", kit):
    bad.append(f"KIT_VERSION {kit!r} is not a release version")
tracked = subprocess.run(["git", "-C", str(root), "ls-files"], capture_output=True, text=True).stdout.split("\n")
for f in tracked:
    if f.endswith((".dylib", ".so", ".a", ".o")):
        bad.append(f"{f}: binaries are never committed, plugins are compiled from source")
for d in sorted((root / "plugins").glob("*/")):
    n = d.name
    for need in ("plugin.json", "README.md", "build.zig", "build.zig.zon", "plugin.zig", "config.zig"):
        if not (d / need).exists():
            bad.append(f"plugins/{n}: missing {need}")
    if not ((d / "LICENSE").exists() or (root / "LICENSE").exists()):
        bad.append(f"plugins/{n}: no license")
    pj = d / "plugin.json"
    if not pj.exists():
        continue
    try:
        p = json.loads(pj.read_text())
    except Exception as e:
        bad.append(f"plugins/{n}/plugin.json: {e}"); continue
    if p.get("name") != n:
        bad.append(f"plugins/{n}/plugin.json: name must equal the folder")
    if p.get("kind") not in ("model", "media", "quant"):
        bad.append(f"plugins/{n}/plugin.json: kind is model, media or quant")
    for k in ("title", "version", "summary", "models", "requires_mlx_serve", "fixture", "license", "maintainers"):
        if k not in p:
            bad.append(f"plugins/{n}/plugin.json: missing {k}")
    if p.get("fixture") == "none" and not p.get("fixture_reason"):
        bad.append(f"plugins/{n}/plugin.json: fixture none needs a fixture_reason")
    title = p.get("title")
    if title is not None and not (isinstance(title, str) and 0 < len(title.strip()) <= 60):
        bad.append(f"plugins/{n}/plugin.json: title is the name people read, 1 to 60 characters")
    home = p.get("homepage")
    if home is not None and not re.fullmatch(r"https://[^\s/]+(/\S*)?", home):
        bad.append(f"plugins/{n}/plugin.json: homepage must be an https URL")
    icon = p.get("icon")
    if icon is not None:
        f = d / str(icon)
        if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*\.png", str(icon)) or not f.is_file():
            bad.append(f"plugins/{n}/plugin.json: icon must name a .png in the plugin folder")
        elif f.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n" or f.stat().st_size > 256 * 1024:
            bad.append(f"plugins/{n}/{icon}: icon must be a PNG of at most 256 KB")
    if not re.fullmatch(r"[a-z0-9_]{1,64}", n):
        bad.append(f"plugins/{n}: folder name must be [a-z0-9_]")
    zon = d / "build.zig.zon"
    if zon.exists():
        z = zon.read_text()
        if f".name = .{n}," not in z:
            bad.append(f"plugins/{n}/build.zig.zon: .name must be .{n}")
        if '.path = "../../.kit/kit"' not in z:
            bad.append(f"plugins/{n}/build.zig.zon: the kit dependency must be .path = \"../../.kit/kit\"")
if bad:
    print("\n".join(bad), file=sys.stderr); sys.exit(1)
print("lint: ok")
P
