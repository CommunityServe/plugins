#!/usr/bin/env bash
# Generates src/mlx_serve/ (git-ignored): the slices of mlx-serve this plugin runs, at the same paths as in mlx-serve's src/
# so their relative imports hold, with the kernels and the other files they embed. scripts/prepare.sh runs it.
# Usage: OVERLAY=<overlay/mlx_serve> scripts/sync-mlx-serve.sh <mlx-serve checkout at MLX_SERVE_REF>
# Two slices: scripts/slice.txt, the model (plugins/deepseek_v41 and what it imports), which is the plugin's root module,
# and scripts/slice-stream.txt, the expert streamer (module `mlx_stream`, its own `sdk` in sdk/weights/streamed/sdk/).
# A slice file the overlay has is kept as the overlay has it (prepare.sh copies the overlay over the result); every other
# is written from the checkout, unchanged but for the edits below, made here and nowhere else:
#   - a `//! copied_from: src/<path>@<sha>` first line;
#   - the imports of what the plugin kit ships by name (the ABI, the SDK's plugin root and helpers, generation, the test
#     fixture, the MLX layer mlx/ffi.zig) pointed at the kit's `mlx_serve` module;
#   - mlx-c handles a copy default-initialises (`mlx.mlx_array = .{}`) given their one field;
#   - every named test taken out, with the comment lines right above it: the plugin runs none of the copies' tests;
#   - the functions and declarations scripts/cut.txt names taken out, with their `///` lines;
#   - blank lines taken out, but one between two lines of multiline strings, which would join them.
# The read pool's C (c_sources below) is copied as it is.
# Files in EDGES are this plugin's own, at the paths of the mlx-serve files they stand in for, and never overwritten.
# Every other file a copy imports gets a stand-in: an empty file, so the import resolves while nothing of it is used.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
dst="$here/src/mlx_serve"

# Ours, at the paths of the mlx-serve files they stand in for: the kernel text lookup without mlx-serve's whole digest list.
EDGES="mlx/kernels.zig"

repo="${1:?mlx-serve checkout}"
sha="$(git -C "$repo" rev-parse HEAD)"
for e in $EDGES; do [ -f "$dst/$e" ] || { echo "the cut edge $e is missing" >&2; exit 1; }; done
python3 - "$repo" "$sha" "$dst" "$here/scripts" $EDGES <<'P'
import os, pathlib, re, sys
repo, sha, dst, scripts, edges = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3]), pathlib.Path(sys.argv[4]), set(sys.argv[5:])
slice_file = str(scripts / "slice.txt")
overlay = pathlib.Path(os.environ["OVERLAY"]) if os.environ.get("OVERLAY") else None
def listed(p):
    return [l.strip() for l in open(p) if l.strip() and not l.startswith("#")]
files = listed(scripts / "slice.txt") + listed(scripts / "slice-stream.txt")
if len(set(files)) != len(files):
    sys.exit("a file is in both slices")
if edges & set(files):
    sys.exit(f"a cut edge is also in a slice: {sorted(edges & set(files))}")
kit = {  # mlx-serve path -> what the kit's `mlx_serve` module calls it
    "abi/plugin.zig": "abi.plugin", "abi/common.zig": "abi.common", "abi/host.zig": "abi.host", "abi/dylib.zig": "abi.dylib",
    "abi/quant.zig": "abi.quant", "abi/media.zig": "abi.media", "sdk/plugin.zig": "sdk.plugin", "sdk/root.zig": "sdk",
    "sdk/config/generation.zig": "generation", "sdk/testing/fixture.zig": "sdk.fixture", "mlx/ffi.zig": "mlx",
}
handles = "mlx_array|mlx_stream|mlx_device|mlx_device_info|mlx_string|mlx_vector_array|mlx_vector_int|mlx_vector_string|" \
          "mlx_map_string_to_array|mlx_map_string_to_string|mlx_closure|mlx_fast_metal_kernel|mlx_fast_metal_kernel_config|" \
          "mlx_fast_cuda_kernel|mlx_fast_cuda_kernel_config|mlx_io_reader|mlx_distributed_group"
imp = re.compile(r'@import\("([^"]+\.zig)"\)')
# The kernels a copy names by its path from src/ (mlx/kernels.zig `source`), Metal only.
kernel_text = re.compile(r'"((?:kernels|plugins/[a-z0-9_]+/kernels)/[^"]+\.(?:metal|h))"')
# What a copy embeds by a literal path relative to itself, and the folders a copy embeds by computed names.
embed_text = re.compile(r'@embedFile\("([^"]+)"\)')
embed_dirs = ["sdk/quant/exl3/stream/kernels"]
# The read pool the streamer calls (build.zig compiles it as C), copied as it is; mlx-serve's other C and C++ beside it
# reaches MLX's private headers, which src/host_shims.zig stands in for.
c_sources = ["sdk/weights/streamed/csrc/q3_lookahead4_exl3.c", "sdk/weights/streamed/csrc/q3_lookahead4.h"]
wanted, kernels, embeds = set(), set(), set(c_sources)
test_head = re.compile(r'^test "((?:\\.|[^"\\])*)"\s*\{')
def code_of(line):
    """The line with its comment, strings and character literals blanked: what its braces are."""
    if line.lstrip().startswith("\\\\"):
        return ""
    out, i, n = [], 0, len(line)
    while i < n:
        c = line[i]
        if line.startswith("//", i):
            break
        if c in "\"'":
            i += 1
            while i < n and line[i] != c:
                i += 2 if line[i] == "\\" else 1
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)
def strip_tests(f, text, keep):
    """`text` without the named tests no build runs (`keep(name)` false), nor the comment lines right above each."""
    lines, out, i, n = text.split("\n"), [], 0, 0
    while i < len(lines):
        m = test_head.match(lines[i])
        if m is None or keep(m.group(1)):
            out.append(lines[i])
            i += 1
            continue
        depth, j = 0, i
        while j < len(lines):
            depth += code_of(lines[j]).count("{") - code_of(lines[j]).count("}")
            if depth <= 0:
                break
            j += 1
        if j == len(lines):
            sys.exit(f"{f}: test {m.group(1)!r} has no end: the test strip needs updating")
        while out and out[-1].lstrip().startswith("//") and not out[-1].startswith("//!"):
            out.pop()
        if j + 1 < len(lines) and lines[j + 1] == "" and (not out or out[-1] == ""):
            j += 1
        i, n = j + 1, n + 1
    return "\n".join(out), n
tests_out = 0
# What no build compiles or names, per file (scripts/cut.txt), and the scanners that find it: top-level functions and
# declarations and those of named containers, by path; what sits inside a function body or a declaration's value is
# that one's.
cut_lists = {}
for l in open(pathlib.Path(slice_file).parent / "cut.txt"):
    if l.strip() and not l.startswith("#"):
        k, v = l.split(":", 1)
        cut_lists[k.strip()] = set(v.split())
fn_head = re.compile(r'^\s*(?:pub\s+)?(?:export\s+)?(inline\s+|noinline\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(')
extern_head = re.compile(r'^\s*(?:pub\s+)?extern\b')
container_head = re.compile(r'^\s*(?:pub\s+)?const\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?::[^=]*)?=\s*(?:extern\s+|packed\s+)?'
                            r'(?:struct|union|enum|opaque)\b[^{};]*\{\s*$')
def fn_end(lines, i, col):
    """The last line of the function whose parameter list opens at lines[i][col]. Braces count once that list has
    closed (a parameter's type can hold a switch or a struct); a return type's braces close with the body's `{` still
    to come on the line, and the function ends where the body's close."""
    paren, brace, opened, j, k = 0, 0, False, i, col
    while j < len(lines):
        code = code_of(lines[j])
        while k < len(code):
            ch = code[k]
            if ch == "(":
                paren += 1
            elif ch == ")":
                paren -= 1
                opened = opened or paren == 0
            elif opened and paren == 0:
                if ch == "{":
                    brace += 1
                elif ch == "}":
                    brace -= 1
                    if brace == 0 and "{" not in code[k + 1:]:
                        return j
                elif ch == ";" and brace == 0:
                    return j
            k += 1
        j, k = j + 1, 0
    sys.exit("a function has no end: the function cut needs updating")
def functions(lines):
    """[(path, first line, last line)] of each function the scanner sees."""
    out, stack, depth, i = [], [], 0, 0
    while i < len(lines):
        m = fn_head.match(lines[i])
        if m and not extern_head.match(lines[i]):
            j = fn_end(lines, i, m.end() - 1)
            out.append((".".join([c for c, _ in stack] + [m.group(2)]), i, j))
            i = j + 1
            continue
        c = container_head.match(lines[i])
        if c:
            stack.append((c.group(1), depth))
        code = code_of(lines[i])
        depth += code.count("{") - code.count("}")
        while stack and depth <= stack[-1][1] and not c:
            stack.pop()
        i += 1
    return out
decl_head = re.compile(r'^\s*(?:pub\s+)?(?:export\s+)?(?:threadlocal\s+)?(?:const|var)\s+([A-Za-z_][A-Za-z0-9_]*)\b')
extern_decl = re.compile(r'^\s*(?:pub\s+)?extern\b.*?\b(?:fn|var|const)\s+([A-Za-z_][A-Za-z0-9_]*)')
block_head = re.compile(r'^\s*(?:test\b|comptime\s*\{)')
def block_end(lines, i):
    """The line where the braces a test or `comptime` block opens on lines[i] close."""
    depth, opened, j = 0, False, i
    while j < len(lines):
        code = code_of(lines[j])
        depth += code.count("{") - code.count("}")
        opened = opened or "{" in code
        if opened and depth <= 0:
            return j
        j += 1
    sys.exit("a block has no end: the cut needs updating")
def decl_end(lines, i):
    """The last line of the declaration starting at lines[i]: where its brackets close and a `;` ends it."""
    depth, j = 0, i
    while j < len(lines):
        for ch in code_of(lines[j]):
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            elif ch == ";" and depth == 0:
                return j
        j += 1
    sys.exit("a declaration has no end: the cut needs updating")
def declarations(lines):
    """[(path, first line, last line)] of each declaration at a container's level, a container's own included."""
    out, stack, depth, i = [], [], 0, 0
    while i < len(lines):
        line, code = lines[i], code_of(lines[i])
        if depth == (stack[-1][1] + 1 if stack else 0):
            m = fn_head.match(line)
            if m and not extern_head.match(line):
                i = fn_end(lines, i, m.end() - 1) + 1
                continue
            if block_head.match(line):
                i = block_end(lines, i) + 1
                continue
            c = container_head.match(line)
            if c:
                stack.append((c.group(1), depth, i))
                depth += code.count("{") - code.count("}")
                i += 1
                continue
            m = extern_decl.match(line) if extern_head.match(line) else decl_head.match(line)
            if m:
                j = decl_end(lines, i)
                out.append((".".join([s[0] for s in stack] + [m.group(1)]), i, j))
                i = j + 1
                continue
        depth += code.count("{") - code.count("}")
        while stack and depth <= stack[-1][1]:
            name, _, start = stack.pop()
            out.append((".".join([s[0] for s in stack] + [name]), start, i))
        i += 1
    return out
def cut_names(f, text):
    paths = cut_lists.get(f)
    if not paths:
        return text, 0
    lines = text.split("\n")
    found = [fn for fn in functions(lines) if fn[0] in paths] + [d for d in declarations(lines) if d[0] in paths]
    missing = sorted(paths - {fn[0] for fn in found})
    if missing:
        sys.exit(f"{f}: scripts/cut.txt names what it lacks: {missing}")
    drop = set()
    for _, a, b in found:
        while a > 0 and lines[a - 1].lstrip().startswith("///"):
            a -= 1
        # a doc comment stands for the declaration below it across blank lines too
        k = a
        while k > 0 and lines[k - 1].strip() == "":
            k -= 1
        if k < a and k > 0 and lines[k - 1].lstrip().startswith("///"):
            a = k
            while a > 0 and lines[a - 1].lstrip().startswith("///"):
                a -= 1
        drop.update(range(a, b + 1))
        if b + 1 < len(lines) and lines[b + 1] == "" and a > 0 and lines[a - 1] == "":
            drop.add(b + 1)
    return "\n".join(l for k, l in enumerate(lines) if k not in drop), len(found)
cut_out = 0
def strip_blanks(text):
    """`text` without its blank lines, but one between two lines of multiline strings, which would join them."""
    lines, out = text.split("\n"), []
    for i, l in enumerate(lines):
        if l.strip() == "":
            before = next((x for x in reversed(out) if x.strip()), "")
            after = next((x for x in lines[i + 1:] if x.strip()), "")
            if before.lstrip().startswith("\\\\") and after.lstrip().startswith("\\\\"):
                out.append(l)
            continue
        out.append(l)
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")
def note(f, text):
    for m in imp.finditer(text):
        wanted.add(os.path.normpath(os.path.join(os.path.dirname(f), m.group(1))))
    for m in kernel_text.finditer(text):
        if (repo / "src" / m.group(1)).is_file():
            kernels.add(m.group(1))
    for m in embed_text.finditer(text):
        t = os.path.normpath(os.path.join(os.path.dirname(f), m.group(1)))
        if (repo / "src" / t).is_file():
            embeds.add(t)
for f in files:
    if overlay is not None and (overlay / f).is_file():
        text = (overlay / f).read_text()
        out = dst / f
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        note(f, text)
        continue
    text = (repo / "src" / f).read_text()
    def to_kit(m):
        t = os.path.normpath(os.path.join(os.path.dirname(f), m.group(1)))
        return f'@import("mlx_serve").{kit[t]}' if t in kit else m.group(0)
    text = imp.sub(to_kit, text)
    text = re.sub(r'\b(mlx\.(?:' + handles + r')) = \.\{\}', r'\1 = .{ .ctx = null }', text)
    text, n = strip_tests(f, text, lambda name: False)
    tests_out += n
    text, n = cut_names(f, text)
    cut_out += n
    text = strip_blanks(text)
    out = dst / f
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"//! copied_from: src/{f}@{sha}\n" + text)
    note(f, text)
for e in edges:
    note(e, (dst / e).read_text())
# The plugin's own code (src/ outside the copies, and the root) names kernels of the same tree.
for p in sorted(dst.parent.rglob("*.zig")) + sorted(dst.parent.parent.glob("*.zig")):
    if dst not in p.parents:
        for m in kernel_text.finditer(p.read_text()):
            if (repo / "src" / m.group(1)).is_file():
                kernels.add(m.group(1))
for d in embed_dirs:
    for p in sorted((repo / "src" / d).rglob("*")):
        if p.is_file():
            embeds.add(str(p.relative_to(repo / "src")))
for k in sorted(kernels | embeds):
    out = dst / k
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes((repo / "src" / k).read_bytes())
stand_ins = sorted(w for w in wanted if w not in files and w not in edges and w not in kit and not w.startswith(".."))
for s in stand_ins:
    out = dst / s
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"//! stand_in: src/{s}@{sha}: the copies import it, this plugin uses nothing of it.\n")
unknown = sorted(set(cut_lists) - set(files))
if unknown:
    sys.exit(f"scripts/cut.txt names files that are not copies: {unknown}")
known = set(files) | edges | set(stand_ins) | kernels | embeds | {"MLX_SERVE_COMMIT"}
for p in sorted(dst.rglob("*")):
    rel = str(p.relative_to(dst))
    if p.is_file() and rel not in known:
        p.unlink()
        print(f"removed {rel} (no longer in the slices)")
print(f"synced {len(files)} files, {len(kernels)} kernels, {len(embeds)} embedded files, {len(stand_ins)} stand-ins from {sha[:8]}; {tests_out} named tests and {cut_out} functions and declarations left out")
P
find "$dst" -type d -empty -delete
echo "$sha" > "$dst/MLX_SERVE_COMMIT"
