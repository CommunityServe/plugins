#!/usr/bin/env bash
# Generates src/mlx_serve/ (git-ignored): the slice of mlx-serve this plugin runs (scripts/slice.txt), at the same
# paths as in mlx-serve's src/ so their relative imports hold, with the Metal kernels they and this plugin's own code
# embed. scripts/prepare.sh runs it.
# Usage: OVERLAY=<overlay/mlx_serve> scripts/sync-mlx-serve.sh <mlx-serve checkout at MLX_SERVE_REF>
# OVERLAY holds what is not mlx-serve's text at that ref: the cut edges (EDGES), and each copy, kernel and abi/quant.zig
# whose text at mlx-serve 1b3e624b (ddalcu/mlx-serve#1005's head), as this script writes it from there, differs from
# what it writes at the ref. A slice file the overlay has is kept as the overlay has it; everything else is written
# from the checkout, so it follows mlx-serve. prepare.sh then copies the overlay over the result.
# The copies are unchanged but for the edits below, made here and nowhere else:
#   - a `//! copied_from: src/<path>@<sha>` first line;
#   - the imports of what the plugin kit ships (abi, the SDK's plugin helpers and root, generation, the test fixture)
#     pointed at the kit's `mlx_serve` module;
#   - mlx-c handles a copy default-initialises (`mlx.mlx_array = .{}`) given their one field, since the kit's `mlx_c`
#     declares them without field defaults;
#   - in the built-in's root (plugins/qwen4_exp/plugin.zig), `pub` on the functions this plugin's table takes from it,
#     and on the model's `ownSlot`, which the prefix cache's restores reach the slot by;
#   - in plugins/qwen4_exp/qwen4_exp.zig, `pub` on the n-gram table's `parse`, which the direct load's open calls;
#   - in plugins/qwen4_exp/trunk/fusion.zig, `pub` on the calls into the trunk format's fused entries and their
#     helpers, which the direct model (src/direct) makes itself;
#   - abi/quant.zig, the quant ABI as #1005 extends it (`QuantTable`'s fused members, read only between the copies and
#     turboexl3's table, both inside this dylib): each declaration the kit's quant ABI has unchanged is re-exported from
#     the kit, so the types the host's tables carry are one type on both sides;
#   - every named test but #1005's EXL3 tests taken out, with the comment lines right above it: a copy keeps the named
#     tests a gpu-test or mlx-free-test filter in build.zig picks, and every unnamed `test {}` block;
#   - the functions and declarations scripts/cut.txt names taken out, with their `///` lines: what no build of the
#     plugin compiles or names (a member as Container.name). A name the copy lacks stops the sync;
#   - blank lines taken out, but one between two lines of multiline strings, which would join them.
# Files in EDGES are this plugin's own, at the paths of the mlx-serve files they stand in for, and never overwritten.
# Every other file a copy imports gets a stand-in: an empty file, so the import resolves while nothing of it is used
# (a use is a compile error, never a silent difference). Zig loads every file a file imports, used or not, so a file
# the plugin's builds reach no code of is a stand-in rather than a copy: slice.txt leaves it out.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
dst="$here/src/mlx_serve"

# Ours, at the paths of the mlx-serve files they stand in for: the cut edges where the slice meets the engine, the host
# and mlx-serve's own MLX layer (which this plugin replaces with mlx-c through the kit's `mlx_c`).
EDGES="env.zig mlx/ffi.zig mlx/array.zig mlx/device.zig mlx/dtype.zig mlx/kernel_cache.zig mlx/kernels.zig sdk/util/log.zig
engines/mlx/transformer.zig engines/mlx/cluster/distributed.zig engines/mlx/plugin/config.zig sdk/blocks/dit/w8a8.zig
plugins/hy_v3/load.zig sdk/util/shared.zig engines/mlx/generate/generator.zig engines/mlx/spec/mtp_head_ref.zig
sdk/ane/offload.zig sdk/quant/formats.zig engines/mlx/load/exl3_bank.zig sdk/kv/cache.zig engines/mlx/forward_ctx.zig
plugins/qwen4_exp/trunk.zig sdk/config/model.zig"

repo="${1:?mlx-serve checkout}"
sha="$(git -C "$repo" rev-parse HEAD)"
for e in $EDGES; do [ -f "$dst/$e" ] || { echo "the cut edge $e is missing" >&2; exit 1; }; done
python3 - "$repo" "$sha" "$dst" "$here/scripts/slice.txt" $EDGES <<'P'
import os, pathlib, re, sys
repo, sha, dst, slice_file, edges = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3]), sys.argv[4], set(sys.argv[5:])
overlay = pathlib.Path(os.environ["OVERLAY"]) if os.environ.get("OVERLAY") else None
files = [l.strip() for l in open(slice_file) if l.strip() and not l.startswith("#")]
if edges & set(files):
    sys.exit(f"a cut edge is also in the slice: {sorted(edges & set(files))}")
kit = {  # mlx-serve path -> what the kit's `mlx_serve` module calls it
    "abi/plugin.zig": "abi.plugin", "abi/common.zig": "abi.common",
    "abi/host.zig": "abi.host", "abi/dylib.zig": "abi.dylib", "sdk/plugin.zig": "sdk.plugin", "sdk/root.zig": "sdk",
    "sdk/config/generation.zig": "generation", "sdk/testing/fixture.zig": "sdk.fixture",
}
handles = "mlx_array|mlx_stream|mlx_device|mlx_device_info|mlx_string|mlx_vector_array|mlx_vector_int|mlx_vector_string|" \
          "mlx_map_string_to_array|mlx_map_string_to_string|mlx_closure|mlx_fast_metal_kernel|mlx_fast_metal_kernel_config|" \
          "mlx_fast_cuda_kernel|mlx_fast_cuda_kernel_config|mlx_io_reader|mlx_distributed_group"
imp = re.compile(r'@import\("([^"]+\.zig)"\)')
# The kernels a copy names, Metal only: mlx/ffi.zig answers the CUDA questions at compile time, so no CUDA kernel is built in.
kernel_text = re.compile(r'"((?:kernels|plugins/qwen4_exp/kernels)/[^"]+\.(?:metal|h))"')
wanted, kernels = set(), set()
# The named tests a build of the plugin runs: the names build.zig's filters pick.
build_zig = (dst.parent.parent / "build.zig").read_text()
test_filters = []
for name in ("gpu_filters", "mlx_free_filters"):
    m = re.search(r'^const ' + name + r' = \[_\]\[\]const u8\{(.*?)\};', build_zig, flags=re.M | re.S)
    if m is None:
        sys.exit(f"build.zig: no `{name}`: the test strip needs updating")
    test_filters += re.findall(r'"((?:\\.|[^"\\])*)"', m.group(1))
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
for f in files:
    if overlay is not None and (overlay / f).is_file():
        # Ours, already as a copy is written: kept as it is, its imports still name what the slice needs.
        text = (overlay / f).read_text()
        out = dst / f
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        for m in imp.finditer(text):
            wanted.add(os.path.normpath(os.path.join(os.path.dirname(f), m.group(1))))
        for m in kernel_text.finditer(text):
            if (repo / "src" / m.group(1)).is_file():
                kernels.add(m.group(1))
        continue
    text = (repo / "src" / f).read_text()
    def to_kit(m):
        t = os.path.normpath(os.path.join(os.path.dirname(f), m.group(1)))
        return f'@import("mlx_serve").{kit[t]}' if t in kit else m.group(0)
    text = imp.sub(to_kit, text)
    text = re.sub(r'\b(mlx\.(?:' + handles + r')) = \.\{\}', r'\1 = .{ .ctx = null }', text)
    if f == "plugins/qwen4_exp/plugin.zig":
        for fn in ("load", "unload", "forwardSettle", "releaseSlot", "meters", "residentBill"):
            text, n = re.subn(r'^fn ' + fn + r'\(', 'pub fn ' + fn + '(', text, flags=re.M)
            if n != 1:
                sys.exit(f"plugins/qwen4_exp/plugin.zig: `fn {fn}(` is not there once: the table's edit needs updating")
        # The model's slot lookup, for the prefix cache's restores (src/cache/aux.zig).
        text, n = re.subn(r'^    fn ownSlot\(', '    pub fn ownSlot(', text, flags=re.M)
        if n != 1:
            sys.exit("plugins/qwen4_exp/plugin.zig: `fn ownSlot(` is not there once: the prefix cache's edit needs updating")
    if f == "plugins/qwen4_exp/qwen4_exp.zig":
        # The n-gram table's parse, for the direct load's open (src/direct/load.zig), which starts no prefetch readers.
        text, n = re.subn(r'^    fn parse\(map: ', '    pub fn parse(map: ', text, flags=re.M)
        if n != 1:
            sys.exit("plugins/qwen4_exp/qwen4_exp.zig: `fn parse(map: ` is not there once: the direct load's edit needs updating")
    if f == "plugins/qwen4_exp/trunk/fusion.zig":
        # The fused entries' calls, for the direct model's layers (src/direct).
        for fn in ("refused", "strm", "own", "freeAll", "callAttnIn", "callAttnPrep", "callLinearPreparedIdx", "callGatedOut", "callQPlanes", "callSharedSwiglu"):
            text, n = re.subn(r'^fn ' + fn + r'\(', 'pub fn ' + fn + '(', text, flags=re.M)
            if n != 1:
                sys.exit(f"plugins/qwen4_exp/trunk/fusion.zig: `fn {fn}(` is not there once: the direct model's edit needs updating")
    text, n = strip_tests(f, text, lambda name: any(x in name for x in test_filters))
    tests_out += n
    text, n = cut_names(f, text)
    cut_out += n
    text = strip_blanks(text)
    out = dst / f
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"//! copied_from: src/{f}@{sha}\n" + text)
    for m in imp.finditer(text):
        wanted.add(os.path.normpath(os.path.join(os.path.dirname(f), m.group(1))))
    for m in kernel_text.finditer(text):
        if (repo / "src" / m.group(1)).is_file():
            kernels.add(m.group(1))
# The quant ABI the copies and turboexl3's table share, over the kit's.
kit_dir = pathlib.Path(os.environ.get("MLX_SERVE_KIT") or (dst.parent.parent.parent.parent / ".kit/kit"))
def decls(text):
    """Each top-level `pub const NAME = ...;`: name -> (start, end, its text without comments or blanks)."""
    out, i, n = {}, 0, len(text)
    for m in re.finditer(r'^pub const (\w+) = ', text, flags=re.M):
        i, depth, body = m.end(), 0, []
        while i < n:
            if text.startswith("//", i):
                i = text.index("\n", i)
                continue
            ch = text[i]
            if ch == '"':
                j = i + 1
                while text[j] != '"':
                    j += 2 if text[j] == "\\" else 1
                body.append(text[i:j + 1]); i = j + 1
                continue
            if ch in "({[": depth += 1
            elif ch in ")}]": depth -= 1
            elif ch == ";" and depth == 0: break
            if not ch.isspace(): body.append(ch)
            i += 1
        out[m.group(1)] = (m.start(), i + 1, "".join(body))
    return out
if overlay is not None and (overlay / "abi/quant.zig").is_file():
    (dst / "abi").mkdir(parents=True, exist_ok=True)
    (dst / "abi/quant.zig").write_text((overlay / "abi/quant.zig").read_text())
else:
    qsrc = (repo / "src/abi/quant.zig").read_text()
    kq = decls((kit_dir / "src/abi/quant.zig").read_text())
    mine = decls(qsrc)
    same = sorted((name for name in mine if name in kq and kq[name][2] == mine[name][2]), key=lambda k: -mine[k][0])
    for name in same:
        a, b, _ = mine[name]
        qsrc = qsrc[:a] + f"pub const {name} = kit_quant.{name};" + qsrc[b:]
    qsrc = qsrc.replace('@import("host.zig")', '@import("mlx_serve").abi.host').replace('@import("common.zig")', '@import("mlx_serve").abi.common')
    qsrc = qsrc.replace('const std = @import("std");', 'const std = @import("std");\nconst kit_quant = @import("mlx_serve").abi.quant;', 1)
    qsrc, n = strip_tests("abi/quant.zig", qsrc, lambda name: any(x in name for x in test_filters))
    tests_out += n
    qsrc = strip_blanks(qsrc)
    (dst / "abi").mkdir(parents=True, exist_ok=True)
    (dst / "abi/quant.zig").write_text(f"//! copied_from: src/abi/quant.zig@{sha}\n" + qsrc)
    print(f"abi/quant.zig: {len(same)} declarations re-exported from the kit ({', '.join(sorted(same))}); its own: {', '.join(sorted(set(mine) - set(same)))}")
# The cut edges' own imports resolve the same way.
for e in edges:
    for m in imp.finditer((dst / e).read_text()):
        wanted.add(os.path.normpath(os.path.join(os.path.dirname(e), m.group(1))))
# The plugin's own code (src/direct and the rest outside the copies) embeds kernels of the same tree.
for p in sorted(dst.parent.rglob("*.zig")) + sorted(dst.parent.parent.glob("*.zig")):
    if dst not in p.parents:
        for m in kernel_text.finditer(p.read_text()):
            if (repo / "src" / m.group(1)).is_file():
                kernels.add(m.group(1))
# hc_prefill_nax.zig names its kernel files by a comptime part: every one of that family.
kernels |= {str(p.relative_to(repo / "src")) for p in (repo / "src/kernels/hc").glob("hc_prefill_*.metal")}
for k in sorted(kernels):
    out = dst / k
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes((repo / "src" / k).read_bytes())
stand_ins = sorted(w for w in wanted if w not in files and w not in edges and w not in kit and w != "abi/quant.zig" and not w.startswith(".."))
for s in stand_ins:
    out = dst / s
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"//! stand_in: src/{s}@{sha}: the copies import it, this plugin uses nothing of it.\n")
unknown = sorted(set(cut_lists) - set(files))
if unknown:
    sys.exit(f"scripts/cut.txt names files that are not copies: {unknown}")
known = set(files) | edges | set(stand_ins) | set(kernels) | {"MLX_SERVE_COMMIT", "abi/quant.zig"}
for p in sorted(dst.rglob("*")):
    rel = str(p.relative_to(dst))
    if p.is_file() and rel not in known:
        p.unlink()
        print(f"removed {rel} (no longer in the slice)")
print(f"synced {len(files)} files, {len(kernels)} kernels, {len(stand_ins)} stand-ins from {sha[:8]}; {tests_out} named tests no build runs and {cut_out} functions and declarations none reaches left out")
P
find "$dst" -type d -empty -delete
# No env levers: the copies reach the environment only through env.zig (ours), but for the variables mlx-serve's own
# tests read (a model or fixture path, a test's context length), which nothing outside a test reaches.
test_vars='QWEN4_FIXTURE[A-Z0-9_]*|QWEN4_[A-Z0-9_]*TEST_MODEL|SINGLE_MTP_TEST_CONTEXT'
reads="$(grep -rnE 'std\.(c|posix)\.getenv\(|getEnvVar|getEnvMap|std\.c\.environ' "$dst" | grep -vE "getenv\(\"($test_vars)\"\)" || true)"
[ -z "$reads" ] || { echo "the copies read the environment:" >&2; echo "$reads" >&2; exit 1; }
echo "$sha" > "$dst/MLX_SERVE_COMMIT"
# MLX's steel headers (src/mlx_steel/), from the checkout's MLX submodule at the commit it pins, when it is checked out.
mlx_pin="$(git -C "$repo" ls-tree HEAD lib/mlx-src | awk '{print $3}')"
steel="$here/src/mlx_steel"
if [ -f "$repo/lib/mlx-src/mlx/backend/metal/kernels/steel/defines.h" ]; then
  [ "$(git -C "$repo/lib/mlx-src" rev-parse HEAD)" = "$mlx_pin" ] || { echo "lib/mlx-src is not at the pinned $mlx_pin" >&2; exit 1; }
  for h in defines.h utils/type_traits.h utils/integral_constant.h gemm/nax.h; do
    mkdir -p "$steel/mlx/backend/metal/kernels/steel/$(dirname "$h")"
    cp "$repo/lib/mlx-src/mlx/backend/metal/kernels/steel/$h" "$steel/mlx/backend/metal/kernels/steel/$h"
  done
  echo "$mlx_pin" > "$steel/MLX_COMMIT"
elif [ "$(tr -d '[:space:]' < "$steel/MLX_COMMIT")" != "$mlx_pin" ]; then
  echo "src/mlx_steel/ holds MLX $(cat "$steel/MLX_COMMIT"), the checkout pins $mlx_pin: sync from a checkout with lib/mlx-src" >&2; exit 1
fi
