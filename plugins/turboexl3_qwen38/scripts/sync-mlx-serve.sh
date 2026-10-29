#!/usr/bin/env bash
# Copies the slice of mlx-serve this plugin runs (scripts/slice.txt) into src/mlx_serve/, at the same paths as in
# mlx-serve's src/, so their relative imports hold, with the Metal kernels they embed.
# Usage: scripts/sync-mlx-serve.sh <mlx-serve checkout>
# The copies are unchanged but for the edits below, made here and nowhere else:
#   - a `//! copied_from: src/<path>@<sha>` first line;
#   - the imports of what the plugin kit ships (abi, the SDK's plugin helpers and root, generation, the test fixture)
#     pointed at the kit's `mlx_serve` module;
#   - mlx-c handles a copy default-initialises (`mlx.mlx_array = .{}`) given their one field, since the kit's `mlx_c`
#     declares them without field defaults;
#   - in the built-in's root (plugins/qwen4_exp/plugin.zig), `pub` on the functions this plugin's table takes from it,
#     and on the model's `ownSlot`, which the prefix cache's restores reach the slot by;
#   - abi/quant.zig, the quant ABI as #1005 extends it (`QuantTable`'s fused members, read only between the copies and
#     turboexl3's table, both inside this dylib): each declaration the kit's quant ABI has unchanged is re-exported from
#     the kit, so the types the host's tables carry are one type on both sides;
#   - in the kernel contract's registry (sdk/testing/kernel_contract/registry.zig), its `areas` cut to REGISTRY_AREAS:
#     the areas whose kernels this plugin ships, which its GPU tests (`zig build gpu-test`) hold to the contract;
#   - every named test no build of the plugin runs taken out, with the comment lines right above it: a copy keeps the
#     named tests of KIT_TESTS (the files whose tests `zig build test` runs) and those a gpu-test or mlx-free-test filter
#     in build.zig picks, and every unnamed `test {}` block. A module's named tests go (no test build roots a module).
# MODULES are directories mlx-serve builds as modules of their own (Sushi's EXL3 decode, `sushi_exl3`): their Zig files
# are copied whole (their test fixtures are not), with no edit but the `copied_from` line, but for MODULE_STAND_INS,
# and of their other files only the kernels a copied Zig file of theirs names. The kernel contract's planted controls
# (`controls/*.metal`) come with its `controls.zig`.
# Files in EDGES are this plugin's own, at the paths of the mlx-serve files they stand in for, and never overwritten.
# Every other file a copy imports gets a stand-in: an empty file, so the import resolves while nothing of it is used
# (a use is a compile error, never a silent difference). Zig loads every file a file imports, used or not, so a file
# the plugin's builds reach no code of is a stand-in rather than a copy: slice.txt leaves it out, MODULE_STAND_INS
# names it inside a module.
# scripts/sync-mlx-serve.sh --check <mlx-serve checkout> <ref>: does <ref> change the slice (or a cut edge's mlx-serve
# file, which is then reviewed by hand) since MLX_SERVE_COMMIT? Exit 0 when the pin holds, 1 with the changed files.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
dst="$here/src/mlx_serve"

# Ours, at the paths of the mlx-serve files they stand in for: the cut edges where the slice meets the engine, the host
# and mlx-serve's own MLX layer (which this plugin replaces with mlx-c through the kit's `mlx_c`).
EDGES="env.zig mlx/ffi.zig mlx/array.zig mlx/device.zig mlx/dtype.zig mlx/kernel_cache.zig mlx/kernels.zig mlx/scope.zig sdk/util/log.zig engines/mlx/transformer.zig engines/mlx/cluster/distributed.zig engines/mlx/plugin/config.zig
plugins/deepseek_v41/deepseek_v41.zig sdk/quant/gguf/claim.zig sdk/quant/nvfp4/modelopt_pack.zig sdk/blocks/dit/w8a8.zig
engines/mlx/runtime/preflight.zig sdk/quant/nvfp4/stream_bank.zig plugins/hy_v3/load.zig sdk/quant/gguf/root.zig
sdk/quant/gguf/format.zig engines/mlx/admission/bills.zig sdk/util/shared.zig engines/mlx/generate/generator.zig
engines/mlx/spec/mtp_head_ref.zig sdk/ane/offload.zig"

MODULES="sdk/quant/exl3/sushi"
# A module's files the plugin's builds reach no code of (its root imports them).
MODULE_STAND_INS="sdk/quant/exl3/sushi/glm_group2.zig sdk/quant/exl3/sushi/glm_prefill_grid.zig
sdk/quant/exl3/sushi/layout.zig sdk/quant/exl3/sushi/rate_groups.zig"

# The copies whose named tests `zig build test` (config.zig's root) runs, all of them.
KIT_TESTS="sdk/vision/mrope.zig plugins/qwen4_exp/ple_table.zig plugins/qwen4_exp/config.zig sdk/util/crt.zig
sdk/quant/exl3/pack_keys.zig"

# The kernel contract's areas this plugin keeps (the registry's paths), in the registry's order.
REGISTRY_AREAS="controls.zig ../../blocks/hc/contract.zig ../../blocks/gdn/contract.zig ../../../engines/mlx/spec/contract.zig
../../../plugins/qwen4_exp/contract.zig ../../blocks/moe/contract.zig"

if [ "${1:-}" = "--check" ]; then
  repo="${2:?mlx-serve checkout}"; ref="${3:?commit or tag}"
  pin="$(tr -d '[:space:]' < "$dst/MLX_SERVE_COMMIT")"
  paths=(); while read -r f; do paths+=("src/$f"); done < <(grep -v '^#' "$here/scripts/slice.txt"; for e in $EDGES; do echo "$e"; done)
  for m in $MODULES; do paths+=("src/$m"); done
  changed="$(git -C "$repo" diff --name-only "$pin" "$ref" -- "${paths[@]}" src/kernels src/plugins/qwen4_exp/kernels)"
  if [ -z "$changed" ]; then echo "pin holds: $ref changes nothing in the slice since ${pin:0:8}"; exit 0; fi
  echo "sync due: $ref changes the slice since ${pin:0:8}:"; echo "$changed" | sed 's/^/  /'; exit 1
fi

repo="${1:?mlx-serve checkout}"
sha="$(git -C "$repo" rev-parse HEAD)"
for e in $EDGES; do [ -f "$dst/$e" ] || { echo "the cut edge $e is missing" >&2; exit 1; }; done
REGISTRY_AREAS="$(echo $REGISTRY_AREAS)" MODULE_STAND_INS="$(echo $MODULE_STAND_INS)" KIT_TESTS="$(echo $KIT_TESTS)" python3 - "$repo" "$sha" "$dst" "$here/scripts/slice.txt" "$MODULES" $EDGES <<'P'
import os, pathlib, re, sys
repo, sha, dst, slice_file, edges = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3]), sys.argv[4], set(sys.argv[6:])
modules = sys.argv[5].split()
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
kernel_text = re.compile(r'"((?:kernels|plugins/qwen4_exp/kernels)/[^"]+\.(?:metal|h|cu))"')
wanted, kernels = set(), set()
# The named tests a build of the plugin runs: KIT_TESTS' files, whole, and the names build.zig's filters pick.
build_zig = (dst.parent.parent / "build.zig").read_text()
test_filters = []
for name in ("gpu_filters", "mlx_free_filters"):
    m = re.search(r'^const ' + name + r' = \[_\]\[\]const u8\{(.*?)\};', build_zig, flags=re.M | re.S)
    if m is None:
        sys.exit(f"build.zig: no `{name}`: the test strip needs updating")
    test_filters += re.findall(r'"((?:\\.|[^"\\])*)"', m.group(1))
kit_tests = set(os.environ["KIT_TESTS"].split())
if kit_tests - set(files):
    sys.exit(f"KIT_TESTS names files outside the slice: {sorted(kit_tests - set(files))}")
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
for f in files:
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
    if f == "sdk/testing/kernel_contract/registry.zig":
        # The areas whose kernels this plugin ships: every other area's import goes.
        keep = os.environ["REGISTRY_AREAS"].split()
        m = re.search(r'^const areas = \.\{\n((?:    @import\("[^"]+"\),\n)+)\};\n', text, flags=re.M)
        if m is None:
            sys.exit("sdk/testing/kernel_contract/registry.zig: no `areas` list: the registry's edit needs updating")
        rows = re.findall(r'@import\("([^"]+)"\)', m.group(1))
        if [a for a in keep if a not in rows]:
            sys.exit(f"sdk/testing/kernel_contract/registry.zig: no area {[a for a in keep if a not in rows]}: REGISTRY_AREAS needs updating")
        text = text[:m.start(1)] + "".join(f'    @import("{a}"),\n' for a in rows if a in keep) + text[m.end(1):]
    text, n = strip_tests(f, text, (lambda name: True) if f in kit_tests else (lambda name: any(x in name for x in test_filters)))
    tests_out += n
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
(dst / "abi").mkdir(parents=True, exist_ok=True)
(dst / "abi/quant.zig").write_text(f"//! copied_from: src/abi/quant.zig@{sha}\n" + qsrc)
print(f"abi/quant.zig: {len(same)} declarations re-exported from the kit ({', '.join(sorted(same))}); its own: {', '.join(sorted(set(mine) - set(same)))}")
# The cut edges' own imports resolve the same way.
for e in edges:
    for m in imp.finditer((dst / e).read_text()):
        wanted.add(os.path.normpath(os.path.join(os.path.dirname(e), m.group(1))))
# hc_prefill_nax.zig names its kernel files by a comptime part: every one of that family.
kernels |= {str(p.relative_to(repo / "src")) for p in (repo / "src/kernels/hc").glob("hc_prefill_*.metal")}
# The kernel contract's controls, embedded by name beside controls.zig.
if "sdk/testing/kernel_contract/controls.zig" in files:
    kernels |= {str(p.relative_to(repo / "src")) for p in (repo / "src/sdk/testing/kernel_contract/controls").glob("*.metal")}
for k in sorted(kernels):
    out = dst / k
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes((repo / "src" / k).read_bytes())
stand_ins = sorted(w for w in wanted if w not in files and w not in edges and w not in kit and w != "abi/quant.zig" and not w.startswith(".."))
for s in stand_ins:
    out = dst / s
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"//! stand_in: src/{s}@{sha}: the copies import it, this plugin uses nothing of it.\n")
module_files = set()
module_stand_ins = set(os.environ["MODULE_STAND_INS"].split())
for mdir in modules:
    srcs = [q for q in sorted((repo / "src" / mdir).rglob("*")) if q.is_file() and "/fixtures/" not in str(q.relative_to(repo / "src"))]
    missing = sorted(m for m in module_stand_ins if m.startswith(mdir + "/") and not (repo / "src" / m).is_file())
    if missing:
        sys.exit(f"MODULE_STAND_INS names files {mdir} does not have: {missing}")
    # What the module's copied Zig files say: a kernel only they name comes along.
    stripped = {}
    for q in srcs:
        rel = str(q.relative_to(repo / "src"))
        if q.suffix == ".zig" and rel not in module_stand_ins:
            stripped[rel], n = strip_tests(rel, q.read_text(), lambda name: False)
            tests_out += n
    named = "".join(stripped.values())
    for q in srcs:
        rel = str(q.relative_to(repo / "src"))
        if q.suffix != ".zig" and f'"{q.relative_to(repo / "src" / mdir)}"' not in named:
            continue
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if rel in module_stand_ins:
            out.write_text(f"//! stand_in: src/{rel}@{sha}: the module's root imports it, this plugin uses nothing of it.\n")
        elif q.suffix == ".zig":
            out.write_text(f"//! copied_from: src/{rel}@{sha}\n" + stripped[rel])
        else:
            out.write_bytes(q.read_bytes())
        module_files.add(rel)
known = set(files) | edges | set(stand_ins) | set(kernels) | module_files | {"MLX_SERVE_COMMIT", "abi/quant.zig"}
for p in sorted(dst.rglob("*")):
    rel = str(p.relative_to(dst))
    if p.is_file() and rel not in known:
        p.unlink()
        print(f"removed {rel} (no longer in the slice)")
print(f"synced {len(files)} files, {len(kernels)} kernels, {len(module_files)} module files, {len(stand_ins)} stand-ins from {sha[:8]}; {tests_out} named tests no build runs left out")
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
