#!/usr/bin/env python3
"""turboexl3_ling: every test block sits in a file whose tests a test binary runs (Python 3.9, stdlib only).

A `test` block in a file that no test binary analyses never runs, yet it reads as coverage: plugin.zig's test at 130592c
was expected red, but plugin.zig is the dylib's root, so the box's red showed one failure where two were expected.

The model of Zig (0.17) this script holds: a test binary runs the test blocks of the files in its root module that its
semantic analysis reaches. That is the root file, then each file a relative `@import` names when analysed code
evaluates that import. Container-level declarations are lazy: `const x = @import("x.zig");` counts only where analysed
code names `x`. A container (`struct {..}`, `enum {..}`, ...) that analysed code names has its fields analysed, and
each of its other declarations only where analysed code names that one: by a member chain (`T.f`, `Self.f`), from a
sibling, or as `.f` anywhere in analysed code, which stands for a method call on a value whose type this script cannot
see. An analysed file's test blocks, `comptime` blocks and exports are always analysed. A member chain (`x.f`,
`const y = x.f;`) makes that member of the imported file analysed. `std.testing.refAllDecls` makes every public
declaration analysed. A root module that build.zig gives other modules as an import (`mlx_host`) has every public
declaration analysed, since those modules name them. A module root's folder bounds its relative imports, and tests of
other modules (`mlx_serve`, `sdk`, `mlx_host`) never count.

The roots come from build.zig: the kit's `test` step over config.zig (`plugin(...)`), and each `b.addTest` root module.
The `test` step's binaries are the MLX-free ones; the others are the GPU binaries. Part B of the restore check
(tests/ling_restore_host_test.zig, EXTERNAL below) is built by the served mlx-serve tree under a test filter, so its
tests count when their names hold that filter. Imports are followed through src/, overlay/ and the generated
src/mlx_serve. An overlay file stands for its generated copy. With no generated tree, imports into it are followed only
where overlay/ has the file. Where this model is not sure, it over-reaches: an import inside a function counts if the
function is named in analysed code, though a comptime branch may skip it, and a `.f` reaches every analysed
container's `f`.

Usage: scripts/test_reach.py [-v] [--selftest] [--digests] [plugin folder, default: this script's plugin]
  -v          one line for each file with test blocks: the roots that reach it
  --selftest  the real tree green, the box's counts (BOX_TOTALS) where its files are the measured ones, and the guard
              on planted copies of the real tree in a temp folder (each plant must go red, or green with the count it
              adds)
  --digests   the digest of each BOX_TOTALS entry for this tree, to record a count the box measured on it
Exit 0: every test block is reached. Exit 1: a test block no root reaches (each one named as file:line), or a plant
the guard missed. Exit 2: the guard cannot model the tree: build.zig names a test whose root it cannot find, or its
parser took a declaration (of a file or of a container) into the one before it.
"""
import argparse
import bisect
import hashlib
import os
import posixpath
import re
import shutil
import sys
import tempfile
from collections import deque

GEN = "src/mlx_serve"
OVERLAY = "overlay/mlx_serve"
SKIP_DIRS = {".git", ".zig-cache", "zig-out", "zig-pkg", ".mlx-serve-src"}
# Part B of the restore check: the served mlx-serve tree builds it, never this plugin's build.zig (EVIDENCE.md, "Tests":
# copied into that tree's src/ as the only test of its src/tests.zig, then `zig build test-build
# -Dtest-filter='turboexl3_ling restore bytes'`). Only its tests whose names hold the filter run.
EXTERNAL = {"tests/ling_restore_host_test.zig": ("restore_host", "turboexl3_ling restore bytes")}
# The box's counts at 4c13ec7 (583523d4's gates): the MLX-free binaries' Build Summary 100/100, ling_kda_gpu's "All 50
# tests passed" and ling_restore_gpu's (the restore check's part A) "All 24 tests passed". --selftest holds the model to
# each where the tree's digest of what those binaries can compile (`digest`) is the one given here: 1dffdcd's, which is
# 4c13ec7's but for test blocks that no binary ran. Elsewhere it says the count was not checked.
BOX_TOTALS = (
    ("MLX-free", 100, "a9cca22c0d21ab8bdf82e91a9b5d75c15ef6c4b1b5fc3a9570c3a161a7204558"),
    ("ling_kda_gpu", 50, "eedc50733c6c1a9fa676d462d09cd8814e33e027d785762db84a1114a147ced4"),
    ("ling_restore_gpu", 24, "daf374057c674c2d456c0073976ac29502062f94a80a1f8bb6a8607bb552c466"),
)

WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
MEMBERS = re.compile(r"(?:\s*\.\s*[A-Za-z_][A-Za-z0-9_]*)+")
IMPORT = re.compile(r'@import\s*\(\s*"([^"\n]*)"\s*\)')
REF_ALL = re.compile(r"\brefAllDecls(?:Recursive)?\s*\(\s*(@This\s*\(\s*\)|[A-Za-z_][A-Za-z0-9_]*)\s*\)")
ALIAS_IMPORT = re.compile(r'^@import\s*\(\s*"([^"\n]*)"\s*\)((?:\s*\.\s*[A-Za-z_][A-Za-z0-9_]*)*)$')
ALIAS_PATH = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)((?:\s*\.\s*[A-Za-z_][A-Za-z0-9_]*)*)$")
TEST_LINE = re.compile(r'^[ \t]*test(?:[ \t]*(?:"|\{)|[ \t]+[A-Za-z_][A-Za-z0-9_]*[ \t]*\{)', re.M)
# A line that zig fmt starts with one of these words, at a container's member indentation (column 0 at a file's top
# level, 4 more spaces for each container), begins a declaration of that container.
DECL_WORDS = r"(?:pub|const|var|fn|export|extern|test|comptime|inline|noinline|threadlocal|usingnamespace)\b"
CONTAINER = re.compile(r"(?:extern\s+|packed\s+)?(?:struct|union|enum|opaque)\b")
ALIAS_THIS = re.compile(r"^@This\s*\(\s*\)$")
FIELD_NAME = re.compile(r'@field\s*\([^,()]*,\s*"([A-Za-z_][A-Za-z0-9_]*)"\s*\)')
LEAD = {"pub", "export", "extern", "inline", "noinline", "threadlocal"}
KINDS = {"const", "var", "fn", "test", "comptime", "usingnamespace"}


class GuardError(Exception):
    pass


def lex(text):
    """(code, kept, strs) for Zig source `text`, both strings as long as it, newlines kept, so offsets and lines hold:
    `kept` has its comments blanked; `code` has its comments, string and character literals and multiline-string lines
    blanked, with a quoted identifier @"x" read as x. `strs` maps each string's opening quote to its raw contents."""
    n = len(text)
    code, kept, strs = list(text), list(text), {}
    i = 0
    while i < n:
        c = text[i]
        if (c == "/" and text.startswith("//", i)) or (c == "\\" and text.startswith("\\\\", i)):
            j = text.find("\n", i)
            j = n if j < 0 else j
            for k in range(i, j):
                code[k] = " "
                if c == "/":
                    kept[k] = " "
            i = j
        elif c == '"' or c == "'":
            j = i + 1
            while j < n and text[j] != c and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            j = min(j, n - 1)
            if c == '"':
                strs[i] = text[i + 1:j]
            for k in range(i, j + 1):
                code[k] = " "
            name = text[i + 1:j]
            if c == '"' and i > 0 and text[i - 1] == "@" and WORD.fullmatch(name):
                code[i - 1:i - 1 + len(name)] = list(name)
            i = j + 1
        else:
            i += 1
    return "".join(code), "".join(kept), strs


def skip_ws(code, i):
    while i < len(code) and code[i].isspace():
        i += 1
    return i


def group_end(code, i):
    """The index just past the bracket group that opens at code[i]."""
    depth = 0
    while i < len(code):
        ch = code[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return len(code)


def scan_to(code, i, stops):
    """The index of the first character of `stops` at bracket depth 0 from code[i], or of an unmatched close."""
    depth = 0
    while i < len(code):
        ch = code[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            if depth == 0:
                return i
            depth -= 1
        elif depth == 0 and ch in stops:
            return i
        i += 1
    return len(code)


def fn_end(code, i):
    """The index just past the function whose name ends at code[i]: its `;` (a prototype), or the close of its body,
    the depth-0 `}` that a new declaration (a word), the container's `}` or the end follows. A return type's braces
    (`error{A}!T`, `struct {..}`) are followed by `!` or the body's `{`."""
    depth = 0
    while i < len(code):
        ch = code[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0 and ch == "}":
                k = skip_ws(code, i + 1)
                if k >= len(code) or code[k] == "_" or code[k] == "}" or code[k].isalpha():
                    return i + 1
        elif depth == 0 and ch == ";":
            return i + 1
        i += 1
    return len(code)


def test_label(kept, strs, j):
    """How a test block reads, from just past its `test`: test "name", test {} (unnamed) or test name (a decltest)."""
    k = skip_ws(kept, j)
    if k in strs:
        return 'test "%s"' % strs[k]
    if kept.startswith("{", k):
        return "test {}"
    m = WORD.match(kept, k)
    return "test " + (m.group(0) if m else "?")


class Item:
    """A container-level declaration: kind (const, var, fn, test, comptime, usingnamespace, field), its name, where its
    name is (name_at), its span [start, end), its value's span (a const or var after `=`), pub or export, the container
    it sits in (parent, None at a file's top level), and for a const whose value is a container (`struct {..}`,
    `enum(u8) {..}`, ...) the span inside its braces (body) and its own declarations (members, by name)."""

    def __init__(self, kind, start, pub, export, parent):
        self.kind, self.start, self.end, self.pub, self.export, self.parent = kind, start, start, pub, export, parent
        self.name, self.name_at, self.value, self.label = None, -1, None, None
        self.body, self.members, self.member_names = None, None, None


def items_of(code, kept, strs, lo, hi, parent):
    """The declarations between code[lo] and code[hi] (a file, or a container's body), in order."""
    items, i = [], lo
    while True:
        i = skip_ws(code, i)
        while i < hi and code[i] in ";,":
            i = skip_ws(code, i + 1)
        if i >= hi:
            break
        if code[i] == "}":  # a stray close: no declaration ends here, skip it
            i += 1
            continue
        pub = export = False
        kind, j = None, i
        while True:
            m = WORD.match(code, j)
            if not m:
                break
            if m.group(0) in LEAD:
                pub, export = pub or m.group(0) == "pub", export or m.group(0) == "export"
                j = skip_ws(code, m.end())
                continue
            if m.group(0) in KINDS:
                kind, j = m.group(0), m.end()
            break
        it = Item(kind or "field", i, pub, export, parent)
        if kind in ("test", "comptime"):
            if kind == "test":
                it.label = test_label(kept, strs, j)
            b = code.find("{", j)
            it.end = group_end(code, b) if b >= 0 else hi
        elif kind == "fn":
            m = WORD.match(code, skip_ws(code, j))
            if m:
                it.name, it.name_at = m.group(0), m.start()
            it.end = fn_end(code, m.end() if m else j)
        elif kind in ("const", "var", "usingnamespace"):
            m = WORD.match(code, skip_ws(code, j)) if kind != "usingnamespace" else None
            if m:
                it.name, it.name_at = m.group(0), m.start()
            e = scan_to(code, j, ";")
            q = scan_to(code, m.end() if m else j, "=;")
            if q < e and code[q] == "=":
                it.value = (q + 1, e)
                v = skip_ws(code, q + 1)
                b = code.find("{", v, e) if kind == "const" and CONTAINER.match(code, v) else -1
                if b >= 0 and not code[group_end(code, b):e].strip():
                    it.body = (b + 1, group_end(code, b) - 1)
                    it.members = items_of(code, kept, strs, it.body[0], it.body[1], it)
                    it.member_names = {}
                    for x in it.members:
                        if x.name and x.kind != "test":
                            it.member_names.setdefault(x.name, x)
            it.end = e + 1
        else:  # a container field, or what this model does not know: up to its `,`
            m = WORD.match(code, i)
            if m:
                it.name, it.name_at = m.group(0), m.start()
            it.end = scan_to(code, i, ",;") + 1
        it.end = max(it.end, i + 1)
        items.append(it)
        i = it.end
    return items


def containers(items):
    """Every container among `items` and inside them, outer ones first."""
    for it in items:
        if it.members is not None:
            yield it
            for c in containers(it.members):
                yield c


class Src:
    """One Zig file: its declarations, its names, and its test blocks (top level or nested) with their lines."""

    def __init__(self, rel, text):
        self.rel, self.text = rel, text
        self.code, self.kept, self.strs = lex(text)
        self.items = items_of(self.code, self.kept, self.strs, 0, len(self.code), None)
        self.names = {}
        for it in self.items:
            if it.name and it.kind != "test":
                self.names.setdefault(it.name, it)
        self.line_starts = [0] + [m.end() for m in re.finditer("\n", text)]
        # The parser's check: each declaration line of the file and of each container is where it put a declaration.
        spans = [(0, len(self.code), 0, self.items)]
        for c in containers(self.items):
            ind = c.start - (self.code.rfind("\n", 0, c.start) + 1)
            spans.append((c.body[0], c.body[1], ind + 4, c.members))
        for lo, hi, ind, items in spans:
            starts = {it.start for it in items}
            for m in re.compile(r"^ {%d}%s" % (ind, DECL_WORDS), re.M).finditer(self.code, lo, hi):
                if m.start() + ind not in starts:
                    raise GuardError("%s:%d: a declaration that this script's parser took into the one before it"
                                     % (rel, self.line(m.start() + ind)))
        top = {it.start: it for it in self.items if it.kind == "test"}
        self.tests = []  # (line, label, top level)
        for m in TEST_LINE.finditer(self.kept):
            at = skip_ws(self.kept, m.start())
            it = top.get(at)
            label = it.label if it else test_label(self.kept, self.strs, at + 4) + " (nested)"
            self.tests.append((self.line(at), label, it is not None))

    def line(self, off):
        return bisect.bisect_right(self.line_starts, off)


class Root:
    def __init__(self, name, path, group, shared):
        self.name, self.path, self.group, self.shared = name, path, group, shared


class Tree:
    """The plugin folder: its files, read once, and how a relative import resolves in it."""

    def __init__(self, root):
        self.root = root
        self.generated = os.path.isdir(os.path.join(root, GEN))
        self.cache = {}

    def disk(self, rel):
        """The file on disk that holds `rel`: itself, or, with no generated tree, overlay's copy of a generated path."""
        p = os.path.join(self.root, rel)
        if os.path.isfile(p):
            return p
        if not self.generated and rel.startswith(GEN + "/"):
            q = os.path.join(self.root, OVERLAY, rel[len(GEN) + 1:])
            if os.path.isfile(q):
                return q
        return None

    def src(self, rel):
        if rel not in self.cache:
            with open(self.disk(rel), encoding="utf-8", errors="replace") as f:
                self.cache[rel] = Src(rel, f.read())
        return self.cache[rel]

    def resolve(self, frm, spec, mod_dir):
        """The file a relative import names from `frm`, inside its module's folder `mod_dir`, or None (a named module,
        a path out of the module, which Zig refuses, or no such file)."""
        if not spec.endswith(".zig"):
            return None
        t = posixpath.normpath(posixpath.join(posixpath.dirname(frm), spec))
        if t == ".." or t.startswith("../") or (mod_dir and not t.startswith(mod_dir + "/")):
            return None
        return t if self.disk(t) else None

    def zig_files(self):
        out = []
        for d, dirs, files in os.walk(self.root):
            dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS)
            for f in sorted(files):
                if f.endswith(".zig"):
                    out.append(os.path.relpath(os.path.join(d, f), self.root).replace(os.sep, "/"))
        return out


def roots_of(tree):
    """The test roots build.zig names: the kit's config.zig test, then each `b.addTest` root module, in order."""
    path = os.path.join(tree.root, "build.zig")
    if not os.path.isfile(path):
        raise GuardError("no build.zig in %s" % tree.root)
    with open(path, encoding="utf-8") as f:
        build = Src("build.zig", f.read())
    code, kept = build.code, build.kept
    roots = []
    # The kit's `plugin` adds the `test` step over the plugin's MLX-free config.zig (kit build.zig `plugin`).
    if re.search(r'@import\s*\(\s*"mlx_serve_kit"\s*\)\s*\.\s*plugin\s*\(', kept):
        roots.append(Root("config", "config.zig", "MLX-free", False))
    for m in re.finditer(r"\bb\s*\.\s*addTest\s*\(", code):
        line = build.line(m.start())
        fn = next((it for it in build.items if it.kind == "fn" and it.start <= m.start() < it.end), None)
        lo, hi = (fn.start, fn.end) if fn else (0, len(code))
        a0 = m.end() - 1
        a = kept[a0:group_end(code, a0)]
        rm = re.search(r"\.root_module\s*=\s*", a)
        if not rm:
            raise GuardError("build.zig:%d: b.addTest with no .root_module" % line)
        value = kept[a0 + rm.end():scan_to(code, a0 + rm.end(), ",}")].strip()
        mod_var = None
        if WORD.fullmatch(value):
            mod_var = value
            d = None
            for c in re.finditer(r"\bconst\s+%s\s*=\s*b\s*\.\s*createModule\s*\(" % re.escape(mod_var), code[:m.start()]):
                if c.start() >= lo:
                    d = c
            if d is None:
                raise GuardError("build.zig:%d: b.addTest's root module %s has no b.createModule before it" % (line, mod_var))
            value = kept[d.end() - 1:group_end(code, d.end() - 1)]
        p = re.search(r'\.root_source_file\s*=\s*b\s*\.\s*path\s*\(\s*"([^"]+)"\s*\)', value)
        if not p:
            raise GuardError("build.zig:%d: b.addTest's root module has no b.path root_source_file" % line)
        nm = re.search(r'\.name\s*=\s*"([^"]+)"', a)
        name = nm.group(1) if nm else posixpath.splitext(posixpath.basename(p.group(1)))[0]
        # The binary's variable: `const v = b.addTest(...)`, or the variable a call of the function that returns it sets.
        v = re.search(r"\bconst\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*$", code[max(lo, m.start() - 200):m.start()])
        var = v.group(1) if v else None
        if var is None and fn is not None and re.search(r"\breturn\s*$", code[max(lo, m.start() - 50):m.start()]):
            c = re.search(r"\bconst\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*%s\s*\(" % re.escape(fn.name), code)
            var = c.group(1) if c else None
        mlx_free = var is not None and re.search(
            r'top_level_steps\s*\.\s*get\s*\(\s*"test"\s*\)[^;]*addRunArtifact\s*\(\s*%s\s*\)' % re.escape(var), kept)
        shared = mod_var is not None and re.search(r"\.module\s*=\s*%s\b" % re.escape(mod_var), code[lo:hi]) is not None
        roots.append(Root(name, posixpath.normpath(p.group(1)), "MLX-free" if mlx_free else "GPU", shared))
    if not roots:
        raise GuardError("build.zig names no test root")
    return roots


def reach(tree, root):
    """The files `root`'s test binary analyses, by the model in this file's doc."""
    mod_dir = posixpath.dirname(root.path)
    files, live, work, memo = set(), set(), deque(), {}
    accessed, waiting = set(), {}  # names analysed code accesses as `.name`; members that wait for theirs

    def mark(f, it):
        if (f, it.start) not in live:
            live.add((f, it.start))
            work.append((f, it))

    def by_name(f, it):
        """A method or constant of a live container, which a `.name` access of analysed code may call or read."""
        if it.name in accessed:
            mark(f, it)
        else:
            waiting.setdefault(it.name, []).append((f, it))

    def access(name):
        if name not in accessed:
            accessed.add(name)
            for f, it in waiting.pop(name, ()):
                mark(f, it)

    def lookup(s, scope, name):
        """The declaration `name` means in the code of item `scope`: its containers' members, inner first, then the
        file's top level."""
        p = scope.parent
        while p is not None:
            if name in p.member_names:
                return p.member_names[name]
            p = p.parent
        return s.names.get(name)

    def path_of(it):
        out = []
        while it is not None:
            out.append(it.name)
            it = it.parent
        return out[::-1]

    def alias(f, it):
        """(file, path from its top level) that a const naming an import, @This() or a member chain stands for."""
        key = (f, it.start)
        if key in memo:
            return memo[key]
        memo[key] = None  # a cycle reads None
        out = None
        if it.kind in ("const", "var") and it.value and it.members is None:
            s = tree.src(f)
            v = s.kept[it.value[0]:it.value[1]].strip()
            m = ALIAS_IMPORT.match(v)
            if m:
                t = tree.resolve(f, m.group(1), mod_dir)
                out = (t, WORD.findall(m.group(2))) if t else None
            elif ALIAS_THIS.match(v):
                out = (f, path_of(it.parent))
            else:
                m = ALIAS_PATH.match(v)
                base = lookup(s, it, m.group(1)) if m else None
                if base is not None and base is not it:
                    rest = WORD.findall(m.group(2))
                    t = alias(f, base)
                    if t:
                        out = (t[0], t[1] + rest)
                    elif base.members is not None:
                        out = (f, path_of(base) + rest)
        memo[key] = out
        return out

    def follow(f, it, rest):
        """`it` analysed, then each member of the chain `rest` after it, through aliases and containers."""
        while it is not None:
            mark(f, it)
            if not rest:
                return
            t = alias(f, it)
            if t:
                seed(t[0], t[1] + list(rest))
                return
            if it.members is None:
                return
            it, rest = it.member_names.get(rest[0]), rest[1:]

    def seed(f, path=()):
        """File `f` analysed, and the declaration at `path` from its top level."""
        s = tree.src(f)
        if f not in files:
            files.add(f)
            is_struct = any(it.kind == "field" for it in s.items)  # a file with fields is a struct type
            for it in s.items:
                if it.kind in ("test", "comptime", "usingnamespace", "field") or it.export:
                    mark(f, it)
                elif is_struct and it.name:
                    by_name(f, it)
        if path:
            follow(f, s.names.get(path[0]), path[1:])

    def all_pub(f, items):
        for it in items:
            if it.pub:
                mark(f, it)

    def scan(f, scope, a, b):
        """What the code between a and b of item `scope` names: declarations, member chains, imports, `.name`s."""
        s = tree.src(f)
        code = s.code
        for m in WORD.finditer(code, a, b):
            name = m.group(0)
            if m.start() == scope.name_at or (m.start() > 0 and code[m.start() - 1] == "@"):
                continue
            k = m.start() - 1
            while k >= a and code[k].isspace():
                k -= 1
            if k >= a and code[k] == "." and not (k > a and code[k - 1] == "."):  # `x.name`, `.name`; not `0..name`
                access(name)
                continue
            target = lookup(s, scope, name)
            if target is not None:
                chain = MEMBERS.match(code, m.end())
                follow(f, target, WORD.findall(chain.group(0)) if chain else [])
        for m in IMPORT.finditer(s.kept, a, b):
            t = tree.resolve(f, m.group(1), mod_dir)
            if t:
                chain = MEMBERS.match(code, m.end())
                seed(t, WORD.findall(chain.group(0)) if chain else [])
        for m in FIELD_NAME.finditer(s.kept, a, b):
            access(m.group(1))
        for m in REF_ALL.finditer(code, a, b):
            if m.group(1).startswith("@This"):
                all_pub(f, scope.parent.members if scope.parent is not None else s.items)
                continue
            base = lookup(s, scope, m.group(1))
            t = alias(f, base) if base is not None else None
            if t and not t[1]:
                seed(t[0])
                all_pub(t[0], tree.src(t[0]).items)
            elif t:
                seed(t[0], t[1])
            elif base is not None:
                mark(f, base)
                if base.members is not None:
                    all_pub(f, base.members)

    def refs(f, it):
        """A live declaration's code. A container's is its own text and its fields', with its comptime blocks, tests
        and exports; its other members wait to be named."""
        if it.members is None:
            scan(f, it, it.start, it.end)
            return
        scan(f, it, it.start, it.body[0])
        scan(f, it, it.body[1], it.end)
        for x in it.members:
            if x.kind in ("test", "comptime", "usingnamespace", "field") or x.export:
                mark(f, x)
            elif x.name:
                by_name(f, x)

    if not tree.disk(root.path):
        raise GuardError("build.zig's test root %s (%s) is not a file" % (root.name, root.path))
    seed(root.path)
    if root.shared:
        all_pub(root.path, tree.src(root.path).items)
    while work:
        refs(*work.popleft())
    return files


def canonical(rel):
    """The path a file compiles as: an overlay file's is its generated copy's."""
    return GEN + rel[len(OVERLAY):] if rel.startswith(OVERLAY + "/") else rel


def closure(tree, roots):
    """Every file that `roots` can import through relative imports, by their text alone (laziness aside): all that
    those test binaries could compile of the plugin."""
    out = set()
    for r in roots:
        mod_dir, seen, todo = posixpath.dirname(r.path), set(), [r.path]
        while todo:
            f = todo.pop()
            if f in seen or not tree.disk(f):
                continue
            seen.add(f)
            for m in IMPORT.finditer(tree.src(f).kept):
                t = tree.resolve(f, m.group(1), mod_dir)
                if t and t not in seen:
                    todo.append(t)
        out |= seen
    return out


def digest(tree, roots):
    """sha256 over build.zig, build.zig.zon and each file of `closure`, by path and contents."""
    h = hashlib.sha256()
    for rel in sorted(closure(tree, roots) | {"build.zig", "build.zig.zon"}):
        p = tree.disk(rel)
        if p:
            with open(p, "rb") as f:
                h.update(("%s %s\n" % (rel, hashlib.sha256(f.read()).hexdigest())).encode())
    return h.hexdigest()


def box_roots(res, name):
    """The roots a BOX_TOTALS entry counts: a group's (MLX-free) or one root's."""
    return [r for r in res["roots"] if r.name == name or r.group == name]


def check(tree):
    """{roots, counts (by root name), reached (file: [root names]), dead [(file, line, label, why)], files, blocks}."""
    roots = roots_of(tree)
    counts, reached = {}, {}
    for r in roots:
        files = reach(tree, r)
        counts[r.name] = sum(len(tree.src(f).tests) for f in files)
        for f in files:
            reached.setdefault(f, []).append(r.name)
    dead, n_files, n_blocks = [], 0, 0
    ext = {}
    for rel in tree.zig_files():
        if rel == "build.zig":
            continue
        s = tree.src(rel)
        if not s.tests:
            continue
        n_files += 1
        n_blocks += len(s.tests)
        if rel in EXTERNAL:
            name, filt = EXTERNAL[rel]
            ext[name] = (rel, len([t for t in s.tests if filt in t[1]]))
            reached.setdefault(rel, []).append(name)
            for line, label, _ in s.tests:
                if filt not in label:
                    dead.append((rel, line, label, "%s runs only tests named with '%s'" % (name, filt)))
            continue
        if canonical(rel) in reached:
            if rel != canonical(rel):
                reached.setdefault(rel, reached[canonical(rel)])
            continue
        why = "no test root reaches %s" % (rel if rel == canonical(rel) else "its generated copy " + canonical(rel))
        for line, label, _ in s.tests:
            dead.append((rel, line, label, why))
    return {"roots": roots, "counts": counts, "external": ext, "reached": reached, "dead": dead,
            "files": n_files, "blocks": n_blocks}


def summary(tree, res):
    parts = []
    for group in ("MLX-free", "GPU"):
        rs = [r for r in res["roots"] if r.group == group]
        if rs:
            parts.append("%s %d (%s)" % (group, sum(res["counts"][r.name] for r in rs),
                                         ", ".join("%s %d" % (r.name, res["counts"][r.name]) for r in rs)))
    for name, (rel, n) in sorted(res["external"].items()):
        parts.append("%s: the plugin's %s holds %d (that binary's own tests are not counted)" % (name, rel, n))
    gen = "" if tree.generated else "; no generated src/mlx_serve (scripts/prepare.sh), so imports into it follow overlay/ only"
    return "test_reach: %s; reachable test blocks by root: %s; %d test blocks in %d files, %d unreached%s" % (
        "red" if res["dead"] else "green", "; ".join(parts), res["blocks"], res["files"], len(res["dead"]), gen)


def report(tree, res, verbose, out):
    if verbose:
        for rel in sorted(set(f for f in tree.zig_files() if f in tree.cache and tree.cache[f].tests)):
            by = res["reached"].get(rel)
            out.write("%s: %d test blocks, %s\n" % (rel, len(tree.cache[rel].tests),
                                                     "reached by " + ", ".join(by) if by else "UNREACHED"))
    for rel, line, label, why in res["dead"]:
        out.write("%s:%d: %s never runs: %s\n" % (rel, line, label, why))
    out.write(summary(tree, res) + "\n")


def copy_tree(src, dst):
    """The plugin's .zig files and build.zig, at the same paths under `dst`."""
    for d, dirs, files in os.walk(src):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
        for f in files:
            if f.endswith(".zig"):
                rel = os.path.relpath(os.path.join(d, f), src)
                os.makedirs(os.path.dirname(os.path.join(dst, rel)), exist_ok=True)
                shutil.copyfile(os.path.join(d, f), os.path.join(dst, rel))


def selftest(plugin, out):
    """The real tree green, and at the box's counts where its files are the measured ones (BOX_TOTALS). Then plants on
    copies of it: a real reached test in a fake file must go red (no import, an unused import, an import from
    plugin.zig, an overlay file with no generated copy, an import named only in a method that nothing calls), and green
    where a root's test block reaches it, with that root's count up by the two blocks."""
    real_tree = Tree(plugin)
    real = check(real_tree)
    ok = True
    if real["dead"]:
        out.write("selftest: the real tree is red\n")
        report(real_tree, real, False, out)
        ok = False
    for name, total, want in BOX_TOTALS:
        rs = box_roots(real, name)
        n = sum(real["counts"][r.name] for r in rs)
        if not rs or digest(real_tree, rs) != want:
            out.write("selftest: %s %d, not checked against the box's %d: what its binaries can compile differs from "
                      "1dffdcd's\n" % (name, n, total))
        elif n == total:
            out.write("selftest: %s %d, the box's count on these files\n" % (name, n))
        else:
            out.write("selftest: %s %d, but the box counted %d on these files: MISSED\n" % (name, n, total))
            ok = False
    # The donor: the first test block, with no import of its own, of the first of the plugin's own files (src/, not
    # the generated tree) that a root reaches and that is not itself a root.
    roots = {r.path for r in real["roots"]}
    donor = None
    for rel in sorted(real["reached"]):
        s = real_tree.cache.get(rel)
        if rel in roots or not rel.startswith("src/") or rel.startswith(GEN + "/") or s is None:
            continue
        top = [it for it in s.items if it.kind == "test" and not IMPORT.search(s.kept, it.start, it.end)]
        if top:
            donor = (rel, s.text[top[0].start:top[0].end])
            break
    if donor is None:
        out.write("selftest: no reached test block outside the roots to plant\n")
        return False
    host = next((r for r in real["roots"] if r.group == "MLX-free" and r.path.startswith("src/")), None)
    if host is None:
        out.write("selftest: no MLX-free root under src/ to import a plant\n")
        return False
    body = "//! test_reach selftest plant: %s's first test block.\nconst std = @import(\"std\");\n%s\n" % donor
    plant = "src/test_reach_plant.zig"
    imp = posixpath.relpath(plant, posixpath.dirname(host.path))
    fn = '\nfn testReachPlant() void {\n    _ = @import("%s");\n}\n' % imp
    # The restore check's part A at 4c13ec7: it names trunk.zig's `Model` as a type, and only `Model.embedding`, which
    # it never calls, names ling_embed.zig, whose container-level `er` is ling_embed_rows.zig's import.
    member = ('\nconst test_reach_plant = @import("%s");\nconst TestReachPlant = struct {\n    rows: u32 = 0,\n'
              '    fn testReachPlantRows(p: TestReachPlant) void {\n        _ = p;\n        _ = test_reach_plant;\n'
              '    }\n};\ntest {\n    const p: TestReachPlant = .{};\n    %s\n}\n')
    # (what, the plant's path, (a file, the text appended to it) or None, the root whose count grows by the plant's
    # test and the appended one, or None: red, at the plant and at any test block appended to a file no root reaches)
    plants = [
        ("a reached test in a file nothing imports", plant, None, None),
        ("an unused container-level import of it in root %s" % host.name, plant,
         (host.path, '\nconst test_reach_plant = @import("%s");\n' % imp), None),
        ("an import of it in a function of root %s that nothing calls" % host.name, plant, (host.path, fn), None),
        ("an import of it in a test block of plugin.zig, the dylib's root", plant,
         ("plugin.zig", '\ntest {\n    _ = @import("%s");\n}\n' % plant), None),
        ("an overlay file with no generated copy", OVERLAY + "/test_reach_plant.zig", None, None),
        ("a container-level import of it in root %s, named only in a method of a struct its test names as a type"
         % host.name, plant, (host.path, member % (imp, "_ = p;")), None),
        ("the same, with the test calling that method", plant,
         (host.path, member % (imp, "p.testReachPlantRows();")), host.name),
        ("an import of it in a function of root %s that a test block calls" % host.name, plant,
         (host.path, fn + "test {\n    testReachPlant();\n}\n"), host.name),
        ("an import of it in a test block of root %s" % host.name, plant,
         (host.path, '\ntest {\n    _ = @import("%s");\n}\n' % imp), host.name),
    ]
    for what, rel, edit, grows in plants:
        tmp = tempfile.mkdtemp(prefix="test_reach.")
        try:
            copy_tree(plugin, tmp)
            os.makedirs(os.path.dirname(os.path.join(tmp, rel)), exist_ok=True)
            with open(os.path.join(tmp, rel), "w", encoding="utf-8") as f:
                f.write(body)
            want = {(rel, 3)}
            if edit:
                p = os.path.join(tmp, edit[0])
                with open(p, encoding="utf-8") as f:
                    lines = f.read().count("\n")
                with open(p, "a", encoding="utf-8") as f:
                    f.write(edit[1])
                if edit[0] not in real["reached"]:
                    want |= {(edit[0], lines + 1 + k) for k, l in enumerate(edit[1].split("\n")) if l.startswith("test")}
            res = check(Tree(tmp))
            got = set((d[0], d[1]) for d in res["dead"])
            if grows is None:
                caught = got == want
                verdict = ("red at " + ", ".join("%s:%d" % x for x in sorted(want)) if caught
                           else "MISSED (dead %s, want %s)" % (sorted(got), sorted(want)))
            else:
                n0, n1 = real["counts"][grows], res["counts"][grows]
                caught = not got and n1 == n0 + 2
                verdict = ("green, %s %d -> %d" % (grows, n0, n1) if caught
                           else "MISSED (dead %s, %s %d -> %d, want %d)" % (sorted(got), grows, n0, n1, n0 + 2))
            out.write("selftest: %s (donor %s): %s\n" % (what, donor[0], verdict))
            ok = ok and caught
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    out.write("selftest: %s; %s\n" % ("pass" if ok else "FAIL", summary(real_tree, real)))
    return ok


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("plugin", nargs="?", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--digests", action="store_true")
    a = ap.parse_args(argv)
    plugin = os.path.abspath(a.plugin)
    try:
        if a.selftest:
            return 0 if selftest(plugin, sys.stdout) else 1
        tree = Tree(plugin)
        res = check(tree)
        if a.digests:
            for name, _, _ in BOX_TOTALS:
                rs = box_roots(res, name)
                sys.stdout.write("%s %d %s\n" % (name, sum(res["counts"][r.name] for r in rs), digest(tree, rs)))
            return 0
        report(tree, res, a.verbose, sys.stdout)
        return 1 if res["dead"] else 0
    except GuardError as e:
        sys.stdout.write("test_reach: cannot model the tree: %s\n" % e)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
