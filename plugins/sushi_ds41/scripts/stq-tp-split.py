#!/usr/bin/env python3
"""Cut a Sushi-STQ DeepSeek-V4.1 pack into one tensor-parallel rank's pack for sushi_ds41's SUSHI_DS41_TP.

    python3 stq-tp-split.py <src pack> <dst dir> <rank> [size=2]

Every cut tensor belongs to one of two families, split in whole grains: attention (wq_b output rows, attn_sink, wo_a
groups, wo_b input) in its o_groups output groups, and the routed and shared experts (gate/up output tiles, down input
tiles) in the 128-wide Hadamard blocks of moe_intermediate_size, so nothing is re-quantized. A family's grains go to
the ranks as evenly as they divide, the first ranks taking one more: 8 groups and 18 blocks give TP2 4/4 and 9/9
(halves), TP3 3/3/2 and 6/6/6, TP4 2/2/2/2 and 5/5/4/4. Each rank's config gets its own num_attention_heads, o_groups
and moe_intermediate_size. Everything else is copied whole; the Engram shards are hard links. Pure Python.
"""
import json
import os
import re
import shutil
import struct
import sys

ITEM = {"U16": 2, "F16": 2, "BF16": 2, "F32": 4, "U8": 1, "I8": 1, "U32": 4, "I32": 4, "I64": 8}
LAYER = r"^(?:layers|mtp)\.\d+\."
EXPERT = r"^(?:model\.layers|mtp)\.\d+\.mlp\.switch_mlp\."
RULES = [
    (LAYER + r"attn\.wq_b\.trellis$", 1),
    (LAYER + r"attn\.wq_b\.svh$", 0),
    (LAYER + r"attn\.attn_sink$", 0),
    (LAYER + r"attn\.wo_a\.(?:trellis|suh|svh)$", 0),
    (LAYER + r"attn\.wo_b\.(?:trellis|suh)$", 0),
    (LAYER + r"ffn\.shared_experts\.w[13]\.trellis$", 1),
    (LAYER + r"ffn\.shared_experts\.w[13]\.svh$", 0),
    (LAYER + r"ffn\.shared_experts\.w2\.(?:trellis|suh)$", 0),
    (EXPERT + r"(?:gate|up)_proj\.g\d+\.trellis_pm$", 1),
    (EXPERT + r"(?:gate|up)_proj\.g\d+\.svh$", 1),
    (EXPERT + r"down_proj\.g\d+\.trellis_pm$", 2),
    (EXPERT + r"down_proj\.g\d+\.suh$", 1),
]
WHOLE_EXPERT = re.compile(EXPERT + r"(?:(?:gate|up)_proj\.g\d+\.suh|down_proj\.g\d+\.svh)$")
TILES = {"trellis": 16, "trellis_pm": 16}


def axis_for(name):
    for pat, ax in RULES:
        if re.match(pat, name):
            return ax
    if "switch_mlp" in name and not WHOLE_EXPERT.match(name):
        raise SystemExit("no rule for expert tensor %s" % name)
    return None


def family(name):
    return "attn" if ".attn." in name else "moe"


def shares(grains, size):
    """How many grains each rank gets: as even as they divide, the first ranks one more."""
    base, extra = divmod(grains, size)
    if base == 0:
        raise SystemExit("%d grains do not give each of %d ranks one" % (grains, size))
    return [base + (r < extra) for r in range(size)]


def grain_range(grains, rank, size):
    s = shares(grains, size)
    lo = sum(s[:rank])
    return lo, lo + s[rank]


def check_grain(name, shape, ax, grains):
    n = shape[ax]
    if n % grains:
        raise SystemExit("%s axis %d (%d) is not %d whole grains" % (name, ax, n, grains))
    leaf = name.rsplit(".", 1)[-1]
    if leaf == "attn_sink" or ".wo_a." in name:
        return
    grain = n // grains
    if leaf == "trellis_pm" and ax == 1:
        width = grain * 2 * 16
    elif leaf in TILES:
        width = grain * 16
    else:
        width = grain
    if width % 128:
        raise SystemExit("%s: a grain is %d wide, off the 128 Hadamard grid" % (name, width))


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def cut(f, base, info, ax, lo, hi, grains):
    shape = info["shape"]
    item = ITEM[info["dtype"]]
    outer = 1
    for d in shape[:ax]:
        outer *= d
    inner = item
    for d in shape[ax + 1:]:
        inner *= d
    run = shape[ax] * inner
    grain = run // grains
    start = base + info["data_offsets"][0]
    for o in range(outer):
        f.seek(start + o * run + lo * grain)
        yield f.read((hi - lo) * grain)


def split_file(src, dst, rank, size, grains):
    header, base = read_header(src)
    meta = header.pop("__metadata__", None)
    plan = []
    out = {}
    off = 0
    for name in sorted(header, key=lambda k: header[k]["data_offsets"][0]):
        info = header[name]
        ax = axis_for(name)
        shape = list(info["shape"])
        if ax is not None:
            n_grains = grains[family(name)]
            check_grain(name, shape, ax, n_grains)
            lo, hi = grain_range(n_grains, rank, size)
            shape[ax] = shape[ax] // n_grains * (hi - lo)
        nbytes = ITEM[info["dtype"]]
        for d in shape:
            nbytes *= d
        out[name] = {"dtype": info["dtype"], "shape": shape, "data_offsets": [off, off + nbytes]}
        plan.append((name, info, ax, nbytes))
        off += nbytes
    if meta is not None:
        out["__metadata__"] = meta
    hb = json.dumps(out, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    cut_n = 0
    with open(src, "rb") as f, open(dst + ".part", "wb") as g:
        g.write(struct.pack("<Q", len(hb)))
        g.write(hb)
        for name, info, ax, nbytes in plan:
            wrote = 0
            if ax is None:
                f.seek(base + info["data_offsets"][0])
                left = nbytes
                while left:
                    chunk = f.read(min(left, 1 << 26))
                    g.write(chunk)
                    left -= len(chunk)
                wrote = nbytes
            else:
                cut_n += 1
                n_grains = grains[family(name)]
                lo, hi = grain_range(n_grains, rank, size)
                for piece in cut(f, base, info, ax, lo, hi, n_grains):
                    g.write(piece)
                    wrote += len(piece)
            if wrote != nbytes:
                raise SystemExit("%s: wrote %d bytes, expected %d" % (name, wrote, nbytes))
    os.replace(dst + ".part", dst)
    return cut_n


def main():
    src, dst, rank = sys.argv[1], sys.argv[2], int(sys.argv[3])
    size = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    if not 0 <= rank < size:
        raise SystemExit("rank must be in [0, %d)" % size)
    t = json.load(open(os.path.join(src, "config.json")))["text_config"]
    heads_per_group = t["num_attention_heads"] // t["o_groups"]
    grains = {"attn": t["o_groups"], "moe": t["moe_intermediate_size"] // 128}
    if t["moe_intermediate_size"] % 128 or t["num_attention_heads"] % t["o_groups"]:
        raise SystemExit("moe_intermediate_size or the heads per group are not whole")
    os.makedirs(dst, exist_ok=True)
    total_cut = 0
    for entry in sorted(os.listdir(src)):
        s, d = os.path.join(src, entry), os.path.join(dst, entry)
        if entry == ".cache" or os.path.exists(d):
            continue
        if entry.startswith("engram-") and entry.endswith(".safetensors"):
            os.link(s, d)
        elif entry.endswith(".safetensors"):
            total_cut += split_file(s, d, rank, size, grains)
            print("split %s" % entry, flush=True)
        elif entry == "config.json":
            c = json.load(open(s))
            t = c["text_config"]
            lo, hi = grain_range(grains["attn"], rank, size)
            t["o_groups"] = hi - lo
            t["num_attention_heads"] = (hi - lo) * heads_per_group
            lo, hi = grain_range(grains["moe"], rank, size)
            t["moe_intermediate_size"] = (hi - lo) * 128
            c["tensor_parallel"] = {"rank": rank, "size": size}
            json.dump(c, open(d, "w"), indent=2)
        elif os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)
    print("TP-SPLIT-OK rank %d of %d: %d tensors cut, into %s" % (rank, size, total_cut, dst))


main()
