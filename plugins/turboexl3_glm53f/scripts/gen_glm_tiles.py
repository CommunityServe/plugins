#!/usr/bin/env python3
"""gen_glm_tiles.py OUT.json: real routed-expert trellis tiles of turboderp's GLM-5.3-Flash self-calibrated EXL3
branches (HTTP Range reads of a few KB, no shard download) with their mul1 inner weights by frac_ref.py (exllamav3
1.6's frac.cu rule; the integer rates at MASK 0, checked equal to exl3_ref.dq_states), for turboexl3_glm53f's zig test
(src/frac_test.zig: turboexl3's format.decodeTile must give the same f16 bits). n 56 (sc_2.60 layer 3, K 3.5), n 40
(sc_2.60 layer 4, K 2.5), and the integer n 32, 48, 64, 80 (sc_2.10 layer 4, sc_3.10 layer 4, sc_3.10 layer 3, sc_4.10
layer 3). numpy + stdlib; frac_ref.py and exl3_ref.py from the DeepSeek repack's tools (queue/tdp/tools)."""
import json, os, struct, sys, urllib.parse, urllib.request

import numpy as np

# FRAC_TOOLS: the folder holding frac_ref.py and exl3_ref.py (the DeepSeek repack's converter tools)
sys.path.insert(0, os.environ.get('FRAC_TOOLS', '.'))
import frac_ref  # noqa: E402

REPO = 'turboderp/GLM-5.3-Flash-exl3'
REVS = {'sc_2.10bpw': '8d2eca0ed5b695108249308bcdc2af0636d5045f', 'sc_2.60bpw': '24781209f2862a939cc21e01d256c943bd1a58a3',
        'sc_3.10bpw': 'b570edf655a2bb00beca23e18d3b98724bfc2db5', 'sc_4.10bpw': 'c6b8ac3f6069173343e253e1cba90536ed2c72dc'}
PICK = [('sc_2.60bpw', 3, 56), ('sc_2.60bpw', 4, 40), ('sc_2.10bpw', 4, 32), ('sc_3.10bpw', 4, 48),
        ('sc_3.10bpw', 3, 64), ('sc_4.10bpw', 3, 80)]
TILES = [0, 1, 2, 3, 143, 1000, 32767]          # a 4096 x 2048 projection has 256 x 128 = 32768 tiles


def get(label, path, a=None, b=None):
    u = f'https://huggingface.co/{REPO}/resolve/{REVS[label]}/{urllib.parse.quote(path)}'
    h = {'User-Agent': 'glm53f_fixture/1'}
    if a is not None:
        h['Range'] = f'bytes={a}-{b - 1}'
    with urllib.request.urlopen(urllib.request.Request(u, headers=h), timeout=120) as r:
        data = r.read()
    if a is not None:
        assert len(data) == b - a, (len(data), b - a)
    return data


index = {}
headers = {}


def tensor(label, name):
    if label not in index:
        index[label] = json.loads(get(label, 'model.safetensors.index.json'))['weight_map']
    shard = index[label][name]
    if (label, shard) not in headers:
        n = struct.unpack('<Q', get(label, shard, 0, 8))[0]
        headers[(label, shard)] = (n, json.loads(get(label, shard, 8, 8 + n)))
    n, h = headers[(label, shard)]
    return shard, 8 + n, h[name]


fix = {'what': "turboderp GLM-5.3-Flash EXL3 routed-expert trellis tiles: stored int16 words and the mul1 inner weight "
               "(f16 bits, row-major) by exllamav3 frac.cu's ring rule (frac_ref.py)", 'tiles': []}
controls = 0
for label, layer, n in PICK:
    for proj in ('gate_proj', 'down_proj'):
        name = f'model.language_model.layers.{layer}.mlp.experts.7.{proj}.trellis'
        shard, base, t = tensor(label, name)
        assert t['dtype'] == 'I16' and t['shape'][-1] == n, (name, t['shape'])
        base += t['data_offsets'][0]
        for ti in TILES:
            a = base + ti * n * 2
            w = np.frombuffer(get(label, shard, a, a + n * 2), np.int16)[None]
            f16 = frac_ref.tiles_f16(w, n)[0]
            if n % 16 == 0:
                bad = frac_ref.selfcheck_integer(w, n)
                assert bad == 0, (name, ti, bad)
                controls += 1
            fix['tiles'].append({'src': f'{label} {shard} {name} tile {ti}', 'n': n,
                                 'words': [int(x) for x in w[0].view(np.uint16)], 'expect': [int(x) for x in f16]})
json.dump(fix, open(sys.argv[1], 'w'), separators=(',', ':'))
print(f'FIXTURE_OK {len(fix["tiles"])} tiles ({controls} integer controls equal to exl3_ref.dq_states) -> {sys.argv[1]}; '
      f'{frac_ref.selftest()}')
