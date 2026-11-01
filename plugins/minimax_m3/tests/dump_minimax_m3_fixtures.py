#!/usr/bin/env python3
"""float64 reference logits for the minimax_m3 conformance fixture, from Hugging Face's MiniMax-M3 text model.

The math is transcribed from transformers' `models/minimax_m3_vl/modeling_minimax_m3_vl.py` (5.14.1): Gemma RMSNorm
(`x * rsqrt(mean(x^2) + eps) * (1 + w)`), per-head QK norm, split-half partial RoPE on the first `head_dim *
partial_rotary_factor` dims, the lightning indexer (unscaled index scores, causal, max-pooled into 128-key blocks, the
query's own block forced in, top-k blocks per index head, each head's choice shared by its KV group), eager softmax
attention, sigmoid routing with the correction bias choosing and the plain sigmoid weighting, SwiGLU-OAI experts
(`(clamp(up) + 1) * g * sigmoid(alpha g)`, gate clamped above) times `routed_scaling_factor` plus the shared expert.
Routed experts are the bank's NVFP4 records dequantized as the format defines them (E2M1 code x E4M3 scale x g/2688).
numpy only, no torch: it runs where transformers' torch model cannot.

  1. zig build test -Dtest-filter="minimax_m3 parity" with MINIMAX_M3_FIXTURE_DUMP=<dir>  (writes the fixture + prompt)
  2. python3 tests/dump_minimax_m3_fixtures.py <dir>                                     (writes reference_logits.f32)
  3. zig build test -Dtest-filter="minimax_m3 parity" with MINIMAX_M3_REFERENCE=<dir>     (compares)
"""
import json, struct, sys
import numpy as np

E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float64)


def e4m3(b):
    b = b.astype(np.int64)
    s = np.where(b & 0x80, -1.0, 1.0)
    e = (b >> 3) & 0xF
    m = (b & 0x7).astype(np.float64)
    v = np.where(e == 0, m / 8.0 * 2.0 ** -6, (1 + m / 8.0) * 2.0 ** (e - 7))
    v = np.where((b & 0x7F) == 0x7F, np.nan, v)
    return s * v


def safetensors(path):
    raw = open(path, "rb").read()
    n = struct.unpack("<Q", raw[:8])[0]
    hdr = json.loads(raw[8:8 + n])
    base = 8 + n
    out = {}
    for name, d in hdr.items():
        if name == "__metadata__":
            continue
        a, b = d["data_offsets"]
        buf = raw[base + a:base + b]
        if d["dtype"] == "BF16":
            u = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16
            arr = u.view(np.float32)
        elif d["dtype"] == "F32":
            arr = np.frombuffer(buf, dtype=np.float32)
        else:
            raise ValueError(f"{name}: dtype {d['dtype']}")
        out[name] = arr.astype(np.float64).reshape(d["shape"])
    return out


def experts(dirp, man):
    """[layer][expert] -> (gate [I,H], up [I,H], down [H,I]) in float64."""
    data = open(f"{dirp}/experts.bin", "rb").read()
    parts = {p["name"]: p for p in man["parts"]}
    stride = man["record_stride"]

    def proj(rec, name):
        w, sc, g = parts[f"{name}.weight"], parts[f"{name}.scales"], parts[f"{name}.global_scale"]
        out, words = w["shape"]
        codes = np.frombuffer(rec[w["offset"]:w["offset"] + w["length"]], dtype=np.uint32).reshape(out, words)
        nib = ((codes[..., None] >> (4 * np.arange(8, dtype=np.uint32))) & 0xF).reshape(out, words * 8)
        val = E2M1[nib & 7] * np.where(nib & 8, -1.0, 1.0)
        scales = e4m3(np.frombuffer(rec[sc["offset"]:sc["offset"] + sc["length"]], dtype=np.uint8).reshape(out, -1))
        gs = struct.unpack("<f", rec[g["offset"]:g["offset"] + 4])[0]
        return val * (np.repeat(scales, 16, axis=1) * (gs / 2688.0))

    out = {}
    for r in man["records"]:
        rec = data[r["offset"]:r["offset"] + r["length"]]
        out.setdefault(r["layer"], {})[r["expert"]] = tuple(proj(rec, n) for n in ("gate_proj", "up_proj", "down_proj"))
    return out


BF16 = False


def bf(x):
    """With --bf16, round to bfloat16 (nearest even) where the plugin stores a bf16 activation."""
    if not BF16:
        return x
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16 << 16).astype(np.uint32)
    return u.view(np.float32).astype(np.float64)


def norm(x, w, eps):
    return bf(x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * (1.0 + w))


def rope(x, cos, sin):
    rd = cos.shape[-1]
    r, p = x[..., :rd], x[..., rd:]
    h = rd // 2
    rot = np.concatenate([-r[..., h:], r[..., :h]], axis=-1)
    return np.concatenate([r * cos + rot * sin, p], axis=-1)


def swiglu(gate, up, alpha, limit):
    gate = np.minimum(gate, limit)
    up = np.clip(up, -limit, limit)
    return bf((up + 1.0) * (gate / (1.0 + np.exp(-gate * alpha))))


def main(dirp, tag=""):
    cfg = json.load(open(f"{dirp}/config.json"))
    c = cfg.get("text_config", cfg)
    sp = c["sparse_attention_config"]
    W = safetensors(f"{dirp}/model.safetensors")
    EX = experts(dirp, json.load(open(f"{dirp}/expert-manifest.json")))
    ids = json.load(open(f"{dirp}/prompt.json"))["ids"]
    S, H, D = len(ids), c["hidden_size"], c["head_dim"]
    nh, nkv, eps = c["num_attention_heads"], c["num_key_value_heads"], c["rms_norm_eps"]
    G = nh // nkv
    ih, idim, bs, topk, local = sp["sparse_num_index_heads"], sp["sparse_index_dim"], sp["sparse_block_size"], sp["sparse_topk_blocks"], sp["sparse_local_block"]
    rd = int(D * c["partial_rotary_factor"])
    inv = 1.0 / (c["rope_theta"] ** (np.arange(0, rd, 2, dtype=np.float64) / rd))
    freqs = np.arange(S, dtype=np.float64)[:, None] * inv[None, :]
    emb = np.concatenate([freqs, freqs], axis=-1)
    cos, sin = np.cos(emb), np.sin(emb)
    alpha, limit = c["swiglu_alpha"], c["swiglu_limit"]
    causal = np.arange(S)[None, :] > np.arange(S)[:, None]  # key after query
    nb = -(-S // bs)
    qblock = np.arange(S) // bs

    x = W["model.embed_tokens.weight"][ids]
    margin = np.full(S, np.inf)  # block selection: 16th less 17th block score, least over layers and heads
    route_margin = np.full(S, np.inf)  # routing: 4th less 5th choice score, least over MoE layers
    for L in range(c["num_hidden_layers"]):
        p = f"model.layers.{L}."
        h = norm(x, W[p + "input_layernorm.weight"], eps)
        q = norm((h @ W[p + "self_attn.q_proj.weight"].T).reshape(S, nh, D), W[p + "self_attn.q_norm.weight"], eps)
        k = norm((h @ W[p + "self_attn.k_proj.weight"].T).reshape(S, nkv, D), W[p + "self_attn.k_norm.weight"], eps)
        v = (h @ W[p + "self_attn.v_proj.weight"].T).reshape(S, nkv, D)
        q, k = rope(q, cos[:, None], sin[:, None]), rope(k, cos[:, None], sin[:, None])
        # Indexer: one block selection per index head.
        iq = norm((h @ W[p + "self_attn.index_q_proj.weight"].T).reshape(S, ih, idim), W[p + "self_attn.index_q_norm.weight"], eps)
        ik = norm(h @ W[p + "self_attn.index_k_proj.weight"].T, W[p + "self_attn.index_k_norm.weight"], eps)
        iq, ik = rope(iq, cos[:, None, :idim], sin[:, None, :idim]), rope(ik, cos[:, :idim], sin[:, :idim])
        keep = np.zeros((ih, S, S), dtype=bool)
        for hh in range(ih):
            sc = iq[:, hh, :] @ ik.T
            sc[causal] = -np.inf
            padded = np.full((S, nb * bs), -np.inf)
            padded[:, :S] = sc
            blocks = padded.reshape(S, nb, bs).max(-1)
            for lb in range(local):
                blocks[np.arange(S), np.maximum(qblock - lb, 0)] = np.inf
            kk = min(topk, nb)
            if nb > topk:
                srt = -np.sort(-blocks, axis=-1)
                with np.errstate(invalid="ignore"):
                    gap = srt[:, topk - 1] - srt[:, topk]
                margin = np.minimum(margin, np.where(np.isfinite(gap), gap, np.inf))
            order = np.argsort(-blocks, axis=-1, kind="stable")[:, :kk]
            chosen = np.take_along_axis(blocks, order, -1) != -np.inf
            bk = np.zeros((S, nb), dtype=bool)
            rows = np.repeat(np.arange(S)[:, None], kk, 1)
            bk[rows[chosen], order[chosen]] = True
            keep[hh] = np.repeat(bk, bs, axis=1)[:, :S] & ~causal
        attn = np.zeros((S, nh, D))
        for hd in range(nh):
            kv = hd // G
            s = (q[:, hd, :] @ k[:, kv, :].T) * D ** -0.5
            s = np.where(keep[hd // (nh // ih)], s, -np.inf)
            s -= s.max(-1, keepdims=True)
            pr = np.exp(s)
            pr /= pr.sum(-1, keepdims=True)
            attn[:, hd, :] = pr @ v[:, kv, :]
        x = bf(x + bf(bf(attn).reshape(S, nh * D) @ W[p + "self_attn.o_proj.weight"].T))
        h = norm(x, W[p + "post_attention_layernorm.weight"], eps)
        if L < 3:
            m = p + "mlp."
            y = swiglu(h @ W[m + "gate_proj.weight"].T, h @ W[m + "up_proj.weight"].T, alpha, limit) @ W[m + "down_proj.weight"].T
        else:
            m = p + "block_sparse_moe."
            se = m + "shared_experts."
            shared = swiglu(h @ W[se + "gate_proj.weight"].T, h @ W[se + "up_proj.weight"].T, alpha, limit) @ W[se + "down_proj.weight"].T
            prob = 1.0 / (1.0 + np.exp(-(h @ W[m + "gate.weight"].T)))
            choice = prob + W[m + "e_score_correction_bias"]
            kt = c["num_experts_per_tok"]
            srt = -np.sort(-choice, axis=-1)
            route_margin = np.minimum(route_margin, srt[:, kt - 1] - srt[:, kt])
            top = np.argsort(-choice, axis=-1, kind="stable")[:, :kt]
            wts = np.take_along_axis(prob, top, -1)
            wts /= wts.sum(-1, keepdims=True)
            routed = np.zeros_like(h)
            for t in range(S):
                for j, e in enumerate(top[t]):
                    gw, uw, dw = EX[L][int(e)]
                    routed[t] += wts[t, j] * (swiglu(gw @ h[t], uw @ h[t], alpha, limit) @ dw.T)
            y = routed * c["routed_scaling_factor"] + shared
        x = bf(x + bf(y))
    logits = norm(x, W["model.norm.weight"], eps) @ W["lm_head.weight"].T
    logits.astype(np.float32).tofile(f"{dirp}/reference_logits{tag}.f32")
    margin.astype(np.float32).tofile(f"{dirp}/selection_margin{tag}.f32")
    route_margin.astype(np.float32).tofile(f"{dirp}/routing_margin{tag}.f32")
    print(json.dumps({"rows": S, "vocab": logits.shape[1], "blocks": nb, "argmax_last": int(logits[-1].argmax())}))


def hf_execute(dirp, dtype_name):
    """The official transformers MiniMaxM3VLForCausalLM on the fixture (CPU, eager attention), in `dtype_name`:
    the residents as stored, the routed experts dequantized from the bank as above and stacked as its 3-D params."""
    import torch
    from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLTextConfig
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3VLForCausalLM
    cfg = json.load(open(f"{dirp}/config.json"))
    tc = MiniMaxM3VLTextConfig(**cfg["text_config"])
    tc._attn_implementation = "eager"
    if dtype_name == "float64":
        tc._experts_implementation = "eager"  # grouped_mm takes no float64
    model = MiniMaxM3VLForCausalLM(tc).eval()
    W = safetensors(f"{dirp}/model.safetensors")
    EX = experts(dirp, json.load(open(f"{dirp}/expert-manifest.json")))
    sd = {"model.embed_tokens.weight": W["model.embed_tokens.weight"], "model.norm.weight": W["model.norm.weight"], "lm_head.weight": W["lm_head.weight"]}
    for L in range(tc.num_hidden_layers):
        p, q = f"model.layers.{L}.", f"model.layers.{L}."
        for n in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            sd[q + n] = W[p + n]
        for n in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
            sd[q + f"self_attn.{n}.weight"] = W[p + f"self_attn.{n}.weight"]
        for n in ("q_proj", "k_proj", "q_norm", "k_norm"):
            sd[q + f"self_attn.indexer.{n}.weight"] = W[p + f"self_attn.index_{n}.weight"]
        if tc.mlp_layer_types[L] == "dense":
            sd[q + "mlp.gate_up_proj.weight"] = np.concatenate([W[p + "mlp.gate_proj.weight"], W[p + "mlp.up_proj.weight"]], 0)
            sd[q + "mlp.down_proj.weight"] = W[p + "mlp.down_proj.weight"]
        else:
            m, se = p + "block_sparse_moe.", p + "block_sparse_moe.shared_experts."
            sd[q + "mlp.gate.weight"] = W[m + "gate.weight"]
            sd[q + "mlp.gate.e_score_correction_bias"] = W[m + "e_score_correction_bias"]
            sd[q + "mlp.shared_experts.gate_up_proj.weight"] = np.concatenate([W[se + "gate_proj.weight"], W[se + "up_proj.weight"]], 0)
            sd[q + "mlp.shared_experts.down_proj.weight"] = W[se + "down_proj.weight"]
            E = tc.num_local_experts
            sd[q + "mlp.experts.gate_up_proj"] = np.stack([np.concatenate([EX[L][e][0], EX[L][e][1]], 0) for e in range(E)])
            sd[q + "mlp.experts.down_proj"] = np.stack([EX[L][e][2] for e in range(E)])
    dt = {"float64": torch.float64, "float32": torch.float32, "bfloat16": torch.bfloat16}[dtype_name]
    want = model.state_dict()
    missing = [k for k in want if k not in sd and not k.endswith("inv_freq")]
    extra = [k for k in sd if k not in want]
    if missing or extra:
        raise SystemExit(f"state dict mismatch: missing {missing[:6]} extra {extra[:6]}")
    model.load_state_dict({k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in sd.items()}, strict=False)
    model = model.to(dt)
    ids = torch.tensor([json.load(open(f"{dirp}/prompt.json"))["ids"]])
    with torch.no_grad():
        logits = model(input_ids=ids, use_cache=False).logits[0].to(torch.float64).numpy()
    tag = {"float64": "_hf", "float32": "_hf_f32", "bfloat16": "_hf_bf16"}[dtype_name]
    logits.astype(np.float32).tofile(f"{dirp}/reference_logits{tag}.f32")
    print(json.dumps({"hf": dtype_name, "rows": logits.shape[0], "argmax_last": int(logits[-1].argmax()),
                      "experts_impl": getattr(tc, "_experts_implementation", None), "torch": torch.__version__}))


def hf_select(scores, position, bs, topk, local):
    """MiniMaxM3VLIndexer.forward's rule for one query: (key mask [heads, keys], boundary tie per head)."""
    heads, keys = scores.shape
    nb = -(-keys // bs)
    sc = scores.astype(np.float64).copy()
    sc[:, np.arange(keys) > position] = -np.inf
    padded = np.full((heads, nb * bs), -np.inf)
    padded[:, :keys] = sc
    blocks = padded.reshape(heads, nb, bs).max(-1)
    for lb in range(local):
        blocks[:, max(position // bs - lb, 0)] = np.inf
    kk = min(topk, nb)
    srt = -np.sort(-blocks, axis=-1)
    tie = (srt[:, kk - 1] == srt[:, kk]) & np.isfinite(srt[:, kk - 1]) if kk < nb else np.zeros(heads, dtype=bool)
    order = np.argsort(-blocks, axis=-1, kind="stable")[:, :kk]
    mask = np.zeros((heads, keys), dtype=bool)
    for h in range(heads):
        for b in order[h]:
            if blocks[h, b] != -np.inf:
                mask[h, b * bs:min((b + 1) * bs, keys)] = True
    mask[:, np.arange(keys) > position] = False
    return mask, tie


def selection(dirp):
    meta = json.load(open(f"{dirp}/selection.json"))
    H, bs, topk, local = meta["heads"], meta["block"], meta["topk"], meta["local"]
    bad = 0
    for ci, pos in enumerate(meta["positions"]):
        keys = pos + 1
        sc = np.fromfile(f"{dirp}/case{ci}.scores.f32", dtype=np.float32).reshape(H, keys)
        want, tie = hf_select(sc, pos, bs, topk, local)
        scalar = np.fromfile(f"{dirp}/case{ci}.scalar.u8", dtype=np.uint8).reshape(H, keys).astype(bool)
        ids = np.fromfile(f"{dirp}/case{ci}.kernel.i32", dtype=np.int32).reshape(H, topk)
        kernel = np.zeros((H, keys), dtype=bool)
        for h in range(H):
            for b in ids[h]:
                if b >= 0:
                    kernel[h, b * bs:min((b + 1) * bs, keys)] = True
        kernel[:, np.arange(keys) > pos] = False
        for h in range(H):
            ok_s, ok_k = (scalar[h] == want[h]).all(), (kernel[h] == want[h]).all()
            if not (ok_s and ok_k):
                bad += 0 if tie[h] else 1
            print(f"pos {pos:5d} blocks {-(-keys // bs):3d} head {h}: scalar {'=' if ok_s else 'DIFF'} kernel {'=' if ok_k else 'DIFF'}"
                  f" selected {int(want[h].sum())} keys{'  (boundary tie: order undefined in torch.topk)' if tie[h] else ''}")
    print(json.dumps({"untied_mismatches": bad}))
    return bad


def routing(dirp):
    """MiniMaxM3VLTopKRouter's rule (float64) against the plugin's routes on the same inputs."""
    meta = json.load(open(f"{dirp}/routing.json"))
    R, H, E, K = meta["rows"], meta["hidden"], meta["experts"], meta["topk"]
    bad = ties = 0
    worst = 0.0
    for ci in range(meta["cases"]):
        f = lambda n, dt, shape: np.fromfile(f"{dirp}/r{ci}.{n}", dtype=dt).reshape(shape)
        x, w, corr = f("x.f32", np.float32, (R, H)).astype(np.float64), f("router.f32", np.float32, (E, H)).astype(np.float64), f("corr.f32", np.float32, (E,)).astype(np.float64)
        ids, wts = f("ids.u16", np.uint16, (R, K)), f("weights.f32", np.float32, (R, K))
        prob = 1.0 / (1.0 + np.exp(-(x @ w.T)))
        choice = prob + corr
        for r in range(R):
            order = np.argsort(-choice[r], kind="stable")
            if choice[r, order[K - 1]] == choice[r, order[K]]:
                ties += 1
                continue
            want = set(order[:K].tolist())
            if set(ids[r].tolist()) != want:
                bad += 1
                print(f"case {ci} row {r}: experts {sorted(ids[r].tolist())} vs reference {sorted(want)}")
                continue
            ref = prob[r, ids[r]] / prob[r, ids[r]].sum() * meta["scaling"]
            worst = max(worst, float(np.max(np.abs(wts[r] - ref) / np.abs(ref))))
    print(json.dumps({"rows": meta["cases"] * R, "set_mismatches": bad, "boundary_ties": ties, "max_rel_weight_err": worst}))
    return bad or worst > 2 ** -7


if __name__ == "__main__":
    if sys.argv[2:3] == ["--hf"]:
        sys.exit(hf_execute(sys.argv[1], sys.argv[3] if len(sys.argv) > 3 else "float64"))
    if sys.argv[2:3] == ["--routing"]:
        sys.exit(1 if routing(sys.argv[1]) else 0)
    if sys.argv[2:3] == ["--selection"]:
        sys.exit(1 if selection(sys.argv[1]) else 0)
    BF16 = "--bf16" in sys.argv[2:]
    main(sys.argv[1], "_bf16" if BF16 else "")
