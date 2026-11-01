#!/usr/bin/env bash
# The HTTP surface on a real pack, through this plugin: serve it under a throwaway HOME, then an answer, a reasoning
# split, a parsed tool call, a streamed request cancelled and a clean one after it, and an unload.
# Usage: MINIMAX_PACK=<prepared pack dir> plugins/minimax_m3/tests/serve_check.sh   (skips without MINIMAX_PACK)
# Needs an mlx-serve on PATH with the MiniMax parser (#1047) and the streamed-plugin load seam, and the plugin built.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
root="$(cd "$here/../.." && pwd)"
[ -n "${MINIMAX_PACK:-}" ] || { echo "SKIP: set MINIMAX_PACK to a prepared MiniMax-M3.1-Flash pack"; exit 0; }
port="${MINIMAX_PORT:-18093}"
home="$(mktemp -d)"
trap 'kill "${pid:-0}" 2>/dev/null || true; wait "${pid:-0}" 2>/dev/null || true; rm -rf "$home"' EXIT
"$root/scripts/install.sh" minimax_m3 --home "$home" >/dev/null
HOME="$home" mlx-serve --model "$MINIMAX_PACK" --serve --host 127.0.0.1 --port "$port" --ctx-size 4096 \
  --max-concurrent 1 --no-mtp --no-pld --prefix-cache-mem 0 --prefix-cache-disk 0 --no-prefix-cache-ram \
  >"$home/server.log" 2>&1 &
pid=$!
python3 - "$port" "$(basename "$MINIMAX_PACK")" "$home/server.log" <<'P'
import json, sys, time, urllib.request
port, model, log = sys.argv[1], sys.argv[2], sys.argv[3]
base = f"http://127.0.0.1:{port}"
def post(path, body, timeout=1800):
    rq = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(rq, timeout=timeout) as r:
        return json.loads(r.read())
for _ in range(600):
    try:
        urllib.request.urlopen(base + "/health", timeout=5); break
    except Exception:
        time.sleep(1)
else:
    sys.exit("FAIL: the server did not come up")
def chat(msgs, **kw):
    return post("/v1/chat/completions", {"model": model, "messages": msgs, "temperature": 0, "stream": False, **kw})
r = chat([{"role": "user", "content": "What is the capital of France? Answer in one short sentence."}], max_tokens=256)
assert "Paris" in r["choices"][0]["message"]["content"], r
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
r = chat([{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}], tools=tools, max_tokens=512)
calls = r["choices"][0]["message"].get("tool_calls") or []
assert r["choices"][0]["finish_reason"] == "tool_calls" and calls, r
assert calls[0]["function"]["name"] == "get_weather" and json.loads(calls[0]["function"]["arguments"]).get("city") == "Paris", calls
rq = urllib.request.Request(base + "/v1/chat/completions", headers={"Content-Type": "application/json"},
    data=json.dumps({"model": model, "messages": [{"role": "user", "content": "Count from 1 to 200."}], "stream": True, "max_tokens": 400}).encode())
with urllib.request.urlopen(rq, timeout=600) as s:
    for i, _ in enumerate(s):
        if i >= 8: break
r = chat([{"role": "user", "content": "Reply with OK."}], max_tokens=64)
m = r["choices"][0]["message"]
assert "OK" in (m.get("content") or ""), r
assert m.get("reasoning_content") and "<mm:think>" not in (m.get("content") or ""), m
u = post("/v1/unload-model", {"model": model})
assert u["model"]["loaded"] is False, u
assert "[plugin] minimax_m3 serves" in open(log).read(), "the plugin did not serve the pack"
print("PASS: answer, tool call, cancel then a clean request with its reasoning split, unload, served by minimax_m3")
P
