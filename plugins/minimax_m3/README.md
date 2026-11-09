# minimax_m3

MiniMax-M3.1-Flash (text) from a prepared pack: the residents in memory, the 128 routed experts streamed from
`experts.bin` (the release's NVFP4 codes, affine Q3, or BF16). It claims a `minimax_m3_vl` / `minimax_m3` /
`minimax_m3_text` pack only when its config declares that expert bank. Vision, the MTP head, distributed spans and
batched decode are refused by name, and so is a load without Metal or on a virtual GPU.

    scripts/build.sh minimax_m3                    # from the repo root: prepare, tests, then ReleaseFast
    scripts/install.sh minimax_m3 --home ~/tmp/h   # under a throwaway HOME
    scripts/check.sh minimax_m3 --model <pack>     # does it claim and parse that pack
    zig build gpu-test -Dmlx-lib=<dir>             # in this folder: the Metal tests, against the server's MLX

Needs an mlx-serve whose host parses MiniMax's reasoning markers and tool calls (ddalcu/mlx-serve#1047) and funds a
streamed plugin's memory and KV precision (the streamed-plugin load seam); without the seam the bank loads the fewest
rows it runs.

- src/mlx_serve (generated, not committed) is the model and mlx-serve's expert streamer: scripts/prepare.sh writes it
  from mlx-serve at MLX_SERVE_REF and copies overlay/mlx_serve over it (NOTICE). A build refuses a tree whose hash is not
  scripts/mlx_serve.sha256.
- src/shims.zig carries the calls mlx-serve links from its MLX C++ shims: the kernel cache, its launch and the
  uninitialised allocation go through the server's Host table; the NVFP4 gather with its per-expert global scale is
  written on mlx-c.

## Prepare a pack

The converters read a local copy of the release (`MiniMaxAI/MiniMax-M3.1-Flash`, actual safetensors), check its pinned
SHA-256s and write a new directory. Python 3.9+; the BF16 bank also needs numpy.

    python3 scripts/prepare_minimax_pack.py --src SNAPSHOT --out PACK \
      --source-revision 54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e --global-scale-convention divisor
    python3 scripts/prepare_minimax_oq3_pack.py --help                              # affine Q3 experts
    python3 scripts/prepare_minimax_pack.py --from-pack PACK --out BF16_PACK --experts bf16

The NVFP4 bank is about 232 GB, Q3 about 181 GB, BF16 about 826 GB. A pack's `expert_streaming.settings` may set
`expert_rows` (a cap on the rows a layer keeps) and `expert_usage_record` (a file every load appends routing counts to).
The model settings' `stream_experts` fixes the row count, `ctx_size` the context, `kv_quant` 8 a Q8 cache.

## Measured

Mac Studio M2 Ultra, 192 GB, macOS 26; served by mlx-serve main 20cb4cc4 with #1047 and the load seam; context 4096,
BF16 KV cache, greedy (seed 0), one request at a time, prefix caches off, no residents seeded. Q3 pack
(`mlx-community/MiniMax-M3.1-Flash-oQ3e-mtp` as a stream pack, 116 expert rows a layer). Before is this plugin as
first published, after is this tree; two runs each on one server binary (the two runs of each agree within 0.1 tok/s).

| Request | Prompt | Prefill tok/s before | after | Decode tok/s before | after |
|---|---|---|---|---|---|
| a Zig file review, 384 tokens (reasoning only) | 3662 | 11.1 | 75.7 | 5.8 | 8.2 |
| a code review, to EOS | 337 | 7.2 | 12.3 | 7.2 | 11.9 |
| a Fibonacci function, to EOS | 218 | 18.0 | 60.0 | 6.9 | 10.4 |

- The prompt runs layer by layer in spans of up to 4096 rows: each span reads the experts it misses once, its
  projections are batched, and its attention is one causal call inside the 16-block budget and tiles of rows past it.
  Decode syncs the GPU less a token, and its SwiGLU-OAI is one kernel (bit-identical to the op chain on finite values
  and signed zeros; NaN stays NaN).
- The Fibonacci answer and the one-line answer are byte for byte the same before and after. The 337-token review
  takes another greedy path; both answers cover the three points it asks for. The float64 parity against
  transformers' `MiniMaxM3VLForCausalLM` (tests/dump_minimax_m3_fixtures.py `--hf`) is unchanged: 2200 rows past 16
  blocks, min cosine 0.99955, no decided argmax miss; the whole-prompt prefill's last row 0.99998. Block selection and
  expert routing match the reference (0 mismatches, 48 of 48 routes).
- The prefill bill prices a span from the geometry; a sparse tile, an expert batch and a row-by-row group free their
  scratch once evaluated.
- tests/serve_check.sh is the HTTP check (`MINIMAX_PACK=<pack>`).

Not measured: other Macs (an M5 is expected to differ; nothing here is M5-specific), the NVFP4 and BF16 banks as an
A/B after these changes, an accuracy benchmark, the long review's final answer, concurrent requests.
