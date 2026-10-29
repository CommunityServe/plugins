# turboexl3_qwen38

turboderp's fully EXL3 Qwen3.8-Flash packs (K2 to K5) through turboexl3, as an outside plugin for mlx-serve, on
mlx-serve's own Qwen3.8 code. Claims only `qwen4_exp` packs whose config has a `trunk_quant` block; every other
Qwen3.8 pack (iQ-MLX, expert-only EXL3 such as Sushi's) keeps the built-in plugin.

    scripts/build.sh turboexl3_qwen38                                # from the repo root: tests, then ReleaseFast
    plugins/turboexl3_qwen38/scripts/sync-mlx-serve.sh <checkout>    # refresh src/mlx_serve/ from mlx-serve
    plugins/turboexl3_qwen38/scripts/sync-mlx-serve.sh --check <checkout> <ref>  # exit 0 = keep the pin
    zig build gpu-test -Dmlx-lib=<dir>      # in the plugin's folder: #1005's kernel contract and GPU tests, against
                                            # the server's MLX (<dir> holds its libmlx.dylib and libmlxc.dylib)

Status: claim, parse, load, forward, the MTP drafter and the prefix cache (RAM), built; K2 to K5 served on an M5 Max.
Not yet: images and the prefix cache's disk tier.

MTP needs two mlx-serve fixes, still open in [ddalcu/mlx-serve#1017](https://github.com/ddalcu/mlx-serve/pull/1017): a
forced commit drains before a native round, and native rounds keep a recurrent model's position. Without the second, an
MTP request that reaches a forced commit (a thinking budget's close, a forced tool-call opener) fails there with "the
position does not follow the slot's state"; without the first, the last forced token enters the model twice and the
answer drifts from that point. With `--no-mtp` it serves correctly on mlx-serve main as it is. #1017 also lets the
engine skip its own pass over a pack the plugin reads itself, and `plugin check --model` check a pack that has no
fixture.

Three interface additions are pending in [ddalcu/mlx-serve#1022](https://github.com/ddalcu/mlx-serve/pull/1022): the
latched MLX error's text (the copies retry a smaller threadgroup on it; an M5 Max never hits those limits), the MTP
depth for a native drafter (`--mtp-depth` does not reach the plugin), and a forced-close hook (#1005 commits a thinking
budget's close through the MTP head; the plugin can once the host calls it).

- The load and forward are mlx-serve's own `qwen4_exp` (#1005's, copied to `src/mlx_serve/` by
  `scripts/sync-mlx-serve.sh`), over mlx-c through the kit's `mlx_c`; each slot runs on state the model keeps itself.
  The copy is cut to what the plugin's builds compile (the dylib, `zig build test`, mlx-free-test, gpu-test), as those
  builds' symbols and the compiler's errors show it: a file with no such code is an empty stand-in, and a test none of
  them runs, or a function none of them compiles (`scripts/cut-functions.txt`), is left out: `src/mlx_serve/` holds
  87,046 lines, 164,603 before the cut.
- MTP: the pack's head drafts as the model's native drafter (`src/spec/drafter.zig`) under #1005's own round policy,
  copied: a prompt-lookup run when the context repeats, else the head's chain at the depth #1005's EV planner picks
  (two-chunk plans priced for EXL3), with the next round's drafts dispatched at each greedy round's tail; one trunk
  forward verifies a round. Greedy rounds and sampled ones (the server accepts those itself). The server's `--mtp-*`
  flags do not reach it; its `mtp` load setting does (`--no-mtp`: no head history, no rounds).
  - #1005's padded head history: a round's rows wait as a stash, and the next head chain appends the whole verify row,
    rejected drafts included, then drafts past them out of their sight.
  - #1005's lazy pre-draft: a greedy round with budget for another builds the next round's first chunk from its own
    verdict before the server reads the round, and keeps it when the read agrees.
  - #1005's PLE id lend: a drafted round's verify gathers its PLE rows from the ids it already holds, `[t1, drafts]`,
    instead of a read that waits for the whole draft chain.
  - Each MTP request ends with #1005's `[spec-stats]` lines: rounds, acceptance, the round-cost table at the request's
    context, lookup rounds, lazy pre-drafts and the chunk-A pricing.
- Prefix cache (RAM): each prompt's state, copied at the start of its prefill's final span (the trunk's KV and
  recurrent entries, the MTP head's history), as the server's module snapshots (`CacheHooks.aux`); a later prompt
  that extends it restores there. The server then holds back the prompt's last 30 tokens as it does for #1005, so
  both split a prompt alike, which greedy bytes depend on.
- The parse is the built-in's, after the EXL3 blocks are checked against what turboexl3 decodes: format `exl3`,
  codebook `mul1` or `mcg`, window 8 to 16, a bank rate of 1 to 8 bits in eighths, top-k at most 32. Anything else
  is refused by name.
- No environment variable changes what it does.

## Measured

MacBook Pro, M5 Max, 128 GB. The 17,743-token coding prompt, greedy, LOW thinking (a 2,048-token budget), at most 16,384
tokens. Cold prefill is the warm-up request's, on a fresh server; decode is the measured request's. #1005 is mlx-serve
PR #1005 at 1b3e624b with its defaults. Each table names the window it comes from and the plugin build.

MTP off, window 2548, the plugin at d832576 (dylib 6f7966adc62a) on mlx-serve main 7d52c147, two runs per side: the same
answer as #1005 on every pack, byte for byte.

| Pack | Decode tok/s, #1005 | Decode tok/s, plugin | Cold prefill tok/s, #1005 | Cold prefill tok/s, plugin | Answer |
|---|---|---|---|---|---|
| K2 | 65.0, 65.0 | 64.8, 64.8 | 2,813, 2,735 | 2,768, 2,764 | 8,235 tokens |
| K3 | 61.0, 61.1 | 61.1, 61.3 | 2,764, 2,632 | 2,662, 2,710 | 8,614 tokens |
| K4 | 59.8, 59.9 | 59.6, 59.7 | 2,769, 2,535 | 2,587, 2,539 | 16,112 tokens |
| K5 | 55.3, 55.3 | 55.3, 55.4 | 2,474, 2,523 | 2,495, 2,532 | 10,937 tokens |

MTP on: the plugin on mlx-serve main 7d52c147 with #1017's two MTP fixes, one run per pack, each row its own window and
build; #1005 where a window ran it at the same settings. The two are compared on speed, not bytes. Greedy rounds accept
by argmax on both sides; the bytes differ only because the rounds' shapes differ: a verify over 1 + n rows is not
bit-equal to n decode steps, and each side plans its rounds from live timings.

| Pack | Window | Plugin build | Decode tok/s, plugin | Decode tok/s, #1005 | Cold prefill tok/s, plugin | Cold prefill tok/s, #1005 |
|---|---|---|---|---|---|---|
| K2 | 2562 | dylib b281345a9212, with the PLE id lend | 115.9 | 115.3 (2552) | 2,759.7 | 2,750.1 (2552) |
| K3 | 2560 | 7c3b922 (dylib f7d54463ecdf), before the PLE id lend | 120.2 | not run | 2,609.5 | not run |
| K4 | 2560 | 7c3b922 (dylib f7d54463ecdf), before the PLE id lend | 123.0 | not run | 2,653.2 | not run |
| K5 | 2554 | d832576 (dylib 6f7966adc62a), before the padded head history, lazy pre-draft and PLE id lend | 115.9 | not run | 2,401.4 | not run |

Not measured: sampled requests, other prompts, other Macs.

`plugin check` needs a conformance fixture pack, which a fully EXL3 pack cannot provide small.
