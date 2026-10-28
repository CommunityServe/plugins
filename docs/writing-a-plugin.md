# Writing a plugin

The server owns HTTP, templates, tool calls, sampling, scheduling and the KV cache. A plugin supplies what is
specific to a model or format. Pick the shallowest seam that works.

| Kind | Teaches the server | Start with |
|---|---|---|
| model | an LLM, vision or embedding arch: config parse, weight load, memory bill, forward | `scripts/new-plugin.sh x` |
| media | an image, audio, video or 3D generator | `scripts/new-plugin.sh x --media` |
| quant | a weight encoding (a tensor layout and its matmul) | `scripts/new-plugin.sh x --quant` |

## The loop

    scripts/new-plugin.sh my_plugin
    scripts/build.sh my_plugin                       # tests, then ReleaseFast
    scripts/check.sh my_plugin --model <pack dir>    # conformance, and does it claim and parse this pack
    scripts/install.sh my_plugin --home ~/tmp/h      # serve it under a throwaway HOME, leave your app alone
    HOME=~/tmp/h mlx-serve --model <pack dir> --port 18082
    HOME=~/tmp/h mlx-serve --model <pack dir> --port 18083 --no-plugins   # the built-in, to diff against

`plugin check` proves the contract, not the numerics. Serve the real pack with the plugin on and off and diff greedy
output.

## What a model plugin is

- `config.zig`: `claims` (which packs) and `parse` (the facts the host needs). Both run with no GPU.
- `plugin.zig`: `bill` (what it allocates), `load`, `forward`, and optional hooks: cache layout and prefix-cache
  snapshots, media (image towers), a native drafter, weight filtering.
- A claim by `model_type` alone replaces the built-in for every pack of that type; a claim by marker (a key or value
  only your packs have) leaves the rest to the built-in. A built-in wins a tie.

## Hooks worth knowing (all in the kit's `src/abi/plugin.zig`)

- `CacheHooks.aux`: prefix-cache snapshots of state the plugin owns.
- `MediaHooks`: `prepare` decodes and plans an image, `tower` encodes it, `placeholder_rows` lays out its tokens.
- `SpecHooks.native_drafter`: a block drafter. `step` is a greedy round; `verify` and `finish` hand the block back so
  the engine does a sampled accept.
- `WeightHooks.keep_key` and `resident_bytes`: a plugin that loads its own tensors tells the host to load none.
- `LoadCtx.settings_json`: the model's settings (`mtp`, `ctx_size`, ...). Size memory for the context it names.

`plugins/sushi_ds41` uses all of these; read it as a worked example of a large streamed model.

## Gotchas

- Plugins load only when `~/.mlx-serve/plugins.json` has `"enabled": true`. The app writes that file; test under a
  throwaway HOME.
- A reinstall changes the hash; install again so the receipt matches. A dylib is never unloaded: restart the server.
- A plugin built against another MLX pin is refused at load, with the reason in the log.
- Outside plugins are macOS only.
- Do not add CLI flags or env levers for one plugin. Use what the host has: the GPU ceiling, `--wired-margin-gib`,
  the per-model `ctx_size`.

The full reference is mlx-serve's plugin docs: `docs/plugins.md`, `plugin-model.md`, `plugin-media.md`,
`plugin-quant.md`.
