# Contributing

Add a plugin, fix one, or improve the tooling. The rules are the same for people and agents; `AGENTS.md` is the
long form and is written to be handed to an agent as is.

## Add a plugin

1. Fork, then `scripts/setup.sh` (pinned Zig and kit).
2. `scripts/new-plugin.sh my_plugin [--media|--quant]`, then fill in `config.zig`, `plugin.zig`, `plugin.json`, `README.md`.
3. `scripts/build.sh my_plugin`, then `scripts/check.sh my_plugin --model <pack dir>`.
4. `scripts/lint.sh && scripts/gen-index.sh && scripts/build.sh --all` (the last one is the gate).
5. Open a PR from `add-my_plugin`. One plugin per PR.

## What a reviewer checks

- It compiles with every other plugin: CI builds all of them, not only yours.
- Narrow claim, MLX-free parse, bounds-checked config, named refusals.
- Tests for what can regress, and `plugin check` passes (or `"fixture": "none"` with a stated reason).
- A README with what it serves, what it does not, and measured numbers with hardware (decode and prefill tok/s first).
- Vendored code is unchanged, with provenance in `NOTICE` and a license that allows it.
- No binaries, no generated files except `plugins.json`.

## Releasing

Tag `<name>-v<x.y.z>` on `main`. The workflow builds the plugin from source and attaches the dylib and its sha256.
Users who can compile should: `scripts/install.sh <name>`.
