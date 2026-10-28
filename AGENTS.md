# CommunityServe/plugins: guide for AI agents

Outside plugins (dylibs) for mlx-serve and sushi: model, media and quant. Each folder under `plugins/` is a standalone
package built against one pinned kit. This file is the whole contract; humans read the same rules in CONTRIBUTING.md.

## Set up (any task)

    scripts/setup.sh          # pinned Zig into .zig-toolchain/, pinned kit into .kit/kit/. Idempotent.

Never use a system Zig and never edit `.zig-toolchain/` or `.kit/`. `KIT_VERSION` names the kit release; the kit's MLX
pin must equal the mlx-serve that loads the plugin. If the release for `KIT_VERSION` has no kit asset yet, point
`MLX_SERVE_KIT` at a staged kit (`zig build kit` in an mlx-serve checkout) and say so in the PR.

## The rules

1. **Every plugin compiles, always.** `scripts/build.sh --all` is green before you push and on every commit of `main`.
   A PR that breaks any plugin, not just its own, does not merge. No exceptions for "work in progress": use a draft PR.
2. **Compile from source.** Never download, commit or run a prebuilt dylib. `scripts/install.sh <name>` builds the
   source you can read, then installs it. Release binaries are a convenience for people without Zig, never the truth.
3. **Pinned toolchain only** (`scripts/setup.sh`). No system Zig, no other kit version.
4. **One plugin per PR.** Touch another plugin's folder or the shared scripts only to fix a bug, and say why.
5. **Narrow claim, MLX-free parse.** `claims` and `parse` run with no GPU. Claim a pack by `model_type` alone only if
   the plugin should replace the built-in for every such pack; otherwise claim by marker so the built-in keeps the rest.
6. **Refuse by name.** An unsupported config or layout is a named error and a `log.err` line, never a fallback to
   another arch and never a crash. Config JSON is untrusted: bounds-check every field.
7. **Bill what you allocate.** Memory the plugin allocates is part of its bill, each term an upper bound it measured.
8. **Tests for behaviour that can regress.** Fail first, then green. One good test beats five; no test that only
   checks a file contains the words you wrote. `zig build test` (or the step `plugin.json` names in `test_step`) runs
   MLX-free, with no model.
9. **Third-party code is vendored unchanged**, with its provenance in `NOTICE`. Edits to it live in the sync script
   that copies it, not in the copy.
10. **Numbers are measured**, on stated hardware, decode and prefill tok/s first. Say what was not measured.
11. **Leave the user's mlx-serve alone while testing.** Install under a throwaway HOME (`scripts/install.sh <name>
    --home ~/tmp/h`). Never run two servers that each load a large pack at once.
12. **Public text is short and plain.** PR titles and bodies say what changed and what was checked. No filler.
13. **Open the PR only when the user asks.** Commit and push on request, never on your own.

## Create a plugin

    scripts/new-plugin.sh my_plugin [--media|--quant]   # plugins/my_plugin/ from the kit's example
    # edit config.zig (claim, parse), plugin.zig (load, bill, forward); fill plugin.json and README.md
    scripts/build.sh my_plugin                          # tests, then ReleaseFast
    scripts/check.sh my_plugin --model <pack dir>       # conformance + does it claim and parse that pack
    scripts/lint.sh && scripts/gen-index.sh && scripts/build.sh --all

Read, in order: `docs/writing-a-plugin.md`, then the kit's own `.kit/kit/src/abi/plugin.zig` (every struct with field
comments) and `.kit/kit/examples/_example/plugin.zig` (a working decoder). Weight names and config keys follow mlx-lm,
so packs load across engines. Run the real pack twice, plugin on and `--no-plugins`, and diff greedy output.

PR: fork, branch `add-<name>`, push, `gh pr create` with the template filled in. The checklist is the rules above.

## Use a plugin (build it, add it to mlx-serve)

    scripts/install.sh <name>                    # compile, then `mlx-serve plugin install`
    scripts/install.sh <name> --home ~/tmp/h     # same, under a throwaway HOME

Plugins load only when `~/.mlx-serve/plugins.json` has `"enabled": true` (the app's Settings > Server writes it).
Restart the server after every reinstall; a reinstall changes the hash, so install again rather than copying a file.
Check it served the pack: the log has `[plugin] <name> serves <model_type>` and `/v1/models` shows `"plugin": "<name>"`.
`plugins.json` at the repo root lists what exists, with the mlx-serve each plugin needs.

## Troubleshooting

- `built against another MLX`: the kit and the server differ. Use the kit of the mlx-serve release you run.
- `has no 'plugin' command`: mlx-serve is too old; plugins need a release that ships the plugin kit.
- Load refused with a hash message: reinstall with `scripts/install.sh`, then restart.
- Build fails right after a kit bump: ABI changed; read the kit's `src/abi/plugin.zig` diff.
