# Agentic setup

Everything here works from a clean checkout with an agent (Claude Code, Codex, Cursor, anything that reads
`AGENTS.md`). Two jobs: build a plugin and send it, or take a plugin and run it in mlx-serve.

## One-time

    git clone https://github.com/CommunityServe/plugins && cd plugins
    scripts/setup.sh

That stages the pinned Zig and the pinned kit inside the checkout. Nothing is installed system-wide.

## Prompt: create a plugin

> Read AGENTS.md and docs/writing-a-plugin.md. Add a plugin for `<model name>` (pack at `<path or HF id>`). Scaffold with
> `scripts/new-plugin.sh`, follow mlx-lm's weight names, write tests first, and keep the claim narrow. Do not stop
> until `scripts/build.sh --all`, `scripts/lint.sh` and `scripts/check.sh <name> --model <pack>` pass. Serve the pack
> with the plugin on and off under a throwaway HOME and diff greedy output. Report decode and prefill tok/s. Do not
> open a PR until I say so.

When you say go: "fork, branch `add-<name>`, push, and open the PR with the template filled in."

## Prompt: use a plugin

> Read AGENTS.md. Build `<plugin name>` from source with `scripts/install.sh <name>` and install it into my mlx-serve.
> Tell me what to set in plugins.json, restart the server on `<pack dir>`, and confirm `/v1/models` shows
> `"plugin": "<name>"`. Use `--home` first if my app is running.

## What an agent must not do

- Download or run a prebuilt dylib. Plugins are compiled here, from source.
- Use a system Zig, a different kit, or edit `.zig-toolchain/` / `.kit/`.
- Push, tag or open a PR without being asked.
- Load two large packs at once, or touch the user's `~/.mlx-serve` while testing.

## Skills

`.claude/skills/create-plugin` and `.claude/skills/install-plugin` hold the same two flows as step lists an agent can
follow without reading anything else.
