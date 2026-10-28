---
name: create-plugin
description: Create a new mlx-serve plugin in this repo and prepare its pull request. Use when asked to add support for a model, media backend or quant format as an outside plugin.
---

1. Read `AGENTS.md` (the rules) and `docs/writing-a-plugin.md`.
2. `scripts/setup.sh`. Never use a system Zig.
3. `scripts/new-plugin.sh <name> [--media|--quant]`. Name is `[a-z0-9_]`.
4. Read the checkpoint's `config.json` and weight index and mlx-lm's model file; keep mlx-lm's names.
5. Write the claim narrow and `parse` MLX-free; write a failing test, then the code. Refuse unsupported configs by name.
6. Fill `plugin.json` and `README.md` (what it serves, what it does not, measured decode and prefill tok/s with hardware).
7. `scripts/build.sh <name>`, `scripts/check.sh <name> --model <pack>`, then serve the pack with the plugin on and
   with `--no-plugins` under a throwaway HOME and diff greedy output.
8. `scripts/lint.sh && scripts/gen-index.sh && scripts/build.sh --all`. All plugins must compile.
9. Stop. Open the PR (fork, branch `add-<name>`, `gh pr create` with the template) only when the user says so.
