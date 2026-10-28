---
name: install-plugin
description: Build a plugin from this repo from source and install it into the user's mlx-serve. Use when asked to add, install or try a plugin.
---

1. Read `docs/using-plugins.md`.
2. `scripts/install.sh <name>` (add `--home <dir>` if the user's app is running or they want to try it first).
   It compiles from source; never download a prebuilt dylib.
3. Tell the user to set `"enabled": true` in `~/.mlx-serve/plugins.json` (or Settings > Server) and restart the server.
4. Confirm: the log shows `[plugin] <name> serves <model_type>` and `/v1/models` shows `"plugin": "<name>"`.
5. If it fails, see Troubleshooting in `AGENTS.md`; do not work around a refusal by editing the kit or the receipt.
