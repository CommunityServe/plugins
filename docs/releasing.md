# Releasing

1. `scripts/build.sh --all`, `scripts/lint.sh`, `scripts/gen-index.sh`; bump `version` in the plugin's `plugin.json`.
2. Merge to `main`, then tag `<name>-v<x.y.z>`.
3. `release.yml` builds that plugin from source on macOS arm64 and attaches `lib<name>.dylib` and its sha256.

A kit bump is its own PR: change `KIT_VERSION`, rebuild everything, fix what the ABI change broke. No plugin merges
against a kit that is not a published mlx-serve release, except with `MLX_SERVE_KIT` and a note saying so.
