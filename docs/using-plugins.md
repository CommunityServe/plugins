# Using a plugin

Plugins are compiled on your machine from the source in this repo. That is on purpose: you can read what you run.

    git clone https://github.com/CommunityServe/plugins && cd plugins
    scripts/install.sh <name>

What it does: stages Zig and the kit (`scripts/setup.sh`), runs the plugin's tests, builds the dylib in ReleaseFast and
runs `mlx-serve plugin install` on it, which shows the file's hash and signer and writes a receipt in
`~/.mlx-serve/plugins/`.

Then:

1. Turn plugins on: `~/.mlx-serve/plugins.json` with `"enabled": true` (the app: Settings > Server).
2. Restart the server on a pack the plugin serves.
3. Check: the log has `[plugin] <name> serves <model_type>` and `/v1/models` shows `"plugin": "<name>"`.

`--no-plugins` starts the server with none, to compare against the built-in. To remove a plugin, delete its
`~/.mlx-serve/plugins/<name>.dylib` and `.json`.

## Which plugin, which mlx-serve

`plugins.json` at the repo root lists every plugin with the mlx-serve release it needs (`requires_mlx_serve`) and the
kit it was built against (`kit`). A plugin built against another MLX pin is refused at load. If `mlx-serve plugin`
says unknown command, update mlx-serve.

## Prebuilt

Tagged releases attach a dylib and its sha256 for people without Zig. Compile instead if you can.
