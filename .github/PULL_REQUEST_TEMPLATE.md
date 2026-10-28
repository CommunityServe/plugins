## What

One or two sentences: the plugin or change, and the packs it serves.

## Checked

- [ ] `scripts/build.sh --all` passes (every plugin compiles, not only this one)
- [ ] `scripts/lint.sh` and `scripts/gen-index.sh` run, `plugins.json` committed
- [ ] `scripts/check.sh <name> --model <pack>` passes (or `"fixture": "none"` with a reason in plugin.json)
- [ ] Plugin on vs `--no-plugins` greedy output diffed on the real pack
- [ ] Narrow claim, MLX-free parse, named refusals
- [ ] Third-party code is unchanged and listed in NOTICE

## Numbers

Hardware, pack, decode and prefill tok/s. Say what you did not measure.
