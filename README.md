<h1 align="center">MLX-Serve Plugins</h1>
<p align="center"><b>Teach <a href="https://github.com/ddalcu/mlx-serve">mlx-serve</a> and sushi a new model, a new format, a new backend. No server rebuild.</b></p>

<p align="center">
  <a href="../../actions/workflows/ci.yml"><img alt="every plugin compiles" src="https://github.com/CommunityServe/plugins/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="macOS arm64" src="https://img.shields.io/badge/macOS-arm64-black">
  <img alt="Zig 0.17" src="https://img.shields.io/badge/Zig-0.17-f7a41d">
  <img alt="license MIT" src="https://img.shields.io/badge/license-MIT-blue">
</p>

A plugin is a small dylib. mlx-serve keeps HTTP, chat templates, tool calls, sampling, scheduling and the KV cache; the
plugin brings what is specific to one model or format. Plugins here are **compiled from source on your machine**, so you
can read what you run.

## 60 seconds

```bash
git clone https://github.com/CommunityServe/plugins && cd plugins
scripts/install.sh sushi_ds41          # pinned Zig + kit, tests, build, mlx-serve plugin install
```

Turn plugins on (`~/.mlx-serve/plugins.json` with `"enabled": true`, or Settings > Server in the app), restart, and the
log says `[plugin] sushi_ds41 serves ...`. Details: [docs/using-plugins.md](docs/using-plugins.md).

## Plugins

| Plugin | Kind | Serves | Needs mlx-serve |
|---|---|---|---|
| [`sushi_ds41`](plugins/sushi_ds41) | model | DeepSeek-V4.1-Flash STQ packs on sushi's model code: DSpark (greedy and sampled), prefix cache, images | see [`plugins.json`](plugins.json) |

`plugins.json` is the machine-readable index (generated from each plugin's `plugin.json`).

## How it fits

```
  request ──► mlx-serve ──► claims a pack? ──► your plugin ──► logits ──► sampling, tools, streaming
              (HTTP, templates,   no: built-in       (config, weights,
               scheduler, KV)                         memory bill, forward)
```

Pick the shallowest seam that works:

| Kind | You write | Start |
|---|---|---|
| **model** | config parse, weight load, memory bill, forward | `scripts/new-plugin.sh my_model` |
| **media** | an image, audio, video or 3D generator | `scripts/new-plugin.sh my_gen --media` |
| **quant** | a weight encoding and its matmul | `scripts/new-plugin.sh my_fmt --quant` |

## Build one (people and agents)

```bash
scripts/setup.sh                              # pinned Zig and kit, inside the checkout
scripts/new-plugin.sh my_plugin               # scaffold from the kit's example
scripts/build.sh my_plugin                    # tests, then ReleaseFast
scripts/check.sh my_plugin --model <pack>     # conformance, and does it claim and parse your pack
scripts/build.sh --all && scripts/lint.sh     # the gate: every plugin compiles
```

Using an AI agent? Hand it [`AGENTS.md`](AGENTS.md). It holds the rules and both flows (create a plugin and send a PR,
build a plugin and add it to mlx-serve), and [docs/agent-guide.md](docs/agent-guide.md) has copy-paste prompts.
`.claude/skills/` carries the same flows as skills.

## The rules, short

1. **Every plugin compiles, always.** CI builds all of them on every PR, and weekly.
2. **Compiled from source.** No committed binaries, no prebuilt installs by default.
3. **Pinned toolchain.** `scripts/setup.sh` stages Zig and the kit; nothing is used from the system.
4. **Narrow claims, named refusals.** A plugin serves only what it can, and says why when it cannot.
5. **Measured numbers.** Hardware stated, decode and prefill tok/s first.

Full list in [CONTRIBUTING.md](CONTRIBUTING.md).

## Layout

```
plugins/<name>/   one standalone package per plugin (build.zig, config.zig, plugin.zig, plugin.json, README)
scripts/          setup, build, check, install, new-plugin, lint, gen-index
docs/             writing-a-plugin, using-plugins, agent-guide, releasing
KIT_VERSION       the mlx-serve kit release every plugin builds against
plugins.json      generated index
```

MIT. Each plugin lists third-party code it vendors in its own `NOTICE`.
