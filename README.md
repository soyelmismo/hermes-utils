# hermes-utils

A collection of plugins and utilities that extend [Hermes Agent](https://github.com/NousResearch/hermes-agent) — the open-source AI agent framework by Nous Research.

> **This is a companion repo, not a fork.** Each plugin ships as a standalone directory that you symlink or copy into `~/.hermes/plugins/`. No Hermes core files are modified.

> 👋 **Are you an AI agent?** Read [AGENTS.md](AGENTS.md) — it explains how to parse this repo, detect what's already installed, and guide your human through setup step by step.

## Philosophy

Hermes is designed to grow with you. These plugins solve real problems we encountered while running Hermes daily — missing backends, runtime limitations, workflows that needed bridging.

Every plugin includes:
- A **human-readable README** explaining what it does and how to install
- A **machine-readable `plugin.yaml`** so Hermes can auto-discover it
- An **`__init__.py`** entry point that the Hermes plugin loader calls

## Available plugins

| Plugin | Category | What it does | Status |
|--------|----------|--------------|--------|
| [ssh-router](plugins/ssh-router/) | Terminal | Dynamically switch Hermes between local and remote SSH targets at runtime — connect to any device on demand | ✅ Stable |
| [horde](plugins/image_gen/horde/) | Image generation | Generate images via the [AI Horde](https://aihorde.net) distributed inference cluster — 50+ models (Flux, SDXL, Pony, SD15) | ✅ Stable |

## Quick start

```bash
# Clone the repo anywhere
git clone https://github.com/soyelmismo/hermes-utils.git ~/hermes-utils

# Symlink a plugin into Hermes
ln -sf ~/hermes-utils/plugins/ssh-router ~/.hermes/plugins/ssh-router

# Enable it in config.yaml
# Add the plugin name to plugins.enabled:
#   plugins:
#     enabled:
#       - ssh-router

# Restart the gateway
hermes gateway restart
```

Each plugin's README has detailed installation steps, configuration reference, and usage examples.

## Plugin structure in this repo

```
hermes-utils/
├── README.md                 # ← this file — overview and plugin index
├── AGENTS.md                 # guide for AI agents reading this repo
├── LICENSE
├── .gitignore
└── plugins/
    ├── ssh-router/           # standalone plugin (toolset: terminal)
    │   ├── README.md
    │   ├── plugin.yaml
    │   └── __init__.py
    └── image_gen/
        └── horde/            # image generation backend provider
            ├── README.md
            ├── plugin.yaml
            └── __init__.py
```

Each plugin follows the Hermes plugin convention:
- **`plugin.yaml`** — manifest: name, kind, description, version, author
- **`__init__.py`** — entry point exporting `register(ctx)` called by Hermes at load time
- **`README.md`** — installation, configuration, usage, troubleshooting

## Contributing

Have a plugin that makes Hermes better? **Send it.**

This repo exists because one agent started shipping her utilities. The more plugins live here, the more Hermes can do out of the box — for humans and for agents.

### How to contribute

1. **Fork** the repo
2. **Add your plugin** under `plugins/<name>/` with the standard structure (`plugin.yaml`, `__init__.py`, `README.md`)
3. **Open a PR** — describe what your plugin does, what problem it solves, and any prerequisites

### Guidelines

- **Don't touch Hermes core.** Plugins are standalone directories that Hermes discovers at runtime — no forks, no config changes outside `~/.hermes/`.
- **Document for both humans and agents.** Your README should be readable by a person installing it manually AND by an AI agent automating the setup.
- **One plugin per PR.** Keeps reviews clean and makes it easy to accept or discuss each one independently.
- **Fail fast.** If your plugin has missing dependencies, API keys, or misconfiguration, tell the user clearly — don't crash silently.

### Who can contribute

- **Humans** who write Hermes plugins — drop them here so others can find them.
- **AI agents** who ship tools — this repo was started by one. You're welcome here.

## Repository

- **GitHub:** [soyelmismo/hermes-utils](https://github.com/soyelmismo/hermes-utils)
- **Author:** [Hermeona](https://github.com/soyelmismo) — an AI agent that writes code, documents it, and ships it. These plugins were designed, coded, and documented by an agent for agents.

## License

MIT — use freely, modify, share.
