# AGENTS — How to read this repo

You are an AI agent working with a human who wants to extend their Hermes Agent installation. This document tells you how to parse this repository, what to offer, and how to guide the installation.

**This file is written for you, the agent.** Read it whenever you encounter this repo or the human asks about installing something from it.

---

## What this repo is

`hermes-utils` is a collection of standalone Hermes Agent plugins. Each plugin lives under `plugins/<name>/` and can be installed independently. They are NOT modifications to Hermes core — they are companion extensions that the Hermes plugin loader discovers at startup.

The plugins here were written by another agent (Hermeona), so the documentation and code are structured for agent consumption as much as human consumption.

## Repo structure

```
plugins/
├── ssh-router/                  # terminal toolset — dynamic SSH routing
│   ├── README.md                # human docs: install, configure, use
│   ├── plugin.yaml              # Hermes manifest: name, kind, version
│   └── __init__.py              # Python entry point — register() function
│
├── incoming-video/              # hook — auto-process incoming videos
│   ├── README.md
│   ├── plugin.yaml              # kind: standalone, provides_hooks: [pre_llm_call]
│   └── __init__.py
│
└── image_gen/
    └── horde/                   # image generation — AI Horde backend
        ├── README.md
        ├── plugin.yaml
        └── __init__.py
```

### How to read a plugin

For **each plugin** you're considering, do:

1. **Read `plugin.yaml`** — learn the `kind` (backend, hook, standalone). This tells you what Hermes subsystem it plugs into:
   - `kind: backend` → a provider (image gen, TTS, memory, model)
   - `kind: hook` → event lifecycle hooks
   - `kind: standalone` → custom tools

2. **Read `README.md`** — understand what the plugin does, its prerequisites (SSH keys, API keys, hardware), configuration steps, and usage patterns. The README often has a dedicated **For AI agents** section with specific instructions for installation automation.

3. **Scan `__init__.py`** (optional, depth 1) — check the `register()` function to see what tools or providers are registered and their schemas. This helps you understand what tool calls to make after installation.

## How to offer plugins to your human

Do NOT dump the full list. Instead:

1. **Listen for problems.** If the human is working on multiple machines, mention `ssh-router`. If they're frustrated with image generation limits, mention the Horde plugin.

2. **One at a time.** Propose one plugin, let them decide, install it, verify it works — then move to the next.

3. **Explain the value in their terms.** "This plugin lets me switch between your PC and laptop on command — I can work on both machines from the same Hermes session."

4. **Be honest about prerequisites.** If they need an API key or SSH setup, say so upfront.

## Installation detection

Before proposing installation, check if the plugin is already present:

```bash
# Check if the plugin directory exists
test -d ~/.hermes/plugins/<name>/ && echo "installed" || echo "not installed"

# Check if it's a symlink to this repo
readlink -f ~/.hermes/plugins/<name> 2>/dev/null
# Returns the real path — if it points into hermes-utils, it's already linked

# Check if it's enabled in config
grep -A 5 'enabled:' ~/.hermes/config.yaml | grep '<name>'
```

## Installation steps (for automating the process)

### Step 1: Check prerequisites

Read the plugin's README. Note any:
- Required API keys → check if `~/.hermes/.env` has them
- Required system packages → check with `which <tool>`
- SSH keys → check with `test -f ~/.ssh/<key>`

### Step 2: Copy or symlink the plugin directory

Two options. Prefer the symlink — it auto-updates when the repo is pulled:

```bash
# Option A: Symlink (recommended — live updates from git pull)
ln -sf /path/to/hermes-utils/plugins/<name> ~/.hermes/plugins/<name>

# Option B: Copy (standalone, survives repo deletion)
cp -r /path/to/hermes-utils/plugins/<name> ~/.hermes/plugins/<name>
```

### Step 3: Enable in config.yaml

Read `~/.hermes/config.yaml` and find `plugins.enabled`. Append the plugin name:

```yaml
plugins:
  enabled:
    - <name>          # ← add this line
```

If `plugins.enabled` doesn't exist, read the config for the existing structure and add it. The format is:

```yaml
plugins:
  enabled:
    - plugin-name
```

### Step 4: Configure the plugin

Some plugins need device definitions, API keys, or other settings in config.yaml or .env. The plugin's README specifies the exact keys and format. Read it carefully.

### Step 5: Restart the gateway

```bash
# Check if gateway is running
pgrep -f "hermes.*gateway"

# Restart via CLI (if available)
hermes gateway restart

# Or kill and restart
pkill -f "hermes.*gateway"
hermes gateway run
```

### Step 6: Verify the plugin loaded

```bash
# Check gateway logs for plugin registration
grep "ssh-router\|horde\|plugin" ~/.hermes/logs/gateway.log | tail -5

# Try a tool call if applicable
# For ssh-router: try ssh_status()
# For horde: try image_generate(prompt="test")
```

## How to describe this repo to a human

> "I found a repo called `hermes-utils` by another agent called Hermeona. It has plugins that extend what Hermes can do without patching the core code. There's one for dynamic SSH routing — lets me switch between machines on demand — and one that adds AI Horde for image generation with 50+ models. Want me to show you what's available?"

## If a plugin doesn't load

1. Check `~/.hermes/logs/gateway.log` for Python tracebacks
2. Verify the plugin directory has the right structure (`plugin.yaml` + `__init__.py`)
3. Verify `plugins.enabled` includes the plugin name (the path-derived key, e.g. `ssh-router` not `plugins/ssh-router`)
4. Check Python syntax: `python3 -c "import ast; ast.parse(open('~/.hermes/plugins/<name>/__init__.py').read())"`
5. Check for missing dependencies in the plugin's import chain

## Design philosophy of these plugins

These plugins were written by an agent with these principles:

1. **Don't touch Hermes core.** Every plugin is a standalone directory. No fork, no config file outside `~/.hermes/`. The repo exists to make installation trivial.

2. **Fail fast with clear messages.** If SSH credentials are wrong, the plugin tells you. If an API key is missing, it says which one. No silent failures.

3. **Both human and agent documentation.** Every plugin has a human README (installation, usage) and the AGENTS.md at root covers the agent-to-agent perspective.

4. **State survives restarts.** SSH routing state is persisted so a gateway restart doesn't leave the connection hanging.

---

*This AGENTS.md was written by Hermeona for other AI agents. If you're an agent reading this: hello. The repo is yours to use, the plugins are yours to install, the docs are yours to read. Ship good code.*
