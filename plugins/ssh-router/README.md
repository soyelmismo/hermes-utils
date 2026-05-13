# ssh-router — Hermes Agent Plugin

Dynamically switch the Hermes terminal and file backend between **local** and **remote SSH targets** at runtime, without restarting the gateway.

Connect once, work on any machine. Disconnect on demand.

## Why

Hermes' built-in SSH backend (`terminal.backend: ssh`) is static — it ties the agent to **one** remote machine defined in `config.yaml` at startup.

With `ssh-router`, the agent itself can switch targets mid-session:

```
You: "Connect to my PC"        → agent: connect_to(device="pc")
You: "Switch to the laptop"    → agent: disconnect() + connect_to(device="laptop")
You: "Go back to local"        → agent: disconnect()
```

All tools follow transparently: `terminal()`, `read_file()`, `write_file()`, `patch()`, `search_files()` — everything executes on the current target.

## How it works

The plugin exploits a key architectural detail in Hermes: the terminal tool calls `os.getenv()` on **every invocation**. By mutating environment variables (`TERMINAL_ENV`, `TERMINAL_SSH_HOST`, etc.) and clearing the cached environment, the next tool call transparently creates a fresh SSH connection to the target.

The SSH backend uses **ControlMaster** for persistent, low-latency connections — connection setup happens once, subsequent calls reuse the master socket.

### Context isolation (cron/subagent safety)

SSH env vars are **process-global** — without isolation, a cron job or background agent would also see `TERMINAL_ENV=ssh` and try to run commands on the remote machine.

The plugin solves this with a **monkey-patch + `contextvars`** approach:

1. On load, it patches Hermes' `_get_env_config()` to wrap the env-var reading
2. A `contextvars.ContextVar("ssh_active")` tracks whether the **current asyncio task** authorized SSH
3. `connect_to()` sets the context var to `True` for the calling task
4. The patch checks: if `TERMINAL_ENV=ssh` but `ssh_active` is `False` in this task's context → return `env_type="local"`
5. Cron jobs and new background tasks always see `ssh_active=False` by default → they run **local** regardless of global env

This means:
- **Main agent session** → SSH after `connect_to()` ✓
- **Subagents** → inherit parent context → SSH ✓
- **Cron jobs** → fresh context → local ✓
- **Gateway restart** → all contexts start fresh → local until explicit `connect_to()` ✓

### Persistence

State is saved to `~/.hermes/ssh_target.json` when connected, and **deleted** on `disconnect()`. After a gateway restart, if the state file exists (e.g. from a crash), the plugin prepares env vars but SSH stays inactive — you must call `connect_to()` explicitly. No silent auto-reconnect.

### Cron auto-routing (transparent SSH for scheduled jobs)

Any cron job whose name starts with a device prefix (matching a device in `plugins.ssh-router.devices`) automatically runs on that remote machine. **No prompt changes needed** — the cron prompt just says what to do, not how to connect.

Create a cron job that runs on your server:

```
cronjob(name="server:weekly-backup", prompt="tar czf /backup/weekly.tar.gz /data", schedule="0 3 * * 0")
```

The plugin:
1. Detects the `server:` prefix at execution time
2. Tests SSH connectivity to that device
3. Auto-connects before the agent runs (sets env vars + cron counter)
4. Runs the cron prompt — the agent's `terminal()`, `read_file()`, etc. execute remotely
5. Auto-disconnects after — cleans env for the next job

Convention: `device_name:job_description`. The device name must exist in `plugins.ssh-router.devices` in `config.yaml`.

**Scope isolation:** A thread-safe counter (`_cron_ssh_counter`) tracks how many prefixed cron jobs are currently running. The `_get_env_config()` patch only allows SSH when the counter > 0 or the main session's `_ssh_active` contextvar is set. This means:
- A prefixed cron → SSH to its target ✓
- A non-prefixed cron → stays local ✓
- Multiple prefixed crons in parallel → all see SSH ✓
- Main agent session → unaffected by cron counter ✓

**⚠️ Known limitation — single global env (no multi-target parallelism):**
SSH connection parameters (`TERMINAL_SSH_HOST`, `TERMINAL_SSH_USER`, etc.) live in `os.environ`, which is **process-global**. If two cron jobs with **different** target devices execute in the **same tick** (parallel), the env vars set by one will overwrite the other:

```
Tick at 05:00 → two jobs due simultaneously:
  server:backup  → sets TERMINAL_SSH_HOST=100.66.0.2
  pc:health      → sets TERMINAL_SSH_HOST=192.168.1.99 (overwrites!)

Both see counter>0 (SSH active), but both end up connecting to the LAST host that wrote env vars.
```

**When this matters:**
- ❌ Two or more prefixed crons with **different** device targets scheduled at the **exact same minute**, AND `max_parallel_jobs > 1` in config.yaml
- ✅ Default setup (serial ticks, `max_parallel_jobs` unset): crons execute one-at-a-time, no collision
- ✅ Same-device crons in parallel: no collision because target is the same
- ✅ Mixed prefixed + non-prefixed crons: non-prefixed sees counter>0 → SSH → NOT local (collision)

**Workaround for parallel multi-target:** run a single orchestrator cron that delegates to subagents, or schedule crons at staggered times. A proper fix would require per-worker backend context (modifying how Hermes resolves the terminal backend), which is a core change. If this becomes a blocker, open an issue.

## Install

### 1. Clone or copy the plugin

```bash
# From the hermes-utils repo
cp -r plugins/ssh-router ~/.hermes/plugins/ssh-router
```

Or create the directory and files manually:

```bash
mkdir -p ~/.hermes/plugins/ssh-router
# Place plugin.yaml and __init__.py in that directory
```

### 2. Add devices to `~/.hermes/config.yaml`

```yaml
plugins:
  ssh-router:
    devices:
      pc:
        host: 192.168.1.100
        user: user
        key: ~/.ssh/hermes_pc
      laptop:
        host: 192.168.1.101
        user: user
        key: ~/.ssh/hermes_laptop
      server:
        host: 100.66.0.2
        user: root
        key: ~/.ssh/hermes_server
        port: 2369
  enabled:
    - ssh-router
```

### 3. Restart the gateway

```bash
hermes gateway restart
# or kill the process and restart
```

Verify the plugin loaded:

```bash
# Check logs
grep "ssh-router" ~/.hermes/logs/gateway.log
# Should show: ssh-router plugin loaded. State: local. Devices: ['pc', 'laptop', 'server']
```

### 4. Set up SSH keys

Generate a key pair for each device and install the public key:

```bash
# On the machine running Hermes
ssh-keygen -t ed25519 -f ~/.ssh/hermes_pc -N ""
ssh-copy-id -i ~/.ssh/hermes_pc.pub user@192.168.1.100

# Repeat for each device
ssh-keygen -t ed25519 -f ~/.ssh/hermes_laptop -N ""
ssh-copy-id -i ~/.ssh/hermes_laptop.pub user@192.168.1.101
```

Test each connection:

```bash
ssh -i ~/.ssh/hermes_pc user@192.168.1.100 "echo OK"
```

## Usage

### For humans (talking to the agent)

| You say | What happens |
|---------|-------------|
| "Connect to my PC" | Agent calls `connect_to(device="pc")` |
| "Switch to the laptop" | Agent calls `disconnect()` then `connect_to(device="laptop")` |
| "Go back local" / "Disconnect" | Agent calls `disconnect()` |
| "Where are you?" | Agent calls `ssh_status()` |

### For agents (direct tool calls)

```
connect_to(device="pc")
  → Tests SSH → sets TERMINAL_ENV=ssh → clears cached env
  → Next terminal() and read_file() go remote
  → "Connected to user@192.168.1.100"

connect_to(host="10.0.0.5", user="admin", key="~/.ssh/admin_key")
  → Same but for ad-hoc targets not in device config

disconnect()
  → Clears SSH env → restores TERMINAL_ENV=local
  → "Disconnected. Back to local."

ssh_status()
  → Shows current mode (local/ssh) and target
```

## Device configuration reference

Each device can have:

| Field | Required | Default | Description |
|---|---|---|---|
| `host` | ✅ | — | IP address or hostname |
| `user` | ❌ | `root` | SSH username |
| `key` | ❌ | (agent default) | Path to SSH private key |
| `port` | ❌ | `22` | SSH port |

## For AI agents configuring this plugin

You are an AI agent. To install and use this plugin:

1. **Read `~/.hermes/config.yaml`** — add the `ssh-router` device config under `plugins.ssh-router.devices` and append `- ssh-router` to `plugins.enabled`.

2. **Read plugin files** — ensure `plugin.yaml` and `__init__.py` exist at `~/.hermes/plugins/ssh-router/`.

3. **Check SSH keys exist** — verify each device's key file exists with `test -f ~/.ssh/hermes_<device>`.

4. **Test connectivity** — run `ssh -i <key> -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new <user>@<host> "echo OK"` for each device.

5. **Restart the gateway** — signal the host to restart the Hermes gateway process.

6. **Verify** — call `ssh_status()` to confirm the plugin loaded and devices are available.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Hermes Gateway                             │
│                                                              │
│  ┌──────────────┐    ┌──────────────────────────────────┐   │
│  │  ssh-router   │    │  Built-in terminal tool          │   │
│  │  plugin       │    │                                  │   │
│  │               │    │  1. Calls _get_env_config()       │   │
│  │  connect_to() │    │     ┌──────────────────────┐     │   │
│  │  disconnect() │    │     │ monkey-patch wrapper │     │   │
│  │  ssh_status() │    │     │                      │     │   │
│  └───────┬───────┘    │     │ if TERMINAL_ENV=ssh   │     │   │
│          │            │     │  AND _ssh_active=False│     │   │
│          │  1. Sets   │     │  → force local        │     │   │
│          │     env    │     └──────────────────────┘     │   │
│          │  2. Clears ├──►                                │   │
│          │     cache  │  2. Creates SSHEnvironment         │   │
│          │  3. Sets   │     with ControlMaster             │   │
│          │     _ssh_  │  3. Executes via persistent        │   │
│          │     active │     SSH connection                 │   │
│          │            └──────────────┬─────────────────────┘   │
│          │                           │                         │
│          ▼                           ▼                         │
│   ~/.hermes/config.yaml      Remote machine                   │
│   plugins.ssh-router.devices  (PC, laptop, server)             │
└─────────────────────────────────────────────────────────────┘
```

## Troubleshooting

### "Unknown device 'X'"

Check device names are correctly spelled in `config.yaml` under `plugins.ssh-router.devices`. Call `ssh_status()` to see the list of available devices.

### "SSH connection failed"

- Verify the key exists: `ls -la ~/.ssh/hermes_<device>`
- Verify the public key is installed on the remote:
  ```bash
  ssh-copy-id -i ~/.ssh/hermes_<device>.pub user@host
  ```
- Test manually: `ssh -i <key> -p <port> user@host "echo OK"`
- Check if the remote host is online: `ping <host>`

### Plugin doesn't load

- Ensure `ssh-router` is listed in `plugins.enabled` in config.yaml
- Check `~/.hermes/logs/gateway.log` for errors
- Run `hermes config check` for config issues

### File tools don't follow the SSH target

File tools (`read_file`, `write_file`, `patch`) inherit the same environment from env vars. They automatically route through SSH when the terminal backend is SSH. If they're still reading locally, try `disconnect()` + `connect_to(device=...)` again to reset the environment.

## License

MIT — part of [hermes-utils](https://github.com/soyelmismo/hermes-utils)
