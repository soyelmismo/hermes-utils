# antigravity-rotator

An account rotation and quota telemetry plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent) and Google Antigravity, supporting both OpenProxy and local backend operation.

> **Zero-Contamination Architecture**: Keeps the `antigravity-subscription-directsdk` provider completely pure and untouched. All failover, quota checks, and credential swapping happen out-of-band via Hermes lifecycle hooks.

## Features

- 🔄 **Automatic Account Rotation**: When a Google Antigravity account hits its 5-hour rolling session limit or weekly limit (`RESOURCE_EXHAUSTED` / `429` / capacity error), Hermes's `transform_api_error_classification` hook queries the active backend, selects the best available account for the model pool, injects credentials into the local `agy` CLI environment, and retries the turn immediately.
- 📊 **Quota Telemetry**: Tracks real-time 5-hour rolling session quotas, weekly quotas, and exact countdowns until quota reset across all registered accounts.
- 🔀 **Hermes Agent Tools**: Exposes `antigravity_list_accounts`, `antigravity_switch_account`, and `antigravity_refresh_quotas` to the agent.
- 💻 **CLI Management**: Includes `hermes antigravity [list|switch|refresh|rotate|login|remove]` subcommands for manual control.

## Installation

```bash
# 1. Symlink the plugin into Hermes
ln -sf /root/code/leagent/hermes-utils/plugins/antigravity-rotator ~/.hermes/plugins/antigravity-rotator

# 2. Enable in ~/.hermes/config.yaml under plugins.enabled:
plugins:
  enabled:
    - antigravity-subscription-directsdk
    - antigravity-rotator
```

## Configuration

In `~/.hermes/config.yaml`:

```yaml
plugins:
  antigravity-rotator:
    backend: "auto"              # "auto", "openproxy", or "local"
    openproxy_url: "http://localhost:8787/admin/api"
    openproxy_token: "op_live_YOUR_TOKEN_HERE"   # or set env OPENPROXY_ADMIN_TOKEN
    auto_rotate: true
    quota_threshold_percent: 95.0
    cooldown_minutes: 15
```

### Backend Resolution (`auto` mode)

In `auto` mode, the plugin checks if OpenProxy credentials can be resolved. If a token is found, the OpenProxy backend is selected; otherwise, the plugin falls back to the local backend.

Credential resolution follows this strict precedence order:
1. Configuration in `~/.hermes/config.yaml` (`openproxy_url`, `openproxy_token`).
2. `OPENPROXY_*` environment variables (`OPENPROXY_ADMIN_URL`, `OPENPROXY_ADMIN_TOKEN`).
3. `SWITCH_ACCOUNT_*` environment variables (`SWITCH_ACCOUNT_API_URL`, `SWITCH_ACCOUNT_TOKEN`).
4. Switch script files (searched in order until the first existing file is found):
   - `ANTIGRAVITY_ROTATOR_SWITCH_SCRIPT` (if set)
   - `~/.agents/switch_account.sh`
   - `~/switch_account.sh`
   - `/root/.agents/switch_account.sh`
   - `/root/switch_account.sh`

Tokens are never logged or exposed in output.

## Pool-Based Selection

Quota tracking distinguishes between two separate capacity pools:
- **gemini pool**: models without 'claude' or 'gpt' in their name.
- **claude_gpt pool**: models with 'claude' or 'gpt' in their name.

When a quota error occurs, the plugin checks the pool corresponding to the requested model. An account is usable for a pool only if both of its known quota windows (5-hour and weekly) have remaining capacity above the threshold and the account is not in cooldown. Accounts with unknown quotas are not excluded, but they are ranked after accounts with confirmed quota.

## Local Backend

The local backend manages multiple Antigravity accounts locally on the filesystem without an external OpenProxy server.

### Environment Variables

| Variable | Description | Default |
|---|---|---|
| `ANTIGRAVITY_ROTATOR_BACKEND` | Backend choice (`auto`, `openproxy`, `local`) | `auto` |
| `OPENPROXY_ADMIN_URL` | OpenProxy admin API URL | `http://localhost:8787/admin/api` |
| `OPENPROXY_ADMIN_TOKEN` | OpenProxy admin authentication token | None |
| `SWITCH_ACCOUNT_API_URL` | Alternative OpenProxy admin API URL | None |
| `SWITCH_ACCOUNT_TOKEN` | Alternative OpenProxy admin authentication token | None |
| `ANTIGRAVITY_ROTATOR_SWITCH_SCRIPT` | Explicit path to OpenProxy switch script | None |
| `ANTIGRAVITY_ACCOUNTS_FILE` | Path to JSON registry file | `~/.hermes/antigravity-accounts.json` |
| `ANTIGRAVITY_ACCOUNTS_DIR` | Directory holding per-account homes | `~/.agy-accounts` |

### Account Onboarding (Login)

Adding a local account requires an interactive terminal session:

```bash
hermes antigravity login --label work-account
```

This command runs `agy` interactively in a staging environment to complete the OAuth browser flow. Upon successful authentication:
- Credentials are verified in the staging directory.
- The active email is extracted from `google_accounts.json`.
- The staging directory is moved to `~/.agy-accounts/<label>`.
- The account is registered in `~/.hermes/antigravity-accounts.json`.
- Any failure or cancellation cleans up the staging directory immediately.

### File-Based Token Storage

The local backend forces `agy` to use file-based token storage rather than the system keyring by assigning the sentinel variable `SSH_CONNECTION="127.0.0.1 0 127.0.0.1 0"` in probe and login environments. This ensures accounts do not overwrite each other's credentials in a shared keyring.

### Save-Back Mechanism

When switching accounts via `apply_account`, if an active account is already loaded, its current real token file and `google_accounts.json` are copied back to its home directory before the target account's token is installed. This ensures any OAuth token refreshes written by `agy` during runtime are preserved.

The target token is written using atomic file replacement (`tempfile` in the same directory followed by `os.replace` with mode `0600`), preserving existing symlinks pointing to the token path.

## CLI Usage

```bash
# List all accounts and quota countdowns (accepts optional --model)
hermes antigravity list --model claude-3-5-sonnet

# Switch to a specific account by ID or label
hermes antigravity switch my-label
hermes antigravity switch 224

# Auto-switch to the account with most remaining quota for a model
hermes antigravity switch --model claude-3-5-sonnet

# Rotate to next account
hermes antigravity rotate --model gemini-2.0-flash

# Refresh live quotas
hermes antigravity refresh

# Onboard a new local account
hermes antigravity login --label account-name

# Remove an account from the local registry (preserves files)
hermes antigravity remove account-name
```

## Agent Tools

- `antigravity_list_accounts(model=...)`: Returns a markdown table of accounts with current usage and reset countdowns.
- `antigravity_switch_account(account_id=..., model=...)`: Switches the active account or auto-rotates.
- `antigravity_refresh_quotas(account_id=...)`: Refreshes quota usage telemetry.
