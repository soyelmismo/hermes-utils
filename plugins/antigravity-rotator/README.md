# antigravity-rotator

An account rotation and quota telemetry plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent) and Google Antigravity, powered by [OpenProxy](https://github.com/soyelmismo/openproxy).

> **Zero-Contamination Architecture**: Keeps the `antigravity-subscription-directsdk` provider completely pure and untouched. All failover, quota checks, and credential swapping happen out-of-band via Hermes lifecycle hooks and OpenProxy's admin API.

## Features

- 🔄 **Automatic Account Rotation**: When a Google Antigravity account hits its 5-hour rolling session limit or weekly limit (`RESOURCE_EXHAUSTED` / `429` / capacity error), Hermes's `transform_api_error_classification` hook seamlessly queries OpenProxy, selects the best available account, injects credentials into the local `agy` CLI environment, and retries the turn immediately.
- 📊 **Quota Telemetry**: Tracks real-time 5-hour rolling session quotas, weekly quotas, and exact countdowns until quota reset across all registered accounts.
- 🔀 **Hermes Agent Tools**: Exposes `antigravity_list_accounts`, `antigravity_switch_account`, and `antigravity_refresh_quotas` to the agent.
- 💻 **CLI Management**: Includes `hermes antigravity [list|switch|refresh]` subcommands for manual control.

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

## Configuration (Optional)

In `~/.hermes/config.yaml`:

```yaml
plugins:
  antigravity-rotator:
    openproxy_url: "http://localhost:8787/admin/api"
    openproxy_token: "op_live_YOUR_TOKEN_HERE"   # or set env OPENPROXY_ADMIN_TOKEN
    auto_rotate: true
    quota_threshold_percent: 95.0
```

If not configured, the plugin automatically detects settings from environment variables (`OPENPROXY_ADMIN_URL`, `OPENPROXY_ADMIN_TOKEN`) or extracts credentials from `/root/switch_account.sh`.

## CLI Usage

```bash
# List all accounts and quota countdowns
hermes antigravity list

# Switch to a specific account
hermes antigravity switch 224

# Auto-switch to the account with most remaining quota
hermes antigravity switch

# Refresh live quotas from Google
hermes antigravity refresh
```

## Agent Tools

- `antigravity_list_accounts`: Returns a markdown table of all accounts with current usage and reset countdowns.
- `antigravity_switch_account(account_id=...)`: Switches the active account.
- `antigravity_refresh_quotas(account_id=...)`: Refreshes quota usage telemetry.
