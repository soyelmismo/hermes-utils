"""
ssh-router plugin for Hermes Agent.

Dynamic SSH routing — switch the terminal/file backend between local and
remote targets at runtime without restarting the gateway.

Usage:
    ssh_router_connect_to(device="pc")                          # named device from config
    ssh_router_connect_to(host="192.168.1.100", user="root")    # explicit params
    ssh_router_disconnect()
    ssh_router_status()

Device definitions go in ~/.hermes/config.yaml:
    plugins:
      ssh-router:
        devices:
          pc:
            host: 192.168.1.100
            user: user
            key: ~/.ssh/hermes_pc

Architecture:
    Hermes' terminal/file tools read their config from environment variables
    (TERMINAL_ENV, TERMINAL_SSH_HOST, etc.) on every tool call.

    This plugin works by:
    1. Mutating os.environ with the new target's SSH config
    2. Calling cleanup_vm() to destroy the cached environment
    3. Monkey-patching tools.terminal_tool._get_env_config() with a
       context-aware wrapper (see _ensure_patch) that overrides env_type
       back to "local" when the current asyncio task is NOT SSH-authorized

    Context isolation (via thread-safe counter):
        Each cron job with a device prefix increments a module-level
        counter on entry and decrements on exit.  The _get_env_config
        patch checks: if TERMINAL_ENV=ssh but NEITHER the main session
        (_ssh_active contextvar) NOR a prefixed cron (_cron_ssh_counter > 0)
        requested it → return env_type="local".

        This means:
        - Main agent session → SSH after ssh_router_connect_to()
        - Cron jobs with prefix → SSH to their target
        - Cron jobs without prefix → local
        - Subagents → inherit parent context → same as parent
        - Gateway restart → fresh state → local until explicit ssh_router_connect_to()

    IMPORTANT LIMITATION: os.environ is process-global.  If two cron jobs
    with DIFFERENT device targets execute in parallel (same tick), the
    env vars set by one overwrite the other.  Both see counter>0 (SSH
    allowed) but both connect to the LAST host that wrote env vars.
    Default scheduling (serial ticks, max_parallel_jobs unset) avoids
    this.  See README for full discussion.

    Cron auto-routing via job name:
        Name a cron job with a device prefix to run it on that remote:
            "server:weekly-backup"  → auto-connects to device 'server'
            "pc:healthcheck"        → auto-connects to device 'pc'
        The prefix must match a device name in plugins.ssh-router.devices.
        Connection and disconnection are transparent — the cron prompt
        never knows about SSH. No extra config, no mapping files.

    Persistence:
        State is saved to ~/.hermes/ssh_target.json during connection,
        and deleted on ssh_router_disconnect(). After a gateway restart, the
        environment is prepared but SSH stays inactive until you
        explicitly call ssh_router_connect_to().
"""

import logging

from .config import _read_state, _load_devices
from .connection import _restore_state
from .patching import _ensure_patch, _patch_cron_scheduler
from .liveness import _ssh_status_hook, _ssh_connection_watchdog
from .handlers import handle_connect_to, handle_disconnect, handle_status

logger = logging.getLogger("ssh-router")

def register(ctx) -> None:
    """Register the ssh-router tools with Hermes."""

    # Apply the monkey-patch for context-isolated SSH routing
    _ensure_patch()

    # Hook cron scheduler for auto-SSH via job name prefix
    _patch_cron_scheduler()

    # Restore previous SSH state if we were connected before restart
    _restore_state()

    # Register pre_llm_call hook to inject SSH status into every turn's context
    ctx.register_hook("pre_llm_call", _ssh_status_hook)

    # Register post_tool_call watchdog to alert on SSH connection failures
    ctx.register_hook("post_tool_call", _ssh_connection_watchdog)

    # ── ssh_router_connect_to ──────────────────────────────────────────────────────
    ctx.register_tool(
        name="ssh_router_connect_to",
        toolset="terminal",
        schema={
            "type": "function",
            "function": {
                "name": "ssh_router_connect_to",
                "description": (
                    "Connect terminal and file backends to a remote machine via SSH. "
                    "After connecting, ALL terminal commands and file operations "
                    "(read_file, write_file, patch, search_files) execute on the "
                    "remote host. Use ssh_router_disconnect() to return to local.\n\n"
                    "Two modes:\n"
                    "  1. Named device: ssh_router_connect_to(device='pc') — looks up host/user/key\n"
                    "     from plugins.ssh-router.devices in config.yaml\n"
                    "  2. Explicit: ssh_router_connect_to(host='...', user='root', key='~/.ssh/key')\n\n"
                    "Available devices are shown in ssh_router_status."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "device": {
                            "type": "string",
                            "description": "Name of a pre-configured device from config.yaml (e.g. 'pc', 'laptop')",
                        },
                        "host": {
                            "type": "string",
                            "description": "Remote host IP or hostname (required if device is not set)",
                        },
                        "user": {
                            "type": "string",
                            "description": "SSH username (default: root)",
                        },
                        "key": {
                            "type": "string",
                            "description": "Path to SSH private key (e.g. ~/.ssh/hermes_key)",
                        },
                        "port": {
                            "type": "integer",
                            "description": "SSH port (default: 22)",
                        },
                    },
                    "oneOf": [
                        {"required": ["device"]},
                        {"required": ["host"]},
                    ],
                },
            },
        },
        handler=handle_connect_to,
        emoji="🔗",
        description="Connect to a remote machine via SSH. All terminal and file operations run on the remote host.",
    )

    # ── ssh_router_disconnect ──────────────────────────────────────────────────────
    ctx.register_tool(
        name="ssh_router_disconnect",
        toolset="terminal",
        schema={
            "type": "function",
            "function": {
                "name": "ssh_router_disconnect",
                "description": (
                    "Disconnect from the current SSH target and return all "
                    "terminal and file operations to local execution."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
        handler=handle_disconnect,
        emoji="🔌",
        description="Disconnect from remote SSH and return to local.",
    )

    # ── ssh_router_status ──────────────────────────────────────────────────────
    ctx.register_tool(
        name="ssh_router_status",
        toolset="terminal",
        schema={
            "type": "function",
            "function": {
                "name": "ssh_router_status",
                "description": "Show the current SSH routing status and list available devices.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
        handler=handle_status,
        emoji="📡",
        description="Show current SSH routing status (local or connected to which host).",
    )

    logger.info(
        "ssh-router plugin loaded. State: %s. Devices: %s",
        _read_state().get("mode", "local"),
        list(_load_devices().keys()),
    )
