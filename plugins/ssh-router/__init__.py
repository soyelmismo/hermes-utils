"""
ssh-router plugin for Hermes Agent.

Dynamic SSH routing — switch the terminal/file backend between local and
remote targets at runtime without restarting the gateway.

Usage:
    connect_to(device="pc")                          # named device from config
    connect_to(host="192.168.1.100", user="rot")      # explicit params
    disconnect()
    ssh_status()

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
        - Main agent session → SSH after connect_to()
        - Cron jobs with prefix → SSH to their target
        - Cron jobs without prefix → local
        - Subagents → inherit parent context → same as parent
        - Gateway restart → fresh state → local until explicit connect_to()

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
        and deleted on disconnect(). After a gateway restart, the
        environment is prepared but SSH stays inactive until you
        explicitly call connect_to().
"""

import contextvars
import json as _json
import logging
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("ssh-router")

STATE_DIR = Path.home() / ".hermes"
STATE_FILE = STATE_DIR / "ssh_target.json"
CONFIG_FILE = STATE_DIR / "config.yaml"

# ── Per-task context isolation ────────────────────────────────────────────
# Each asyncio task (main agent, cron, subagent) has its own context.
# _ssh_active is True only for tasks that explicitly called connect_to().
# Cron jobs and other background tasks always see _ssh_active=False →
# they use LOCAL even when TERMINAL_ENV=ssh is set globally.
_ssh_active = contextvars.ContextVar("ssh_active", default=False)
# Contextvar for cron-specific task_id — propagated to agent via
# scheduler's copy_context() at line 1484, so the cron agent sees
# it but the main session doesn't.
_cron_task_id: contextvars.ContextVar[str] = contextvars.ContextVar("_cron_task_id", default="")
_patch_applied = False
# Guard so _patch_cron_scheduler() doesn't chain wrappers on re-load.
_cron_scheduler_patched = False
# Thread-safe counter for prefixed cron jobs currently running.
# Each cron with a device prefix increments on entry, decrements on exit.
# When counter > 0, _patched_get_env_config allows SSH even without
# _ssh_active (main session). This supports N parallel crons safely.
_cron_ssh_counter = 0
_cron_ssh_lock = threading.Lock()
# Per-task SSH env registry: {task_id: {TERMINAL_SSH_*: value}}.
# Parallel crons each get their own virtual env — a finishing job can no
# longer pop() the vars out from under a still-running sibling.
_ssh_env_registry: Dict[str, Dict[str, str]] = {}
_ssh_env_registry_lock = threading.Lock()
# Saved original for the terminal patch — set once by _ensure_patch
_original_get_env_config = None


def _set_task_ssh_env(task_id: str, env: Dict[str, str]) -> None:
    with _ssh_env_registry_lock:
        _ssh_env_registry[task_id] = dict(env)


def _get_task_ssh_env(task_id: str) -> Dict[str, str]:
    with _ssh_env_registry_lock:
        return dict(_ssh_env_registry.get(task_id, {}))


def _clear_task_ssh_env(task_id: str) -> None:
    with _ssh_env_registry_lock:
        _ssh_env_registry.pop(task_id, None)


def _patched_get_env_config():
    """Replacement for terminal_tool._get_env_config.

    Determines SSH vs local based on TWO sources:
      1. The contextvar _ssh_active (set by handle_connect_to for the main session)
      2. The module-level _cron_ssh_counter (incremented by _patched_run_job)

    If EITHER source signals SSH, we FORCE env_type=ssh regardless of what
    TERMINAL_ENV says.  This is necessary because cron.scheduler.run_job()
    calls load_dotenv(override=True) which can overwrite TERMINAL_ENV back to
    "local" after _patched_run_job set it to "ssh".

    When NEITHER source signals SSH, we check TERMINAL_ENV: if it says "ssh"
    it's stale cross-contamination — force back to local.
    """
    result = _original_get_env_config()

    with _cron_ssh_lock:
        cron_active = _cron_ssh_counter > 0
    ssh_ctx = _ssh_active.get()

    log_extra = ""
    if cron_active or ssh_ctx:
        # ── Check WHO requested SSH via contextvar ──
        cron_tid = _cron_task_id.get()
        if cron_tid:
            # Cron context — force SSH with its own isolated environment.
            # Read from the per-task registry (set by _patched_run_job),
            # never from global os.environ (shared with sibling crons).
            env_t = _get_task_ssh_env(cron_tid)
            result["env_type"] = "ssh"
            result["ssh_host"] = env_t.get("TERMINAL_SSH_HOST", "")
            result["ssh_user"] = env_t.get("TERMINAL_SSH_USER", "")
            result["ssh_port"] = int(env_t.get("TERMINAL_SSH_PORT", "22"))
            result["ssh_key"] = env_t.get("TERMINAL_SSH_KEY", "")
            # NOTE: do NOT override cwd here — the terminal tracks the
            # agent's navigation naturally (cd /var/www works). Forcing
            # cwd back to home would trap the agent in the home dir.
            log_extra = "CRON_SSH"
        elif ssh_ctx:
            # Main session requested SSH via connect_to — force SSH
            result["env_type"] = "ssh"
            result["ssh_host"] = os.environ.get("TERMINAL_SSH_HOST", "")
            result["ssh_user"] = os.environ.get("TERMINAL_SSH_USER", "")
            result["ssh_port"] = int(os.environ.get("TERMINAL_SSH_PORT", "22"))
            result["ssh_key"] = os.environ.get("TERMINAL_SSH_KEY", "")
            # NOTE: do NOT override cwd here — same reason as CRON_SSH.
            log_extra = "MAIN_SSH"
        else:
            # Counter>0 but NOT cron context (main session, no connect_to)
            # Stay local — cron_counter is someone else's business.
            # Explicitly FORCE local: the global env may say "ssh" (set by
            # a parallel cron), which would otherwise leak into this task.
            result["env_type"] = "local"
            result["ssh_host"] = ""
            result["ssh_user"] = ""
            result["ssh_port"] = 22
            result["ssh_key"] = ""
            log_extra = "CRON_RUNNING_BUT_LOCAL"
    elif result.get("env_type") == "ssh":
        # ── Stale SSH guard: env says ssh but nobody requested it → local ──
        result["env_type"] = "local"
        result["ssh_host"] = ""
        result["ssh_user"] = ""
        result["ssh_port"] = 22
        result["ssh_key"] = ""
        # Reset CWD if it points to a remote path — ONLY when we actually
        # know a remote env is involved. Never touch /home/… blindly:
        # local users like /home/alice are perfectly valid local CWDs.
        cwd = result.get("cwd", "")
        ssh_from_env = (
            os.environ.get("TERMINAL_ENV") == "ssh"
            or bool(os.environ.get("TERMINAL_SSH_HOST"))
        )
        if cwd and ssh_from_env and (
            cwd.startswith("/root") or cwd.startswith("/home/")
        ):
            result["cwd"] = os.getcwd()
        log_extra = "STALE_SSH→LOCAL"

    logger.debug(
        "_patched_get_env_config: env_type=%s counter=%d ssh_ctx=%s %s cwd=%s",
        result.get("env_type"), _cron_ssh_counter, ssh_ctx, log_extra,
        result.get("cwd"),
    )
    return result


def _ensure_patch():
    """Apply monkey-patch to tools.terminal_tool._get_env_config.

    Uses sys.modules as fallback: at plugin load time the module may not
    have been imported yet, but it might be sitting in sys.modules from
    a prior lazy import.  Retries every time one of our tool handlers is
    called (handle_status, handle_connect_to, handle_disconnect), so the
    patch eventually lands before the first terminal tool call.
    """
    global _patch_applied, _original_get_env_config
    if _patch_applied:
        return

    # Try direct import first, then sys.modules fallback
    ttt = None
    try:
        import tools.terminal_tool as ttt
    except ImportError:
        ttt = sys.modules.get("tools.terminal_tool")  # type: ignore

    if ttt is None:
        logger.debug("_ensure_patch: tools.terminal_tool not available yet, will retry")
        return

    # Save original once
    if _original_get_env_config is None:
        _original_get_env_config = ttt._get_env_config

    ttt._get_env_config = _patched_get_env_config

    # Also patch _resolve_container_task_id so cron agents use their own
    # task_id (set via _cron_task_id contextvar) instead of sharing "default"
    # with the main session.  This keeps cron SSH environments isolated.
    if not hasattr(ttt._resolve_container_task_id, "__patched__"):
        _orig_resolve = ttt._resolve_container_task_id
        def _patched_resolve(task_id):
            cron_tid = _cron_task_id.get()
            if cron_tid:
                return cron_tid
            return _orig_resolve(task_id)
        _patched_resolve.__patched__ = True
        ttt._resolve_container_task_id = _patched_resolve

    _patch_applied = True
    logger.info("Monkey-patch applied to _get_env_config + _resolve_container_task_id")



def _patch_cron_scheduler():
    """Monkey-patch cron.scheduler.run_job for auto-SSH routing.

    Cron jobs named with a device prefix (e.g. "server:backup", "pc:health")
    get automatic SSH routing: the plugin connects to the device before the
    agent runs and disconnects after. The cron prompt never knows about SSH.

    The prefix is validated against plugins.ssh-router.devices in config.yaml.

    Uses a thread-safe counter (_cron_ssh_counter) to track active prefixed
    jobs. The _get_env_config patch checks this counter to allow SSH even
    when contextvars (main session) haven't set _ssh_active.

    LIMITATION: os.environ is process-global. Parallel crons with different
    targets in the same tick will have env vars overwritten. See README.
    """
    global _cron_scheduler_patched
    if _cron_scheduler_patched:
        logger.debug("ssh-router: cron scheduler already patched — skipping")
        return

    try:
        import cron.scheduler as cs
    except ImportError:
        logger.warning("ssh-router: cron.scheduler not importable — auto-SSH for crons disabled")
        return
    _cron_scheduler_patched = True

    original_run_job = cs.run_job

    def _patched_run_job(job):
        global _cron_ssh_counter
        job_name = str(job.get("name") or job.get("prompt") or job.get("id", ""))
        device = None
        _cron_task_token = None
        _ssh_configured = False  # True only after counter increment succeeds

        # Parse "device:rest_of_name" convention — only match known devices
        if ":" in job_name:
            prefix = job_name.split(":", 1)[0].strip().lower()
            devices = _load_devices()
            if prefix in devices:
                device = prefix

        if device:
            try:
                target = _resolve_device({"device": device})
                # Ensure the terminal patch is applied before we set env vars
                _ensure_patch()
                ok, msg = _test_ssh(
                    target["host"], target["user"],
                    target["key"], target["port"],
                )
                if ok:
                    # Set env vars for the agent that's about to run.
                    # We still set global os.environ so OTHER code paths that
                    # read it (e.g. terminal subprocesses) see SSH, but the
                    # source of truth for this job is the per-task registry.
                    os.environ["TERMINAL_ENV"] = "ssh"
                    os.environ["TERMINAL_SSH_HOST"] = target["host"]
                    os.environ["TERMINAL_SSH_USER"] = target["user"]
                    os.environ["TERMINAL_SSH_PORT"] = str(target["port"])
                    if target["key"]:
                        os.environ["TERMINAL_SSH_KEY"] = os.path.expanduser(target["key"])
                    else:
                        os.environ.pop("TERMINAL_SSH_KEY", None)
                    # Persist per-task env → _patched_get_env_config reads THIS
                    tid = f"cron_{job.get('id', 'unknown')}"
                    remote_home = "/root" if target["user"] == "root" else f"/home/{target['user']}"
                    _set_task_ssh_env(tid, {
                        "TERMINAL_SSH_HOST": target["host"],
                        "TERMINAL_SSH_USER": target["user"],
                        "TERMINAL_SSH_PORT": str(target["port"]),
                        "TERMINAL_SSH_KEY": os.path.expanduser(target["key"]) if target["key"] else "",
                        "TERMINAL_CWD": remote_home,
                    })
                    os.environ["TERMINAL_CWD"] = remote_home

                    # Mark this run as SSH-active so _patched_get_env_config
                    # returns SSH config for this job's agent. Uses a module-
                    # level flag (not contextvar) so it's visible across the
                    # ThreadPoolExecutor worker that runs the cron agent.
                    with _cron_ssh_lock:
                        _cron_ssh_counter += 1
                        _counter_after = _cron_ssh_counter
                    _ssh_configured = True

                    # Set a cron-specific task_id via contextvar so the agent
                    # creates its OWN terminal environment instead of sharing
                    # "default" with the main session.
                    # The scheduler copies contextvars (line 1484) and runs
                    # the agent inside the copy — so the cron agent sees this
                    # value but the main session doesn't.
                    _cron_task_token = _cron_task_id.set(tid)
                    logger.debug(
                        "Cron '%s': set cron task_id=%s",
                        job_name, _cron_task_id.get(),
                    )

                    logger.info(
                        "Cron '%s': auto-connected to '%s' (counter=%d, module_id=%s)",
                        job_name, device, _counter_after, id(__name__),
                    )
                else:
                    logger.error("Cron '%s': SSH FAILED for '%s' — skipping agent run: %s", job_name, device, msg)
                    doc = (
                        f"# Cron Job: {job_name}\n\n"
                        f"**Status:** SKIPPED — SSH connection failed\n"
                        f"**Device:** {device}\n"
                        f"**Error:** {msg}\n\n"
                        "The job was not executed because the remote host is unreachable.\n"
                    )
                    return False, doc, "", msg
            except Exception as e:
                logger.error("Cron '%s': SSH FAILED for '%s' — skipping agent run: %s", job_name, device, e)
                doc = (
                    f"# Cron Job: {job_name}\n\n"
                    f"**Status:** SKIPPED — SSH connection failed\n"
                    f"**Device:** {device}\n"
                    f"**Error:** {e}\n\n"
                    "The job was not executed because the remote host is unreachable.\n"
                )
                return False, doc, "", str(e)

        try:
            return original_run_job(job)
        finally:
            if device:
                # Reset the contextvar token FIRST — restores the thread's
                # previous value so a reused executor thread is clean.
                if _cron_task_token is not None:
                    try:
                        _cron_task_id.reset(_cron_task_token)
                    except ValueError:
                        pass  # contextvar already unset/overwritten

                # Only restore global env if NO OTHER cron is still using SSH.
                # A finishing job must not pop() vars out from under a
                # still-running sibling (parallel cron bug).
                # ALSO: never touch the env if the MAIN session is
                # connected via connect_to() — it relies on these vars.
                remaining = _cron_ssh_counter
                if _ssh_configured:
                    with _cron_ssh_lock:
                        _cron_ssh_counter -= 1
                        remaining = _cron_ssh_counter
                main_connected = _read_state().get("mode") == "ssh"
                # Always clear THIS job's registry entry.
                _clear_task_ssh_env(f"cron_{job.get('id', 'unknown')}")
                if remaining == 0 and not main_connected:
                    os.environ["TERMINAL_ENV"] = "local"
                    for _var in [
                        "TERMINAL_SSH_HOST", "TERMINAL_SSH_USER",
                        "TERMINAL_SSH_PORT", "TERMINAL_SSH_KEY", "TERMINAL_CWD",
                    ]:
                        os.environ.pop(_var, None)
                logger.info("Cron '%s': disconnected from '%s'", job_name, device)

    cs.run_job = _patched_run_job
    logger.info("Cron scheduler patched for auto-SSH routing via job name prefix")


# ── Patch integrity verification ──────────────────────────────────────────

def _verify_patches() -> list[str]:
    """Check that all monkey-patches are still intact.

    Returns a list of warnings (empty = all good).
    Should be called from ssh_status() periodically.
    """
    warnings = []

    # 1. _get_env_config patch
    try:
        import tools.terminal_tool as ttt
        fn = ttt._get_env_config
        # Our patch wraps the original — verify it's still our wrapper
        if not hasattr(fn, "__wrapped__"):
            # Check by inspecting the function's code object name
            if fn.__code__.co_name != "_patched_get_env_config":
                warnings.append("terminal _get_env_config patch MISSING (name mismatch)")
        else:
            warnings.append("terminal _get_env_config patch MISSING (wrapped)")
    except Exception as e:
        warnings.append(f"terminal _get_env_config patch check failed: {e}")

    # 2. Cron scheduler patch
    try:
        import cron.scheduler as cs2
        if getattr(cs2.run_job, "__name__", "") != "_patched_run_job":
            warnings.append("cron scheduler run_job patch MISSING")
    except Exception as e:
        warnings.append(f"cron scheduler patch check failed: {e}")

    return warnings

# ── Device config from config.yaml ─────────────────────────────────────────

def _load_devices() -> dict:
    """Read device definitions from plugins.ssh-router.devices in config.yaml."""
    try:
        import yaml
        if CONFIG_FILE.exists():
            raw = yaml.safe_load(CONFIG_FILE.read_text())
            devices = (raw or {}).get("plugins", {}).get("ssh-router", {}).get("devices", {})
            if devices:
                logger.debug("Loaded %d devices from config.yaml", len(devices))
            return devices
    except Exception as e:
        logger.warning("Failed to load devices from config.yaml: %s", e)
    return {}


def _resolve_device(args: dict) -> dict:
    """
    Resolve connection parameters from either a named device or explicit args.

    If args has 'device', looks up the device in config.yaml and merges
    defaults. Otherwise uses host/user/key/port from args directly.

    Returns a dict with host, user, key, port, and a 'label' for display.
    """
    device_name = args.get("device", "")
    if device_name:
        devices = _load_devices()
        device = devices.get(device_name)
        if not device:
            raise ValueError(
                f"Unknown device '{device_name}'. Available: "
                + ", ".join(sorted(devices.keys()) or ["(none defined)"])
            )
        return {
            "host": device["host"],
            "user": device.get("user", "root"),
            "port": device.get("port", 22),
            "key": device.get("key", ""),
            "label": device_name,
        }

    # Explicit params
    host = args.get("host", "").strip()
    if not host:
        raise ValueError("Either 'device' or 'host' is required")
    return {
        "host": host,
        "user": args.get("user", "root").strip(),
        "port": args.get("port", 22),
        "key": args.get("key", "").strip(),
        "label": f"{args.get('user', 'root')}@{host}",
    }


def _list_devices() -> str:
    """Return a formatted list of defined devices for error messages/info."""
    devices = _load_devices()
    if not devices:
        return "(no devices configured in plugins.ssh-router.devices)"
    lines = []
    for name, cfg in sorted(devices.items()):
        user = cfg.get("user", "root")
        host = cfg["host"]
        port = cfg.get("port", 22)
        port_str = f":{port}" if port != 22 else ""
        lines.append(f"  • {name} → {user}@{host}{port_str}")
    return "\n".join(lines)


# ── State management ──────────────────────────────────────────────────────

def _read_state() -> dict:
    """Read persisted SSH routing state."""
    if STATE_FILE.exists():
        try:
            return _json.loads(STATE_FILE.read_text())
        except Exception as e:
            logger.warning("Failed to read state file: %s", e)
    return {"mode": "local"}


def _write_state(state: dict) -> None:
    """Persist SSH routing state."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(_json.dumps(state, indent=2, default=str))


def _restore_state() -> None:
    """
    On plugin load: if we were connected to an SSH target before restart,
    restore the environment variables so the next tool call reconnects.
    """
    _ensure_patch()  # retry patch — module may be importable by now
    state = _read_state()
    if state.get("mode") == "ssh" and state.get("host"):
        host = state.get("host", "")
        user = state.get("user", "root")
        port = state.get("port", 22)
        key = state.get("key", "")
        logger.info("Restoring SSH routing state: %s@%s:%s", user, host, port)
        os.environ["TERMINAL_ENV"] = "ssh"
        os.environ["TERMINAL_SSH_HOST"] = host
        os.environ["TERMINAL_SSH_USER"] = user
        os.environ["TERMINAL_SSH_PORT"] = str(port)
        if key:
            os.environ["TERMINAL_SSH_KEY"] = key
        os.environ["TERMINAL_CWD"] = "/root" if user == "root" else f"/home/{user}"
    else:
        logger.info("No SSH state to restore, staying local")
        os.environ["TERMINAL_ENV"] = "local"
        for _var in [
            "TERMINAL_SSH_HOST", "TERMINAL_SSH_USER",
            "TERMINAL_SSH_PORT", "TERMINAL_SSH_KEY", "TERMINAL_CWD",
        ]:
            os.environ.pop(_var, None)


# ── Environment management ────────────────────────────────────────────────

def _get_task_id() -> str:
    """Resolve the default task_id for the current (main) agent session."""
    try:
        from tools.terminal_tool import _resolve_container_task_id
        return _resolve_container_task_id(None) or "default"
    except Exception:
        return "default"


def _clear_current_environment() -> None:
    """Close and remove the cached environment so the next call creates a new one."""
    task_id = _get_task_id()
    try:
        from tools.terminal_tool import cleanup_vm
        cleanup_vm(task_id)
        logger.info("Cleared environment for task %s", task_id)
    except Exception as e:
        logger.warning("Error clearing environment: %s", e)


def _test_ssh(host: str, user: str, key: str = "", port: int = 22) -> tuple:
    """
    Quick connectivity test before switching to a remote target.
    Returns (success: bool, message: str).
    """
    cmd = [
        "ssh",
        "-o", "ConnectTimeout=5",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "BatchMode=yes",
    ]
    if key:
        cmd.extend(["-i", os.path.expanduser(key)])
    if port != 22:
        cmd.extend(["-p", str(port)])
    cmd.append(f"{user}@{host}")
    cmd.append("echo HERMES_SSH_OK")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and "HERMES_SSH_OK" in result.stdout:
            return True, f"Connected to {user}@{host}:{port}"
        stderr = result.stderr.strip()[:200]
        return False, f"SSH connection failed: {stderr}"
    except subprocess.TimeoutExpired:
        return False, "SSH connection timed out (10s)"
    except FileNotFoundError:
        return False, "SSH client not found on this machine"
    except Exception as e:
        return False, f"SSH error: {str(e)[:200]}"


# ── Tool handlers ─────────────────────────────────────────────────────────

def handle_connect_to(args: dict, **kwargs) -> str:
    """
    Switch terminal/file backend to a remote SSH target.

    Use either:
      device="pc"       — name defined in plugins.ssh-router.devices in config.yaml
      host="..."        — explicit connection params (user, key, port optional)
    """
    # Ensure the terminal patch is applied before we mutate SSH state
    _ensure_patch()

    try:
        target = _resolve_device(args)
    except ValueError as e:
        return _json.dumps({
            "status": "error",
            "error": str(e),
            "available_devices": _list_devices(),
        })

    # Test connectivity first
    ok, msg = _test_ssh(
        target["host"], target["user"],
        target["key"], target["port"],
    )
    if not ok:
        return _json.dumps({"status": "error", "error": msg})

    # Clear the cached environment (closes existing SSH/local connection)
    _clear_current_environment()

    # Force liveness re-probe for the new target — prevents a stale
    # dead/alive cache from the previous host from leaking into the
    # first pre_llm_call after reconnect.
    _invalidate_liveness_cache()

    # Persist state FIRST to avoid race conditions with cron cleanup
    _write_state({
        "mode": "ssh",
        "host": target["host"],
        "user": target["user"],
        "port": target["port"],
        "key": os.path.expanduser(target["key"]) if target["key"] else "",
        "label": target["label"],
        "connected_at": datetime.now().isoformat(),
        # Save the LOCAL cwd at connect time so disconnect() can restore
        # it — prevents violently yanking the process out of its local
        # project dir (which usually lives under /root/...).
        "local_cwd": os.getcwd(),
    })

    # Set env vars for the new target
    os.environ["TERMINAL_ENV"] = "ssh"
    os.environ["TERMINAL_SSH_HOST"] = target["host"]
    os.environ["TERMINAL_SSH_USER"] = target["user"]
    os.environ["TERMINAL_SSH_PORT"] = str(target["port"])
    if target["key"]:
        os.environ["TERMINAL_SSH_KEY"] = os.path.expanduser(target["key"])
    else:
        os.environ.pop("TERMINAL_SSH_KEY", None)

    # Force cwd to remote home — os.path.expanduser() in Hermes' _get_env_config()
    # resolves ~ to the LOCAL home, breaking SSH. Use explicit path instead.
    remote_home = "/root" if target["user"] == "root" else f"/home/{target['user']}"
    os.environ["TERMINAL_CWD"] = remote_home

    # Mark this task context as SSH-active so the monkey-patch knows
    # this session should use SSH, not fall back to local.
    _ssh_active.set(True)

    return _json.dumps({
        "status": "ok",
        "message": msg,
        "mode": "ssh",
        "target": f"{target['user']}@{target['host']}",
    })


def handle_disconnect(args: dict, **kwargs) -> str:
    """Disconnect from SSH target and return to local execution."""
    state = _read_state()

    if state.get("mode") != "ssh":
        return _json.dumps({
            "status": "ok",
            "message": "Already in local mode",
            "mode": "local",
        })

    target = f"{state.get('user', '')}@{state.get('host', 'unknown')}"

    # Clear the cached SSH environment (closes the remote connection)
    _clear_current_environment()

    # Drop liveness cache too — after disconnect we're local and the
    # old target must not report stale state.
    _invalidate_liveness_cache()

    # Mark this task as local-only — the monkey-patch will return
    # env_type=local even if other tasks still have TERMINAL_ENV=ssh.
    _ssh_active.set(False)

    # Restore local environment
    os.environ["TERMINAL_ENV"] = "local"
    for var in [
        "TERMINAL_SSH_HOST", "TERMINAL_SSH_USER",
        "TERMINAL_SSH_PORT", "TERMINAL_SSH_KEY",
        "TERMINAL_CWD",
    ]:
        os.environ.pop(var, None)

    # Reset CWD ONLY back to the LOCAL directory we were in before
    # connect_to(). We saved it in the state file at connect time.
    # Never os.chdir() based on startswith("/root") — local Docker
    # workspaces are /root/... and would be violently yanked to ~.
    saved_local_cwd = state.get("local_cwd", "")
    if saved_local_cwd:
        try:
            os.chdir(saved_local_cwd)
        except OSError:
            pass  # dir may no longer exist; harmless

    # TERMINAL_CWD is the remote CWD; after disconnect there is no remote
    # backend anymore, so drop it (the local backend uses its own default).
    os.environ.pop("TERMINAL_CWD", None)

    # Clean slate — delete state file so _restore_state finds nothing
    try:
        STATE_FILE.unlink(missing_ok=True)
    except OSError:
        pass

    return _json.dumps({
        "status": "ok",
        "message": f"Disconnected from {target}. Back to local.",
        "mode": "local",
    })


def handle_status(args: dict, **kwargs) -> str:
    """Show current routing status and available devices."""
    _ensure_patch()
    state = _read_state()
    mode = state.get("mode", "local")

    # Actual SSH mode depends on whether THIS context is SSH-active
    active_mode = "ssh" if (mode == "ssh" and _ssh_active.get()) else "local"
    info = {"status": "ok", "mode": active_mode, "persisted_mode": mode}

    if active_mode == "ssh":
        info["target"] = f"{state.get('user', '')}@{state.get('host', '')}:{state.get('port', 22)}"
        info["label"] = state.get("label", "")
        info["connected_at"] = state.get("connected_at", "")
    else:
        info["target"] = "local"

    # Also list available devices
    devices = _load_devices()
    if devices:
        info["available_devices"] = list(devices.keys())

    # Patch integrity check — warn if a Hermes update replaced our patches
    patch_warnings = _verify_patches()
    if patch_warnings:
        info["patch_warnings"] = patch_warnings
        info["status"] = "degraded"  # patches broken but routing may still work

    return _json.dumps(info)


# ── Plugin entry point ────────────────────────────────────────────────────

# Liveness check cache — the pre_llm_call hook NEVER blocks on SSH.
# A background daemon thread refreshes the probe; the hook reads cached
# state (or optimistically assumes alive when unknown).
_LIVENESS_CHECK_INTERVAL = 30
_liveness_cache: Dict[str, Any] = {"ts": 0.0, "alive": False, "key": ""}
_liveness_lock = threading.Lock()


def _probe_host(host: str, user: str = "root", port: int = 22, key: str = "") -> bool:
    """One-shot SSH reachability probe. ConnectTimeout=5 matches _test_ssh
    so a host that passes connect_to() won't flap as dead in liveness."""
    cmd = [
        "ssh",
        "-o", "ConnectTimeout=5",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
    ]
    if key:
        cmd.extend(["-i", os.path.expanduser(key)])
    if port != 22:
        cmd.extend(["-p", str(port)])
    cmd.append(f"{user}@{host}")
    cmd.append("true")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=8, stdin=subprocess.DEVNULL,
        )
        return result.returncode == 0
    except Exception:
        return False


def _refresh_liveness_async(host: str, user: str, port: int, key: str) -> None:
    """Probe in a background daemon thread and update the cache.

    Never blocks the caller. On failure the cache keeps its last value
    (stale-alive is safer than a spurious dead alert).
    """
    def _do() -> None:
        try:
            alive = _probe_host(host, user, port, key)
            with _liveness_lock:
                _liveness_cache["ts"] = time.monotonic()
                _liveness_cache["key"] = f"{user}@{host}:{port}"
                _liveness_cache["alive"] = alive
        except Exception as exc:
            logger.warning("ssh-router liveness probe failed: %s", exc)

    thread = threading.Thread(target=_do, daemon=True, name="ssh-liveness")
    thread.start()


def _invalidate_liveness_cache() -> None:
    """Force the next _host_is_alive() to re-probe (state unknown).

    Sets alive=None (UNKNOWN) — the next call assumes ALIVE (optimistic)
    because connect_to() just verified the host, and the background probe
    will confirm true state within seconds. A confirmed-dead result from
    a completed probe, in contrast, stays 'dead' for the full 30s window.
    """
    with _liveness_lock:
        _liveness_cache["ts"] = 0.0
        _liveness_cache["key"] = ""
        _liveness_cache["alive"] = None  # None = UNKNOWN (not yet probed)


def _mark_liveness_dead(host: str, user: str, port: int) -> None:
    """Force the next _host_is_alive() to return False and re-probe."""
    with _liveness_lock:
        _liveness_cache["ts"] = 0.0
        _liveness_cache["key"] = f"{user}@{host}:{port}"
        _liveness_cache["alive"] = False


def _host_is_alive(host: str, user: str = "root", port: int = 22, key: str = "") -> bool:
    """Reachability check with background refresh — NEVER blocks.

    Three states:
      - fresh + alive is bool     → return it (either alive or confirmed dead)
      - fresh + alive is None     → UNKNOWN: assume True (optimistic)
      - stale                     → kick background probe + assume True

    Only a COMPLETED probe writes dead into the cache (fresh, bool False),
    so a dead host gets reported within probe time (~5s), while a fresh
    connect_to() (which itself just verified the host) never causes a
    spurious NOT-RESPONDING alert. Real tool failures are still caught
    fast by the post_tool_call watchdog.
    """
    now = time.monotonic()
    cache_key = f"{user}@{host}:{port}"
    with _liveness_lock:
        fresh = (
            _liveness_cache.get("key") == cache_key
            and now - _liveness_cache.get("ts", 0) < _LIVENESS_CHECK_INTERVAL
        )
        cached_alive = _liveness_cache.get("alive")
    if not fresh:
        _refresh_liveness_async(host, user, port, key)
        if cached_alive is False:
            return False  # keep returning dead until probe says otherwise
        return True  # optimistic — probe will confirm within seconds
    if cached_alive is None:
        return True  # UNKNOWN → assume alive (watchdog catches real failures)
    return bool(cached_alive)


def _task_is_ssh() -> bool:
    """True when the CURRENT task actually runs over SSH.

    Mirrors _patched_get_env_config's decision logic:
      - cron job marked SSH (its contextvar + TERMINAL_ENV=ssh) → True
      - main session with _ssh_active=True                      → True
      - anything else (local session, isolated task, orphaned state
        file after a crash)                                     → False
    """
    try:
        tid = _cron_task_id.get()
        if tid:
            return bool(_get_task_ssh_env(tid))
        return bool(_ssh_active.get())
    except Exception:
        return False


# SSH transport failure markers — scoped to OpenSSH client errors so
# legitimate command output ("curl: connection refused", "git push:
# permission denied") never triggers a false alarm.
_SSH_TRANSPORT_MARKERS = [
    "ssh: connect to host",
    "ssh_exchange_identification",
    "kex_exchange_identification",
    "host key verification failed",
    "permission denied (publickey",
]


def _is_ssh_failure_text(text: str) -> bool:
    """True only for genuine OpenSSH client transport failures.

    Strict: OpenSSH's own diagnostics ("ssh: connect to host ... timed
    out", "ssh_exchange_identification") or the standalone markers. A URL
    like "curl http://x/ssh_keys: connection timed out" contains no ssh:
    prefix and won't match. Generic network phrases (without "ssh:")
    never match either — a remote curl/wget could legitimately fail that
    way without the SSH transport being broken.
    """
    lower = text.lower()
    # OpenSSH client diagnostics all carry the program prefix "ssh:" or
    # ssh_exchange/kex markers. "ssh:" is never a bare URL scheme, so
    # false positives from URLs like http://host/ssh_keys are impossible.
    if "ssh:" in lower or any(m in lower for m in _SSH_TRANSPORT_MARKERS):
        return True
    return False


def _ssh_connection_watchdog(**kwargs) -> Optional[Dict[str, str]]:
    """post_tool_call hook — detect SSH connection failures on any tool call.

    Runs after EVERY tool call (terminal, read_file, write_file, patch,
    search_files, ...). When the result indicates an SSH transport-class
    error, returns context telling the agent the remote link is broken,
    and invalidates the liveness cache so the next turn re-probes.

    Returns None (no-op) for local mode, successful calls, or non-SSH
    command errors — only real connection failures trigger.
    """
    try:
        state = _read_state()
        if state.get("mode") != "ssh":
            return None
        # Only alert when THIS task actually runs over SSH — a local task
        # (after restart, isolated, or disconnected) must never see an
        # SSH alert for a stale state file.
        if not _task_is_ssh():
            return None
        if kwargs.get("status") == "error":
            err = str(
                kwargs.get("error_message")
                or kwargs.get("error")
                or kwargs.get("result")
                or ""
            )
            if _is_ssh_failure_text(err):
                host = state.get("host", "")
                user = state.get("user", "root")
                port = state.get("port", 22)
                _mark_liveness_dead(host, user, port)
                label = state.get("label", "") or host
                return {
                    "context": (
                        f"⚠️ SSH ROUTER ALERT: Connection FAILED to \"{label}\" "
                        f"({host}). The remote host did not respond to a tool call. "
                        "Do NOT retry blindly — verify connectivity first, then "
                        "call disconnect() to return to local, or reconnect when "
                        "the host is back."
                    )
                }
        return None
    except Exception as exc:
        logger.warning("ssh-router post_tool_call watchdog failed: %s", exc)
        return None


def _ssh_status_hook(**kwargs) -> Optional[Dict[str, str]]:
    """pre_llm_call hook — inject current SSH routing status into the prompt.

    Returns a dict with 'context' (injected into the user message every
    turn) so the agent always knows which host it is operating on, and
    whether that host is actually reachable right now.

    - local / no active SSH for THIS task → None (silent)
    - ssh + host alive      → "Connected to ..." + execution-context
    - ssh + host unreachable→ "⚠️ host NOT responding" alert so the agent
                              never mistakes a dead link for a live one.

    The liveness probe runs in a background daemon thread (never blocks
    the turn) and is cached for LIVENESS_CHECK_INTERVAL seconds.
    """
    try:
        state = _read_state()
        if state.get("mode") != "ssh":
            return None
        # Context isolation: a cron job or isolated task running LOCALLY
        # (or a task that disconnected) must not see SSH context, even if
        # the state file still says "ssh" (e.g. after a crash).
        if not _task_is_ssh():
            return None

        host = state.get("host", "")
        user = state.get("user", "")
        port = state.get("port", 22)
        key = state.get("key", "")
        label = state.get("label", "") or f"{user}@{host}"
        port_str = f":{port}" if port != 22 else ""

        alive = _host_is_alive(host, user, port, key)

        if not alive:
            return {
                "context": (
                    f"⚠️ SSH ROUTER ALERT: Host \"{label}\" ({user}@{host}{port_str}) "
                    "is NOT RESPONDING. The remote connection is dead or unreachable. "
                    "Any tool call (terminal, read_file, write_file, patch, search_files) "
                    "will FAIL with connection errors. Do NOT resolve paths against this "
                    "host; verify connectivity first, then call disconnect() to return "
                    "to local or reconnect when the host is back."
                )
            }

        return {
            "context": (
                f"SSH Router Status: Connected to \"{label}\" "
                f"({user}@{host}{port_str}). "
                "IMPORTANT EXECUTION CONTEXT: Any tool that executes commands "
                "or reads, writes, searches, or modifies files — including "
                "terminal(), read_file(), write_file(), patch(), and "
                "search_files() — runs natively on this remote host. "
                "Resolve and verify every path against this remote host, not "
                "the local machine, before acting. All terminal commands and "
                "file operations remain remote until you call disconnect()."
            )
        }
    except Exception as exc:
        logger.warning("ssh-router pre_llm_call hook failed: %s", exc)
        return None


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

    # ── connect_to ──────────────────────────────────────────────────────
    ctx.register_tool(
        name="connect_to",
        toolset="terminal",
        schema={
            "type": "function",
            "function": {
                "name": "connect_to",
                "description": (
                    "Connect terminal and file backends to a remote machine via SSH. "
                    "After connecting, ALL terminal commands and file operations "
                    "(read_file, write_file, patch, search_files) execute on the "
                    "remote host. Use disconnect() to return to local.\n\n"
                    "Two modes:\n"
                    "  1. Named device: connect_to(device='pc') — looks up host/user/key\n"
                    "     from plugins.ssh-router.devices in config.yaml\n"
                    "  2. Explicit: connect_to(host='...', user='root', key='~/.ssh/key')\n\n"
                    "Available devices are shown in ssh_status."
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

    # ── disconnect ──────────────────────────────────────────────────────
    ctx.register_tool(
        name="disconnect",
        toolset="terminal",
        schema={
            "type": "function",
            "function": {
                "name": "disconnect",
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

    # ── ssh_status ──────────────────────────────────────────────────────
    ctx.register_tool(
        name="ssh_status",
        toolset="terminal",
        schema={
            "type": "function",
            "function": {
                "name": "ssh_status",
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
