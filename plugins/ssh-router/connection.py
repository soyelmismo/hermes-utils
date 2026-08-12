import os
import subprocess
import logging
from .config import _read_state, get_remote_home

logger = logging.getLogger("ssh-router")

def set_ssh_env(host: str, user: str, port: int, key: str) -> None:
    """Centralized function to set SSH environment variables globally."""
    os.environ["TERMINAL_ENV"] = "ssh"
    os.environ["TERMINAL_SSH_HOST"] = host
    os.environ["TERMINAL_SSH_USER"] = user
    os.environ["TERMINAL_SSH_PORT"] = str(port)
    if key:
        os.environ["TERMINAL_SSH_KEY"] = os.path.expanduser(key)
    else:
        os.environ.pop("TERMINAL_SSH_KEY", None)
    os.environ["TERMINAL_CWD"] = get_remote_home(user)

def clear_ssh_env() -> None:
    """Centralized function to clear SSH environment variables globally."""
    os.environ["TERMINAL_ENV"] = "local"
    for var in [
        "TERMINAL_SSH_HOST", "TERMINAL_SSH_USER",
        "TERMINAL_SSH_PORT", "TERMINAL_SSH_KEY", "TERMINAL_CWD",
    ]:
        os.environ.pop(var, None)

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

def _restore_state() -> None:
    """
    On plugin load: if we were connected to an SSH target before restart,
    restore the environment variables so the next tool call reconnects.
    """
    from .patching import _ensure_patch
    _ensure_patch()  # retry patch — module may be importable by now
    state = _read_state()
    if state.get("mode") == "ssh" and state.get("host"):
        host = state.get("host", "")
        user = state.get("user", "root")
        port = state.get("port", 22)
        key = state.get("key", "")
        logger.info("Restoring SSH routing state: %s@%s:%s", user, host, port)
        set_ssh_env(host, user, port, key)
    else:
        logger.info("No SSH state to restore, staying local")
        clear_ssh_env()
