import os
import signal
import subprocess
import logging
import threading
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

_CLEANUP_TIMEOUT = 15

def _clear_current_environment() -> None:
    """Close and remove the cached environment so the next call creates a new one."""
    task_id = _get_task_id()
    try:
        from tools.terminal_tool import cleanup_vm
        thread = threading.Thread(target=cleanup_vm, args=(task_id,), daemon=True)
        thread.start()
        thread.join(timeout=_CLEANUP_TIMEOUT)
        if thread.is_alive():
            logger.warning("cleanup_vm timed out after %ds for task %s — proceeding anyway", _CLEANUP_TIMEOUT, task_id)
        else:
            logger.info("Cleared environment for task %s", task_id)
    except Exception as e:
        logger.warning("Error clearing environment: %s", e)

def _test_ssh(host: str, user: str, key: str = "", port: int = 22) -> tuple:
    """
    Quick connectivity test before switching to a remote target.
    Returns (success: bool, message: str).
    Uses start_new_session + os.killpg to avoid pipe-inheritance hangs.
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
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                proc.kill()
            proc.wait(timeout=3)
            return False, "SSH connection timed out (10s)"
        stdout_text = stdout.decode(errors="replace")
        stderr_text = stderr.decode(errors="replace")
        if proc.returncode == 0 and "HERMES_SSH_OK" in stdout_text:
            return True, f"Connected to {user}@{host}:{port}"
        return False, f"SSH connection failed: {stderr_text.strip()[:200]}"
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
