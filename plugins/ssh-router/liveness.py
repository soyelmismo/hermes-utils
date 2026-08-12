import os
import signal
import time
import subprocess
import threading
import logging
from typing import Dict, Any, Optional

from . import context as ctx
from .config import _read_state

logger = logging.getLogger("ssh-router")

_LIVENESS_CHECK_INTERVAL = 30
_liveness_cache: Dict[str, Any] = {"ts": 0.0, "alive": False, "key": ""}
_liveness_lock = threading.Lock()

def _probe_host(host: str, user: str = "root", port: int = 22, key: str = "") -> bool:
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
        proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                proc.kill()
            proc.wait(timeout=3)
            return False
        return proc.returncode == 0
    except Exception:
        return False

def _refresh_liveness_async(host: str, user: str, port: int, key: str) -> None:
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
    with _liveness_lock:
        _liveness_cache["ts"] = 0.0
        _liveness_cache["key"] = ""
        _liveness_cache["alive"] = None

def _mark_liveness_dead(host: str, user: str, port: int) -> None:
    with _liveness_lock:
        _liveness_cache["ts"] = 0.0
        _liveness_cache["key"] = f"{user}@{host}:{port}"
        _liveness_cache["alive"] = False

def _host_is_alive(host: str, user: str = "root", port: int = 22, key: str = "") -> bool:
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
            return False
        return True
    if cached_alive is None:
        return True
    return bool(cached_alive)

_SSH_TRANSPORT_MARKERS = [
    "ssh: connect to host",
    "ssh_exchange_identification",
    "kex_exchange_identification",
    "host key verification failed",
    "permission denied (publickey",
]

def _is_ssh_failure_text(text: str) -> bool:
    lower = text.lower()
    if "ssh:" in lower or any(m in lower for m in _SSH_TRANSPORT_MARKERS):
        return True
    return False

def _ssh_connection_watchdog(**kwargs) -> Optional[Dict[str, str]]:
    try:
        state = _read_state()
        if state.get("mode") != "ssh":
            return None
        if not ctx._task_is_ssh():
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
    try:
        state = _read_state()
        if state.get("mode") != "ssh":
            return None
        if not ctx._task_is_ssh():
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
