import contextvars
import threading
from typing import Dict

_ssh_active = contextvars.ContextVar("ssh_active", default=False)
_cron_task_id: contextvars.ContextVar[str] = contextvars.ContextVar("_cron_task_id", default="")

_patch_applied = False
_cron_scheduler_patched = False

_cron_ssh_counter = 0
_cron_ssh_lock = threading.Lock()

_ssh_env_registry: Dict[str, Dict[str, str]] = {}
_ssh_env_registry_lock = threading.Lock()

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

def _task_is_ssh() -> bool:
    """True when the CURRENT task actually runs over SSH."""
    try:
        tid = _cron_task_id.get()
        if tid:
            return bool(_get_task_ssh_env(tid))
        return bool(_ssh_active.get())
    except Exception:
        return False
