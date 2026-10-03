"""Account registry and cross-platform file locking for local Antigravity accounts.

Handles JSON state persistence with atomic file writes (0600),
directory permissions (0700), fail-open on corruption, and cross-platform locking.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import tempfile
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@contextmanager
def file_lock(lock_path: str, timeout: float = 10.0):
    """Acquire a cross-platform exclusive file lock with timeout.

    Uses msvcrt.locking on Windows (nt) and fcntl.flock on POSIX,
    imported locally so the module can be imported anywhere.
    """
    dir_path = os.path.dirname(lock_path)
    if dir_path:
        os.makedirs(dir_path, mode=0o700, exist_ok=True)

    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    deadline = time.monotonic() + timeout
    try:
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    msvcrt.locking(lock_fd, msvcrt.LK_NBLCK, 1)
                    break
                except (IOError, OSError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"Could not acquire lock {lock_path} within {timeout}s"
                        )
                    time.sleep(0.05)
            try:
                yield lock_fd
            finally:
                with contextlib.suppress(Exception):
                    msvcrt.locking(lock_fd, msvcrt.LK_UNLCK, 1)
        else:
            try:
                import fcntl
            except ImportError:
                fcntl = None

            if fcntl is not None:
                while True:
                    try:
                        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except (IOError, OSError):
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                f"Could not acquire lock {lock_path} within {timeout}s"
                            )
                        time.sleep(0.05)
            try:
                yield lock_fd
            finally:
                if fcntl is not None:
                    with contextlib.suppress(Exception):
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)
    finally:
        with contextlib.suppress(Exception):
            os.close(lock_fd)


def load_state(path: str) -> Dict[str, Any]:
    """Load accounts state from JSON. Fail-open on corruption."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            if "accounts" not in data or not isinstance(data["accounts"], list):
                data["accounts"] = []
            if "active_account" not in data:
                data["active_account"] = None
            return data
    except FileNotFoundError:
        pass
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        logger.warning("Corrupt account state at %s, resetting: %s", path, exc)
    return {"accounts": [], "active_account": None}


def save_state(path: str, data: Dict[str, Any]) -> None:
    """Save accounts state atomically with 0600 permissions."""
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, mode=0o700, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dir_path or None, prefix=".accounts_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def sanitize_folder_name(label: str) -> str:
    """Sanitize a label for use as a directory name."""
    s = re.sub(r"[^a-zA-Z0-9_-]", "_", label.strip()).lstrip(".-")
    return (s or "account")[:64]


def find_account(state: Dict[str, Any], label: str) -> Optional[Dict[str, Any]]:
    """Locate an account dict by label."""
    for a in state.get("accounts", []):
        if a.get("label") == label:
            return a
    return None


def register_account_entry(
    state_file: str,
    lock_path: str,
    label: str,
    home_dir: str,
    email: str = "",
) -> Dict[str, Any]:
    """Register a new account entry in the state file under lock."""
    with file_lock(lock_path):
        state = load_state(state_file)
        accounts = state.setdefault("accounts", [])
        for a in accounts:
            if a.get("label") == label:
                raise RuntimeError(f"Account '{label}' already exists")
        entry = {
            "label": label,
            "home_dir": home_dir,
            "email": email,
            "enabled": True,
            "cooldown_until": None,
            "last_used": None,
        }
        accounts.append(entry)
        save_state(state_file, state)
    return entry


def remove_account_entry(
    state_file: str,
    lock_path: str,
    label: str,
) -> bool:
    """Remove an account entry from the registry without deleting files."""
    with file_lock(lock_path):
        state = load_state(state_file)
        accounts = state.get("accounts", [])
        before = len(accounts)
        state["accounts"] = [a for a in accounts if a.get("label") != label]
        if len(state["accounts"]) == before:
            return False
        if state.get("active_account") == label:
            state["active_account"] = None
        save_state(state_file, state)
    return True


def update_account_cooldown(
    state_file: str,
    lock_path: str,
    label: str,
    cooldown_until_iso: Optional[str],
) -> None:
    """Update cooldown_until for an account under lock."""
    with file_lock(lock_path):
        state = load_state(state_file)
        for a in state.get("accounts", []):
            if a.get("label") == label:
                a["cooldown_until"] = cooldown_until_iso
                break
        save_state(state_file, state)
