"""Environment and subprocess probing for local Antigravity accounts.

Handles agy binary lookup, private browser-blocking shims (0700),
account probe environment construction, quota telemetry, and model checks.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional

try:
    from .quota import pools_from_usage_json
except ImportError:
    from quota import pools_from_usage_json

logger = logging.getLogger(__name__)

SHIM_SCRIPT_CONTENT = (
    '#!/bin/sh\n'
    'echo "blocked browser launch (antigravity-rotator probe)" >&2\n'
    'exit 1\n'
)


def find_agy() -> str:
    """Locate the agy CLI executable."""
    found = shutil.which("agy")
    if found:
        return found
    for candidate in (
        os.path.expanduser("~/.gemini/antigravity-cli/bin/agy"),
        os.path.expanduser("~/.local/bin/agy"),
    ):
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError(
        "agy binary not found in PATH, ~/.gemini/antigravity-cli/bin/, or ~/.local/bin/"
    )


def nobrowser_shim_dir(cache_dir: Optional[str] = None) -> str:
    """Create and return a private directory with open/xdg-open shims (0700).

    Shims write to stderr and exit 1 so agy immediately recognizes the browser
    cannot be opened rather than hanging until timeout. Rewrites shims if
    content differs.
    """
    base = cache_dir or os.path.expanduser("~/.cache/antigravity-rotator")
    shim_dir = os.path.join(base, "nobrowser")
    os.makedirs(shim_dir, mode=0o700, exist_ok=True)

    for cmd in ("open", "xdg-open"):
        p = os.path.join(shim_dir, cmd)
        needs_write = True
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    if f.read() == SHIM_SCRIPT_CONTENT:
                        needs_write = False
            except OSError:
                pass
        if needs_write:
            with open(p, "w", encoding="utf-8") as f:
                f.write(SHIM_SCRIPT_CONTENT)
            os.chmod(p, 0o755)

    return shim_dir


def probe_env(home_dir: str, cache_dir: Optional[str] = None) -> Dict[str, str]:
    """Build environment dictionary for probing an account's agy instance.

    Sets HOME, USERPROFILE, and HOMEPATH to home_dir.
    Removes ANTIGRAVITY_CONFIG_DIR.
    ALWAYS assigns SSH_CONNECTION="127.0.0.1 0 127.0.0.1 0" to force file token storage.
    Blocks browser via BROWSER=/usr/bin/false and private 0700 shims for open/xdg-open.
    Uses os.pathsep for cross-platform PATH construction.
    """
    env = os.environ.copy()
    env["HOME"] = home_dir
    env["USERPROFILE"] = home_dir
    env["HOMEPATH"] = home_dir
    env.pop("ANTIGRAVITY_CONFIG_DIR", None)

    # Always assign; empty inherited value breaks file token storage
    env["SSH_CONNECTION"] = "127.0.0.1 0 127.0.0.1 0"
    env["BROWSER"] = "/usr/bin/false"

    shim_dir = nobrowser_shim_dir(cache_dir)
    current_path = env.get("PATH", "/usr/bin:/bin")
    env["PATH"] = f"{shim_dir}{os.pathsep}{current_path}"
    return env


def fetch_usage_for_home(
    home_dir: str,
    cache: Optional[Dict[str, Any]] = None,
    cache_ttl: float = 60.0,
) -> Dict[str, Dict[str, Any]]:
    """Fetch and parse quota usage for an account home directory."""
    cache_key = f"usage:{home_dir}"
    if cache is not None:
        cached = cache.get(cache_key)
        if cached:
            ts, pools = cached
            if time.monotonic() - ts < cache_ttl:
                return pools

    try:
        agy = find_agy()
    except FileNotFoundError:
        return {}

    env = probe_env(home_dir)
    try:
        proc = subprocess.run(
            [agy, "-p", "/usage", "--output-format", "json"],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            stdin=subprocess.DEVNULL,
        )
        if proc.returncode != 0:
            logger.warning("agy usage failed for %s: %s", home_dir, proc.stderr[:200])
            return {}
        pools = pools_from_usage_json(proc.stdout)
        if cache is not None:
            cache[cache_key] = (time.monotonic(), pools)
        return pools
    except subprocess.TimeoutExpired:
        logger.warning("agy usage timed out for %s", home_dir)
        return {}
    except Exception as exc:
        logger.warning("agy usage error for %s: %s", home_dir, exc)
        return {}


def list_models_for_home(
    home_dir: str,
    cache: Optional[Dict[str, Any]] = None,
    cache_ttl: float = 21600.0,
    fail_ttl: float = 300.0,
) -> Optional[List[str]]:
    """List available models for an account (cached 6h, failures 5m)."""
    cache_key = f"models:{home_dir}"
    if cache is not None:
        cached = cache.get(cache_key)
        if cached:
            ts, models, was_failure = cached
            ttl = fail_ttl if was_failure else cache_ttl
            if time.monotonic() - ts < ttl:
                return models

    try:
        agy = find_agy()
    except FileNotFoundError:
        return None

    env = probe_env(home_dir)
    try:
        proc = subprocess.run(
            [agy, "models"],
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
            stdin=subprocess.DEVNULL,
        )
        if proc.returncode != 0:
            if cache is not None:
                cache[cache_key] = (time.monotonic(), None, True)
            return None

        models: List[str] = []
        for line in proc.stdout.strip().split("\n"):
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("-"):
                model = line.lstrip("* ").split()[0] if line.lstrip("* ") else ""
                if model:
                    models.append(model)

        if cache is not None:
            cache[cache_key] = (time.monotonic(), models, False)
        return models
    except subprocess.TimeoutExpired:
        if cache is not None:
            cache[cache_key] = (time.monotonic(), None, True)
        return None
    except Exception:
        if cache is not None:
            cache[cache_key] = (time.monotonic(), None, True)
        return None


def account_supports_model(models: Optional[List[str]], model_name: str) -> bool:
    """Check if model matches supported list by exact name or prefix.

    Unknown models list (None) returns True.
    """
    if models is None or not model_name:
        return True

    ml = model_name.lower()
    for m in models:
        m_low = m.lower()
        if m_low == ml or m_low.startswith(f"{ml}-") or ml.startswith(f"{m_low}-"):
            return True
    return False
