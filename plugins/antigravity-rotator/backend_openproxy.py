"""OpenProxy configuration and credential resolution helpers."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from .constants import (
        DEFAULT_API_URL,
        DEFAULT_SWITCH_SCRIPT_PATH,
        SWITCH_SCRIPT_ENV_VAR,
    )
except ImportError:
    from constants import (
        DEFAULT_API_URL,
        DEFAULT_SWITCH_SCRIPT_PATH,
        SWITCH_SCRIPT_ENV_VAR,
    )

logger = logging.getLogger(__name__)

SWITCH_SCRIPT_PATH = DEFAULT_SWITCH_SCRIPT_PATH


def _config_bases() -> tuple[Path, ...]:
    """Directories searched for agy CLI state (primary home first, /root fallback)."""
    return (Path.home(), Path("/root"))


def get_switch_script_candidates() -> List[Path]:
    """Return ordered list of switch script candidate paths to search."""
    candidates: List[Path] = []
    env_script = os.getenv(SWITCH_SCRIPT_ENV_VAR)
    if env_script and env_script.strip():
        candidates.append(Path(env_script.strip()))
    try:
        home = Path.home()
        candidates.append(home / ".agents" / "switch_account.sh")
        candidates.append(home / "switch_account.sh")
    except Exception:
        pass
    candidates.append(Path("/root/.agents/switch_account.sh"))
    candidates.append(Path("/root/switch_account.sh"))
    return candidates


def resolve_switch_script_path(
    script_path: Optional[Path | str] = None,
) -> Optional[Path]:
    """Resolve switch script path from candidates or explicit path."""
    if script_path is not None:
        p = Path(script_path)
        return p if p.is_file() else None

    for candidate in get_switch_script_candidates():
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def _read_credentials_from_script(
    script_path: Optional[Path | str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Extract API_URL and TOKEN from switch_account.sh candidate if present.

    Never logs or exposes tokens.
    """
    resolved = resolve_switch_script_path(script_path)
    if not resolved:
        return None, None
    try:
        content = resolved.read_text(encoding="utf-8")
        url: Optional[str] = None
        token: Optional[str] = None

        url_patterns = (
            r'(?:export\s+)?SWITCH_ACCOUNT_API_URL=(?:["\']([^"\']+)["\']|([^\s#]+))',
            r'(?:export\s+)?API_URL=(?:["\']([^"\']+)["\']|([^\s#]+))',
        )
        for pattern in url_patterns:
            for m in re.finditer(pattern, content):
                val = (m.group(1) or m.group(2) or "").strip()
                if val and not val.startswith("$"):
                    url = val
                    break
            if url:
                break

        token_patterns = (
            r'(?:export\s+)?SWITCH_ACCOUNT_TOKEN=(?:["\']([^"\']+)["\']|([^\s#]+))',
            r'(?:export\s+)?TOKEN=(?:["\']([^"\']+)["\']|([^\s#]+))',
        )
        for pattern in token_patterns:
            for m in re.finditer(pattern, content):
                val = (m.group(1) or m.group(2) or "").strip()
                if val and not val.startswith("$"):
                    token = val
                    break
            if token:
                break

        return url, token
    except Exception as exc:
        logger.debug("Failed to parse switch script %s: %s", resolved, exc)
        return None, None


def resolve_openproxy_credentials(
    config_url: Optional[str] = None,
    config_token: Optional[str] = None,
    script_path: Optional[Path | str] = None,
) -> Tuple[str, Optional[str]]:
    """Resolve OpenProxy (API_URL, TOKEN) adhering to precedence order:
    1. config (explicit arguments)
    2. OPENPROXY_* environment variables
    3. SWITCH_ACCOUNT_* environment variables
    4. script candidate files
    5. default URL / None token

    Never logs or exposes tokens.
    """
    script_url, script_token = _read_credentials_from_script(script_path)

    url = (
        (config_url or "").strip()
        or (os.getenv("OPENPROXY_ADMIN_URL") or "").strip()
        or (os.getenv("SWITCH_ACCOUNT_API_URL") or "").strip()
        or (script_url or "").strip()
        or DEFAULT_API_URL
    ).rstrip("/")

    token = (
        (config_token or "").strip()
        or (os.getenv("OPENPROXY_ADMIN_TOKEN") or "").strip()
        or (os.getenv("SWITCH_ACCOUNT_TOKEN") or "").strip()
        or (script_token or "").strip()
        or None
    )

    return url, token
