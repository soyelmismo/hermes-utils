"""Account onboarding and interactive login for local Antigravity accounts.

Handles interactive login requiring a TTY, isolated staging environments,
token verification, active email extraction without decoding id_tokens,
and safe cleanup on failure or cancellation.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, Optional

try:
    from .constants import GOOGLE_ACCOUNTS_FILE, TOKEN_FILENAMES
    from .local_probe import find_agy
    from .local_registry import register_account_entry, sanitize_folder_name
except ImportError:
    from constants import GOOGLE_ACCOUNTS_FILE, TOKEN_FILENAMES
    from local_probe import find_agy
    from local_registry import register_account_entry, sanitize_folder_name

logger = logging.getLogger(__name__)


def resolve_token_file(base_dir: str) -> Optional[str]:
    """Find the first existing token file in <base_dir>/.gemini/antigravity-cli/."""
    token_dir = os.path.join(base_dir, ".gemini", "antigravity-cli")
    for name in TOKEN_FILENAMES:
        p = os.path.join(token_dir, name)
        if os.path.exists(p):
            return p
    return None


def login_account(
    label: str,
    accounts_dir: str,
    state_file: str,
    lock_path: str,
    timeout: int = 300,
) -> Dict[str, Any]:
    """Perform interactive login for an Antigravity account into a dedicated HOME.

    Requires an interactive TTY.
    Creates a staging directory (mode 0700) within accounts_dir.
    Runs agy with SSH_CONNECTION sentinel, keeping browser unblocked for OAuth.
    Validates token presence, extracts email from google_accounts.json active field.
    Renames staging directory to final label and registers account in state file.
    Ensures staging directory is cleaned up in a finally block on any failure,
    timeout, or KeyboardInterrupt.
    """
    if not sys.stdin.isatty():
        raise RuntimeError(
            "Login requires an interactive terminal (TTY). "
            "Run this command directly in a terminal session."
        )

    safe_label = sanitize_folder_name(label)
    os.makedirs(accounts_dir, mode=0o700, exist_ok=True)

    final_dir = os.path.join(accounts_dir, safe_label)
    if os.path.exists(final_dir):
        raise RuntimeError(f"Account directory already exists: {final_dir}")

    staging = os.path.join(accounts_dir, f".staging-{safe_label}")
    os.makedirs(staging, mode=0o700, exist_ok=True)

    try:
        agy = find_agy()

        env = os.environ.copy()
        env["HOME"] = staging
        env["USERPROFILE"] = staging
        env["HOMEPATH"] = staging
        env.pop("ANTIGRAVITY_CONFIG_DIR", None)
        env["SSH_CONNECTION"] = "127.0.0.1 0 127.0.0.1 0"

        logger.info("Starting interactive agy login for '%s'...", safe_label)
        proc = subprocess.run([agy], env=env, timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(f"agy login exited with return code {proc.returncode}")

        token_path = resolve_token_file(staging)
        if not token_path:
            raise RuntimeError(
                f"Login completed but no token file found under {staging}/.gemini/antigravity-cli/"
            )

        email = ""
        ga_path = os.path.join(staging, ".gemini", GOOGLE_ACCOUNTS_FILE)
        if os.path.isfile(ga_path):
            try:
                with open(ga_path, "r", encoding="utf-8") as f:
                    ga_data = json.load(f)
                email = str(ga_data.get("active") or "").strip()
            except (json.JSONDecodeError, OSError) as exc:
                logger.debug("Could not read active email from %s: %s", ga_path, exc)

        os.rename(staging, final_dir)

        entry = register_account_entry(
            state_file=state_file,
            lock_path=lock_path,
            label=safe_label,
            home_dir=final_dir,
            email=email,
        )
        logger.info("Successfully onboarded account '%s' (email: %s)", safe_label, email)
        return entry

    finally:
        if os.path.exists(staging):
            shutil.rmtree(staging, ignore_errors=True)
