"""OpenProxy client and Antigravity account rotation engine.

Communicates with OpenProxy's admin API to track 5-hour session quotas,
weekly quotas, refresh telemetry, and inject OAuth credentials directly
into the local agy CLI environment.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "http://localhost:8787/admin/api"
DEFAULT_TOKEN = "op_live_REDACTED"
SWITCH_SCRIPT_PATH = Path("/root/switch_account.sh")


def _read_credentials_from_script(script_path: Path = SWITCH_SCRIPT_PATH) -> tuple[str | None, str | None]:
    """Extract API_URL and TOKEN from switch_account.sh if present."""
    if not script_path.is_file():
        return None, None
    try:
        content = script_path.read_text(encoding="utf-8")
        url_match = re.search(r'API_URL=["\']([^"\']+)["\']', content)
        token_match = re.search(r'TOKEN=["\']([^"\']+)["\']', content)
        url = url_match.group(1).strip() if url_match else None
        token = token_match.group(1).strip() if token_match else None
        return url, token
    except Exception as exc:
        logger.debug("Failed to parse switch script %s: %s", script_path, exc)
        return None, None


def format_remaining_time(iso_str: str | None) -> str:
    """Format an ISO 8601 timestamp as human-readable remaining time."""
    if not iso_str or iso_str == "N/A":
        return "N/A"
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        diff = (dt - now).total_seconds()
        if diff <= 0:
            return "ready"
        d = int(diff // 86400)
        h = int((diff % 86400) // 3600)
        m = int((diff % 3600) // 60)
        parts = []
        if d > 0:
            parts.append(f"{d}d")
        if h > 0:
            parts.append(f"{h}h")
        parts.append(f"{m}m")
        return " ".join(parts)
    except Exception:
        return str(iso_str)


class OpenProxyRotator:
    """Manages Antigravity accounts and performs automated credential rotation."""

    def __init__(
        self,
        api_url: str | None = None,
        token: str | None = None,
        quota_threshold_percent: float = 95.0,
    ) -> None:
        script_url, script_token = _read_credentials_from_script()
        self.api_url = (
            api_url
            or os.getenv("OPENPROXY_ADMIN_URL")
            or script_url
            or DEFAULT_API_URL
        ).rstrip("/")
        self.token = (
            token
            or os.getenv("OPENPROXY_ADMIN_TOKEN")
            or script_token
            or DEFAULT_TOKEN
        ).strip()
        self.quota_threshold_percent = quota_threshold_percent
        self._last_switched_id: int | None = None

    def _call(self, method: str, path: str, data: dict[str, Any] | None = None) -> Any:
        """Call OpenProxy admin API endpoint."""
        url = f"{self.api_url}{path}"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "User-Agent": "hermes-antigravity-rotator/1.0",
        }
        body = json.dumps(data).encode("utf-8") if data is not None else None
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw.strip() else {}

    def list_accounts(self) -> list[dict[str, Any]]:
        """Fetch and return all antigravity provider accounts."""
        try:
            data = self._call("GET", "/accounts")
            if not isinstance(data, list):
                logger.warning("OpenProxy /accounts returned non-list: %s", data)
                return []
            accounts = [a for a in data if a.get("provider_id") == "antigravity"]
            # Mark active status if known
            active_id = self.get_active_account_id()
            for a in accounts:
                a["is_active"] = (a.get("id") == active_id)
            return accounts
        except Exception as exc:
            logger.error("Failed to list accounts from OpenProxy (%s): %s", self.api_url, exc)
            return []

    def get_active_account_id(self) -> int | None:
        """Return the ID of the currently active account if known."""
        if self._last_switched_id is not None:
            return self._last_switched_id

        # Try matching active email from google_accounts.json
        for base in (Path.home(), Path("/root")):
            accounts_file = base / ".gemini" / "google_accounts.json"
            if accounts_file.is_file():
                try:
                    data = json.loads(accounts_file.read_text(encoding="utf-8"))
                    active_email = data.get("active")
                    if active_email:
                        for acc in self.list_accounts():
                            if acc.get("email") == active_email or acc.get("label") == active_email:
                                self._last_switched_id = acc.get("id")
                                return self._last_switched_id
                except Exception:
                    pass
        return None

    def refresh_quota(self, account_id: int) -> dict[str, Any]:
        """Request live quota telemetry refresh for an account."""
        try:
            return self._call("POST", f"/accounts/{account_id}/refresh-quota")
        except Exception as exc:
            logger.warning("Failed to refresh quota for account %s: %s", account_id, exc)
            return {"error": str(exc)}

    def apply_account(self, account_id: int) -> dict[str, Any]:
        """Apply account credentials to local agy CLI configuration."""
        res = self._call("POST", f"/accounts/{account_id}/apply-local-cli")
        self._last_switched_id = account_id

        # Sync any non-symlinked isolated workspace tokens
        self._sync_isolated_workspace_tokens()
        return res

    def _sync_isolated_workspace_tokens(self) -> None:
        """Propagate updated token to existing isolated workspace homes if not symlinked."""
        real_token = Path.home() / ".gemini" / "antigravity-cli" / "antigravity-oauth-token"
        if not real_token.is_file():
            real_token = Path("/root/.gemini/antigravity-cli/antigravity-oauth-token")
        if not real_token.is_file():
            return

        # Check /tmp for active agy_* isolated workspace homes
        tmp_dir = Path("/tmp")
        try:
            for agy_dir in tmp_dir.glob("agy_*"):
                isolated_token = agy_dir / "home" / ".gemini" / "antigravity-cli" / "antigravity-oauth-token"
                if isolated_token.is_file() and not isolated_token.is_symlink():
                    with contextlib.suppress(Exception):
                        shutil.copy2(real_token, isolated_token)
        except Exception as exc:
            logger.debug("Isolated token sync ignored: %s", exc)

    def find_best_candidate(self, exclude_id: int | None = None) -> dict[str, Any] | None:
        """Find the account with the most remaining quota capacity."""
        accounts = self.list_accounts()
        if not accounts:
            return None

        current_id = exclude_id if exclude_id is not None else self.get_active_account_id()

        candidates = []
        for a in accounts:
            aid = a.get("id")
            if current_id is not None and aid == current_id and len(accounts) > 1:
                continue

            sess_used = a.get("quota_session_used") or 0
            sess_limit = a.get("quota_session_limit") or 1000
            weekly_used = a.get("quota_weekly_used") or 0
            weekly_limit = a.get("quota_weekly_limit") or 1000

            # Skip accounts with 0 remaining or at capacity
            if sess_limit > 0 and sess_used >= sess_limit:
                continue
            if weekly_limit > 0 and weekly_used >= weekly_limit:
                continue

            remaining_sess = max(0, sess_limit - sess_used)
            remaining_weekly = max(0, weekly_limit - weekly_used)
            candidates.append((remaining_sess, remaining_weekly, sess_used, a))

        if candidates:
            # Sort: highest remaining session quota first, then lowest session used
            candidates.sort(key=lambda item: (-item[0], item[2]))
            return candidates[0][3]

        # If all candidates exhausted, find account with earliest session reset
        earliest_reset_acc = None
        earliest_ts = None
        now_iso = datetime.now(timezone.utc).isoformat()
        for a in accounts:
            reset_at = a.get("quota_session_reset_at")
            if reset_at:
                if earliest_ts is None or reset_at < earliest_ts:
                    earliest_ts = reset_at
                    earliest_reset_acc = a

        return earliest_reset_acc

    def rotate_to_next_account(self, current_account_id: int | None = None) -> tuple[dict[str, Any], str]:
        """Select best candidate, inject into agy CLI, and return account + description."""
        active_id = current_account_id if current_account_id is not None else self.get_active_account_id()
        best = self.find_best_candidate(exclude_id=active_id)
        if not best:
            raise RuntimeError("No Antigravity accounts configured in OpenProxy.")

        best_id = best.get("id")
        email = best.get("email") or best.get("label") or f"account-{best_id}"
        sess_used = best.get("quota_session_used") or 0
        sess_lim = best.get("quota_session_limit") or 1000
        reset_time = format_remaining_time(best.get("quota_session_reset_at"))

        self.apply_account(best_id)

        msg = (
            f"Rotated Antigravity account to [{best_id}] {email} "
            f"(session: {sess_used}/{sess_lim}, reset: {reset_time})"
        )
        logger.info("[antigravity-rotator] %s", msg)
        return best, msg

    def format_status_table(self) -> str:
        """Render a formatted markdown table of all Antigravity accounts."""
        accounts = self.list_accounts()
        if not accounts:
            return "No Antigravity accounts found in OpenProxy."

        active_id = self.get_active_account_id()
        lines = [
            "### 🔄 Google Antigravity Accounts (OpenProxy Pool)",
            "",
            "| Active | ID | Account / Email | 5-Hour Session | Session Reset | Weekly Quota | Weekly Reset |",
            "|:---:|:---:|:---|:---:|:---:|:---:|:---:|",
        ]
        for a in accounts:
            aid = a.get("id")
            is_active = "👉 **ACTIVE**" if aid == active_id else ""
            label = a.get("email") or a.get("label") or "N/A"
            sess = f"{a.get('quota_session_used') or 0} / {a.get('quota_session_limit') or '∞'}"
            sess_reset = format_remaining_time(a.get("quota_session_reset_at"))
            weekly = f"{a.get('quota_weekly_used') or 0} / {a.get('quota_weekly_limit') or '∞'}"
            weekly_reset = format_remaining_time(a.get("quota_weekly_reset_at"))
            lines.append(f"| {is_active} | `{aid}` | {label} | {sess} | {sess_reset} | {weekly} | {weekly_reset} |")

        return "\n".join(lines)
