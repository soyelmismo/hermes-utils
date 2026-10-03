"""Selection engine and account rotators for Antigravity.

Coordinates quota evaluation across Gemini and Claude/GPT pools,
evaluates account usability gates, ranks candidates, and drives
credential switching across OpenProxy and local file backends.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn, Optional
import urllib.request

try:
    from .backend_local import LocalBackend
    from .backend_openproxy import (
        DEFAULT_API_URL, _config_bases, _read_credentials_from_script,
        get_switch_script_candidates, resolve_openproxy_credentials, resolve_switch_script_path,
    )
    from .constants import (
        DEFAULT_SWITCH_SCRIPT_PATH, ISOLATED_HOME_GLOBS, ISOLATED_TMP_ROOT, TOKEN_FILENAMES,
    )
    from .quota import (
        calculate_score, is_usable_for_pool, pool_for_model, pools_from_openproxy,
    )
except ImportError:
    from backend_local import LocalBackend
    from backend_openproxy import (
        DEFAULT_API_URL, _config_bases, _read_credentials_from_script,
        get_switch_script_candidates, resolve_openproxy_credentials, resolve_switch_script_path,
    )
    from constants import (
        DEFAULT_SWITCH_SCRIPT_PATH, ISOLATED_HOME_GLOBS, ISOLATED_TMP_ROOT, TOKEN_FILENAMES,
    )
    from quota import (
        calculate_score, is_usable_for_pool, pool_for_model, pools_from_openproxy,
    )

logger = logging.getLogger(__name__)

DEFAULT_TOKEN: str | None = None
SWITCH_SCRIPT_PATH = DEFAULT_SWITCH_SCRIPT_PATH
_TOKEN_FILENAMES = TOKEN_FILENAMES
_ISOLATED_TMP_ROOT = ISOLATED_TMP_ROOT
_ISOLATED_HOME_GLOBS = ISOLATED_HOME_GLOBS


def format_remaining_time(iso_str: str | None) -> str:
    """Format an ISO 8601 timestamp as human-readable remaining time."""
    if not iso_str or iso_str == "N/A":
        return "N/A"
    try:
        dt = datetime.fromisoformat(str(iso_str).replace("Z", "+00:00"))
        diff = (dt - datetime.now(timezone.utc)).total_seconds()
        if diff <= 0:
            return "ready"
        d, rem = divmod(int(diff), 86400)
        h, m = divmod(rem // 60, 60)
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
    """OpenProxy account rotator and common selection engine."""

    def __init__(
        self,
        api_url: str | None = None,
        token: str | None = None,
        quota_threshold_percent: float = 95.0,
        cooldown_minutes: int = 15,
        script_path: Path | str | None = None,
    ) -> None:
        self.api_url, self.token = resolve_openproxy_credentials(
            config_url=api_url,
            config_token=token,
            script_path=script_path,
        )
        self.quota_threshold_percent = quota_threshold_percent
        self.cooldown_minutes = cooldown_minutes
        self._last_switched_id: int | str | None = None

    def _call(self, method: str, path: str, data: dict[str, Any] | None = None) -> Any:
        if not self.token:
            raise RuntimeError(
                "OpenProxy admin token not configured (config openproxy_token, "
                "env OPENPROXY_ADMIN_TOKEN, env SWITCH_ACCOUNT_TOKEN, or switch_account.sh script)."
            )
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "User-Agent": "hermes-antigravity-rotator/1.1",
        }
        body = json.dumps(data).encode("utf-8") if data is not None else None
        req = urllib.request.Request(f"{self.api_url}{path}", data=body, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw.strip() else {}

    def list_accounts(self, mark_active: bool = True) -> list[dict[str, Any]]:
        try:
            data = self._call("GET", "/accounts")
            if not isinstance(data, list):
                return []
            accounts = [a for a in data if a.get("provider_id") == "antigravity"]
            active_id = self.get_active_account_id() if mark_active else None
            for a in accounts:
                a["is_active"] = (a.get("id") == active_id)
                if "pools" not in a:
                    a["pools"] = pools_from_openproxy(a)
            return accounts
        except Exception as exc:
            logger.error("Failed to list accounts from OpenProxy (%s): %s", self.api_url, exc)
            return []

    def get_active_account_id(self) -> Any:
        if self._last_switched_id is not None:
            return self._last_switched_id
        for base in _config_bases():
            sf = base / ".gemini" / "antigravity-cli" / "active_account.json"
            if sf.is_file():
                try:
                    aid = json.loads(sf.read_text(encoding="utf-8")).get("account_id")
                    if aid is not None:
                        self._last_switched_id = int(aid) if str(aid).isdigit() else aid
                        return self._last_switched_id
                except Exception:
                    pass
        for base in _config_bases():
            af = base / ".gemini" / "google_accounts.json"
            if af.is_file():
                try:
                    act = json.loads(af.read_text(encoding="utf-8")).get("active")
                    if act:
                        for acc in self.list_accounts(mark_active=False):
                            if acc.get("email") == act or acc.get("label") == act:
                                aid = acc.get("id")
                                self._last_switched_id = int(aid) if str(aid).isdigit() else aid
                                return self._last_switched_id
                except Exception:
                    pass
        return None

    def refresh_quota(self, account_id: Any) -> dict[str, Any]:
        try:
            return self._call("POST", f"/accounts/{account_id}/refresh-quota")
        except Exception as exc:
            return {"error": str(exc)}

    def apply_account(self, account_id: Any) -> dict[str, Any]:
        res = self._call("POST", f"/accounts/{account_id}/apply-local-cli")
        if isinstance(res, dict) and not res.get("success"):
            raise RuntimeError(f"OpenProxy failed to apply account [{account_id}]: {res}")
        aid = int(account_id) if str(account_id).isdigit() else account_id
        self._last_switched_id = aid
        self._persist_active_state(aid)
        self._sync_isolated_workspace_tokens()
        return res

    def _persist_active_state(self, account_id: Any) -> None:
        for base in _config_bases():
            sf = base / ".gemini" / "antigravity-cli" / "active_account.json"
            try:
                sf.parent.mkdir(parents=True, exist_ok=True)
                sf.write_text(
                    json.dumps({"account_id": account_id, "switched_at": datetime.now(timezone.utc).isoformat()}),
                    encoding="utf-8",
                )
                break
            except Exception:
                pass

    def _resolve_real_token(self) -> Path | None:
        for base in _config_bases():
            td = base / ".gemini" / "antigravity-cli"
            for name in TOKEN_FILENAMES:
                if (td / name).is_file():
                    return td / name
        return None

    def _sync_isolated_workspace_tokens(self, tmp_root: Path | None = None) -> None:
        rt = self._resolve_real_token()
        if not rt:
            return
        for pattern in ISOLATED_HOME_GLOBS:
            for ad in (tmp_root or ISOLATED_TMP_ROOT).glob(pattern):
                td = ad / "home" / ".gemini" / "antigravity-cli"
                for name in TOKEN_FILENAMES:
                    it = td / name
                    if it.is_file() and not it.is_symlink():
                        with contextlib.suppress(Exception):
                            shutil.copy2(rt, it)

    def _has_remaining_quota(self, account: dict[str, Any], pool_key: str = "gemini") -> bool:
        pools = account.get("pools") or pools_from_openproxy(account)
        return is_usable_for_pool(pools, pool_key, self.quota_threshold_percent)

    @staticmethod
    def _quota_remaining(account: dict[str, Any]) -> tuple[int, int, int]:
        su, sl = account.get("quota_session_used") or 0, account.get("quota_session_limit") or 1000
        wu, wl = account.get("quota_weekly_used") or 0, account.get("quota_weekly_limit") or 1000
        return (max(0, sl - su), max(0, wl - wu), su)

    def _earliest_reset_account(self, accounts: list[dict[str, Any]], pool_key: str = "gemini") -> dict[str, Any] | None:
        earliest_acc, earliest_ts = None, None
        for a in accounts:
            pools = a.get("pools") or pools_from_openproxy(a)
            reset_at = pools.get(pool_key, {}).get("5h", {}).get("reset_at") or a.get("quota_session_reset_at")
            if reset_at and (earliest_ts is None or str(reset_at) < earliest_ts):
                earliest_ts, earliest_acc = str(reset_at), a
        return earliest_acc

    @staticmethod
    def _is_rate_limited(account: dict[str, Any]) -> bool:
        if account.get("in_cooldown"):
            return True
        lu = account.get("rate_limited_until") or account.get("cooldown_until")
        if not lu:
            return False
        try:
            return datetime.fromisoformat(str(lu).replace("Z", "+00:00")) > datetime.now(timezone.utc)
        except Exception:
            return False

    def find_best_candidates(
        self,
        exclude_id: Any = None,
        allow_exhausted: bool = False,
        model: str = "",
        pool_key: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        accounts = self.list_accounts()
        if not accounts:
            return []
        target_pool = pool_key or pool_for_model(model)
        cid = exclude_id if exclude_id is not None else self.get_active_account_id()

        candidates = []
        for a in accounts:
            aid = a.get("id")
            if cid is not None and str(aid) == str(cid) and len(accounts) > 1:
                continue
            if self._is_rate_limited(a):
                continue

            # Model support check (Point 6)
            if model and hasattr(self, "backend") and hasattr(self.backend, "list_models"):
                models = self.backend.list_models(str(aid))
                if not self.backend.account_supports_model(models, model):
                    continue

            pools = a.get("pools") or pools_from_openproxy(a)
            a["pools"] = pools
            if not is_usable_for_pool(pools, target_pool, self.quota_threshold_percent):
                continue
            candidates.append((calculate_score(pools, target_pool), a))

        if candidates:
            candidates.sort(key=lambda item: item[0], reverse=True)
            return [item[1] for item in candidates]
        if allow_exhausted:
            fb = self._earliest_reset_account(accounts, pool_key=target_pool)
            if fb is not None:
                return [fb]
        return []

    def find_best_candidate(
        self,
        exclude_id: Any = None,
        allow_exhausted: bool = False,
        model: str = "",
        pool_key: Optional[str] = None,
    ) -> dict[str, Any] | None:
        c = self.find_best_candidates(
            exclude_id=exclude_id,
            allow_exhausted=allow_exhausted,
            model=model,
            pool_key=pool_key,
        )
        return c[0] if c else None

    def _refresh_and_still_usable(self, account: dict[str, Any], pool_key: str = "gemini") -> dict[str, Any] | None:
        aid = account.get("id")
        refresh = self.refresh_quota(aid)
        if not isinstance(refresh, dict) or "error" in refresh:
            return account
        merged = {**account, **refresh}
        if "pools" in refresh and isinstance(refresh["pools"], dict) and refresh["pools"]:
            pools = refresh["pools"]
        elif "pools" in merged and isinstance(merged["pools"], dict) and merged["pools"] and not any(k.startswith("quota_session_") for k in refresh):
            pools = merged["pools"]
        else:
            pools = pools_from_openproxy(merged)
        merged["pools"] = pools
        if self._is_rate_limited(merged) or not is_usable_for_pool(pools, pool_key, self.quota_threshold_percent):
            return None
        return merged

    def rotate_to_next_account(
        self,
        current_account_id: Any = None,
        model: str = "",
        quota_exhausted: bool = False,
    ) -> tuple[dict[str, Any], str]:
        pool_key = pool_for_model(model)
        aid = current_account_id if current_account_id is not None else self.get_active_account_id()

        if quota_exhausted and aid is not None and hasattr(self, "backend") and hasattr(self.backend, "set_cooldown"):
            ap = None
            for a in self.list_accounts():
                if str(a.get("id")) == str(aid):
                    ap = a.get("pools")
                    break
            self.backend.set_cooldown(str(aid), pool_key, ap, fallback_minutes=self.cooldown_minutes)

        candidates = self.find_best_candidates(exclude_id=aid, model=model)
        if not candidates:
            self._raise_no_usable_account(aid, pool_key=pool_key)

        for cand in candidates:
            best = self._refresh_and_still_usable(cand, pool_key=pool_key)
            if best is None:
                continue
            bid = best.get("id")
            email = best.get("email") or best.get("label") or f"account-{bid}"
            pools = best.get("pools") or pools_from_openproxy(best)
            w5h = pools.get(pool_key, {}).get("5h", {})
            f5h = w5h.get("remaining_fraction")
            rem_5h = f"{int(f5h * 100)}%" if f5h is not None else "?"
            rst = format_remaining_time(w5h.get("reset_at") or best.get("quota_session_reset_at"))
            self.apply_account(bid)
            msg = f"Rotated Antigravity account to [{bid}] {email} (pool: {pool_key}, 5h: {rem_5h}, reset: {rst})"
            logger.info("[antigravity-rotator] %s", msg)
            return best, msg

        self._raise_no_usable_account(aid, pool_key=pool_key)

    def _raise_no_usable_account(self, exclude_id: Any, pool_key: str = "gemini") -> NoReturn:
        fallback = self.find_best_candidate(exclude_id=exclude_id, allow_exhausted=True, pool_key=pool_key)
        if fallback is None:
            raise RuntimeError("No Antigravity accounts configured in OpenProxy.")
        pools = fallback.get("pools") or pools_from_openproxy(fallback)
        iso = (
            pools.get(pool_key, {}).get("5h", {}).get("reset_at")
            or fallback.get("quota_session_reset_at")
            or "unknown"
        )
        raise RuntimeError(f"All Antigravity accounts exhausted; earliest session reset: {iso} ({format_remaining_time(iso)})")

    def format_status_table(self, model: str = "") -> str:
        accounts = self.list_accounts()
        backend_name = "Local Pool" if isinstance(self, LocalRotator) else "OpenProxy Pool"
        if not accounts:
            return f"No Antigravity accounts found in {backend_name}."
        pool_key = pool_for_model(model)
        aid = self.get_active_account_id()
        lines = [
            f"### 🔄 Google Antigravity Accounts ({backend_name} - {pool_key.upper()})",
            "",
            "| Active | ID | Account / Email | 5-Hour Quota | Session Reset | Weekly Quota | Weekly Reset |",
            "|:---:|:---:|:---|:---:|:---:|:---:|:---:|",
        ]
        for a in accounts:
            acc_id = a.get("id")
            act = "👉 **ACTIVE**" if str(acc_id) == str(aid) else ""
            lbl = a.get("email") or a.get("label") or "N/A"
            pools = a.get("pools") or pools_from_openproxy(a)
            pool = pools.get(pool_key, {})
            w5h = pool.get("5h", {})
            f5h = w5h.get("remaining_fraction")
            su = f"{int(f5h * 100)}%" if f5h is not None else f"{a.get('quota_session_used') or 0} / {a.get('quota_session_limit') or '∞'}"
            sr = format_remaining_time(w5h.get("reset_at") or a.get("quota_session_reset_at"))
            wwk = pool.get("weekly", {})
            fwk = wwk.get("remaining_fraction")
            wu = f"{int(fwk * 100)}%" if fwk is not None else f"{a.get('quota_weekly_used') or 0} / {a.get('quota_weekly_limit') or '∞'}"
            wr = format_remaining_time(wwk.get("reset_at") or a.get("quota_weekly_reset_at"))
            lines.append(f"| {act} | `{acc_id}` | {lbl} | {su} | {sr} | {wu} | {wr} |")
        return "\n".join(lines)


class LocalRotator(OpenProxyRotator):
    """Local file-based account rotator."""

    def __init__(
        self,
        state_file: Optional[str] = None,
        accounts_dir: Optional[str] = None,
        quota_threshold_percent: float = 95.0,
        cooldown_minutes: int = 15,
        backend: Any = None,
    ) -> None:
        super().__init__(quota_threshold_percent=quota_threshold_percent, cooldown_minutes=cooldown_minutes)
        self.backend = backend or LocalBackend(state_file=state_file, accounts_dir=accounts_dir)

    def list_accounts(self, mark_active: bool = True) -> list[dict[str, Any]]:
        accounts = self.backend.list_accounts()
        if not mark_active:
            for a in accounts:
                a["is_active"] = False
        return accounts

    def get_active_account_id(self) -> Any:
        return self.backend.get_active_account_id()

    def refresh_quota(self, account_id: Any) -> dict[str, Any]:
        return self.backend.refresh_quota(str(account_id))

    def apply_account(self, account_id: Any) -> dict[str, Any]:
        aid = str(account_id)
        res = self.backend.apply_account(aid)
        self._last_switched_id = aid
        return res

    def login(self, label: str, timeout: int = 300) -> dict[str, Any]:
        return self.backend.login(label, timeout=timeout)

    def remove_account(self, label: str) -> bool:
        return self.backend.remove_account(label)


AntigravityRotator = OpenProxyRotator


def resolve_backend_choice(
    config_choice: Optional[str] = None,
    openproxy_token: Optional[str] = None,
    script_path: Optional[Path | str] = None,
) -> str:
    choice = (config_choice or os.environ.get("ANTIGRAVITY_ROTATOR_BACKEND", "auto")).lower().strip()
    if choice in ("openproxy", "local"):
        return choice
    _, resolved_token = resolve_openproxy_credentials(
        config_token=openproxy_token,
        script_path=script_path,
    )
    return "openproxy" if resolved_token else "local"


def create_rotator(
    backend_choice: Optional[str] = None,
    openproxy_url: Optional[str] = None,
    openproxy_token: Optional[str] = None,
    quota_threshold_percent: float = 95.0,
    cooldown_minutes: int = 15,
    state_file: Optional[str] = None,
    accounts_dir: Optional[str] = None,
    script_path: Optional[Path | str] = None,
) -> OpenProxyRotator:
    choice = resolve_backend_choice(
        config_choice=backend_choice,
        openproxy_token=openproxy_token,
        script_path=script_path,
    )
    if choice == "openproxy":
        return OpenProxyRotator(
            api_url=openproxy_url,
            token=openproxy_token,
            quota_threshold_percent=quota_threshold_percent,
            cooldown_minutes=cooldown_minutes,
            script_path=script_path,
        )
    return LocalRotator(
        state_file=state_file,
        accounts_dir=accounts_dir,
        quota_threshold_percent=quota_threshold_percent,
        cooldown_minutes=cooldown_minutes,
    )

