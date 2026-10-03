"""Local file-based backend for antigravity-rotator.

Manages Antigravity accounts without OpenProxy, storing state in
~/.hermes/antigravity-accounts.json and per-account HOME directories
under ~/.agy-accounts/<label>/.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import logging
import os
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .constants import (
        GOOGLE_ACCOUNTS_FILE,
        ISOLATED_HOME_GLOBS,
        ISOLATED_TMP_ROOT,
        TOKEN_FILENAMES,
    )
    from .local_login import login_account, resolve_token_file
    from .local_probe import (
        account_supports_model,
        fetch_usage_for_home,
        list_models_for_home,
    )
    from .local_registry import (
        file_lock,
        find_account,
        load_state,
        register_account_entry,
        remove_account_entry,
        sanitize_folder_name,
        save_state,
        update_account_cooldown,
    )
except ImportError:
    from constants import (
        GOOGLE_ACCOUNTS_FILE,
        ISOLATED_HOME_GLOBS,
        ISOLATED_TMP_ROOT,
        TOKEN_FILENAMES,
    )
    from local_login import login_account, resolve_token_file
    from local_probe import (
        account_supports_model,
        fetch_usage_for_home,
        list_models_for_home,
    )
    from local_registry import (
        file_lock,
        find_account,
        load_state,
        register_account_entry,
        remove_account_entry,
        sanitize_folder_name,
        save_state,
        update_account_cooldown,
    )

logger = logging.getLogger(__name__)


def _copy_token_atomic(src: str, dst: str) -> None:
    """Copy a token file atomically with 0600 permissions. Never log content."""
    content = Path(src).read_bytes()
    dst_dir = os.path.dirname(dst)
    if dst_dir:
        os.makedirs(dst_dir, mode=0o700, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dst_dir or None, prefix=".token_", suffix=".tmp")
    try:
        os.write(fd, content)
        os.close(fd)
        os.chmod(tmp, 0o600)
        os.replace(tmp, dst)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _copy_if_exists(src: str, dst: str) -> None:
    """Copy an optional metadata file atomically if source exists."""
    if os.path.isfile(src):
        _copy_token_atomic(src, dst)


class LocalBackend:
    """File-based account backend managing local agy installations."""

    def __init__(
        self,
        state_file: Optional[str] = None,
        accounts_dir: Optional[str] = None,
    ) -> None:
        self._state_file = state_file or os.environ.get(
            "ANTIGRAVITY_ACCOUNTS_FILE",
            os.path.expanduser("~/.hermes/antigravity-accounts.json"),
        )
        self._accounts_dir = accounts_dir or os.environ.get(
            "ANTIGRAVITY_ACCOUNTS_DIR",
            os.path.expanduser("~/.agy-accounts"),
        )
        self._lock_path = self._state_file + ".lock"
        self._cache: Dict[str, Any] = {}

    # -- Backend Interface ---------------------------------------------------

    def list_accounts(self) -> List[Dict[str, Any]]:
        """Return normalized account dicts with quota pools (parallel probe max 4 workers)."""
        state = load_state(self._state_file)
        active = state.get("active_account")
        enabled_accounts = [a for a in state.get("accounts", []) if a.get("enabled", True)]
        if not enabled_accounts:
            return []

        def _get_pools(home: str) -> Dict[str, Dict[str, Any]]:
            return fetch_usage_for_home(home, cache=self._cache, cache_ttl=60.0)

        home_to_pools: Dict[str, Dict[str, Any]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(enabled_accounts))) as executor:
            future_to_home = {
                executor.submit(_get_pools, str(a.get("home_dir", ""))): str(a.get("home_dir", ""))
                for a in enabled_accounts
            }
            for future in concurrent.futures.as_completed(future_to_home):
                home = future_to_home[future]
                try:
                    home_to_pools[home] = future.result()
                except Exception as exc:
                    logger.debug("Quota probe failed for %s: %s", home, exc)
                    home_to_pools[home] = {}

        now = datetime.now(timezone.utc)
        result: List[Dict[str, Any]] = []
        for acct in enabled_accounts:
            label = str(acct.get("label", ""))
            home_dir = str(acct.get("home_dir", ""))
            cooldown = acct.get("cooldown_until")

            in_cooldown = False
            if cooldown:
                try:
                    cd_dt = datetime.fromisoformat(str(cooldown).replace("Z", "+00:00"))
                    in_cooldown = cd_dt > now
                except (ValueError, AttributeError):
                    pass

            pools = home_to_pools.get(home_dir, {})

            result.append({
                "id": label,
                "label": label,
                "email": acct.get("email", ""),
                "home_dir": home_dir,
                "enabled": True,
                "is_active": label == active,
                "cooldown_until": cooldown if in_cooldown else None,
                "in_cooldown": in_cooldown,
                "last_used": acct.get("last_used"),
                "pools": pools,
            })
        return result

    def refresh_quota(self, account_id: str) -> Dict[str, Any]:
        """Force refresh quota telemetry for a specific account."""
        state = load_state(self._state_file)
        acct = find_account(state, str(account_id))
        if not acct:
            return {"error": f"Account '{account_id}' not found"}
        home_dir = acct.get("home_dir", "")
        self._cache.pop(f"usage:{home_dir}", None)
        pools = fetch_usage_for_home(home_dir, cache=self._cache, cache_ttl=0.0)
        return {"pools": pools}

    def apply_account(self, account_id: str) -> Dict[str, Any]:
        """Switch active account: save-back current, copy target credentials."""
        target_label = str(account_id)
        with file_lock(self._lock_path):
            state = load_state(self._state_file)
            target = find_account(state, target_label)
            if not target:
                raise RuntimeError(f"Account '{target_label}' not found in registry")

            current_active = state.get("active_account")
            if current_active is None:
                # Backup pre-existing host credentials before first apply
                self._backup_host_credentials_if_needed(state)
            elif current_active != target_label:
                current_acct = find_account(state, current_active)
                if current_acct:
                    self._save_back_credentials(current_acct.get("home_dir", ""))

            # Copy target credentials to primary real path
            target_home = target.get("home_dir", "")
            self._install_credentials_from(target_home)

            # Update state
            now_iso = datetime.now(timezone.utc).isoformat()
            state["active_account"] = target_label
            for a in state.get("accounts", []):
                if a.get("label") == target_label:
                    a["last_used"] = now_iso
                    break
            save_state(self._state_file, state)

        # Synchronize isolated workspace environments
        self._sync_isolated_workspace_tokens()
        return {"success": True, "account_id": target_label}

    def get_active_account_id(self) -> Optional[str]:
        """Return label of currently active account."""
        state = load_state(self._state_file)
        return state.get("active_account")

    def set_cooldown(
        self,
        account_id: str,
        pool_key: str,
        pools: Optional[Dict[str, Dict[str, Any]]] = None,
        fallback_minutes: float = 15.0,
    ) -> str:
        """Set cooldown on an account to pool reset time, or fallback minutes."""
        now = datetime.now(timezone.utc)
        earliest_reset: Optional[datetime] = None

        if pools:
            pool = pools.get(pool_key, {})
            for wkey in ("5h", "weekly"):
                rst = pool.get(wkey, {}).get("reset_at")
                if not rst:
                    continue
                try:
                    dt_obj = datetime.fromisoformat(str(rst).replace("Z", "+00:00"))
                    if dt_obj > now:
                        if earliest_reset is None or dt_obj < earliest_reset:
                            earliest_reset = dt_obj
                except (ValueError, AttributeError):
                    pass

        if earliest_reset is not None:
            cooldown_dt = earliest_reset
        else:
            cooldown_dt = now + timedelta(minutes=fallback_minutes)

        cooldown_iso = cooldown_dt.isoformat()
        update_account_cooldown(
            state_file=self._state_file,
            lock_path=self._lock_path,
            label=str(account_id),
            cooldown_until_iso=cooldown_iso,
        )
        return cooldown_iso

    # -- Onboarding and Administration ---------------------------------------

    def login(self, label: str, timeout: int = 300) -> Dict[str, Any]:
        """Interactive account login into dedicated home."""
        return login_account(
            label=label,
            accounts_dir=self._accounts_dir,
            state_file=self._state_file,
            lock_path=self._lock_path,
            timeout=timeout,
        )

    def register_account(
        self, label: str, home_dir: str, email: str = ""
    ) -> Dict[str, Any]:
        """Register an existing home directory under a label."""
        return register_account_entry(
            state_file=self._state_file,
            lock_path=self._lock_path,
            label=sanitize_folder_name(label),
            home_dir=home_dir,
            email=email,
        )

    def remove_account(self, label: str) -> bool:
        """Remove account from registry (files remain intact)."""
        return remove_account_entry(
            state_file=self._state_file,
            lock_path=self._lock_path,
            label=label,
        )

    def list_models(self, account_id: str) -> Optional[List[str]]:
        """List models supported by an account."""
        state = load_state(self._state_file)
        acct = find_account(state, str(account_id))
        if not acct:
            return None
        return list_models_for_home(acct.get("home_dir", ""), cache=self._cache)

    account_supports_model = staticmethod(account_supports_model)

    # -- Credential Management -----------------------------------------------

    def _backup_host_credentials_if_needed(self, state: Dict[str, Any]) -> None:
        """Backup existing host credentials before first apply when active_account is None."""
        if state.get("active_account") is not None:
            return

        real_token = resolve_token_file(os.path.expanduser("~"))
        real_ga = os.path.expanduser(f"~/.gemini/{GOOGLE_ACCOUNTS_FILE}")
        has_ga = os.path.isfile(real_ga)

        if not real_token and not has_ga:
            return

        ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(self._accounts_dir, f".host-backup-{ts_str}")
        backup_cli = os.path.join(backup_dir, ".gemini", "antigravity-cli")
        os.makedirs(backup_cli, mode=0o700, exist_ok=True)
        os.chmod(backup_dir, 0o700)
        os.chmod(backup_cli, 0o700)

        if real_token:
            dst_token = os.path.join(backup_cli, os.path.basename(real_token))
            _copy_token_atomic(real_token, dst_token)

        if has_ga:
            dst_ga = os.path.join(backup_dir, ".gemini", GOOGLE_ACCOUNTS_FILE)
            _copy_token_atomic(real_ga, dst_ga)

        logger.info("[antigravity-rotator] Backed up pre-existing host credentials to %s", backup_dir)

    def _save_back_credentials(self, home_dir: str) -> None:
        """Copy active token and google_accounts.json back to account HOME."""
        if not home_dir:
            return
        real_token = resolve_token_file(os.path.expanduser("~"))
        if real_token:
            dst_token_dir = os.path.join(home_dir, ".gemini", "antigravity-cli")
            os.makedirs(dst_token_dir, mode=0o700, exist_ok=True)
            # Use existing token filename in account's home if one exists
            existing_in_home = resolve_token_file(home_dir)
            if existing_in_home:
                dst_token = existing_in_home
            else:
                dst_token = os.path.join(dst_token_dir, os.path.basename(real_token))
            _copy_token_atomic(real_token, dst_token)

        real_ga = os.path.expanduser(f"~/.gemini/{GOOGLE_ACCOUNTS_FILE}")
        dst_ga = os.path.join(home_dir, ".gemini", GOOGLE_ACCOUNTS_FILE)
        _copy_if_exists(real_ga, dst_ga)

    def _install_credentials_from(self, home_dir: str) -> None:
        """Install account credentials to primary agy environment."""
        src_token = resolve_token_file(home_dir)
        if not src_token:
            raise RuntimeError(
                f"No token file found under {home_dir}/.gemini/antigravity-cli/"
            )

        primary_dir = os.path.expanduser("~/.gemini/antigravity-cli")
        os.makedirs(primary_dir, mode=0o700, exist_ok=True)

        existing_primary = resolve_token_file(os.path.expanduser("~"))
        if existing_primary:
            dst_token = existing_primary
        else:
            dst_token = os.path.join(primary_dir, os.path.basename(src_token))

        _copy_token_atomic(src_token, dst_token)

        src_ga = os.path.join(home_dir, ".gemini", GOOGLE_ACCOUNTS_FILE)
        dst_ga = os.path.expanduser(f"~/.gemini/{GOOGLE_ACCOUNTS_FILE}")
        _copy_if_exists(src_ga, dst_ga)

    def _sync_isolated_workspace_tokens(
        self, tmp_root: Optional[Path] = None
    ) -> None:
        """Update tokens in active isolated workspaces created by directsdk provider."""
        real_token_str = resolve_token_file(os.path.expanduser("~"))
        if not real_token_str:
            return
        real_token = Path(real_token_str)
        search_root = tmp_root or ISOLATED_TMP_ROOT
        try:
            for pattern in ISOLATED_HOME_GLOBS:
                for agy_dir in search_root.glob(pattern):
                    token_dir = agy_dir / "home" / ".gemini" / "antigravity-cli"
                    for name in TOKEN_FILENAMES:
                        iso_token = token_dir / name
                        if iso_token.is_file() and not iso_token.is_symlink():
                            with contextlib.suppress(Exception):
                                shutil.copy2(real_token, iso_token)
        except Exception as exc:
            logger.debug("Isolated workspace token sync ignored: %s", exc)
