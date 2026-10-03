"""Tests verifying orchestrator review fixes (points 1 to 13)."""

from __future__ import annotations

import os
import stat
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from backend_local import LocalBackend
from backend_openproxy import (
    _read_credentials_from_script,
    resolve_openproxy_credentials,
    resolve_switch_script_path,
)
from local_probe import nobrowser_shim_dir
from local_registry import file_lock
from rotator import LocalRotator, OpenProxyRotator, resolve_backend_choice


# ---------------------------------------------------------------------------
# Point 1: Host credentials backup on first apply
# ---------------------------------------------------------------------------

def test_host_credentials_backup_on_first_apply(tmp_path, monkeypatch):
    """Before first apply, existing host credentials are backed up to .host-backup-<timestamp>."""
    user_home = tmp_path / "user_home"
    primary_cli = user_home / ".gemini" / "antigravity-cli"
    primary_cli.mkdir(parents=True)
    (primary_cli / "antigravity-oauth-token").write_bytes(b"HOST_ORIGINAL_TOKEN")
    (user_home / ".gemini" / "google_accounts.json").write_text('{"active": "host@example.com"}', encoding="utf-8")

    monkeypatch.setenv("HOME", str(user_home))

    accounts_dir = tmp_path / "accounts"
    acc_home = accounts_dir / "account-1"
    (acc_home / ".gemini" / "antigravity-cli").mkdir(parents=True)
    (acc_home / ".gemini" / "antigravity-cli" / "jetski-standalone-oauth-token").write_bytes(b"NEW_TOKEN_1")

    state_file = tmp_path / "accounts.json"
    backend = LocalBackend(state_file=str(state_file), accounts_dir=str(accounts_dir))
    backend.register_account("account-1", str(acc_home), "acc1@example.com")

    # First apply when active_account is None
    backend.apply_account("account-1")

    # Find backup directories matching .host-backup-*
    backup_dirs = list(accounts_dir.glob(".host-backup-*"))
    assert len(backup_dirs) == 1
    backup_dir = backup_dirs[0]

    # Verify directory permissions are 0700
    assert stat.S_IMODE(os.stat(backup_dir).st_mode) == 0o700

    # Verify token and google_accounts.json are backed up with 0600
    backed_token = backup_dir / ".gemini" / "antigravity-cli" / "antigravity-oauth-token"
    assert backed_token.is_file()
    assert stat.S_IMODE(os.stat(backed_token).st_mode) == 0o600
    assert backed_token.read_bytes() == b"HOST_ORIGINAL_TOKEN"

    backed_ga = backup_dir / ".gemini" / "google_accounts.json"
    assert backed_ga.is_file()
    assert backed_ga.read_text(encoding="utf-8") == '{"active": "host@example.com"}'

    # Subsequent apply with active_account already set should NOT create another backup
    backend.apply_account("account-1")
    assert len(list(accounts_dir.glob(".host-backup-*"))) == 1


# ---------------------------------------------------------------------------
# Point 2: Cross-platform locking without module-level fcntl
# ---------------------------------------------------------------------------

def test_windows_locking_without_fcntl(monkeypatch, tmp_path):
    """Module can be imported and locked on Windows simulation without fcntl."""
    # Simulate environment where fcntl is unavailable
    monkeypatch.setitem(sys.modules, "fcntl", None)
    monkeypatch.setattr(os, "name", "nt")

    mock_msvcrt = types.ModuleType("msvcrt")
    mock_msvcrt.LK_NBLCK = 1
    mock_msvcrt.LK_UNLCK = 2
    mock_msvcrt.locking = MagicMock()
    monkeypatch.setitem(sys.modules, "msvcrt", mock_msvcrt)

    lock_file = tmp_path / "test.lock"
    with file_lock(str(lock_file), timeout=1.0):
        pass

    assert mock_msvcrt.locking.call_count == 2
    mock_msvcrt.locking.assert_any_call(mock_msvcrt.locking.call_args_list[0][0][0], mock_msvcrt.LK_NBLCK, 1)
    mock_msvcrt.locking.assert_any_call(mock_msvcrt.locking.call_args_list[1][0][0], mock_msvcrt.LK_UNLCK, 1)


# ---------------------------------------------------------------------------
# Point 3: Shim script exits 1 and rewrites differing content
# ---------------------------------------------------------------------------

def test_nobrowser_shim_exits_1_and_rewrites(tmp_path):
    """nobrowser shims write to stderr, exit 1, and rewrite stale content."""
    cache_dir = tmp_path / "cache"
    shim_dir = cache_dir / "nobrowser"
    shim_dir.mkdir(parents=True)

    # Put a stale shim that exits 0
    stale_file = shim_dir / "open"
    stale_file.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

    res_dir = nobrowser_shim_dir(str(cache_dir))
    assert res_dir == str(shim_dir)

    # File should have been overwritten with exit 1 script
    content = stale_file.read_text(encoding="utf-8")
    assert "exit 1" in content
    assert "blocked browser launch (antigravity-rotator probe)" in content


# ---------------------------------------------------------------------------
# Point 4: _has_remaining_quota accepts pool_key
# ---------------------------------------------------------------------------

def test_has_remaining_quota_accepts_pool_key():
    """OpenProxyRotator._has_remaining_quota evaluates the specified pool_key."""
    rotator = OpenProxyRotator(api_url="http://mock", token="mock")
    account = {
        "pools": {
            "gemini": {"5h": {"remaining_fraction": 0.80, "reset_at": None}},
            "claude_gpt": {"5h": {"remaining_fraction": 0.01, "reset_at": None}},  # below 95% threshold (0.05)
        }
    }
    assert rotator._has_remaining_quota(account, pool_key="gemini") is True
    assert rotator._has_remaining_quota(account, pool_key="claude_gpt") is False


# ---------------------------------------------------------------------------
# Point 5: LocalRotator _refresh_and_still_usable preserves pools dict
# ---------------------------------------------------------------------------

def test_local_rotator_refresh_and_still_usable_preserves_pools():
    """LocalRotator uses refresh['pools'] and excludes candidate if pool is exhausted."""
    mock_backend = MagicMock()
    rotator = LocalRotator(backend=mock_backend)

    cand = {
        "id": "acc-local",
        "pools": {"claude_gpt": {"5h": {"remaining_fraction": 0.80, "reset_at": None}}},
    }

    # Live refresh reveals Claude is exhausted
    mock_backend.refresh_quota.return_value = {
        "pools": {"claude_gpt": {"5h": {"remaining_fraction": 0.0, "reset_at": None}}}
    }

    res = rotator._refresh_and_still_usable(cand, pool_key="claude_gpt")
    assert res is None  # Candidate excluded post-refresh


# ---------------------------------------------------------------------------
# Point 6: account_supports_model filtering in candidate selection
# ---------------------------------------------------------------------------

def test_account_supports_model_candidate_filtering():
    """Candidate selection excludes accounts whose known supported models do not include the model."""
    mock_backend = MagicMock()
    rotator = LocalRotator(backend=mock_backend)

    accounts = [
        {"id": "acc-gemini", "pools": {"claude_gpt": {"5h": {"remaining_fraction": 0.80, "reset_at": None}}}},
        {"id": "acc-claude", "pools": {"claude_gpt": {"5h": {"remaining_fraction": 0.80, "reset_at": None}}}},
        {"id": "acc-unknown", "pools": {"claude_gpt": {"5h": {"remaining_fraction": 0.80, "reset_at": None}}}},
    ]
    mock_backend.list_accounts.return_value = accounts

    def fake_list_models(aid):
        if aid == "acc-gemini":
            return ["gemini-2.0-flash"]
        if aid == "acc-claude":
            return ["claude-3-5-sonnet"]
        return None  # acc-unknown: models unknown

    mock_backend.list_models.side_effect = fake_list_models
    mock_backend.account_supports_model.side_effect = lambda models, m: (
        True if models is None else any(m in mod for mod in models)
    )

    candidates = rotator.find_best_candidates(model="claude-3-5-sonnet")
    candidate_ids = [c["id"] for c in candidates]

    # acc-gemini is excluded; acc-claude and acc-unknown (unknown permitted) are present
    assert "acc-gemini" not in candidate_ids
    assert "acc-claude" in candidate_ids
    assert "acc-unknown" in candidate_ids


# ---------------------------------------------------------------------------
# Point 7: Cooldown only set when quota_exhausted is True
# ---------------------------------------------------------------------------

def test_rotate_to_next_account_cooldown_only_on_quota_exhausted():
    """Cooldown is only applied to the outgoing account when quota_exhausted=True."""
    mock_backend = MagicMock()
    rotator = LocalRotator(backend=mock_backend)

    accounts = [
        {"id": "curr", "pools": {"gemini": {"5h": {"remaining_fraction": 0.10, "reset_at": None}}}},
        {"id": "next", "pools": {"gemini": {"5h": {"remaining_fraction": 0.90, "reset_at": None}}}},
    ]
    mock_backend.list_accounts.return_value = accounts
    mock_backend.get_active_account_id.return_value = "curr"

    # Manual switch/rotate (quota_exhausted=False) -> no cooldown
    rotator.rotate_to_next_account(current_account_id="curr", quota_exhausted=False)
    mock_backend.set_cooldown.assert_not_called()

    # Error failover rotation (quota_exhausted=True) -> sets cooldown
    rotator.rotate_to_next_account(current_account_id="curr", quota_exhausted=True)
    mock_backend.set_cooldown.assert_called_once()


# ---------------------------------------------------------------------------
# Point 12: _save_back_credentials preserves account's existing token filename
# ---------------------------------------------------------------------------

def test_save_back_preserves_existing_token_filename(tmp_path, monkeypatch):
    """Save-back writes to the token filename already present in account HOME."""
    user_home = tmp_path / "user_home"
    primary_cli = user_home / ".gemini" / "antigravity-cli"
    primary_cli.mkdir(parents=True)
    # Primary environment has jetski filename
    (primary_cli / "jetski-standalone-oauth-token").write_bytes(b"NEW_REFRESHED_BYTES")
    monkeypatch.setenv("HOME", str(user_home))

    # Account HOME was onboarded with legacy antigravity-oauth-token filename
    acc_home = tmp_path / "accounts" / "my_acc"
    acc_cli = acc_home / ".gemini" / "antigravity-cli"
    acc_cli.mkdir(parents=True)
    (acc_cli / "antigravity-oauth-token").write_bytes(b"OLD_BYTES")

    backend = LocalBackend(state_file=str(tmp_path / "accounts.json"), accounts_dir=str(tmp_path / "accounts"))
    backend._save_back_credentials(str(acc_home))

    # Must update antigravity-oauth-token and NOT create jetski-standalone-oauth-token
    assert (acc_cli / "antigravity-oauth-token").read_bytes() == b"NEW_REFRESHED_BYTES"
    assert not (acc_cli / "jetski-standalone-oauth-token").exists()


# ---------------------------------------------------------------------------
# Point 13: LocalBackend list_accounts parallel probes
# ---------------------------------------------------------------------------

def test_local_backend_parallel_probes(tmp_path):
    """LocalBackend probes accounts in parallel using ThreadPoolExecutor."""
    state_file = tmp_path / "accounts.json"
    accounts_dir = tmp_path / "accounts"
    backend = LocalBackend(state_file=str(state_file), accounts_dir=str(accounts_dir))

    for i in range(3):
        home = accounts_dir / f"acc-{i}"
        home.mkdir(parents=True)
        backend.register_account(f"acc-{i}", str(home), f"acc{i}@example.com")

    with patch("backend_local.fetch_usage_for_home", return_value={"gemini": {"5h": {"remaining_fraction": 0.7, "reset_at": None}}}) as mock_fetch:
        accounts = backend.list_accounts()
        assert len(accounts) == 3
        assert mock_fetch.call_count == 3
        for a in accounts:
            assert a["pools"]["gemini"]["5h"]["remaining_fraction"] == 0.7


# ---------------------------------------------------------------------------
# Revision 2: Switch script candidate resolution & credential precedence
# ---------------------------------------------------------------------------

def test_switch_script_candidate_home_agents_detected(tmp_path, monkeypatch):
    """Candidate ~/.agents/switch_account.sh is detected when HOME is tmp_path."""
    user_home = tmp_path / "user_home"
    agents_dir = user_home / ".agents"
    agents_dir.mkdir(parents=True)
    script = agents_dir / "switch_account.sh"
    script.write_text(
        'API_URL="https://home-agents.example/admin/api"\nTOKEN="tok_home_agents_123"\n',
        encoding="utf-8",
    )

    monkeypatch.setenv("HOME", str(user_home))
    monkeypatch.delenv("ANTIGRAVITY_ROTATOR_SWITCH_SCRIPT", raising=False)
    monkeypatch.delenv("OPENPROXY_ADMIN_URL", raising=False)
    monkeypatch.delenv("OPENPROXY_ADMIN_TOKEN", raising=False)
    monkeypatch.delenv("SWITCH_ACCOUNT_API_URL", raising=False)
    monkeypatch.delenv("SWITCH_ACCOUNT_TOKEN", raising=False)

    resolved_path = resolve_switch_script_path()
    assert resolved_path == script

    url, token = _read_credentials_from_script()
    assert url == "https://home-agents.example/admin/api"
    assert token == "tok_home_agents_123"

    # OpenProxyRotator picks up the script credentials
    rotator = OpenProxyRotator()
    assert rotator.api_url == "https://home-agents.example/admin/api"
    assert rotator.token == "tok_home_agents_123"

    # Auto mode selects openproxy
    assert resolve_backend_choice("auto") == "openproxy"


def test_switch_script_env_var_custom_path(tmp_path, monkeypatch):
    """ANTIGRAVITY_ROTATOR_SWITCH_SCRIPT is preferred over other candidates."""
    custom_script = tmp_path / "custom_switch.sh"
    custom_script.write_text(
        'export SWITCH_ACCOUNT_API_URL="https://custom.example/api"\n'
        'export SWITCH_ACCOUNT_TOKEN="tok_custom_999"\n',
        encoding="utf-8",
    )

    user_home = tmp_path / "home_with_agents"
    agents_dir = user_home / ".agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "switch_account.sh").write_text(
        'API_URL="https://should-not-pick.example/api"\nTOKEN="tok_ignored"\n',
        encoding="utf-8",
    )

    monkeypatch.setenv("HOME", str(user_home))
    monkeypatch.setenv("ANTIGRAVITY_ROTATOR_SWITCH_SCRIPT", str(custom_script))
    monkeypatch.delenv("OPENPROXY_ADMIN_URL", raising=False)
    monkeypatch.delenv("OPENPROXY_ADMIN_TOKEN", raising=False)
    monkeypatch.delenv("SWITCH_ACCOUNT_API_URL", raising=False)
    monkeypatch.delenv("SWITCH_ACCOUNT_TOKEN", raising=False)

    resolved_path = resolve_switch_script_path()
    assert resolved_path == custom_script

    url, token = resolve_openproxy_credentials()
    assert url == "https://custom.example/api"
    assert token == "tok_custom_999"


def test_switch_account_env_vars_recognized(monkeypatch):
    """SWITCH_ACCOUNT_API_URL and SWITCH_ACCOUNT_TOKEN env vars are recognized."""
    monkeypatch.setenv("SWITCH_ACCOUNT_API_URL", "https://switch-env.example/api")
    monkeypatch.setenv("SWITCH_ACCOUNT_TOKEN", "tok_switch_env_456")
    monkeypatch.delenv("OPENPROXY_ADMIN_URL", raising=False)
    monkeypatch.delenv("OPENPROXY_ADMIN_TOKEN", raising=False)
    monkeypatch.delenv("ANTIGRAVITY_ROTATOR_SWITCH_SCRIPT", raising=False)

    url, token = resolve_openproxy_credentials()
    assert url == "https://switch-env.example/api"
    assert token == "tok_switch_env_456"

    rotator = OpenProxyRotator()
    assert rotator.api_url == "https://switch-env.example/api"
    assert rotator.token == "tok_switch_env_456"


def test_openproxy_credential_precedence_order(tmp_path, monkeypatch):
    """Verify precedence: config > OPENPROXY_* > SWITCH_ACCOUNT_* > script."""
    script = tmp_path / "script.sh"
    script.write_text(
        'API_URL="http://from-script"\nTOKEN="tok_script"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("ANTIGRAVITY_ROTATOR_SWITCH_SCRIPT", str(script))
    monkeypatch.delenv("OPENPROXY_ADMIN_URL", raising=False)
    monkeypatch.delenv("OPENPROXY_ADMIN_TOKEN", raising=False)
    monkeypatch.delenv("SWITCH_ACCOUNT_API_URL", raising=False)
    monkeypatch.delenv("SWITCH_ACCOUNT_TOKEN", raising=False)

    # 1. Only script
    url, token = resolve_openproxy_credentials()
    assert url == "http://from-script"
    assert token == "tok_script"

    # 2. SWITCH_ACCOUNT_* overrides script
    monkeypatch.setenv("SWITCH_ACCOUNT_API_URL", "http://from-switch-env")
    monkeypatch.setenv("SWITCH_ACCOUNT_TOKEN", "tok_switch_env")
    url, token = resolve_openproxy_credentials()
    assert url == "http://from-switch-env"
    assert token == "tok_switch_env"

    # 3. OPENPROXY_* overrides SWITCH_ACCOUNT_*
    monkeypatch.setenv("OPENPROXY_ADMIN_URL", "http://from-openproxy-env")
    monkeypatch.setenv("OPENPROXY_ADMIN_TOKEN", "tok_openproxy_env")
    url, token = resolve_openproxy_credentials()
    assert url == "http://from-openproxy-env"
    assert token == "tok_openproxy_env"

    # 4. Config overrides all
    url, token = resolve_openproxy_credentials(
        config_url="http://from-config",
        config_token="tok_config",
    )
    assert url == "http://from-config"
    assert token == "tok_config"


def test_resolve_backend_choice_auto_sources(tmp_path, monkeypatch):
    """Auto mode chooses openproxy for each token source, and local when no token exists."""
    user_home = tmp_path / "empty_home"
    user_home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(user_home))
    monkeypatch.delenv("ANTIGRAVITY_ROTATOR_SWITCH_SCRIPT", raising=False)
    monkeypatch.delenv("OPENPROXY_ADMIN_TOKEN", raising=False)
    monkeypatch.delenv("SWITCH_ACCOUNT_TOKEN", raising=False)

    # Simulate environment where no script files exist
    with patch("backend_openproxy.get_switch_script_candidates", return_value=[]):
        # 1. No token anywhere -> chooses local
        assert resolve_backend_choice("auto") == "local"

        # 2. Config token -> chooses openproxy
        assert resolve_backend_choice("auto", openproxy_token="conf_tok") == "openproxy"

        # 3. OPENPROXY_ADMIN_TOKEN in env -> chooses openproxy
        monkeypatch.setenv("OPENPROXY_ADMIN_TOKEN", "op_tok")
        assert resolve_backend_choice("auto") == "openproxy"
        monkeypatch.delenv("OPENPROXY_ADMIN_TOKEN")

        # 4. SWITCH_ACCOUNT_TOKEN in env -> chooses openproxy
        monkeypatch.setenv("SWITCH_ACCOUNT_TOKEN", "sw_tok")
        assert resolve_backend_choice("auto") == "openproxy"
        monkeypatch.delenv("SWITCH_ACCOUNT_TOKEN")

    # 5. Script candidate with token -> chooses openproxy
    script = user_home / ".agents" / "switch_account.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text('TOKEN="tok_from_script"\n', encoding="utf-8")
    assert resolve_backend_choice("auto") == "openproxy"

    # 6. Explicit choice overrides auto token detection
    assert resolve_backend_choice("local") == "local"
    with patch("backend_openproxy.get_switch_script_candidates", return_value=[]):
        assert resolve_backend_choice("openproxy") == "openproxy"

