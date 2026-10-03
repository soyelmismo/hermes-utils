"""Tests for LocalBackend, registry persistence, probe environment, credential swapping, and onboarding."""

from __future__ import annotations

import json
import os
import stat
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from backend_local import LocalBackend
from local_login import login_account
from local_probe import nobrowser_shim_dir, probe_env
from local_registry import (
    file_lock,
    load_state,
    register_account_entry,
    save_state,
)


# ---------------------------------------------------------------------------
# 4. Atomic Registry / 0600 / Fail-open on corrupt JSON
# ---------------------------------------------------------------------------

def test_registry_atomic_0600_permissions_and_fail_open(tmp_path):
    """Registry writes atomically with 0600, dir 0700, and fails open on corruption."""
    state_file = tmp_path / "sub_dir" / "accounts.json"

    # Save initial state
    data = {"accounts": [{"label": "test1", "enabled": True}], "active_account": "test1"}
    save_state(str(state_file), data)

    # Check directory permissions (0700)
    dir_stat = os.stat(state_file.parent)
    assert stat.S_IMODE(dir_stat.st_mode) == 0o700

    # Check file permissions (0600)
    file_stat = os.stat(state_file)
    assert stat.S_IMODE(file_stat.st_mode) == 0o600

    # Verify loaded state matches
    loaded = load_state(str(state_file))
    assert loaded["active_account"] == "test1"
    assert len(loaded["accounts"]) == 1

    # Corrupt the JSON file content
    state_file.write_text("{ corrupt json: not valid !!", encoding="utf-8")

    # Must fail-open and return empty state without crashing
    fail_open = load_state(str(state_file))
    assert fail_open == {"accounts": [], "active_account": None}


# ---------------------------------------------------------------------------
# 5. Probe environment (SSH_CONNECTION, BROWSER, private shim dir, no CONFIG_DIR)
# ---------------------------------------------------------------------------

def test_probe_environment_security_and_sentinels(tmp_path, monkeypatch):
    """Probe env forces SSH_CONNECTION sentinel, blocks browser, shims open/xdg-open 0700, removes CONFIG_DIR."""
    home_dir = str(tmp_path / "account_home")
    cache_dir = str(tmp_path / "cache")

    # Inherited environment with empty SSH_CONNECTION and ANTIGRAVITY_CONFIG_DIR set
    monkeypatch.setenv("SSH_CONNECTION", "")
    monkeypatch.setenv("ANTIGRAVITY_CONFIG_DIR", "/some/custom/config/dir")
    monkeypatch.setenv("BROWSER", "/usr/bin/firefox")

    env = probe_env(home_dir, cache_dir=cache_dir)

    # 1. HOME/USERPROFILE/HOMEPATH set to home_dir
    assert env["HOME"] == home_dir
    assert env["USERPROFILE"] == home_dir
    assert env["HOMEPATH"] == home_dir

    # 2. ANTIGRAVITY_CONFIG_DIR must be removed
    assert "ANTIGRAVITY_CONFIG_DIR" not in env

    # 3. SSH_CONNECTION must be assigned unconditionally (never setdefault)
    assert env["SSH_CONNECTION"] == "127.0.0.1 0 127.0.0.1 0"

    # 4. BROWSER set to /usr/bin/false
    assert env["BROWSER"] == "/usr/bin/false"

    # 5. Private shim dir created with mode 0700 and prepended to PATH
    shim_dir = Path(cache_dir) / "nobrowser"
    assert stat.S_IMODE(os.stat(shim_dir).st_mode) == 0o700
    assert env["PATH"].startswith(str(shim_dir))

    # 6. Shim scripts exist and are executable
    for cmd in ("open", "xdg-open"):
        shim_file = shim_dir / cmd
        assert shim_file.is_file()
        assert os.access(shim_file, os.X_OK)
        # Execute shim; must exit 1 and report blocked browser to stderr
        res = subprocess.run([str(shim_file)], capture_output=True)
        assert res.returncode == 1
        assert b"blocked browser launch (antigravity-rotator probe)" in res.stderr


# ---------------------------------------------------------------------------
# 7. apply_account with save-back, real token selection, os.replace preserving symlink
# ---------------------------------------------------------------------------

def test_apply_account_save_back_and_symlink_preservation(tmp_path, monkeypatch):
    """apply_account performs save-back to previous active, selects token filename, and preserves symlinks."""
    user_home = tmp_path / "user_home"
    primary_cli = user_home / ".gemini" / "antigravity-cli"
    primary_cli.mkdir(parents=True)

    # Real token file in primary environment (jetski-standalone-oauth-token priority)
    primary_token = primary_cli / "jetski-standalone-oauth-token"
    primary_token.write_bytes(b"INITIAL_TOKEN_ACTIVE_A")

    primary_ga = user_home / ".gemini" / "google_accounts.json"
    primary_ga.write_text('{"active": "a@example.com"}', encoding="utf-8")

    monkeypatch.setenv("HOME", str(user_home))

    # External symlink pointing to primary token (such as directsdk isolated symlinks)
    ext_dir = tmp_path / "external_client"
    ext_dir.mkdir()
    ext_symlink = ext_dir / "token_symlink"
    ext_symlink.symlink_to(primary_token)

    # Setup accounts directories
    acc_a_home = tmp_path / "accounts" / "acc-a"
    (acc_a_home / ".gemini" / "antigravity-cli").mkdir(parents=True)

    acc_b_home = tmp_path / "accounts" / "acc-b"
    (acc_b_home / ".gemini" / "antigravity-cli").mkdir(parents=True)
    b_token = acc_b_home / ".gemini" / "antigravity-cli" / "jetski-standalone-oauth-token"
    b_token.write_bytes(b"CREDENTIALS_FOR_B")
    (acc_b_home / ".gemini" / "google_accounts.json").write_text('{"active": "b@example.com"}', encoding="utf-8")

    state_file = tmp_path / "accounts.json"
    backend = LocalBackend(state_file=str(state_file), accounts_dir=str(tmp_path / "accounts"))
    backend.register_account("acc-a", str(acc_a_home), "a@example.com")
    backend.register_account("acc-b", str(acc_b_home), "b@example.com")

    # Account A is active
    state = load_state(str(state_file))
    state["active_account"] = "acc-a"
    save_state(str(state_file), state)

    # Simulate runtime token refresh by agy while Account A was active
    primary_token.write_bytes(b"REFRESHED_TOKEN_A")

    # Apply Account B
    backend.apply_account("acc-b")

    # 1. Save-back: Account A's home must have received the refreshed token and google_accounts.json
    saved_a_token = acc_a_home / ".gemini" / "antigravity-cli" / "jetski-standalone-oauth-token"
    assert saved_a_token.is_file()
    assert saved_a_token.read_bytes() == b"REFRESHED_TOKEN_A"
    assert (acc_a_home / ".gemini" / "google_accounts.json").read_text(encoding="utf-8") == '{"active": "a@example.com"}'

    # 2. Primary token now has Account B's credentials
    assert primary_token.read_bytes() == b"CREDENTIALS_FOR_B"
    assert primary_ga.read_text(encoding="utf-8") == '{"active": "b@example.com"}'

    # 3. External symlink is still a valid symlink and sees the new token
    assert ext_symlink.is_symlink()
    assert ext_symlink.read_bytes() == b"CREDENTIALS_FOR_B"

    # 4. Active account in state is now acc-b
    assert backend.get_active_account_id() == "acc-b"


# ---------------------------------------------------------------------------
# 8. Cooldown with reset time and 15m fallback
# ---------------------------------------------------------------------------

def test_cooldown_with_reset_and_15m_fallback(tmp_path):
    """set_cooldown uses earliest reset of pool if known, else falls back to now + 15 min."""
    state_file = tmp_path / "accounts.json"
    backend = LocalBackend(state_file=str(state_file), accounts_dir=str(tmp_path / "accounts"))
    backend.register_account("test-acc", str(tmp_path / "home"), "test@example.com")

    now = datetime.now(timezone.utc)
    future_reset = (now + timedelta(hours=2)).replace(microsecond=0).isoformat()

    # Case 1: Pool has a known future reset time
    pools_with_reset = {
        "gemini": {
            "5h": {"remaining_fraction": 0.0, "reset_at": future_reset},
            "weekly": {"remaining_fraction": 0.5, "reset_at": None},
        }
    }
    cd1 = backend.set_cooldown("test-acc", "gemini", pools=pools_with_reset)
    assert cd1 == future_reset

    # Verify persisted in registry
    state = load_state(str(state_file))
    assert state["accounts"][0]["cooldown_until"] == future_reset

    # Case 2: Pool has no known reset time -> fallback 15 min
    pools_no_reset = {
        "claude_gpt": {
            "5h": {"remaining_fraction": 0.0, "reset_at": None},
            "weekly": {"remaining_fraction": 0.0, "reset_at": None},
        }
    }
    cd2 = backend.set_cooldown("test-acc", "claude_gpt", pools=pools_no_reset, fallback_minutes=15)
    cd2_dt = datetime.fromisoformat(cd2.replace("Z", "+00:00"))
    expected_fallback = now + timedelta(minutes=15)
    assert abs((cd2_dt - expected_fallback).total_seconds()) < 5


# ---------------------------------------------------------------------------
# 9. Login (staging cleanup on failure/KeyboardInterrupt, email extraction, no TTY)
# ---------------------------------------------------------------------------

def test_login_requires_tty():
    """Login without an interactive TTY raises RuntimeError."""
    with patch("sys.stdin.isatty", return_value=False):
        with pytest.raises(RuntimeError, match="interactive terminal \\(TTY\\)"):
            login_account(
                label="acc",
                accounts_dir="/tmp/test_accs",
                state_file="/tmp/test.json",
                lock_path="/tmp/test.lock",
            )


def test_login_success_email_extraction_and_staging_rename(tmp_path):
    """Login extracts email from google_accounts.json active field and renames staging."""
    accounts_dir = tmp_path / "accounts"
    state_file = tmp_path / "accounts.json"
    lock_path = tmp_path / "accounts.lock"

    def mock_agy_run(cmd, env, timeout):
        # Simulate agy creating credentials in staging HOME
        staging_home = Path(env["HOME"])
        cli_dir = staging_home / ".gemini" / "antigravity-cli"
        cli_dir.mkdir(parents=True)
        (cli_dir / "jetski-standalone-oauth-token").write_bytes(b"NEW_OAUTH_TOKEN")
        (staging_home / ".gemini" / "google_accounts.json").write_text(
            json.dumps({"active": "developer@gmail.com", "accounts": ["developer@gmail.com"]}),
            encoding="utf-8",
        )
        return MagicMock(returncode=0)

    with (
        patch("sys.stdin.isatty", return_value=True),
        patch("local_login.find_agy", return_value="/bin/mock_agy"),
        patch("subprocess.run", side_effect=mock_agy_run),
    ):
        result = login_account(
            label="my_label",
            accounts_dir=str(accounts_dir),
            state_file=str(state_file),
            lock_path=str(lock_path),
        )

    assert result["label"] == "my_label"
    assert result["email"] == "developer@gmail.com"

    # Final directory exists and staging directory is gone
    final_dir = accounts_dir / "my_label"
    assert final_dir.is_dir()
    assert (final_dir / ".gemini" / "antigravity-cli" / "jetski-standalone-oauth-token").is_file()
    assert not (accounts_dir / ".staging-my_label").exists()


def test_login_failure_cleans_up_staging(tmp_path):
    """Staging directory is deleted upon subprocess error or KeyboardInterrupt."""
    accounts_dir = tmp_path / "accounts"
    state_file = tmp_path / "accounts.json"
    lock_path = tmp_path / "accounts.lock"

    # Case A: Subprocess returns non-zero code
    def mock_fail_run(cmd, env, timeout):
        staging = Path(env["HOME"])
        (staging / "partial_file").write_text("temp", encoding="utf-8")
        return MagicMock(returncode=1)

    with (
        patch("sys.stdin.isatty", return_value=True),
        patch("local_login.find_agy", return_value="/bin/mock_agy"),
        patch("subprocess.run", side_effect=mock_fail_run),
    ):
        with pytest.raises(RuntimeError, match="return code 1"):
            login_account(
                label="failed_acc",
                accounts_dir=str(accounts_dir),
                state_file=str(state_file),
                lock_path=str(lock_path),
            )
    assert not (accounts_dir / ".staging-failed_acc").exists()

    # Case B: KeyboardInterrupt during authentication
    def mock_interrupt_run(cmd, env, timeout):
        staging = Path(env["HOME"])
        (staging / "auth_in_progress").write_text("temp", encoding="utf-8")
        raise KeyboardInterrupt()

    with (
        patch("sys.stdin.isatty", return_value=True),
        patch("local_login.find_agy", return_value="/bin/mock_agy"),
        patch("subprocess.run", side_effect=mock_interrupt_run),
    ):
        with pytest.raises(KeyboardInterrupt):
            login_account(
                label="interrupted_acc",
                accounts_dir=str(accounts_dir),
                state_file=str(state_file),
                lock_path=str(lock_path),
            )
    assert not (accounts_dir / ".staging-interrupted_acc").exists()
