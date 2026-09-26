"""Unit tests for antigravity-rotator plugin."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

import rotator as rotator_module
from rotator import (
    OpenProxyRotator,
    format_remaining_time,
    _read_credentials_from_script,
)
import __init__ as plugin_module


def test_format_remaining_time():
    assert format_remaining_time(None) == "N/A"
    assert format_remaining_time("N/A") == "N/A"
    # Past date
    assert format_remaining_time("2020-01-01T00:00:00Z") == "ready"


def test_read_credentials_from_script(tmp_path):
    script = tmp_path / "switch_account.sh"
    script.write_text(
        'API_URL="http://test:8787/admin/api"\nTOKEN="op_test_12345"\n',
        encoding="utf-8",
    )
    url, token = _read_credentials_from_script(script)
    assert url == "http://test:8787/admin/api"
    assert token == "op_test_12345"


def test_is_antigravity_provider():
    assert plugin_module._is_antigravity_provider("antigravity")
    assert plugin_module._is_antigravity_provider("antigravity-subscription-directsdk")
    assert plugin_module._is_antigravity_provider("agy")
    assert plugin_module._is_antigravity_provider("antigravity-directsdk")
    assert not plugin_module._is_antigravity_provider("openai")
    assert not plugin_module._is_antigravity_provider(None)


def test_is_quota_error():
    assert plugin_module._is_quota_error("RESOURCE_EXHAUSTED: quota limit reached")
    assert plugin_module._is_quota_error("You have exhausted your capacity on this model")
    assert plugin_module._is_quota_error("Rate limit exceeded (429)")
    assert plugin_module._is_quota_error("request 429 too many")
    assert plugin_module._is_quota_error(None, RuntimeError("capacity exhausted"))
    assert not plugin_module._is_quota_error("SyntaxError in input")
    assert not plugin_module._is_quota_error(None, ValueError("invalid argument"))
    # Wrong-token guards: "429" must be a standalone token, not part of a number
    assert not plugin_module._is_quota_error("file size 4290 bytes")
    assert not plugin_module._is_quota_error("object id 42910 not found")


def test_is_quota_error_individual_quota_message():
    # Real production message from agy on 21-sep: must trigger rotation
    assert plugin_module._is_quota_error(
        "Individual quota reached. Please upgrade your subscription to increase your limits. Resets in 1h52m1s"
    )


def test_find_best_candidate():
    rotator = OpenProxyRotator(api_url="http://mock", token="mock")
    mock_accounts = [
        {
            "id": 1,
            "provider_id": "antigravity",
            "email": "full@example.com",
            "quota_session_used": 1000,
            "quota_session_limit": 1000,
            "quota_weekly_used": 100,
            "quota_weekly_limit": 1000,
        },
        {
            "id": 2,
            "provider_id": "antigravity",
            "email": "medium@example.com",
            "quota_session_used": 200,
            "quota_session_limit": 1000,
            "quota_weekly_used": 200,
            "quota_weekly_limit": 1000,
        },
        {
            "id": 3,
            "provider_id": "antigravity",
            "email": "fresh@example.com",
            "quota_session_used": 10,
            "quota_session_limit": 1000,
            "quota_weekly_used": 10,
            "quota_weekly_limit": 1000,
        },
    ]

    with patch.object(rotator, "list_accounts", return_value=mock_accounts):
        best = rotator.find_best_candidate(exclude_id=None)
        assert best is not None
        assert best["id"] == 3  # fresh has most remaining quota

        # If fresh is excluded
        best_no_fresh = rotator.find_best_candidate(exclude_id=3)
        assert best_no_fresh is not None
        assert best_no_fresh["id"] == 2


def test_find_best_candidate_exhausted_pool_behavior():
    rotator = OpenProxyRotator(api_url="http://mock", token="mock")
    mock_accounts = [
        {
            "id": 1,
            "provider_id": "antigravity",
            "quota_session_used": 1000,
            "quota_session_limit": 1000,
            "quota_session_reset_at": "2026-09-22T10:00:00Z",
        },
        {
            "id": 2,
            "provider_id": "antigravity",
            "quota_session_used": 1000,
            "quota_session_limit": 1000,
            "quota_session_reset_at": "2026-09-22T04:00:00Z",
        },
    ]

    with patch.object(rotator, "list_accounts", return_value=mock_accounts):
        # Default: an exhausted pool yields no candidate at all
        assert rotator.find_best_candidate() is None

        # Opt-in fallback: earliest session reset (for operator-driven switches)
        best = rotator.find_best_candidate(allow_exhausted=True)
        assert best is not None
        assert best["id"] == 2  # resets earlier


def test_plugin_registration():
    ctx = MagicMock()
    ctx.get_config.side_effect = lambda k, d=None: d

    plugin_module.register(ctx)

    ctx.register_hook.assert_called()
    assert ctx.register_tool.call_count == 3
    ctx.register_cli_command.assert_called_once()


def test_hook_triggers_rotation_on_quota_error():
    ctx = MagicMock()
    registered_hooks = {}

    def mock_register_hook(name, cb):
        registered_hooks[name] = cb

    ctx.register_hook.side_effect = mock_register_hook
    ctx.get_config.side_effect = lambda k, d=None: d

    plugin_module.register(ctx)

    hook_fn = registered_hooks["transform_api_error_classification"]

    # Non-antigravity provider should be ignored
    res = hook_fn(provider="openai", error_message="rate limit exceeded")
    assert res is None

    # Non-quota error on antigravity should be ignored
    res = hook_fn(provider="antigravity", error_message="malformed json request")
    assert res is None

    # Quota error on antigravity should trigger rotation
    with patch("rotator.OpenProxyRotator.rotate_to_next_account") as mock_rotate:
        mock_rotate.return_value = ({"id": 224, "email": "test@gmail.com"}, "Rotated account to 224")
        res = hook_fn(
            provider="antigravity-subscription-directsdk",
            status_code=429,
            error_message="RESOURCE_EXHAUSTED",
        )
        assert res is not None
        assert res["reason"] == "rate_limit"
        assert res["retryable"] is True
        assert res["should_rotate_credential"] is True
        mock_rotate.assert_called_once()


def test_hook_returns_none_when_rotation_fails():
    ctx = MagicMock()
    registered_hooks = {}

    def mock_register_hook(name, cb):
        registered_hooks[name] = cb

    ctx.register_hook.side_effect = mock_register_hook
    ctx.get_config.side_effect = lambda k, d=None: d

    plugin_module.register(ctx)
    hook_fn = registered_hooks["transform_api_error_classification"]

    # Rotation failure (e.g. all accounts exhausted) must not hijack classification
    with patch(
        "rotator.OpenProxyRotator.rotate_to_next_account",
        side_effect=RuntimeError("All Antigravity accounts exhausted; earliest session reset: ..."),
    ):
        res = hook_fn(
            provider="antigravity-subscription-directsdk",
            status_code=429,
            error_message="RESOURCE_EXHAUSTED",
        )
    assert res is None


# ── Fix 1: no infinite recursion when resolving the active account ───────────

def test_list_accounts_mark_active_false_skips_active_lookup(monkeypatch):
    rotator = OpenProxyRotator(api_url="http://mock", token="mock")
    lookups = []

    def fake_active_id():
        lookups.append(1)
        return 99

    monkeypatch.setattr(rotator, "get_active_account_id", fake_active_id)
    monkeypatch.setattr(
        rotator, "_call", lambda *a, **k: [{"id": 99, "provider_id": "antigravity"}]
    )

    accounts = rotator.list_accounts(mark_active=False)
    assert lookups == []  # active state never consulted
    assert accounts[0]["is_active"] is False

    accounts = rotator.list_accounts()  # default marks active
    assert lookups == [1]
    assert accounts[0]["is_active"] is True


def test_get_active_account_id_no_recursion(tmp_path, monkeypatch):
    monkeypatch.setattr(rotator_module, "_config_bases", lambda: (tmp_path,))
    gemini_dir = tmp_path / ".gemini"
    gemini_dir.mkdir()
    (gemini_dir / "google_accounts.json").write_text(
        json.dumps({"active": "user@example.com"}), encoding="utf-8"
    )
    # No active_account.json → the google_accounts.json fallback path runs

    rotator = OpenProxyRotator(api_url="http://mock", token="mock")
    accounts = [{"id": 7, "provider_id": "antigravity", "email": "user@example.com"}]

    def fake_list_accounts(mark_active=True):
        # Emulate the real list_accounts, which consults the active state
        # when asked to mark it — this recursed against the old code.
        if mark_active:
            rotator.get_active_account_id()
        return [dict(a) for a in accounts]

    monkeypatch.setattr(rotator, "list_accounts", fake_list_accounts)

    assert rotator.get_active_account_id() == 7
    assert rotator._last_switched_id == 7


# ── Fix 3: apply_account only records state on real success ──────────────────

def test_apply_account_success_persists_state(tmp_path, monkeypatch):
    monkeypatch.setattr(rotator_module, "_config_bases", lambda: (tmp_path,))
    rotator = OpenProxyRotator(api_url="http://mock", token="mock")
    monkeypatch.setattr(rotator, "_call", lambda *a, **k: {"success": True})

    res = rotator.apply_account(42)

    assert res == {"success": True}
    assert rotator._last_switched_id == 42
    state_file = tmp_path / ".gemini" / "antigravity-cli" / "active_account.json"
    assert json.loads(state_file.read_text(encoding="utf-8"))["account_id"] == 42


def test_apply_account_failure_raises_and_skips_state(tmp_path, monkeypatch):
    monkeypatch.setattr(rotator_module, "_config_bases", lambda: (tmp_path,))
    rotator = OpenProxyRotator(api_url="http://mock", token="mock")
    monkeypatch.setattr(
        rotator, "_call", lambda *a, **k: {"success": False, "error": "token rejected"}
    )

    with pytest.raises(RuntimeError, match="token rejected"):
        rotator.apply_account(42)

    assert rotator._last_switched_id is None
    assert not (tmp_path / ".gemini" / "antigravity-cli" / "active_account.json").exists()


# ── Fix 4: rotation refuses to switch to an exhausted account ────────────────

def test_rotate_all_exhausted_raises_without_applying():
    rotator = OpenProxyRotator(api_url="http://mock", token="mock")
    exhausted = [
        {
            "id": 1,
            "provider_id": "antigravity",
            "quota_session_used": 1000,
            "quota_session_limit": 1000,
            "quota_session_reset_at": "2026-09-22T10:00:00Z",
        },
        {
            "id": 2,
            "provider_id": "antigravity",
            "quota_session_used": 1000,
            "quota_session_limit": 1000,
            "quota_session_reset_at": "2026-09-22T04:00:00Z",
        },
    ]

    with patch.object(rotator, "list_accounts", return_value=exhausted), patch.object(
        rotator, "apply_account"
    ) as mock_apply:
        with pytest.raises(RuntimeError, match="All Antigravity accounts exhausted"):
            rotator.rotate_to_next_account(current_account_id=1)
    mock_apply.assert_not_called()


def test_rotate_no_accounts_configured_raises():
    rotator = OpenProxyRotator(api_url="http://mock", token="mock")
    with patch.object(rotator, "list_accounts", return_value=[]):
        with pytest.raises(RuntimeError, match="No Antigravity accounts configured"):
            rotator.rotate_to_next_account(current_account_id=1)


def test_rotate_applies_best_candidate():
    rotator = OpenProxyRotator(api_url="http://mock", token="mock")
    accounts = [
        {
            "id": 1,
            "provider_id": "antigravity",
            "email": "exhausted@example.com",
            "quota_session_used": 1000,
            "quota_session_limit": 1000,
            "quota_session_reset_at": "2026-09-22T04:00:00Z",
        },
        {
            "id": 2,
            "provider_id": "antigravity",
            "email": "fresh@example.com",
            "quota_session_used": 100,
            "quota_session_limit": 1000,
            "quota_weekly_used": 100,
            "quota_weekly_limit": 1000,
        },
    ]

    with patch.object(rotator, "list_accounts", return_value=accounts), patch.object(
        rotator, "apply_account", return_value={"success": True}
    ) as mock_apply:
        best, msg = rotator.rotate_to_next_account(current_account_id=1)

    assert best["id"] == 2
    assert "fresh@example.com" in msg
    mock_apply.assert_called_once_with(2)


# ── Fix 5: quota_threshold_percent is enforced ───────────────────────────────

def test_find_best_candidate_threshold_filters_near_exhausted():
    rotator = OpenProxyRotator(api_url="http://mock", token="mock")  # default 95.0
    accounts = [
        {
            "id": 1,
            "provider_id": "antigravity",
            "quota_session_used": 960,  # 96% >= 95% threshold
            "quota_session_limit": 1000,
        },
        {
            "id": 2,
            "provider_id": "antigravity",
            "quota_session_used": 940,  # 94% < 95% threshold
            "quota_session_limit": 1000,
        },
    ]

    with patch.object(rotator, "list_accounts", return_value=accounts):
        assert rotator.find_best_candidate()["id"] == 2
        # Account 1 is not a candidate at all under the default threshold
        assert rotator.find_best_candidate(exclude_id=2) is None

    # threshold=100.0 preserves the legacy "exactly at capacity" filter only
    rotator.quota_threshold_percent = 100.0
    with patch.object(rotator, "list_accounts", return_value=accounts):
        assert rotator.find_best_candidate(exclude_id=2)["id"] == 1


# ── Fix 6: ranking considers weekly remaining quota ──────────────────────────

def test_find_best_candidate_ranks_by_weekly_remaining():
    rotator = OpenProxyRotator(api_url="http://mock", token="mock")
    accounts = [
        {
            "id": 1,
            "provider_id": "antigravity",
            "quota_session_used": 100,
            "quota_session_limit": 1000,
            "quota_weekly_used": 900,
            "quota_weekly_limit": 1000,
        },
        {
            "id": 2,
            "provider_id": "antigravity",
            "quota_session_used": 100,
            "quota_session_limit": 1000,
            "quota_weekly_used": 100,
            "quota_weekly_limit": 1000,
        },
    ]

    with patch.object(rotator, "list_accounts", return_value=accounts):
        # Equal session remaining → the account with more weekly quota wins
        assert rotator.find_best_candidate()["id"] == 2


# ── Fix 2: isolated workspace token sync ─────────────────────────────────────

def _make_fake_home(home_dir: Path) -> Path:
    token_dir = home_dir / ".gemini" / "antigravity-cli"
    token_dir.mkdir(parents=True)
    (token_dir / "jetski-standalone-oauth-token").write_text("NEW-TOKEN", encoding="utf-8")
    return home_dir


def test_sync_isolated_workspace_tokens(tmp_path, monkeypatch):
    home_dir = _make_fake_home(tmp_path / "fakehome")
    monkeypatch.setattr(rotator_module, "_config_bases", lambda: (home_dir,))

    tmp_root = tmp_path / "tmproot"

    # hermes_agy_* home holding the current token name as a regular file
    current_name = tmp_root / "hermes_agy_abc" / "home" / ".gemini" / "antigravity-cli"
    current_name.mkdir(parents=True)
    (current_name / "jetski-standalone-oauth-token").write_text("OLD", encoding="utf-8")

    # agy_* home holding the legacy token name
    legacy_name = tmp_root / "agy_legacy" / "home" / ".gemini" / "antigravity-cli"
    legacy_name.mkdir(parents=True)
    (legacy_name / "antigravity-oauth-token").write_text("OLD", encoding="utf-8")

    # symlinked token must be left alone (it already sees the new token)
    symlinked = tmp_root / "hermes_agy_sym" / "home" / ".gemini" / "antigravity-cli"
    symlinked.mkdir(parents=True)
    symlink_path = symlinked / "jetski-standalone-oauth-token"
    symlink_path.symlink_to(home_dir / ".gemini" / "antigravity-cli" / "jetski-standalone-oauth-token")

    # hardlinked token must not crash the sync (SameFileError is suppressed)
    hardlinked = tmp_root / "hermes_agy_hard" / "home" / ".gemini" / "antigravity-cli"
    hardlinked.mkdir(parents=True)
    os.link(
        home_dir / ".gemini" / "antigravity-cli" / "jetski-standalone-oauth-token",
        hardlinked / "jetski-standalone-oauth-token",
    )

    rotator = OpenProxyRotator(api_url="http://mock", token="mock")
    rotator._sync_isolated_workspace_tokens(tmp_root=tmp_root)

    assert (current_name / "jetski-standalone-oauth-token").read_text(encoding="utf-8") == "NEW-TOKEN"
    assert (legacy_name / "antigravity-oauth-token").read_text(encoding="utf-8") == "NEW-TOKEN"
    assert symlink_path.is_symlink()  # still a symlink, not replaced
    assert (hardlinked / "jetski-standalone-oauth-token").read_text(encoding="utf-8") == "NEW-TOKEN"


def test_sync_isolated_workspace_tokens_noop_without_token(tmp_path, monkeypatch):
    monkeypatch.setattr(rotator_module, "_config_bases", lambda: (tmp_path,))
    tmp_root = tmp_path / "tmproot"
    stale = tmp_root / "hermes_agy_stale" / "home" / ".gemini" / "antigravity-cli"
    stale.mkdir(parents=True)
    (stale / "jetski-standalone-oauth-token").write_text("OLD", encoding="utf-8")

    rotator = OpenProxyRotator(api_url="http://mock", token="mock")
    rotator._sync_isolated_workspace_tokens(tmp_root=tmp_root)

    # No real token available → nothing is touched
    assert (stale / "jetski-standalone-oauth-token").read_text(encoding="utf-8") == "OLD"


def _mk_account(aid, sess_used=0, sess_lim=1000, wk_used=0, wk_lim=1000, **kw):
    acc = {
        "id": aid,
        "provider_id": "antigravity",
        "email": f"a{aid}@example.com",
        "quota_session_used": sess_used,
        "quota_session_limit": sess_lim,
        "quota_weekly_used": wk_used,
        "quota_weekly_limit": wk_lim,
    }
    acc.update(kw)
    return acc


def test_rate_limited_until_future_is_skipped():
    from datetime import datetime, timedelta, timezone
    rotator = OpenProxyRotator(api_url="http://mock", token="mock")
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    accounts = [
        _mk_account(1, rate_limited_until=future),   # benched → must be skipped
        _mk_account(2, quota_session_used=500, rate_limited_until=past),  # expired bench → usable
    ]
    with patch.object(rotator, "list_accounts", return_value=accounts):
        best = rotator.find_best_candidate(exclude_id=None)
        assert best is not None and best["id"] == 2


def test_rotate_skips_candidate_still_exhausted_after_refresh():
    rotator = OpenProxyRotator(api_url="http://mock", token="mock")
    accounts = [_mk_account(1, sess_used=10), _mk_account(2, sess_used=900)]
    with (
        patch.object(rotator, "list_accounts", return_value=accounts),
        patch.object(rotator, "refresh_quota") as m_refresh,
        patch.object(rotator, "apply_account") as m_apply,
    ):
        # Account 1 was the best by cached data, but live refresh says full
        m_refresh.side_effect = lambda aid: (
            {"quota_session_used": 1000, "quota_session_limit": 1000} if aid == 1 else {}
        )
        m_apply.return_value = {"success": True}
        best, msg = rotator.rotate_to_next_account(current_account_id=99)
        assert best["id"] == 2
        m_apply.assert_called_once_with(2)
        assert "[2]" in msg


def test_rotate_refresh_failure_proceeds_with_cached_data():
    rotator = OpenProxyRotator(api_url="http://mock", token="mock")
    accounts = [_mk_account(1, sess_used=10)]
    with (
        patch.object(rotator, "list_accounts", return_value=accounts),
        patch.object(rotator, "refresh_quota", return_value={"error": "boom"}),
        patch.object(rotator, "apply_account", return_value={"success": True}) as m_apply,
    ):
        best, _ = rotator.rotate_to_next_account(current_account_id=99)
        assert best["id"] == 1
        m_apply.assert_called_once_with(1)
