"""Unit tests for antigravity-rotator plugin."""

import json
from unittest.mock import MagicMock, patch
import pytest

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
    assert plugin_module._is_quota_error(None, RuntimeError("capacity exhausted"))
    assert not plugin_module._is_quota_error("SyntaxError in input")
    assert not plugin_module._is_quota_error(None, ValueError("invalid argument"))


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


def test_find_best_candidate_all_exhausted_fallback():
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
        best = rotator.find_best_candidate()
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
