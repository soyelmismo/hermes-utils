"""Tests for pool-aware quota selection, error classification patterns, /usage parsing, and backend auto resolution."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

import __init__ as plugin_module
from quota import (
    calculate_score,
    is_usable_for_pool,
    pool_for_model,
    pools_from_openproxy,
    pools_from_usage_json,
)
from rotator import (
    AntigravityRotator,
    LocalRotator,
    OpenProxyRotator,
    create_rotator,
    resolve_backend_choice,
)


# ---------------------------------------------------------------------------
# 1. Pool selection (Claude exhausted + Gemini full)
# ---------------------------------------------------------------------------

def test_pool_selection_claude_exhausted_gemini_full():
    """Claude exhausted + Gemini full chooses another account for Claude and does not rotate away for Gemini."""
    accounts = [
        {
            "id": 1,
            "label": "acc-1",
            "email": "acc1@example.com",
            "pools": {
                "gemini": {
                    "5h": {"remaining_fraction": 0.90, "reset_at": "2026-10-04T12:00:00Z"},
                    "weekly": {"remaining_fraction": 0.90, "reset_at": None},
                },
                "claude_gpt": {
                    "5h": {"remaining_fraction": 0.0, "reset_at": "2026-10-04T05:00:00Z"},  # Exhausted
                    "weekly": {"remaining_fraction": 0.50, "reset_at": None},
                },
            },
        },
        {
            "id": 2,
            "label": "acc-2",
            "email": "acc2@example.com",
            "pools": {
                "gemini": {
                    "5h": {"remaining_fraction": 0.40, "reset_at": "2026-10-04T12:00:00Z"},
                    "weekly": {"remaining_fraction": 0.50, "reset_at": None},
                },
                "claude_gpt": {
                    "5h": {"remaining_fraction": 0.80, "reset_at": "2026-10-04T06:00:00Z"},  # Usable
                    "weekly": {"remaining_fraction": 0.80, "reset_at": None},
                },
            },
        },
    ]

    rotator = OpenProxyRotator(api_url="http://mock", token="mock")

    with patch.object(rotator, "list_accounts", return_value=accounts):
        # Gemini model: Account 1 has 90% remaining > Account 2 (40%)
        best_gemini = rotator.find_best_candidate(exclude_id=None, model="gemini-2.5-pro")
        assert best_gemini is not None
        assert best_gemini["id"] == 1

        # Claude model: Account 1 is exhausted for Claude, Account 2 is chosen
        best_claude = rotator.find_best_candidate(exclude_id=None, model="claude-3-7-sonnet")
        assert best_claude is not None
        assert best_claude["id"] == 2

        # If Account 1 is current for Gemini, Account 1 is excluded if exclude_id=1, but if no rotation needed:
        # Candidate search for Gemini excluding 2 picks 1
        assert rotator.find_best_candidate(exclude_id=2, model="gemini-2.0-flash")["id"] == 1


# ---------------------------------------------------------------------------
# 2. Unknown quota ordering
# ---------------------------------------------------------------------------

def test_unknown_quota_orders_after_known():
    """Unknown quota is not excluded but orders after known quota."""
    accounts = [
        {
            "id": "known_acc",
            "pools": {
                "gemini": {
                    "5h": {"remaining_fraction": 0.20, "reset_at": "2026-10-04T10:00:00Z"},
                    "weekly": {"remaining_fraction": 0.50, "reset_at": None},
                }
            },
        },
        {
            "id": "unknown_acc",
            "pools": {
                "gemini": {
                    "5h": {"remaining_fraction": None, "reset_at": None},
                    "weekly": {"remaining_fraction": None, "reset_at": None},
                }
            },
        },
    ]

    rotator = OpenProxyRotator(api_url="http://mock", token="mock")

    with patch.object(rotator, "list_accounts", return_value=accounts):
        # Both are usable (gate does not exclude unknown)
        candidates = rotator.find_best_candidates(exclude_id=None, model="gemini-pro")
        assert len(candidates) == 2
        # Known account (20%) ranks ahead of unknown account
        assert candidates[0]["id"] == "known_acc"
        assert candidates[1]["id"] == "unknown_acc"

    # When known account is exhausted (0%), unknown account passes the gate and is selected
    exhausted_known = [
        {
            "id": "exhausted_acc",
            "pools": {
                "gemini": {
                    "5h": {"remaining_fraction": 0.0, "reset_at": "2026-10-04T10:00:00Z"},
                    "weekly": {"remaining_fraction": 0.0, "reset_at": None},
                }
            },
        },
        accounts[1],  # unknown_acc
    ]
    with patch.object(rotator, "list_accounts", return_value=exhausted_known):
        best = rotator.find_best_candidate(exclude_id=None, model="gemini-pro")
        assert best is not None
        assert best["id"] == "unknown_acc"


# ---------------------------------------------------------------------------
# 3. Error patterns (transient throttling vs quota exhaustion)
# ---------------------------------------------------------------------------

def test_error_patterns_throttling_no_rotate_resource_exhausted_yes():
    """Transient throttling patterns do not trigger rotation; RESOURCE_EXHAUSTED and 429 do."""
    from __init__ import _is_quota_error

    # Transient throttling messages must NOT be classified as quota exhaustion
    assert not _is_quota_error("Rate limit exceeded: please slow down")
    assert not _is_quota_error("ratelimit: transient backoff")
    assert not _is_quota_error("Too many requests, try again later")
    assert not _is_quota_error("server error: too many requests")

    # Quota exhaustion errors MUST be classified as quota errors
    assert _is_quota_error("RESOURCE_EXHAUSTED: quota limit reached")
    assert _is_quota_error("resource exhausted")
    assert _is_quota_error("Quota exceeded for project")
    assert _is_quota_error("quota_exceeded: session limit")
    assert _is_quota_error("Individual quota reached. Resets in 1h")
    assert _is_quota_error("You have exhausted your capacity on this model")
    assert _is_quota_error("Billing hard limit reached")
    assert not _is_quota_error("token limit reached")  # Context length error, not quota
    assert _is_quota_error("out of quota")

    # Standalone 429 token must match even without explicit quota wording
    assert _is_quota_error("Error HTTP 429: client request limit")
    assert _is_quota_error("status 429")

    # False positives on numbers containing 429 must NOT match
    assert not _is_quota_error("item 4290 not found")
    assert not _is_quota_error("error code 14299")


def test_hook_classification_throttling_ignored_exhaustion_retries():
    """Hook transform_api_error_classification ignores throttling and retries exhaustion."""
    ctx = MagicMock()
    registered_hooks = {}

    def mock_register_hook(name, cb):
        registered_hooks[name] = cb

    ctx.register_hook.side_effect = mock_register_hook
    ctx.get_config.side_effect = lambda k, d=None: d

    plugin_module.register(ctx)
    hook_fn = registered_hooks["transform_api_error_classification"]

    # Throttling error (without status_code 429) -> ignored (returns None)
    res_throttle = hook_fn(
        provider="antigravity-subscription-directsdk",
        status_code=503,
        error_message="Too many requests, slowing down",
    )
    assert res_throttle is None

    # Quota exhausted -> triggers rotation and returns retryable classification
    with patch("rotator.OpenProxyRotator.rotate_to_next_account") as mock_rotate:
        mock_rotate.return_value = ({"id": 2}, "Rotated to account 2")
        res_exhaust = hook_fn(
            provider="antigravity-subscription-directsdk",
            status_code=200,
            error_message="RESOURCE_EXHAUSTED: quota reached",
        )
        assert res_exhaust is not None
        assert res_exhaust["retryable"] is True
        assert res_exhaust["should_rotate_credential"] is True
        mock_rotate.assert_called_once()


# ---------------------------------------------------------------------------
# 6. Parser /usage to pools
# ---------------------------------------------------------------------------

def test_usage_parser_to_pools():
    """Parse agy /usage JSON into normalized gemini and claude_gpt pools."""
    raw_usage = json.dumps({
        "groups": [
            {
                "name": "Gemini 2.0 & Experimental",
                "buckets": [
                    {
                        "window": "5h session",
                        "remaining": 750,
                        "limit": 1000,
                        "reset_at": "2026-10-04T05:00:00Z",
                    },
                    {
                        "window": "weekly limit",
                        "remaining": 4000,
                        "limit": 5000,
                        "reset_at": "2026-10-09T00:00:00Z",
                    },
                ],
            },
            {
                "name": "Claude 3.5 & GPT-4o 3P Models",
                "buckets": [
                    {
                        "window": "5h",
                        "remaining": 150,
                        "limit": 500,
                        "reset_at": "2026-10-04T08:30:00Z",
                    },
                    {
                        "window": "weekly",
                        "remaining": 0,
                        "limit": 1000,
                        "reset_at": "2026-10-07T12:00:00Z",
                    },
                ],
            },
        ]
    })

    pools = pools_from_usage_json(raw_usage)
    assert "gemini" in pools
    assert "claude_gpt" in pools

    # Gemini pool assertions
    assert pools["gemini"]["5h"]["remaining_fraction"] == 0.75
    assert pools["gemini"]["5h"]["reset_at"].startswith("2026-10-04T05:00:00")
    assert pools["gemini"]["weekly"]["remaining_fraction"] == 0.8
    assert pools["gemini"]["weekly"]["reset_at"].startswith("2026-10-09T00:00:00")

    # Claude/GPT pool assertions
    assert pools["claude_gpt"]["5h"]["remaining_fraction"] == 0.3
    assert pools["claude_gpt"]["5h"]["reset_at"].startswith("2026-10-04T08:30:00")
    assert pools["claude_gpt"]["weekly"]["remaining_fraction"] == 0.0
    assert pools["claude_gpt"]["weekly"]["reset_at"].startswith("2026-10-07T12:00:00")


# ---------------------------------------------------------------------------
# 10. Backend auto resolution
# ---------------------------------------------------------------------------

def test_backend_auto_resolution(monkeypatch, tmp_path):
    """Auto selects openproxy if token is resoluble via config, env, or script; else local."""
    script_path = tmp_path / "switch_account.sh"

    # 1. Config explicit: openproxy and local
    assert resolve_backend_choice(config_choice="openproxy", script_path=script_path) == "openproxy"
    assert resolve_backend_choice(config_choice="local", script_path=script_path) == "local"

    # 2. Auto with config token -> openproxy
    assert resolve_backend_choice(config_choice="auto", openproxy_token="op_cfg_token", script_path=script_path) == "openproxy"

    # 3. Auto with env token -> openproxy
    monkeypatch.setenv("OPENPROXY_ADMIN_TOKEN", "op_env_token")
    assert resolve_backend_choice(config_choice="auto", script_path=script_path) == "openproxy"
    monkeypatch.delenv("OPENPROXY_ADMIN_TOKEN", raising=False)

    # 4. Auto with script token -> openproxy
    script_path.write_text('API_URL="http://test"\nTOKEN="op_script_tok"\n', encoding="utf-8")
    assert resolve_backend_choice(config_choice="auto", script_path=script_path) == "openproxy"

    # 5. Auto with no token anywhere -> local
    script_path.unlink()
    assert resolve_backend_choice(config_choice="auto", script_path=script_path) == "local"

    # Factory creates correct class
    rot_local = create_rotator(backend_choice="local", state_file=str(tmp_path / "accts.json"))
    assert isinstance(rot_local, LocalRotator)

    rot_op = create_rotator(backend_choice="openproxy", openproxy_token="token")
    assert isinstance(rot_op, OpenProxyRotator)
    assert not isinstance(rot_op, LocalRotator)
