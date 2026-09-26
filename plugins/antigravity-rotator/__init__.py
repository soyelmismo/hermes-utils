"""Google Antigravity account rotator plugin for Hermes Agent.

Provides automatic account failover when 5-hour session or weekly quotas
are reached in Google Antigravity, seamlessly swapping credentials via
OpenProxy without altering or polluting the provider plugin.
"""

from __future__ import annotations

import logging
import re
from typing import Any

try:
    from .cli import handle_cli, setup_cli
    from .rotator import OpenProxyRotator
except ImportError:
    from cli import handle_cli, setup_cli
    from rotator import OpenProxyRotator


logger = logging.getLogger(__name__)

__version__ = "1.0.0"

_AGY_PROVIDERS = {
    "antigravity",
    "antigravity-subscription-directsdk",
    "agy",
    "antigravity-directsdk",
}

_QUOTA_ERROR_PATTERNS = (
    "resource_exhausted",
    "resource exhausted",
    "quota exceeded",
    "quota_exceeded",
    "quota reached",
    "individual quota",
    "exhausted your capacity",
    "exceeded your current quota",
    "capacity exhausted",
    "rate limit",
    "ratelimit",
    "too many requests",
    "out of quota",
)
# "429" as a standalone token only (avoid matching sizes/IDs like 4290)
_QUOTA_STATUS_CODE_PATTERN = re.compile(r"\b429\b")


def _is_antigravity_provider(provider: str | None) -> bool:
    """Check if the provider name corresponds to the Antigravity provider."""
    if not provider:
        return False
    p = str(provider).strip().lower()
    return p in _AGY_PROVIDERS or "antigravity" in p or "agy" in p


def _is_quota_error(error_message: str | None, error: Exception | None = None) -> bool:
    """Detect if an API error signifies quota exhaustion or rate limiting."""
    haystack = []
    if error_message:
        haystack.append(str(error_message).lower())
    if error:
        haystack.append(str(error).lower())
        haystack.append(type(error).__name__.lower())
    full_text = " ".join(haystack)
    if any(pat in full_text for pat in _QUOTA_ERROR_PATTERNS):
        return True
    return _QUOTA_STATUS_CODE_PATTERN.search(full_text) is not None


def register(ctx: Any) -> None:
    """Hermes plugin entry point."""
    # Resolve configuration
    api_url = ctx.get_config("openproxy_url")
    token = ctx.get_config("openproxy_token")
    auto_rotate = ctx.get_config("auto_rotate", True)
    threshold = float(ctx.get_config("quota_threshold_percent", 95.0))

    rotator = OpenProxyRotator(
        api_url=api_url,
        token=token,
        quota_threshold_percent=threshold,
    )

    # ── Hook: transform_api_error_classification ─────────────────────────────
    def _on_transform_api_error_classification(
        *,
        provider: str = "",
        model: str = "",
        status_code: int | None = None,
        error_message: str = "",
        error: Exception | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """Intercept provider API failures before classification."""
        if not auto_rotate:
            return None

        if not _is_antigravity_provider(provider):
            return None

        if status_code != 429 and not _is_quota_error(error_message, error):
            return None

        try:
            _, msg = rotator.rotate_to_next_account()
            return {
                "reason": "rate_limit",
                "retryable": True,
                "should_rotate_credential": True,
                "message": f"Antigravity quota exhausted. {msg}. Retrying turn...",
            }
        except Exception as exc:
            logger.warning("[antigravity-rotator] Auto-rotation failed: %s", exc)
            return None

    ctx.register_hook("transform_api_error_classification", _on_transform_api_error_classification)

    # ── Tool: antigravity_list_accounts ──────────────────────────────────────
    def _handle_list_accounts() -> str:
        return rotator.format_status_table()

    ctx.register_tool(
        name="antigravity_list_accounts",
        toolset="antigravity",
        schema={
            "type": "function",
            "function": {
                "name": "antigravity_list_accounts",
                "description": (
                    "List all Google Antigravity accounts registered in OpenProxy, "
                    "including 5-hour rolling session quota, weekly quota, and reset countdowns."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
        handler=_handle_list_accounts,
        emoji="🔄",
        description="List all Google Antigravity accounts with live quota status.",
    )

    # ── Tool: antigravity_switch_account ─────────────────────────────────────
    def _handle_switch_account(account_id: int | None = None) -> str:
        if account_id is not None:
            try:
                rotator.apply_account(account_id)
            except RuntimeError as exc:
                return f"Failed to switch to account [{account_id}]: {exc}"
            return f"Successfully switched agy CLI to account [{account_id}]."
        try:
            _, msg = rotator.rotate_to_next_account()
        except RuntimeError as exc:
            return str(exc)
        return msg

    ctx.register_tool(
        name="antigravity_switch_account",
        toolset="antigravity",
        schema={
            "type": "function",
            "function": {
                "name": "antigravity_switch_account",
                "description": (
                    "Switch the local agy CLI to a specific Google Antigravity account ID, "
                    "or auto-rotate to the account with the most remaining quota if no ID is specified."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "account_id": {
                            "type": "integer",
                            "description": "Optional account ID to switch to. If omitted, picks the best available account.",
                        },
                    },
                },
            },
        },
        handler=_handle_switch_account,
        emoji="🔀",
        description="Switch or auto-rotate the active Google Antigravity account.",
    )

    # ── Tool: antigravity_refresh_quotas ─────────────────────────────────────
    def _handle_refresh_quotas(account_id: int | None = None) -> str:
        if account_id is not None:
            res = rotator.refresh_quota(account_id)
            return f"Quota refreshed for [{account_id}]: {res}"
        accounts = rotator.list_accounts()
        for a in accounts:
            aid = a.get("id")
            if aid is not None:
                rotator.refresh_quota(aid)
        return rotator.format_status_table()

    ctx.register_tool(
        name="antigravity_refresh_quotas",
        toolset="antigravity",
        schema={
            "type": "function",
            "function": {
                "name": "antigravity_refresh_quotas",
                "description": "Refresh live quota telemetry for Google Antigravity accounts from Google via OpenProxy.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "account_id": {
                            "type": "integer",
                            "description": "Optional specific account ID to refresh. If omitted, refreshes all accounts.",
                        },
                    },
                },
            },
        },
        handler=_handle_refresh_quotas,
        emoji="⏳",
        description="Refresh live quota telemetry for Antigravity accounts.",
    )

    # ── CLI Command: hermes antigravity ──────────────────────────────────────
    ctx.register_cli_command(
        name="antigravity",
        help="Inspect and switch Google Antigravity accounts via OpenProxy",
        setup_fn=setup_cli,
        handler_fn=handle_cli,
        description="Operator CLI for Antigravity account rotation and quota telemetry.",
    )

    logger.info("[antigravity-rotator] plugin registered (transform_api_error_classification hook, tools, CLI)")
