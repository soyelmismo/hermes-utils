"""CLI subcommands for hermes antigravity [list|switch|refresh]."""

from __future__ import annotations

import argparse
from typing import Any

try:
    from .rotator import OpenProxyRotator
except ImportError:
    from rotator import OpenProxyRotator



def setup_cli(parser: argparse.ArgumentParser) -> None:
    """Configure argparse subparsers for `hermes antigravity`."""
    sub = parser.add_subparsers(dest="antigravity_action", help="Antigravity account management")

    # list / status
    sub.add_parser("list", help="Display all Antigravity accounts and quota telemetry")
    sub.add_parser("status", help="Alias for list")

    # switch
    p_switch = sub.add_parser("switch", help="Switch local agy CLI to a specific account or auto-rotate")
    p_switch.add_argument(
        "account_id",
        type=int,
        nargs="?",
        default=None,
        help="Account ID to switch to (if omitted, auto-selects account with most remaining quota)",
    )

    # refresh
    p_refresh = sub.add_parser("refresh", help="Refresh quota usage stats from Google via OpenProxy")
    p_refresh.add_argument(
        "account_id",
        type=int,
        nargs="?",
        default=None,
        help="Account ID to refresh (if omitted, refreshes all Antigravity accounts)",
    )


def handle_cli(args: argparse.Namespace) -> int:
    """Execute `hermes antigravity ...` subcommand."""
    rotator = OpenProxyRotator()
    action = getattr(args, "antigravity_action", "") or "list"

    if action in ("list", "status"):
        print(rotator.format_status_table())
        return 0

    if action == "switch":
        target_id = getattr(args, "account_id", None)
        if target_id is not None:
            res = rotator.apply_account(target_id)
            if res.get("success"):
                print(f"✅ Successfully switched agy CLI to account [{target_id}].")
            else:
                print(f"❌ Failed to switch to account [{target_id}]: {res}")
        else:
            _, msg = rotator.rotate_to_next_account()
            print(f"🔄 {msg}")
        return 0

    if action == "refresh":
        target_id = getattr(args, "account_id", None)
        if target_id is not None:
            res = rotator.refresh_quota(target_id)
            print(f"Quota refreshed for [{target_id}]: {res}")
        else:
            print("Refreshing quota telemetry for all Antigravity accounts...")
            for a in rotator.list_accounts():
                aid = a.get("id")
                if aid is not None:
                    rotator.refresh_quota(aid)
            print(rotator.format_status_table())
        return 0

    print(rotator.format_status_table())
    return 0
