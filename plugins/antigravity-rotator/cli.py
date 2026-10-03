"""CLI subcommands for hermes antigravity [list|switch|refresh|rotate|login|remove]."""

from __future__ import annotations

import argparse
from typing import Optional

try:
    from .rotator import OpenProxyRotator, create_rotator
except ImportError:
    from rotator import OpenProxyRotator, create_rotator


def setup_cli(parser: argparse.ArgumentParser) -> None:
    """Configure argparse subparsers for `hermes antigravity`."""
    sub = parser.add_subparsers(dest="antigravity_action", help="Antigravity account management")

    # list / status
    p_list = sub.add_parser("list", help="Display all Antigravity accounts and quota telemetry")
    p_list.add_argument("--model", default="", help="Optional model name to check pool quota for")

    p_status = sub.add_parser("status", help="Alias for list")
    p_status.add_argument("--model", default="", help="Optional model name to check pool quota for")

    # rotate
    p_rotate = sub.add_parser("rotate", help="Manually rotate to next available account with quota")
    p_rotate.add_argument("--model", default="", help="Optional model name to check pool quota for")

    # switch
    p_switch = sub.add_parser("switch", help="Switch local agy CLI to a specific account or auto-rotate")
    p_switch.add_argument(
        "account_id",
        type=str,
        nargs="?",
        default=None,
        help="Account ID or label to switch to (if omitted, auto-selects account with most quota)",
    )
    p_switch.add_argument("--model", default="", help="Target model name for auto-rotation selection")

    # refresh
    p_refresh = sub.add_parser("refresh", help="Refresh quota usage stats from provider")
    p_refresh.add_argument(
        "account_id",
        type=str,
        nargs="?",
        default=None,
        help="Account ID or label to refresh (if omitted, refreshes all Antigravity accounts)",
    )

    # login
    p_login = sub.add_parser("login", help="Log in and onboard a new local Antigravity account")
    p_login.add_argument("--label", required=True, help="Unique label for the new account")

    # remove
    p_remove = sub.add_parser("remove", help="Remove an account from local registry (files preserved)")
    p_remove.add_argument("label", help="Label of the account to remove")


def handle_cli(args: argparse.Namespace, rotator: Optional[OpenProxyRotator] = None) -> int:
    """Execute `hermes antigravity ...` subcommand."""
    if rotator is None:
        rotator = create_rotator()

    action = getattr(args, "antigravity_action", "") or "list"
    model = getattr(args, "model", "") or ""

    if action in ("list", "status"):
        print(rotator.format_status_table(model=model))
        return 0

    if action == "rotate":
        try:
            _, msg = rotator.rotate_to_next_account(model=model)
            print(f"🔄 {msg}")
            return 0
        except RuntimeError as exc:
            print(f"❌ {exc}")
            return 1

    if action == "switch":
        target_id = getattr(args, "account_id", None)
        if target_id is not None:
            try:
                rotator.apply_account(target_id)
                print(f"✅ Successfully switched agy CLI to account [{target_id}].")
                return 0
            except RuntimeError as exc:
                print(f"❌ Failed to switch to account [{target_id}]: {exc}")
                return 1
        try:
            _, msg = rotator.rotate_to_next_account(model=model)
            print(f"🔄 {msg}")
            return 0
        except RuntimeError as exc:
            print(f"❌ {exc}")
            return 1

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
            print(rotator.format_status_table(model=model))
        return 0

    if action == "login":
        label = getattr(args, "label", "")
        if hasattr(rotator, "login"):
            try:
                res = rotator.login(label)
                print(f"✅ Successfully onboarded account '{res.get('label')}' (email: {res.get('email', 'N/A')}).")
                return 0
            except Exception as exc:
                print(f"❌ Login failed: {exc}")
                return 1
        print("❌ Login subcommand is only supported when using the local backend.")
        return 1

    if action == "remove":
        label = getattr(args, "label", "")
        if hasattr(rotator, "remove_account"):
            if rotator.remove_account(label):
                print(f"✅ Account '{label}' removed from local registry.")
                return 0
            print(f"❌ Account '{label}' not found in registry.")
            return 1
        print("❌ Remove subcommand is only supported when using the local backend.")
        return 1

    print(rotator.format_status_table(model=model))
    return 0
