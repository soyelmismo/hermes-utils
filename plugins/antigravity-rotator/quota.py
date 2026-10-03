"""Quota parsing, pool normalization, and scoring for antigravity-rotator.

Normalized pool format per account:
    pools = {
        "gemini": {
            "5h": {"remaining_fraction": float|None, "reset_at": iso|None},
            "weekly": {"remaining_fraction": float|None, "reset_at": iso|None},
        },
        "claude_gpt": { ... },
    }

None means unknown (never assume 100%).
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pool determination
# ---------------------------------------------------------------------------

def pool_for_model(model: str) -> str:
    """Return 'claude_gpt' if model contains 'claude' or 'gpt', else 'gemini'."""
    if not model:
        return "gemini"
    m = model.lower()
    if "claude" in m or "gpt" in m:
        return "claude_gpt"
    return "gemini"


# ---------------------------------------------------------------------------
# OpenProxy -> normalized pools
# ---------------------------------------------------------------------------

def _parse_reset_time(raw: Any) -> Optional[str]:
    """Coerce a reset value to ISO 8601 UTC string, or None."""
    if raw is None or raw == "" or raw == "N/A":
        return None
    s = str(raw).strip()
    try:
        dt_obj = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt_obj.isoformat()
    except (ValueError, AttributeError):
        pass

    try:
        ts = float(s)
        if ts > 1e11:
            ts /= 1000.0
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except (ValueError, TypeError, OverflowError):
        pass

    total = 0
    for value, unit in re.findall(r"(\d+)\s*([hHmMsS])", s):
        v = int(value)
        if unit.lower() == "h":
            total += v * 3600
        elif unit.lower() == "m":
            total += v * 60
        elif unit.lower() == "s":
            total += v
    if total > 0:
        return (
            datetime.now(timezone.utc).replace(microsecond=0)
            + timedelta(seconds=total)
        ).isoformat()
    return None


def _fraction(used: Any, limit: Any) -> Optional[float]:
    """Compute remaining_fraction = (limit - used) / limit. None if unknowable."""
    try:
        u, lim = float(used), float(limit)
    except (TypeError, ValueError):
        return None
    if lim <= 0:
        return 0.0
    return max(0.0, (lim - u) / lim)


def pools_from_openproxy(account: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Map OpenProxy aggregate + model_detail fields to normalized pools.

    Aggregate quota_session_* / quota_weekly_* -> gemini pool.
    model_detail entries with 'claude' or 'gpt' in model_id -> claude_gpt pool.
    """
    pools: Dict[str, Dict[str, Any]] = {}

    sess_frac = _fraction(
        account.get("quota_session_used"),
        account.get("quota_session_limit"),
    )
    sess_reset = _parse_reset_time(account.get("quota_session_reset_at"))
    wk_frac = _fraction(
        account.get("quota_weekly_used"),
        account.get("quota_weekly_limit"),
    )
    wk_reset = _parse_reset_time(account.get("quota_weekly_reset_at"))

    gemini: Dict[str, Any] = {}
    if sess_frac is not None or sess_reset is not None:
        gemini["5h"] = {"remaining_fraction": sess_frac, "reset_at": sess_reset}
    if wk_frac is not None or wk_reset is not None:
        gemini["weekly"] = {"remaining_fraction": wk_frac, "reset_at": wk_reset}
    if gemini:
        pools["gemini"] = gemini

    details = account.get("model_detail") or []
    if not isinstance(details, list):
        details = []

    claude_session: Optional[Dict[str, Any]] = None
    claude_weekly: Optional[Dict[str, Any]] = None

    for detail in details:
        if not isinstance(detail, dict):
            continue
        mid = str(detail.get("model_id", "")).lower()
        if "claude" not in mid and "gpt" not in mid:
            continue

        d_frac = _fraction(detail.get("session_used"), detail.get("session_limit"))
        d_reset = _parse_reset_time(detail.get("session_reset_at"))
        if d_frac is not None or d_reset is not None:
            if claude_session is None or (
                d_frac is not None
                and (
                    claude_session["remaining_fraction"] is None
                    or d_frac < claude_session["remaining_fraction"]
                )
            ):
                claude_session = {"remaining_fraction": d_frac, "reset_at": d_reset}

        w_frac = _fraction(detail.get("weekly_used"), detail.get("weekly_limit"))
        w_reset = _parse_reset_time(detail.get("weekly_reset_at"))
        if w_frac is not None or w_reset is not None:
            if claude_weekly is None or (
                w_frac is not None
                and (
                    claude_weekly["remaining_fraction"] is None
                    or w_frac < claude_weekly["remaining_fraction"]
                )
            ):
                claude_weekly = {"remaining_fraction": w_frac, "reset_at": w_reset}

    claude_gpt: Dict[str, Any] = {}
    if claude_session is not None:
        claude_gpt["5h"] = claude_session
    if claude_weekly is not None:
        claude_gpt["weekly"] = claude_weekly
    if claude_gpt:
        pools["claude_gpt"] = claude_gpt

    return pools


# ---------------------------------------------------------------------------
# Local backend: agy /usage JSON -> normalized pools
# ---------------------------------------------------------------------------

def _find_groups(data: Any, depth: int = 0) -> Optional[List[dict]]:
    """Recursively locate the groups array in /usage JSON."""
    if depth > 5:
        return None
    if isinstance(data, dict):
        if "groups" in data and isinstance(data["groups"], list):
            groups = data["groups"]
            if (
                groups
                and isinstance(groups[0], dict)
                and any(k in groups[0] for k in ("name", "buckets", "models"))
            ):
                return groups
        for value in data.values():
            result = _find_groups(value, depth + 1)
            if result is not None:
                return result
    elif isinstance(data, list):
        if (
            data
            and isinstance(data[0], dict)
            and any(k in data[0] for k in ("name", "buckets", "models"))
        ):
            return data
        for item in data:
            result = _find_groups(item, depth + 1)
            if result is not None:
                return result
    return None


def pools_from_usage_json(raw_output: str) -> Dict[str, Dict[str, Any]]:
    """Parse 'agy -p /usage --output-format json' output into normalized pools."""
    if not raw_output or not raw_output.strip():
        return {}
    try:
        data = json.loads(raw_output.strip())
    except json.JSONDecodeError:
        m = re.search(r"[\[{].*[\]}]", raw_output, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
            except json.JSONDecodeError:
                return {}
        else:
            return {}

    groups = _find_groups(data)
    if not groups:
        return {}

    pools: Dict[str, Dict[str, Any]] = {}

    for group in groups:
        if not isinstance(group, dict):
            continue
        name = group.get("name", "").lower()
        if "gemini" in name:
            pool_key = "gemini"
        elif any(k in name for k in ("claude", "gpt", "3p", "third")):
            pool_key = "claude_gpt"
        else:
            continue

        buckets = group.get("buckets", [])
        if not isinstance(buckets, list):
            continue

        pool_data: Dict[str, Any] = pools.get(pool_key, {})
        for bucket in buckets:
            if not isinstance(bucket, dict):
                continue
            window = bucket.get("window", "").lower()
            if "5h" in window or "session" in window:
                wkey = "5h"
            elif "week" in window:
                wkey = "weekly"
            else:
                continue

            remaining = bucket.get("remaining")
            limit = bucket.get("limit")
            reset = bucket.get("reset_at", bucket.get("reset", bucket.get("resets_at")))

            frac = None
            if remaining is not None and limit is not None:
                try:
                    r, lim = float(remaining), float(limit)
                    frac = (r / lim) if lim > 0 else 0.0
                except (TypeError, ValueError):
                    pass

            pool_data[wkey] = {
                "remaining_fraction": frac,
                "reset_at": _parse_reset_time(reset) if reset else None,
            }

        if pool_data:
            pools[pool_key] = pool_data

    return pools


# ---------------------------------------------------------------------------
# Usability gate
# ---------------------------------------------------------------------------

def is_usable_for_pool(
    pools: Dict[str, Dict[str, Any]],
    pool_key: str,
    threshold_percent: float = 95.0,
) -> bool:
    """True if the account has quota above the exhaustion threshold for pool_key.

    Threshold formula: remaining_fraction > (1 - threshold_percent / 100).
    An account is usable when every known window in the pool satisfies this.
    Unknown windows do NOT exclude the account (they sort after known).
    """
    threshold_remaining = (
        (1.0 - threshold_percent / 100.0)
        if threshold_percent > 1.0
        else (1.0 - threshold_percent)
    )
    pool = pools.get(pool_key, {})
    if not pool:
        return True

    for wkey in ("5h", "weekly"):
        window = pool.get(wkey, {})
        frac = window.get("remaining_fraction")
        if frac is not None and frac <= threshold_remaining:
            return False
    return True


# ---------------------------------------------------------------------------
# Score / ordering
# ---------------------------------------------------------------------------

def calculate_score(
    pools: Dict[str, Dict[str, Any]],
    pool_key: str,
) -> Tuple[int, float, int, float, float]:
    """Score for selection:

    (is_5h_known, 5h_remaining, is_weekly_known, weekly_remaining, -reset_proximity).
    Higher is better. Unknown values sort after known.
    Sooner reset 5h is preferred on tie-break (-proximity is higher).
    """
    pool = pools.get(pool_key, {})

    five_h = pool.get("5h", {})
    rem_5h = five_h.get("remaining_fraction")
    is_5h_known = 1 if rem_5h is not None else 0
    rem_5h_val = rem_5h if rem_5h is not None else -1.0

    weekly = pool.get("weekly", {})
    rem_wk = weekly.get("remaining_fraction")
    is_wk_known = 1 if rem_wk is not None else 0
    rem_wk_val = rem_wk if rem_wk is not None else -1.0

    reset_str = five_h.get("reset_at")
    if reset_str:
        try:
            reset_dt = datetime.fromisoformat(reset_str.replace("Z", "+00:00"))
            proximity = (reset_dt - datetime.now(timezone.utc)).total_seconds()
        except (ValueError, AttributeError):
            proximity = float("inf")
    else:
        proximity = float("inf")

    return (is_5h_known, rem_5h_val, is_wk_known, rem_wk_val, -proximity)


def earliest_pool_reset(
    accounts_pools: List[Tuple[Any, Dict[str, Dict[str, Any]]]],
    pool_key: str,
) -> Optional[str]:
    """Find the nearest reset ISO timestamp for pool_key across accounts."""
    best: Optional[datetime] = None
    for _id, pools in accounts_pools:
        pool = pools.get(pool_key, {})
        for wkey in ("5h", "weekly"):
            rst = pool.get(wkey, {}).get("reset_at")
            if not rst:
                continue
            try:
                dt_obj = datetime.fromisoformat(str(rst).replace("Z", "+00:00"))
                if best is None or dt_obj < best:
                    best = dt_obj
            except (ValueError, AttributeError):
                pass
    return best.isoformat() if best else None
