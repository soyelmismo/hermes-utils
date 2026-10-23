"""Plugin registration and stable public exports."""
from __future__ import annotations
import re
from typing import Any

try:
    from .engine import SegmentedMicroCompactEngine
    from .markers import (
        SEGMENT_MARKER_KEY,
        SEGMENT_PREFIX,
        SEGMENT_END_MARKER,
        TOOL_THREAD_MARKER,
        TOOL_THREAD_PREFIX,
        TOOL_THREAD_END,
        sanitize_tool_context,
    )
    from .compat_patch import apply_runtime_compat_patches
except (ImportError, ValueError):
    from engine import SegmentedMicroCompactEngine
    from markers import (
        SEGMENT_MARKER_KEY,
        SEGMENT_PREFIX,
        SEGMENT_END_MARKER,
        TOOL_THREAD_MARKER,
        TOOL_THREAD_PREFIX,
        TOOL_THREAD_END,
        sanitize_tool_context,
    )
    from compat_patch import apply_runtime_compat_patches

# Apply in-memory compatibility hooks on import so Hermes Agent core stays unmodified
apply_runtime_compat_patches()

__all__ = [
    "SegmentedMicroCompactEngine",
    "SEGMENT_MARKER_KEY",
    "SEGMENT_PREFIX",
    "SEGMENT_END_MARKER",
    "TOOL_THREAD_MARKER",
    "TOOL_THREAD_PREFIX",
    "TOOL_THREAD_END",
    "register",
    "strip_leaked_tool_summary",
]


_LEGACY_TOOL_SUMMARY_RE = re.compile(
    r"^[ \t]*\[TOOL WORK SUMMARY[^\]\r\n]*\]"
    r"[^\r\n]*\r?\n[\s\S]*?\[END TOOL WORK SUMMARY\]",
    re.MULTILINE,
)


def strip_leaked_tool_summary(response_text: str, **kwargs: Any) -> str | None:
    """Scrub plugin-owned tool context and complete legacy blocks, including mid-reply.

    Returns ``None`` when nothing was removed, so the host never records a
    transform (and never prints its post-stream transform notice) for a reply we
    left untouched. Trailing whitespace alone must not count as a change.
    """
    if not isinstance(response_text, str):
        return None
    text = sanitize_tool_context(response_text)
    text = _LEGACY_TOOL_SUMMARY_RE.sub("", text)
    if text == response_text:
        return None
    return text.strip()


def register(ctx) -> None:
    from hermes_cli.config import load_config
    compression = load_config().get("compression", {})
    enabled = compression.get("tool_thread_compact", False) is True
    ctx.register_context_engine(SegmentedMicroCompactEngine(tool_thread_compact=enabled))
    if hasattr(ctx, "register_hook"):
        ctx.register_hook("transform_llm_output", strip_leaked_tool_summary)
