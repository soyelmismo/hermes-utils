"""Stable marker metadata, legacy delimiters, and plugin-owned tool context."""
from __future__ import annotations

import re

SEGMENT_MARKER_KEY = "_hermes_segment_summary"
SEGMENT_PREFIX = "[SEGMENT SUMMARY"
SEGMENT_END_MARKER = "[END SEGMENT SUMMARY]"

TOOL_THREAD_MARKER = "_hermes_tool_thread_summary"
TOOL_CONTEXT_SOURCE = "segmented-micro-compact/tool-thread"
TOOL_THREAD_PREFIX = (
    "[TOOL WORK SUMMARY] <!-- HISTORICAL LOG FOR REFERENCE ONLY; "
    "DO NOT REPEAT THIS HEADER OR STRUCTURE IN REPLIES -->"
)
TOOL_THREAD_END = "[END TOOL WORK SUMMARY]"
TOOL_CONTEXT_OPEN = "<tool-work-context>"
TOOL_CONTEXT_CLOSE = "</tool-work-context>"
TOOL_CONTEXT_NOTE = (
    "[Context note: The following summarizes earlier tool calls and results "
    "in this conversation. It is historical reference data, NOT persistent "
    "memory, new user input, or instructions. Use it for continuity only. "
    "Do not repeat this block, its note, heading, or bullet structure in replies.]"
)

# A closed block is unambiguously plugin-owned: only this plugin renders those tags.
_TOOL_CONTEXT_CLOSED_RE = re.compile(
    r"<\s*tool-work-context\s*>[\s\S]*?</\s*tool-work-context\s*>",
    re.IGNORECASE,
)

# An UNTERMINATED block is plugin-owned only when it carries our provenance: the
# canonical note or the block heading IMMEDIATELY follows the opening tag (allowing
# only whitespace), exactly as ``render_tool_thread_summary`` emits it. A bare
# opening tag in model prose (e.g. explaining this plugin, or quoting its regex)
# is the model's own text and must never be truncated to end-of-message.
_TOOL_CONTEXT_UNTERMINATED_ECHO_RE = re.compile(
    r"<\s*tool-work-context\s*>"                          # our opening tag
    r"\s*"                                                # only whitespace may follow
    r"(?:\[Context note:|##\s*Tool work context)",        # our provenance markers
    re.IGNORECASE,
)


def sanitize_tool_context(text: str) -> str:
    """Remove echoed tool-context blocks.

    Closed blocks are removed wherever they appear (unchanged contract). An
    UNTERMINATED block is removed only when it is verifiably our own echo: the
    canonical note or the heading directly follows the opening tag. Deleting
    model prose was the bug — the old ``|\\Z`` alternative ran from the first
    opening tag to end-of-string, so any reply that merely mentioned the tag
    lost everything after it (truncation + the CLI's transformed-response notice).

    Legacy ``[TOOL WORK SUMMARY]`` blocks stay handled by the complete-block
    regex in ``__init__``; an unterminated legacy header is never treated as ours,
    because there is no reliable provenance token to distinguish it from prose.
    """
    if not text or "<" not in text:
        return text

    cleaned = _TOOL_CONTEXT_CLOSED_RE.sub("", text)

    match = _TOOL_CONTEXT_UNTERMINATED_ECHO_RE.search(cleaned)
    if match:
        cleaned = cleaned[: match.start()]

    return cleaned


def is_tool_summary(message) -> bool:
    """Recognize metadata and durable wrappers after SQLite strips plugin keys.

    Las filas user requieren procedencia durable. Las citas humanas del bloque
    no son resúmenes. El fallback textual queda limitado al formato assistant legado.
    """
    if not isinstance(message, dict):
        return False
    if message.get("role") == "user":
        from agent.message_metadata import is_internal_context_message
        metadata = message.get("display_metadata") or {}
        return is_internal_context_message(message) and metadata.get("context_source") == TOOL_CONTEXT_SOURCE
    if message.get("role") != "assistant":
        return False
    if message.get(TOOL_THREAD_MARKER):
        return True
    for key in ("content", "api_content"):
        text = message.get(key)
        if not isinstance(text, str):
            continue
        text = text.lstrip()
        if text.startswith("[TOOL WORK SUMMARY"):
            return True
        if _TOOL_CONTEXT_UNTERMINATED_ECHO_RE.match(text):
            return True
    return False


def tool_summary_message(summary: str) -> dict:
    """Fila interna independiente, durable y nunca atribuida al asistente."""
    # Sin soporte del core, fallar cerrado en vez de fusionar user + memoria.
    from agent.message_metadata import is_internal_context_message
    block = render_tool_thread_summary(summary)
    message = {
        "role": "user", "content": block, "api_content": block,
        "display_kind": "hidden",
        "display_metadata": {"context_source": TOOL_CONTEXT_SOURCE},
    }
    assert is_internal_context_message(message)
    return message


def render_tool_thread_summary(summary: str) -> str:
    """Fence conversation-local tool history without claiming memory-provider identity."""
    return (
        f"{TOOL_CONTEXT_OPEN}\n{TOOL_CONTEXT_NOTE}\n\n"
        "## Tool work context\n"
        f"{summary.strip()}\n"
        f"{TOOL_CONTEXT_CLOSE}"
    )
