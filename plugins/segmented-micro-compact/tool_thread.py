"""Optional, fail-closed compaction of completed tool threads in conversation history.

Scans for fully resolved assistant+tool threads (including those split or bounded
by out-of-band user steer messages) and compacts them into durable reference markers.
"""
from __future__ import annotations

import logging
from typing import Any

from agent.model_metadata import estimate_messages_tokens_rough

try:
    from .markers import TOOL_THREAD_MARKER, render_tool_thread_summary
    from .persistence import commit_tool_thread
except (ImportError, ValueError):
    from markers import TOOL_THREAD_MARKER, render_tool_thread_summary
    from persistence import commit_tool_thread

logger = logging.getLogger(__name__)


def find_tool_threads(messages: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Find all resolved, contiguous tool thread spans [start, end) in messages.

    A tool thread span begins on an assistant message with function tool_calls,
    continues through matching tool results, and completes when all tool calls
    are answered without dangling IDs or unhandled interleavings.
    """
    if not messages or not all(isinstance(m, dict) for m in messages):
        return []

    spans: list[tuple[int, int]] = []
    i = 0
    n = len(messages)

    while i < n:
        row = messages[i]
        prev = messages[i - 1] if i > 0 else None
        # A valid tool thread start cannot immediately follow an unhandled tool
        # result or an assistant message with unresolved/pending tool calls.
        prev_ok = prev is None or (prev.get("role") != "tool" and not prev.get("tool_calls"))

        if (
            isinstance(row, dict)
            and row.get("role") == "assistant"
            and isinstance(row.get("tool_calls"), list)
            and row.get("tool_calls")
            and not row.get(TOOL_THREAD_MARKER)
            and not row.get("function_call")
            and prev_ok
        ):
            start = i
            pending: set[str] = set()
            seen: set[str] = set()
            valid = True

            while i < n:
                curr = messages[i]
                if not isinstance(curr, dict) or curr.get(TOOL_THREAD_MARKER) or curr.get("function_call"):
                    valid = False
                    break
                role = curr.get("role")
                if role == "assistant":
                    calls = curr.get("tool_calls")
                    if calls:
                        # Cannot start a new batch of tool calls while earlier calls are still pending
                        if pending or not isinstance(calls, list):
                            valid = False
                            break
                        for call in calls:
                            if not isinstance(call, dict) or call.get("type", "function") != "function":
                                valid = False
                                break
                            ident = call.get("id")
                            fn = call.get("function")
                            if (
                                not isinstance(ident, str)
                                or not ident
                                or ident in seen
                                or not isinstance(fn, dict)
                                or not fn.get("name")
                            ):
                                valid = False
                                break
                            seen.add(ident)
                            pending.add(ident)
                        if not valid:
                            break
                    else:
                        # Assistant text response without tool_calls closes the tool thread
                        break
                elif role == "tool":
                    ident = curr.get("tool_call_id")
                    if not isinstance(ident, str) or ident not in pending:
                        valid = False
                        break
                    pending.remove(ident)
                else:
                    # Boundary reached (e.g. out-of-band user steer or next user message)
                    break
                i += 1

            end = i
            # To be a valid completed tool thread:
            # - Must not have encountered malformed rows (valid == True)
            # - All tool calls must be fully answered (not pending)
            # - At least one tool call/result pair was processed (bool(seen))
            # - Must not end mid-execution on a tool message at EOF (end == n on role 'tool' is unfinished)
            if valid and not pending and seen:
                if end < n:
                    spans.append((start, end))
        else:
            i += 1

    return spans


def find_tool_thread(messages: list[dict[str, Any]]) -> tuple[int, int] | None:
    """Return the first uncompacted resolved tool thread span in messages, or None."""
    spans = find_tool_threads(messages)
    return spans[0] if spans else None


def compact_tool_thread(engine: Any, messages: list) -> list:
    """Compact all resolved tool thread spans in messages."""
    failed_spans: set[tuple[int, int]] = set()
    attempts = 0
    max_attempts = 25

    while attempts < max_attempts:
        candidate_spans = [s for s in find_tool_threads(messages) if s not in failed_spans]
        if not candidate_spans:
            break
        span = candidate_spans[0]
        start, end = span
        rows = messages[start:end]

        try:
            skip, watermark = engine._micro_start_watermark(messages)
            if skip:
                break
            text = engine._serialize_for_summary(rows)
            summary = engine._tool_thread_summarize_one(text)
            if not isinstance(summary, str) or not summary.strip():
                failed_spans.add(span)
                attempts += 1
                continue

            # Reject malformed prose or unexpectedly verbose model output.
            bullets = [b.strip() for b in summary.strip().splitlines() if b.strip()]
            if (
                not bullets
                or len(bullets) > 8
                or any(not b.startswith("- ") for b in bullets)
                or len(summary) > 2400
            ):
                failed_spans.add(span)
                attempts += 1
                continue

            block = render_tool_thread_summary(summary)
            marker = {
                "role": "assistant",
                # Keep the reference visible in history before the final response.
                # The explicit sidecar preserves the same bytes on request replay.
                "content": block,
                "api_content": block,
                TOOL_THREAD_MARKER: True,
            }

            before = estimate_messages_tokens_rough(rows)
            after = estimate_messages_tokens_rough([marker])
            if after >= before:
                # Compression would not save tokens; skip this span
                failed_spans.add(span)
                attempts += 1
                continue

            result = messages[:start] + [marker] + messages[end:]
            if not commit_tool_thread(engine, messages, result, watermark):
                failed_spans.add(span)
                attempts += 1
                continue

            # Translate the old-exchange cursor if this splice shortened its prefix.
            cursor = engine._micro_compact_cursor
            if cursor >= end:
                engine._micro_compact_cursor = cursor - (end - start) + 1
            elif cursor > start:
                engine._micro_compact_cursor = start

            engine._flush_scan_cursor_invalidated = True
            engine._tool_thread_passes += 1
            engine._tool_thread_tokens_saved += before - after
            logger.info("Tool-thread compaction: rows=%d->1 tokens_est=%d->%d", len(rows), before, after)

            # Succeeded: update messages and reset failed_spans for the newly-indexed messages
            messages = result
            failed_spans.clear()
            attempts += 1

        except Exception:
            logger.warning("Tool-thread summary failed; keeping original transcript", exc_info=True)
            failed_spans.add(span)
            attempts += 1

    return messages
