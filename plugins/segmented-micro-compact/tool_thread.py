"""Optional, fail-closed compaction of completed tool threads in conversation history.

Scans for fully resolved assistant+tool threads (including those split or bounded
by out-of-band user steer messages) and compacts them into durable reference markers.
"""
from __future__ import annotations

import logging
from typing import Any

from agent.model_metadata import estimate_messages_tokens_rough

try:
    from .markers import TOOL_THREAD_MARKER, tool_summary_message
    from .persistence import commit_tool_thread
    from .skill_retention import compaction_parts
except (ImportError, ValueError):
    from markers import TOOL_THREAD_MARKER, tool_summary_message
    from persistence import commit_tool_thread
    from skill_retention import compaction_parts

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


def _valid_summary(summary: Any) -> bool:
    if not isinstance(summary, str) or not summary.strip() or len(summary) > 2400:
        return False
    bullets = [line.strip() for line in summary.splitlines() if line.strip()]
    return 1 <= len(bullets) <= 8 and all(line.startswith("- ") for line in bullets)


def _batch_summaries(raw: Any, count: int) -> list[str] | None:
    """Reject missing, duplicate, unexpected IDs and malformed output atomically."""
    import json
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, list) or len(parsed) != count:
        return None
    found = {}
    for item in parsed:
        if not isinstance(item, dict) or set(item) != {"id", "summary"}:
            return None
        ident, summary = item["id"], item["summary"]
        if type(ident) is not int or not 0 <= ident < count or ident in found:
            return None
        if not _valid_summary(summary):
            return None
        found[ident] = summary
    return [found[i] for i in range(count)]


def compact_tool_thread(engine: Any, messages: list) -> list:
    """Plan from ORIGINAL rows, call once, validate everything, commit once.

    Never feed a produced marker to another summarizer. Independent spans are
    batched so OOB user boundaries/final answers stay in their original positions.
    Bounded input/output budgets defer excess spans to later turns, not retries
    within this pass. Reentrant invocations cannot start another summary call.
    """
    import hashlib
    if getattr(engine, "_tool_thread_inflight", False):
        return messages
    engine._tool_thread_inflight = True
    fingerprints = []
    failed = None
    try:
        failed = getattr(engine, "_tool_thread_failed_inputs", None)
        if failed is None:
            engine._tool_thread_failed_inputs = failed = {}
        spans, items, fingerprints = [], [], []
        max_spans = max(1, min(25, (engine.max_summary_tokens or 1500) // 1500))
        input_budget = max(1, int(engine.context_length * 0.5) - 1500 * max_spans - 1000)
        used = 0
        candidates = (
            part for start, end in find_tool_threads(messages)
            for part in compaction_parts(messages, start, end)
        )
        for start, end, retained, rows in candidates:
            if any(engine._is_context_summary_message(row) for row in rows):
                continue
            text = engine._serialize_for_summary(rows)
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            size = estimate_messages_tokens_rough(rows)
            if digest in failed or size + used > input_budget:
                continue
            spans.append((start, end, retained))
            items.append({"id": len(items), "text": text})
            fingerprints.append(digest)
            used += size
            if len(spans) >= max_spans:
                break
        if not spans:
            return messages
        skip, watermark = engine._micro_start_watermark(messages)
        if skip:
            return messages
        if len(items) == 1:
            raw = engine._tool_thread_summarize_one(items[0]["text"])
            summaries = [raw] if _valid_summary(raw) else None
        else:
            raw = engine._tool_thread_summarize_batch(items)
            summaries = _batch_summaries(raw, len(items))
        if summaries is None:
            for digest in fingerprints:
                failed[digest] = True
            while len(failed) > 128:
                del failed[next(iter(failed))]
            return messages

        result = []
        cursor = engine._micro_compact_cursor
        last = 0
        saved = 0
        accepted = 0
        for (start, end, retained), summary, digest in zip(spans, summaries, fingerprints):
            marker = tool_summary_message(summary)
            marker[TOOL_THREAD_MARKER] = True
            before = estimate_messages_tokens_rough(messages[start:end])
            replacement = retained + [marker]
            after = estimate_messages_tokens_rough(replacement)
            result.extend(messages[last:start])
            if after >= before:
                result.extend(messages[start:end])
                failed[digest] = True
            else:
                result.extend(replacement)
                saved += before - after
                accepted += 1
                if engine._micro_compact_cursor >= end:
                    cursor -= end - start - len(replacement)
                elif engine._micro_compact_cursor > start:
                    # El cursor es una frontera, no el índice del marcador.
                    # Los spans posteriores al cursor original no lo desplazan.
                    cursor = len(result)
            last = end
        result.extend(messages[last:])
        while len(failed) > 128:
            del failed[next(iter(failed))]
        if not accepted or not commit_tool_thread(engine, messages, result, watermark):
            return messages
        engine._micro_compact_cursor = cursor
        engine._flush_scan_cursor_invalidated = True
        engine._tool_thread_passes += 1
        engine._tool_thread_tokens_saved += saved
        logger.info("Tool-thread batch: spans=%d calls=1 tokens_saved=%d", accepted, saved)
        return result
    except Exception:
        if failed is not None:
            for digest in fingerprints:
                failed[digest] = True
            while len(failed) > 128:
                del failed[next(iter(failed))]
        logger.warning("Tool-thread pass failed; keeping original transcript", exc_info=True)
        return messages
    finally:
        engine._tool_thread_inflight = False
