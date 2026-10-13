"""Segment cadence and independent marker splicing, inheriting general compaction."""
from __future__ import annotations
import copy
import logging
from typing import Any, Optional
from agent.context_compressor import ContextCompressor

try:
    from .markers import SEGMENT_MARKER_KEY, SEGMENT_PREFIX, SEGMENT_END_MARKER, TOOL_THREAD_MARKER, is_tool_summary
    from .prompts import build_segment_summary_prompt
    from .summarizer import SummarizerMixin
except (ImportError, ValueError):
    from markers import SEGMENT_MARKER_KEY, SEGMENT_PREFIX, SEGMENT_END_MARKER, TOOL_THREAD_MARKER, is_tool_summary
    from prompts import build_segment_summary_prompt
    from summarizer import SummarizerMixin

logger = logging.getLogger(__name__)


class SegmentedMicroCompactEngine(SummarizerMixin, ContextCompressor):
    engine_name = "segmented-micro-compact"

    def __init__(self, model: str = "", **kwargs: Any):
        # The plugin loader instantiates the engine via register(ctx) with NO model;
        # the host calls update_model(model, context_length, ...) right after selection
        # (see agent_init._build_context_engine). Make model optional so construction
        # succeeds and let update_model fill the real values in.
        self._tool_thread_compact_enabled = kwargs.pop("tool_thread_compact", False) is True
        self._tool_thread_passes = 0
        self._tool_thread_tokens_saved = 0
        self._tool_thread_failed_inputs = {}
        self._tool_thread_inflight = False
        super().__init__(model=model, **kwargs)

    @property
    def name(self) -> str:
        return self.engine_name

    def clone_for_agent(self) -> SegmentedMicroCompactEngine:
        """Return a clean per-agent instance preserving plugin configuration."""
        return copy.deepcopy(self)

    @classmethod
    def _is_segment_marker(cls, msg: Any) -> bool:
        if not isinstance(msg, dict):
            return False
        if bool(msg.get(SEGMENT_MARKER_KEY)):
            return True
        c = msg.get("content")
        return isinstance(c, str) and c.startswith(SEGMENT_PREFIX)

    @classmethod
    def classify_summary_content(cls, content: Any) -> Optional[str]:
        if isinstance(content, str) and content.lstrip().startswith(SEGMENT_PREFIX):
            return "standalone"
        return super().classify_summary_content(content)

    @classmethod
    def _is_context_summary_message(cls, message: Any) -> bool:
        if not isinstance(message, dict):
            return False
        if cls._is_segment_marker(message) or is_tool_summary(message):
            return True
        return super()._is_context_summary_message(message)

    def _resolve_compact_cursor(self, messages: list[dict[str, Any]], head_end: int, tail_start: int) -> int:
        """Index of the first message not yet absorbed into segment summaries:
        the in-memory cursor when valid, else just past the last summary marker."""
        if head_end < self._micro_compact_cursor < tail_start:
            return self._micro_compact_cursor
        summaries = (i for i in range(head_end, tail_start) if self._is_context_summary_message(messages[i]))
        last = max(summaries, default=-1)
        cursor = last + 1 if last >= head_end else head_end
        self._micro_compact_cursor = cursor
        return cursor

    def _cursor_after_splice(self, result: list[dict[str, Any]], fallback: int) -> int:
        """Cursor position just past the newest summary marker in result."""
        try:
            from agent.micro_compaction import _is_summary_marker
        except Exception:
            _is_summary_marker = lambda m: False
        return next(
            (idx + 1 for idx in range(len(result) - 1, -1, -1)
             if self._is_segment_marker(result[idx]) or _is_summary_marker(result[idx])),
            fallback,
        )

    def _render_segment_marker_content(self, summary_text: str, index: int) -> str:
        """Render a segment marker body.

        The built-in wraps micro markers with its own heading/end markers and the
        batch pass looks for those; we deliberately use our OWN prefix + end marker
        and DO NOT set the built-in's summary-metadata key, so batch compaction
        treats a segment marker as ordinary conversation to fold into the next
        handoff rather than as a summary to replace.
        """
        return f"{SEGMENT_PREFIX} #{index}]\n{summary_text.strip()}\n{SEGMENT_END_MARKER}"

    def _segment_count(self, messages: list) -> int:
        return sum(1 for m in messages if self._is_segment_marker(m))

    def _splice_micro_compact_result(
        self, messages: list[dict[str, Any]], splice_start: int, splice_end: int,
        supersede: bool = True,
    ) -> list[dict[str, Any]]:
        """Replace ``messages[splice_start:splice_end]`` with ONE NEW segment marker.

        This is the whole point of the plugin. The built-in version keeps only the
        newest marker (``supersede`` -> drop every older micro marker). Here
        ``supersede`` is ignored: we never delete prior segment markers, we only
        replace the absorbed exchange with a fresh, independent summary marker.
        """
        summary_text = self._micro_compact_rolling_summary
        if not summary_text.strip():
            return messages

        index = self._segment_count(messages) + 1
        marker = {
            "role": "assistant",
            "content": self._render_segment_marker_content(summary_text, index),
            SEGMENT_MARKER_KEY: True,
        }
        result = messages[:splice_start] + [marker] + messages[splice_end:]
        # Merge user turns left dangling by the splice so alternation stays valid
        # (same repair the built-in performs; reuse its own implementation).
        try:
            result = self._merge_adjacent_user_turns(result)
        except Exception:
            logger.debug("segment splice: adjacent-user merge skipped", exc_info=True)
        return result

    def _build_segment_summary_prompt(self, exchange_text):
        return build_segment_summary_prompt(exchange_text)

    def _micro_compact(self, messages: list[dict[str, Any]], *, turn_session_id: Optional[str] = None):
        """One pass: summarize the next un-absorbed exchange into its OWN marker.

        Cadence (``_micro_compact_every_n_turns``) is honoured exactly like the
        built-in: each pass still costs one prompt-cache break. The difference from
        the built-in is only WHAT gets written: a new independent segment marker
        instead of an update to a single rolling marker.
        """
        if not self._micro_compact_enabled:
            return messages
        if self._tool_thread_compact_enabled:
            # Alternative policy, NOT a preprocessing stage for segment summaries.
            # Tool work is summarized once; user/final messages remain verbatim.
            try:
                from .tool_thread import compact_tool_thread
            except (ImportError, ValueError):
                from tool_thread import compact_tool_thread
            return compact_tool_thread(self, messages)

        every_n = max(1, int(self._micro_compact_every_n_turns or 1))
        if every_n > 1:
            self._micro_compact_turns_since_pass += 1
            if self._micro_compact_turns_since_pass < every_n:
                return messages
            self._micro_compact_turns_since_pass = 0

        if len(messages) < 4:
            return messages

        exchange = self._next_exchange(messages)
        if exchange is None:
            return messages
        exchange_start, exchange_end = exchange

        # Pre-pass baseline and watermark for durable DB commit
        _held = list(messages)
        _skip, _start_watermark = self._micro_start_watermark(_held)
        if _skip:
            return messages
        _pre_cursor = self._micro_compact_cursor

        # Context serialization: include the preceding user prompt (if present) so
        # the summarizer knows what question or steer was being addressed, but
        # splice ONLY exchange_start:exchange_end so user turns remain verbatim.
        ctx_start = exchange_start - 1 if (
            exchange_start > 0 and messages[exchange_start - 1].get("role") == "user"
        ) else exchange_start

        try:
            exchange_text = self._serialize_for_summary(messages[ctx_start:exchange_end])
        except Exception:
            exchange_text = self._fallback_serialize(messages[ctx_start:exchange_end])

        updated = self._segment_summarize_one(exchange_text)
        if updated is None:
            self._record_micro_failure(exchange_start, exchange_end)
            return messages

        self._micro_compact_rolling_summary = updated
        self._reset_micro_failure_tracking()

        result = self._splice_micro_compact_result(messages, exchange_start, exchange_end, supersede=False)
        self._micro_compact_cursor = self._cursor_after_splice(result, exchange_start + 1)
        if not self._sync_micro_compact_to_db(result, held=_held, start_watermark=_start_watermark):
            self._micro_compact_cursor = _pre_cursor
            return messages

        self._micro_compact_passes += 1
        return result

    @staticmethod
    def _fallback_serialize(rows: list[dict[str, Any]]) -> str:
        """Minimal serializer used only if the built-in helper ever changes shape."""
        parts = []
        for m in rows:
            if not isinstance(m, dict):
                continue
            role = m.get("role", "?")
            content = m.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            parts.append(f"[{role}] {content}")
        return "\n\n".join(parts)

    def on_session_reset(self) -> None:
        super().on_session_reset()
        self._tool_thread_failed_inputs.clear()
        self._tool_thread_inflight = False
