"""In-memory compatibility patch for internal context boundaries in Hermes Agent.

Replaces the need for modifying Hermes Agent core repository files directly.
Applied dynamically when the segmented-micro-compact plugin is registered/imported.
"""
from __future__ import annotations
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_PATCHED = False


def is_internal_context_message(message: Any) -> bool:
    """Reconoce una fila user interna por procedencia durable, nunca por su prosa."""
    if not isinstance(message, dict) or message.get("role") != "user":
        return False
    metadata = message.get("display_metadata")
    return (
        message.get("display_kind") in ("internal_context", "hidden")
        and isinstance(metadata, dict)
        and isinstance(metadata.get("context_source"), str)
        and bool(metadata["context_source"].strip())
    )


def apply_runtime_compat_patches() -> bool:
    """Apply memory monkeypatches to Hermes Agent core modules idempotently."""
    global _PATCHED
    if _PATCHED:
        return True

    try:
        import agent.message_metadata as mm
        import agent.agent_runtime_helpers as arh
        import agent.context_compressor as cc
        import agent.micro_compaction as mc
        import agent.turn_request_assembly as tra
    except ImportError as e:
        logger.warning("Could not import Hermes Agent core modules for compat patch: %s", e)
        return False

    # 1. message_metadata: export is_internal_context_message
    if not hasattr(mm, "is_internal_context_message"):
        mm.is_internal_context_message = is_internal_context_message

    # 2. agent_runtime_helpers: wrap _merge_consecutive_users
    orig_merge_consecutive = arh._merge_consecutive_users

    def patched_merge_consecutive_users(messages: list[dict]) -> tuple[list[dict], int]:
        from agent.context_compressor import _DB_PERSISTED_MARKER, split_user_originated_turn
        from agent.conversation_compression_archive import MERGED_DURABLE_ROWS
        from agent.turn_context import drop_stale_api_content
        from hermes_state import SessionDB

        def _plain_text(content: Any) -> bool:
            return isinstance(content, str) and not content.startswith(SessionDB._CONTENT_JSON_PREFIX)

        repairs = 0
        merged: list[dict] = []
        for msg in messages:
            prev = merged[-1] if merged and isinstance(merged[-1], dict) else None
            if (
                prev is not None and prev.get("role") == "user"
                and isinstance(msg, dict) and msg.get("role") == "user"
                and not is_internal_context_message(prev) and not is_internal_context_message(msg)
                and split_user_originated_turn(prev)[0] is None
                and prev.get("display_kind") != arh.STEER_DISPLAY_KIND
                and _plain_text(prev.get("content", "")) and _plain_text(msg.get("content", ""))
            ):
                prev_content, new_content = prev.get("content", ""), msg.get("content", "")
                merged_content = (
                    (prev_content + "\n\n" + new_content) if prev_content and new_content else (prev_content or new_content)
                )
                had_api_sidecar = "api_content" in prev
                if (prev.get(_DB_PERSISTED_MARKER) or prev.get(MERGED_DURABLE_ROWS)) and msg.get(_DB_PERSISTED_MARKER):
                    prev[MERGED_DURABLE_ROWS] = int(prev.get(MERGED_DURABLE_ROWS) or 1) + 1
                else:
                    prev.pop(MERGED_DURABLE_ROWS, None)
                prev["content"] = merged_content
                if prev_content:
                    prev[arh.MERGED_TURN_PREFIX] = prev_content
                drop_stale_api_content(prev)

                if prev.get("display_kind") and not msg.get("display_kind"):
                    marker_row_id = prev.get("_row_id")
                    prev.pop("display_kind", None)
                    if msg.get("_row_id") is not None:
                        if isinstance(marker_row_id, int) and not isinstance(marker_row_id, bool):
                            absorbed_ids = prev.setdefault("_absorbed_row_ids", [])
                            if marker_row_id not in absorbed_ids:
                                absorbed_ids.append(marker_row_id)
                        prev["_row_id"] = msg["_row_id"]
                    prev.pop(_DB_PERSISTED_MARKER, None)

                if merged_content != prev_content or had_api_sidecar:
                    prev.pop(_DB_PERSISTED_MARKER, None)
                arh._remember_absorbed_row(prev, msg, folded=True)
                repairs += 1
                continue
            merged.append(msg)
        return merged, repairs

    arh._merge_consecutive_users = patched_merge_consecutive_users
    arh._SEQUENCE_REPAIR_PASSES = tuple(
        patched_merge_consecutive_users if fn == orig_merge_consecutive or getattr(fn, "__name__", "") == "_merge_consecutive_users" else fn
        for fn in arh._SEQUENCE_REPAIR_PASSES
    )

    # 3. agent_runtime_helpers: wrap drop_thinking_only_and_merge_users
    def patched_drop_thinking_only_and_merge_users(
        messages: list[dict[str, Any]], *, drop_codex_reasoning_items: bool = True,
        drop_nudge_marker: Optional[str] = None,
        internal_context_contents: frozenset[str] = frozenset(),
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        if not messages:
            return messages
        if not internal_context_contents and messages:
            internal_context_contents = frozenset(
                text for m in messages if is_internal_context_message(m)
                for text in (m.get("api_content"), m.get("content")) if isinstance(text, str)
            )

        kept = [
            m for m in messages
            if not (drop_nudge_marker is not None and m.get("role") == "user" and m.get("content") == drop_nudge_marker)
            and not arh._ra().AIAgent._is_thinking_only_assistant(m, drop_codex_reasoning_items=drop_codex_reasoning_items)
        ]
        dropped = len(messages) - len(kept)
        merged: list[dict[str, Any]] = []
        merges = 0
        for m in kept:
            prev = merged[-1] if merged else None
            content = arh._UNMERGEABLE
            if (
                prev is not None and prev.get("role") == "user" and m.get("role") == "user"
                and not (isinstance(prev.get("content"), str) and prev["content"] in internal_context_contents)
                and not (isinstance(m.get("content"), str) and m["content"] in internal_context_contents)
            ):
                content = arh._merge_user_content(prev.get("content", ""), m.get("content", ""))
            if content is arh._UNMERGEABLE:
                merged.append(m)
            else:
                merged[-1] = {**prev, "content": content}
                merges += 1
        if dropped == 0 and merges == 0:
            return messages
        arh._ra().logger.debug(
            "Pre-call sanitizer: dropped %d thinking-only assistant turn(s), "
            "merged %d adjacent user message(s)", dropped, merges,
        )
        return merged

    arh.drop_thinking_only_and_merge_users = patched_drop_thinking_only_and_merge_users

    # 4. context_compressor: wrap _build_verbatim_user_section & _is_synthetic_compression_user_turn
    orig_build_verbatim = cc._build_verbatim_user_section

    def patched_build_verbatim_user_section(turns: list[dict[str, Any]]) -> str:
        filtered_turns = [msg for msg in turns if not is_internal_context_message(msg)]
        return orig_build_verbatim(filtered_turns)

    cc._build_verbatim_user_section = patched_build_verbatim_user_section

    orig_is_synthetic = cc.ContextCompressor._is_synthetic_compression_user_turn

    @classmethod
    def patched_is_synthetic_compression_user_turn(cls, message: Any) -> bool:
        if is_internal_context_message(message):
            return True
        return orig_is_synthetic(message)

    cc.ContextCompressor._is_synthetic_compression_user_turn = patched_is_synthetic_compression_user_turn

    # 5. micro_compaction: wrap _merge_adjacent_user_turns
    def patched_mc_merge_adjacent_user_turns(self, result: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from agent.turn_context import drop_stale_api_content

        def _plain_user(m: Any) -> bool:
            return (
                isinstance(m, dict) and m.get("role") == "user" and not mc._is_summary_marker(m)
                and not is_internal_context_message(m)
                and isinstance(m.get("content"), str)
            )

        merged: list[dict[str, Any]] = []
        for msg in result:
            if (
                merged
                and _plain_user(merged[-1])
                and _plain_user(msg)
            ):
                drop_stale_api_content(merged[-1])
                merged[-1] = {
                    **merged[-1],
                    "content": merged[-1]["content"] + "\n\n" + msg["content"],
                }
            else:
                merged.append(msg)
        return merged

    mc.MicroCompactionMixin._merge_adjacent_user_turns = patched_mc_merge_adjacent_user_turns

    # 6. turn_request_assembly: wrap assemble_api_request
    orig_assemble = tra.assemble_api_request

    def patched_assemble_api_request(agent: Any, *args: Any, **kwargs: Any) -> Any:
        if hasattr(agent, "_drop_thinking_only_and_merge_users"):
            agent._drop_thinking_only_and_merge_users = patched_drop_thinking_only_and_merge_users
        return orig_assemble(agent, *args, **kwargs)

    tra.assemble_api_request = patched_assemble_api_request

    _PATCHED = True
    logger.info("segmented-micro-compact: in-memory core compat patches applied successfully")
    return True
