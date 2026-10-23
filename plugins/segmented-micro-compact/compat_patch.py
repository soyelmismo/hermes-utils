"""In-memory compatibility patch for internal context boundaries in Hermes Agent.

Replaces the need for modifying Hermes Agent core repository files directly.
Applied dynamically when the segmented-micro-compact plugin is registered/imported.
"""
from __future__ import annotations
import logging
from typing import Any, Mapping

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
        # Si no hay mensajes con rol user, dejar pasar
        if not messages:
            return orig_merge_consecutive(messages)
        # Wrap custom logic
        merged = []
        absorbed_count = 0
        prev = None
        for msg in messages:
            if (
                prev is not None and prev.get("role") == "user"
                and isinstance(msg, dict) and msg.get("role") == "user"
                and not is_internal_context_message(prev) and not is_internal_context_message(msg)
                and arh.split_user_originated_turn(prev)[0] is None
            ):
                # Usar lógica original de fusión
                merged[-1] = arh._merge_user_pair(prev, msg)
                prev = merged[-1]
                absorbed_count += 1
            else:
                merged.append(msg)
                prev = msg
        return merged, absorbed_count

    arh._merge_consecutive_users = patched_merge_consecutive_users

    # 3. agent_runtime_helpers: wrap drop_thinking_only_and_merge_users
    orig_drop_thinking = arh.drop_thinking_only_and_merge_users

    def patched_drop_thinking_only_and_merge_users(
        messages: list[dict[str, Any]], *, drop_codex_reasoning_items: bool = True,
        drop_nudge_marker: Any = None,
        internal_context_contents: frozenset[str] = frozenset(),
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        # Extraer contenidos si no se pasaron explícitamente pero hay mensajes internos
        if not internal_context_contents and messages:
            internal_context_contents = frozenset(
                text for m in messages if is_internal_context_message(m)
                for text in (m.get("api_content"), m.get("content")) if isinstance(text, str)
            )

        kept = [m for m in messages if not arh._is_thinking_only_assistant(m, drop_codex=drop_codex_reasoning_items, drop_nudge=drop_nudge_marker)]
        merged: list[dict[str, Any]] = []
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
    orig_mc_merge = mc.MicroCompactionMixin._merge_adjacent_user_turns

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
        # Asegurar que el método del agente use el drop_thinking parcheado
        if hasattr(agent, "_drop_thinking_only_and_merge_users"):
            agent._drop_thinking_only_and_merge_users = patched_drop_thinking_only_and_merge_users
        return orig_assemble(agent, *args, **kwargs)

    tra.assemble_api_request = patched_assemble_api_request

    _PATCHED = True
    logger.info("segmented-micro-compact: in-memory core compat patches applied successfully")
    return True
