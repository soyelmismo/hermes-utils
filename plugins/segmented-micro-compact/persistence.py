"""Fail-closed watermark-aware tool-thread database transaction."""
from __future__ import annotations
import logging
from typing import Any
logger = logging.getLogger(__name__)

def commit_tool_thread(engine: Any, original: list, result: list, watermark: int | None) -> bool:
    """Use Hermes' watermark/coverage transaction, but NEVER swallow a DB failure."""
    db = getattr(engine, "_session_db", None)
    sid = getattr(engine, "_session_id", "")
    if not db or not sid:
        return True
    sync_fn = getattr(engine, "_sync_micro_compact_to_db", None)
    if callable(sync_fn):
        return bool(sync_fn(result, held=original, start_watermark=watermark))
    try:
        from agent.context_compressor import stamp_db_persisted_markers, _DB_PERSISTED_MARKER
        from agent.conversation_compression_archive import coverage_for_commit
        covered_ids, unresolved_held = coverage_for_commit(db, sid, original)
        carried = [m for m in result if isinstance(m, dict) and m.get(_DB_PERSISTED_MARKER)]
        db.archive_and_compact(
            sid, result, carried_messages=carried, watermark=watermark,
            covered_ids=covered_ids, unresolved_held=unresolved_held,
        )
        stamp_db_persisted_markers(result)
    except Exception:
        logger.warning("Tool-thread commit failed; keeping original transcript", exc_info=True)
        return False
    return True
