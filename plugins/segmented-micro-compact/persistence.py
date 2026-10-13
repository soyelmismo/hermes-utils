"""Fail-closed watermark/coverage transaction shared by both compaction policies."""
from __future__ import annotations
import logging
from typing import Any

logger = logging.getLogger(__name__)


def commit_tool_thread(engine: Any, original: list, result: list, watermark: int | None) -> bool:
    """Commit directly: core sync currently swallows non-generation DB errors."""
    db = getattr(engine, "_session_db", None)
    sid = getattr(engine, "_session_id", "")
    if not db or not sid:
        return True
    try:
        from agent.context_compressor import (
            stamp_db_persisted_markers, _DB_PERSISTED_MARKER, _archive_watermark_for,
        )
        from agent.conversation_compression_archive import coverage_for_commit
        covered_ids = unresolved_held = None
        if watermark is not None:
            watermark = _archive_watermark_for(db, sid, original, watermark)
            covered_ids, unresolved_held = coverage_for_commit(db, sid, original)
        carried = [m for m in result if isinstance(m, dict) and m.get(_DB_PERSISTED_MARKER)]
        db.archive_and_compact(
            sid, result, carried_messages=carried, watermark=watermark,
            covered_ids=covered_ids, unresolved_held=unresolved_held,
        )
        stamp_db_persisted_markers(result)
    except Exception:
        logger.warning("Compaction commit failed; keeping original transcript", exc_info=True)
        return False
    return True
