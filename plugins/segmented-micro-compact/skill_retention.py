"""Conserva solo el prefijo inicial puro de lecturas de skills de cada turno."""
from __future__ import annotations

import json
from typing import Any


def _only_reads_skills(call: dict[str, Any]) -> bool:
    """No proyectar lotes mixtos: cambiar la fila también rompe el prefijo."""
    fn = call.get("function", {})
    if not isinstance(fn, dict):
        return False
    name = fn.get("name") or ""
    if name.rsplit(".", 1)[-1] == "skill_view":
        return True
    if name not in ("multi_tool_use.parallel", "multi_tool_use_parallel"):
        return False
    try:
        args = fn.get("arguments", "{}")
        args = json.loads(args) if isinstance(args, str) else args
        children = args.get("tool_uses", [])
        return isinstance(children, list) and bool(children) and all(
            isinstance(child, dict)
            and str(child.get("recipient_name", "")).rsplit(".", 1)[-1] == "skill_view"
            for child in children
        )
    except (TypeError, ValueError, AttributeError):
        return False


def initial_skill_batches(messages: list) -> set[int]:
    """Índices de batches puros anteriores a cualquier otra tool del turno.

    Solo un user humano abre otro turno. Un resumen interno cuenta como tools
    previas y no reabre la protección al repetir el pase o recuperar SQLite.
    """
    from agent.message_metadata import is_internal_context_message
    try:
        from .markers import is_tool_summary
    except (ImportError, ValueError):
        from markers import is_tool_summary

    protected = set()
    initial = False
    for index, row in enumerate(messages):
        if is_tool_summary(row) or is_internal_context_message(row):
            initial = False
        elif row.get("role") == "user":
            initial = True
        elif row.get("role") == "assistant" and row.get("tool_calls"):
            calls = row["tool_calls"]
            pure = isinstance(calls, list) and all(
                isinstance(call, dict) and isinstance(call.get("id"), str)
                and bool(call["id"]) and _only_reads_skills(call)
                for call in calls
            )
            if initial and pure:
                protected.add(index)
            else:
                initial = False
        elif row.get("function_call"):
            initial = False
    return protected


def compaction_parts(messages: list, start: int, end: int) -> list[tuple[int, int, list, list]]:
    """Deja intacto el prefijo protegido y resume el resto del hilo normalmente."""
    protected = initial_skill_batches(messages)
    parts = []
    pending_start = start
    pending_rows = []
    i = start
    while i < end:
        j = i + 1
        while j < end and messages[j].get("role") == "tool":
            j += 1
        if i in protected:
            if pending_rows:
                parts.append((pending_start, i, [], pending_rows))
            pending_rows = []
            pending_start = j
        else:
            pending_rows.extend(messages[i:j])
        i = j
    if pending_rows:
        parts.append((pending_start, end, [], pending_rows))
    return parts
