"""Behavioural tests for the segmented-micro-compact context engine.

Strategy: the engine subclasses Hermes' ContextCompressor. Importing the real one
drags in the full auxiliary-client import chain (credential pools, provider
workers, network probes), which hangs a bare pytest worker. So we import the real
``ContextCompressor`` ONCE here (proven to import in the Hermes venv) and test the
ENGINE'S OWN BEHAVIOUR against it — the transcript-level contract Miguel asked
for:

  1. segment markers accumulate and are never superseded
  2. an earlier segment is never re-summarized after it is written
  3. the summary prompt is isolated (no cumulative running summary)
  4. general (threshold) compaction still runs on a transcript with segments
  5. the engine is a valid ContextEngine / ContextCompressor and registers itself

Run:
  cd /usr/local/lib/hermes-agent && source venv/bin/activate
  python -m pytest /root/.hermes/plugins/segmented-micro-compact/test_segmented_micro_compact.py -q
"""

from __future__ import annotations

import importlib.util
import sys

# NOTE: no `import pytest` — this file is a self-contained behavioural runner
# (see _run_all at the bottom). The engine subclasses the real Hermes
# ContextCompressor; running its compaction inside a bare pytest worker on this
# host is unreliable, so the assertions run directly against the real classes.

PLUGIN_DIR = "/root/.hermes/plugins/segmented-micro-compact"
INIT_FILE = f"{PLUGIN_DIR}/__init__.py"

# Import the plugin module in the Hermes venv interpreter (agent.* resolvable).
_spec = importlib.util.spec_from_file_location(
    "_segmented_micro_compact_under_test", INIT_FILE,
    submodule_search_locations=[PLUGIN_DIR],
)
plugin_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = plugin_mod
_spec.loader.exec_module(plugin_mod)

SegmentedEngine = plugin_mod.SegmentedMicroCompactEngine
SEGMENT_MARKER_KEY = plugin_mod.SEGMENT_MARKER_KEY
SEGMENT_PREFIX = plugin_mod.SEGMENT_PREFIX
SEGMENT_END_MARKER = plugin_mod.SEGMENT_END_MARKER


def _make_engine(**over) -> "SegmentedEngine":
    eng = SegmentedEngine(
        model="fake-model", threshold_percent=0.8, protect_first_n=1, protect_last_n=25,
        summary_target_ratio=0.3, quiet_mode=True, config_context_length=100000,
    )
    # Pin the resolved window so no /models network probe can fire during tests
    # (get_model_context_length is only called when _resolved_context_length is None).
    eng.context_length = 100000
    eng._micro_compact_enabled = True
    eng._micro_compact_every_n_turns = 1
    for k, v in over.items():
        setattr(eng, k, v)
    return eng


def _exchanges(n: int):
    msgs = [{"role": "system", "content": "SYSTEM PROMPT — never touched"}]
    for i in range(n):
        msgs.append({"role": "user", "content": f"question {i}: do the thing"})
        msgs.append({"role": "assistant", "content": f"answer {i}: done " + "x" * 40})
        msgs.append({"role": "tool", "content": f"tool result {i}"})
        msgs.append({"role": "assistant", "content": f"closing {i}"})
    msgs.append({"role": "user", "content": "latest user message"})
    msgs.append({"role": "user", "content": "and another"})
    return msgs


# --------------------------------------------------------------------------- 1


def test_engine_is_valid_context_engine_and_compressor():
    from agent.context_engine import ContextEngine
    from agent.context_compressor import ContextCompressor

    eng = _make_engine()
    assert isinstance(eng, ContextEngine)
    assert isinstance(eng, ContextCompressor)
    eng.update_from_response({"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
    assert eng.should_compress(1) in (True, False)
    assert callable(eng.compress)


def test_register_selects_engine():
    calls = {}

    class FakeCtx:
        def register_context_engine(self, e):
            calls["engine"] = e

    plugin_mod.register(FakeCtx())
    assert isinstance(calls["engine"], SegmentedEngine)


# --------------------------------------------------------------------------- 2


def test_markers_accumulate_and_are_never_superseded():
    eng = _make_engine()
    counter = {"n": 0}

    def fake_summarize(text):
        counter["n"] += 1
        return f"UNIQUE-SEGMENT-BODY-{counter['n']}"

    eng._segment_summarize_one = fake_summarize
    eng._serialize_for_summary = lambda rows: "SERIALIZED"

    msgs = _exchanges(6)
    counts = []
    for _ in range(4):
        msgs = eng._micro_compact(msgs)
        counts.append(len([m for m in msgs if eng._is_segment_marker(m)]))

    assert counts == sorted(counts), f"marker count went down: {counts}"
    assert counts[-1] >= 2, f"expected accumulated segments, got {counts}"

    bodies = [m["content"] for m in msgs if eng._is_segment_marker(m)]
    assert len(set(bodies)) == len(bodies), "a segment body was overwritten"
    for i, body in enumerate(bodies, start=1):
        assert f"#{i}]" in body, f"segments not numbered independently: {body!r}"


# --------------------------------------------------------------------------- 3


def test_earlier_segments_never_reach_the_summarizer():
    eng = _make_engine()
    seen = []

    def spy(text):
        seen.append(text)
        return f"BODY-{len(seen)}"

    eng._segment_summarize_one = spy

    msgs = _exchanges(8)
    for _ in range(5):
        msgs = eng._micro_compact(msgs)

    for i, text in enumerate(seen):
        for j in range(i):
            assert f"BODY-{j+1}" not in text, (
                f"pass {i} re-summarized content of segment {j+1}"
            )


def test_prompt_is_isolated_not_cumulative():
    eng = _make_engine()
    captured = {}

    def spy(text):
        captured["prompt"] = eng._build_segment_summary_prompt(text)
        return "BODY"

    eng._segment_summarize_one = spy
    eng._micro_compact(_exchanges(4))

    prompt_text = captured["prompt"][1]["content"]
    assert "Current Running Summary" not in prompt_text
    assert "Next Exchange to Merge" not in prompt_text
    assert "not cumulative" in prompt_text


# --------------------------------------------------------------------------- 4


def test_general_compaction_still_works_with_segments():
    """The inherited threshold compaction must run on a transcript containing our
    segment markers and still produce a valid, non-growing transcript with the
    system head protected. We stub only the aux summary call (the single network
    seam) so the test is deterministic and offline."""
    from unittest.mock import patch

    eng = _make_engine()
    msgs = [{"role": "system", "content": "SYS"}]
    msgs.append({"role": "user", "content": "u0"})
    msgs.append({"role": "assistant",
                 "content": f"{SEGMENT_PREFIX} #1]\nSEG-ONE\n{SEGMENT_END_MARKER}",
                 SEGMENT_MARKER_KEY: True})
    msgs.append({"role": "assistant",
                 "content": f"{SEGMENT_PREFIX} #2]\nSEG-TWO\n{SEGMENT_END_MARKER}",
                 SEGMENT_MARKER_KEY: True})
    for i in range(40):
        msgs.append({"role": "user", "content": f"q{i}"})
        msgs.append({"role": "assistant", "content": f"a{i} " + "y" * 300})

    # Stub the built-in's summary call (offline, deterministic) so the REAL boundary
    # and splice logic runs over a transcript that already has segment markers.
    with patch.object(eng, "_generate_summary", lambda *a, **k: "HANDOFF SUMMARY", create=True):
        before = len(msgs)
        compacted = eng.compress(msgs, current_tokens=10**9, force=True)

    assert isinstance(compacted, list)
    assert compacted, "compaction returned an empty transcript"
    assert len(compacted) <= before, "general compaction grew the transcript"
    # The system head is protected at position 0 (the built-in may append a
    # compaction note to it, so we match the prefix, not equality).
    head_content = str(compacted[0].get("content", ""))
    assert compacted[0].get("role") == "system", "head is no longer the system prompt"
    assert head_content.startswith("SYS"), f"system head content lost: {head_content[:40]!r}"
    joined = " ".join(str(m.get("content", "")) for m in compacted if isinstance(m, dict))
    assert "HANDOFF SUMMARY" in joined or "SEG-ONE" in joined or "SEG-TWO" in joined


# --------------------------------------------------------------------------- 5


def test_splice_ignores_supersede_and_keeps_older_markers():
    eng = _make_engine()
    eng._micro_compact_rolling_summary = "NEW BODY"
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "old user"},
        {"role": "assistant",
         "content": f"{SEGMENT_PREFIX} #1]\nOLD-ONE\n{SEGMENT_END_MARKER}",
         SEGMENT_MARKER_KEY: True},
        {"role": "user", "content": "recent"},
        {"role": "assistant", "content": "to absorb " + "z" * 30},
    ]
    result = eng._splice_micro_compact_result(msgs, 4, 5, supersede=True)
    markers = [m for m in result if eng._is_segment_marker(m)]
    assert len(markers) == 2, "older marker deleted despite supersede=True"
    assert "OLD-ONE" in markers[0]["content"]
    assert "NEW BODY" in markers[1]["content"]


def test_cadence_is_honoured():
    eng = _make_engine(_micro_compact_every_n_turns=10)
    eng._segment_summarize_one = lambda t: "BODY"
    msgs = _exchanges(6)
    for _ in range(9):
        msgs = eng._micro_compact(msgs)
    assert not [m for m in msgs if eng._is_segment_marker(m)], "compacted before cadence"
    msgs = eng._micro_compact(msgs)
    assert len([m for m in msgs if eng._is_segment_marker(m)]) == 1, "exactly one pass on turn 10"


def test_summary_failure_keeps_transcript_untouched():
    eng = _make_engine()
    eng._segment_summarize_one = lambda t: None
    msgs = _exchanges(5)
    before = [dict(m) for m in msgs]
    out = eng._micro_compact(msgs)
    assert out == msgs
    assert len(out) == len(before)
    assert not [m for m in out if eng._is_segment_marker(m)]


def test_sqlite_reloaded_markers_advance_cursor_and_accumulate():
    """Simulates reloading from SQLite where in-memory keys (SEGMENT_MARKER_KEY)
    are stripped. The engine must recognize the segment marker via its durable
    prefix, advance cursor past it, and produce [SEGMENT SUMMARY #2] on the next exchange."""
    eng = _make_engine(protect_last_n=2)
    summaries = []

    def spy(text):
        summaries.append(text)
        return f"- did: step {len(summaries)}"

    eng._segment_summarize_one = spy

    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "q0"},
        # Persisted segment marker without SEGMENT_MARKER_KEY (standard SQLite row):
        {"role": "assistant", "content": f"{SEGMENT_PREFIX} #1]\n- did: step 0\n{SEGMENT_END_MARKER}"},
        {"role": "user", "content": "q1: what is next"},
        {"role": "assistant", "content": "a1: answering next " + "w" * 40},
        {"role": "user", "content": "q2: keep going"},
        {"role": "assistant", "content": "a2: still going " + "w" * 40},
        {"role": "user", "content": "q3: tail user"},
        {"role": "assistant", "content": "a3: tail answer " + "w" * 40},
    ]

    # Verify marker recognition without in-memory key
    assert eng._is_segment_marker(msgs[2]) is True
    assert eng._is_context_summary_message(msgs[2]) is True

    # Next exchange must target the assistant answer at index 4, NOT re-absorb index 2
    exchange = eng._next_exchange(msgs)
    assert exchange == (4, 5), f"Expected exchange (4, 5), got {exchange}"

    # Perform micro_compact pass
    compacted = eng._micro_compact(msgs)

    # Segment markers must accumulate: #1 and #2 both present
    markers = [m for m in compacted if eng._is_segment_marker(m)]
    assert len(markers) == 2, f"Expected 2 markers, got {len(markers)}"
    assert "#1]" in markers[0]["content"]
    assert "#2]" in markers[1]["content"]

    # The original #1 marker must not be rewritten
    assert "- did: step 0" in markers[0]["content"]
    assert "- did: step 1" in markers[1]["content"]

    # User messages must be preserved verbatim
    assert compacted[1]["content"] == "q0"
    assert compacted[3]["content"] == "q1: what is next"

    # Summarizer received user query in context for q1
    assert "q1: what is next" in summaries[0]


def _run_all() -> int:
    """Run every ``test_*`` function directly (no pytest worker).

    Why not ``pytest.main`` here: this engine subclasses the real Hermes
    ContextCompressor, and running its ``_micro_compact`` inside a bare pytest
    worker on this host kills the worker (an interaction between the Hermes shell
    wrapper's sys.path handling and the venv), even though the identical code runs
    cleanly as a plain interpreter. Running the assertions directly against the
    real classes keeps the behavioural contract honest and green.
    """
    import inspect
    passed = failed=0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            if "monkeypatch" in inspect.signature(fn).parameters:
                # No test here needs pytest's monkeypatch anymore; skip if it does.
                print(f"SKIP {name} (needs pytest monkeypatch)")
                continue
            fn()
            print(f"PASS {name}")
            passed += 1
        except Exception as exc:
            import traceback
            print(f"FAIL {name}: {exc}")
            traceback.print_exc()
            failed += 1
    print(f"\n=== {passed} passed, {failed} failed ===")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
