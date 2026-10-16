"""Regresiones offline del prefijo inicial de skills, prune y replay."""
from copy import deepcopy
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
from unittest.mock import patch

from verify_tool_thread import engine, rows
from smc_checks.skill_retention import initial_skill_batches
from smc_checks.markers import is_tool_summary, tool_summary_message
from agent.agent_runtime_helpers import repair_message_sequence
from agent.turn_context import build_api_messages
from agent.turn_finalizer import _micro_compact_after_turn
from hermes_state import SessionDB


def skill_rows(ident='skill', body=None):
    return [
        {'role': 'assistant', 'content': 'Cargo la skill', 'tool_calls': [
            {'id': ident, 'type': 'function', 'function': {
                'name': 'skill_view', 'arguments': '{"name":"test-skill"}'}}]},
        {'role': 'tool', 'tool_call_id': ident,
         'content': body if body is not None else 'SKILL_INTEGRA\n' + 's' * 12000},
    ]


def assert_pairs(messages):
    pending = set()
    for row in messages:
        if row.get('tool_calls'):
            assert not pending
            pending.update(c['id'] for c in row['tool_calls'])
        elif row.get('role') == 'tool':
            assert row['tool_call_id'] in pending
            pending.remove(row['tool_call_id'])
        else:
            assert not pending
    assert not pending


def prefix_rows():
    source = rows()
    source[1:1] = skill_rows()
    return source


def capture_engine():
    e = engine()
    inputs = []
    summary = '- ran: fichero y tests -> correctos'
    e._tool_thread_summarize_one = lambda text: inputs.append(text) or summary
    e._tool_thread_summarize_batch = lambda items: inputs.extend(i['text'] for i in items) or json.dumps([
        {'id': i['id'], 'summary': summary} for i in items])
    return e, inputs


def check_initial_skill_keeps_identical_prefix():
    for body in ('skill corta', 'SKILL_INTEGRA\n' + 's' * 12000):
        source = prefix_rows()
        source[2]['content'] = body
        frozen = deepcopy(source)
        e, inputs = capture_engine()
        out = e._micro_compact(source)
        assert out is not source and source == frozen
        assert out[:3] == source[:3]
        assert all(a is b for a, b in zip(out[:3], source[:3]))
        assert not any('skill_view' in text or 'SKILL_INTEGRA' in text for text in inputs)
        assert is_tool_summary(out[3]) and out[-1] is source[-1]
        assert_pairs(out)
        for _ in range(12):
            assert e._micro_compact(out) is out


def check_late_skill_summarized_with_preceding_tools():
    for index in (3, 5):
        source = rows()
        source[index:index] = skill_rows()
        e, inputs = capture_engine()
        out = e._micro_compact(source)
        assert len(inputs) == 1 and 'skill_view' in inputs[0]
        assert [m['role'] for m in out] == ['user', 'user', 'assistant']
        assert not any(m.get('tool_calls') or m.get('role') == 'tool' for m in out)
        assert_pairs(out)


def check_consecutive_initial_skills_survive_before_tools():
    source = rows()
    source[1:1] = skill_rows('s1') + skill_rows('s2')
    out = engine()._micro_compact(source)
    assert out[:5] == source[:5]
    assert all(a is b for a, b in zip(out[:5], source[:5]))
    assert is_tool_summary(out[5])
    assert_pairs(out)


def check_late_skill_after_retained_prefix_is_summarized():
    source = prefix_rows()
    source[-1:-1] = skill_rows('late')
    e, inputs = capture_engine()
    out = e._micro_compact(source)
    assert out[:3] == source[:3]
    assert len(inputs) == 1 and 'late' in inputs[0]
    assert [c['id'] for m in out for c in m.get('tool_calls') or []] == ['skill']
    assert_pairs(out)


def check_mixed_batch_is_summarized_without_projection():
    for first in (True, False):
        source = rows()
        skill = skill_rows()
        source[1]['tool_calls'].insert(0 if first else 1, skill[0]['tool_calls'][0])
        source.insert(2, skill[1])
        e, inputs = capture_engine()
        out = e._micro_compact(source)
        assert len(inputs) == 1 and 'skill_view' in inputs[0]
        assert len(out) == 3 and is_tool_summary(out[1])
        assert_pairs(out)


def check_wrappers_preserve_only_pure_initial_skills():
    for name in ('multi_tool_use.parallel', 'multi_tool_use_parallel'):
        for mixed in (True, False):
            source = prefix_rows()
            children = [{'recipient_name': 'functions.skill_view', 'parameters': {'name': 'test'}}]
            if mixed:
                children.append({'recipient_name': 'functions.read_file', 'parameters': {}})
            source[1]['tool_calls'][0]['function'] = {
                'name': name, 'arguments': json.dumps({'tool_uses': children})}
            out = engine()._micro_compact(source)
            assert (source[2] in out) is not mixed
            if not mixed:
                assert out[:3] == source[:3]
            assert_pairs(out)


def check_summary_boundary_never_reopens_skill_protection():
    source = rows()[:1] + [tool_summary_message('- ran: tools previas -> resumidas')] + skill_rows() + [rows()[-1]]
    assert initial_skill_batches(source) == set()
    out = engine()._micro_compact(source)
    assert not any(m.get('tool_calls') for m in out)
    assert out[1] is source[1]


def check_new_human_turn_resets_prefix_protection():
    source = rows() + prefix_rows()
    protected = initial_skill_batches(source)
    assert protected == {len(rows()) + 1}
    out = engine()._micro_compact(source)
    assert any(m.get('tool_call_id') == 'skill' for m in out)
    assert_pairs(out)


def check_prune_protects_initial_but_not_late_skill():
    source = prefix_rows()
    source[-1:-1] = skill_rows('late')
    frozen = deepcopy(source)
    out, count = engine()._prune_old_tool_results(source, protect_tail_count=0, protect_tail_tokens=100)
    assert source == frozen and count > 0
    assert out[2] is source[2]
    # El core puede proteger por sí mismo la skill reciente. El plugin no añade
    # protección a la tardía: debe coincidir con el resultado ordinario del core.
    from agent.context_compressor import ContextCompressor
    baseline, _ = ContextCompressor._prune_old_tool_results(
        engine(), source, protect_tail_count=0, protect_tail_tokens=100)
    late = next(m for m in out if m.get('tool_call_id') == 'late')
    assert late == next(m for m in baseline if m.get('tool_call_id') == 'late')


def check_only_initial_skills_no_aux_no_prune():
    source = [rows()[0]] + skill_rows('s1') + skill_rows('s2') + [rows()[-1]]
    e = engine()
    e._tool_thread_summarize_one = lambda _: (_ for _ in ()).throw(AssertionError('no aux'))
    assert e._micro_compact(source) is source
    assert not e._tool_thread_failed_inputs
    out, count = e._prune_old_tool_results(source, protect_tail_count=0, protect_tail_tokens=100)
    assert count == 0 and out == source


def check_missing_call_id_does_not_abort_prune():
    source = rows()
    source[1]['tool_calls'].append({'function': {'name': 'skill_view', 'arguments': '{}'}})
    out, count = engine()._prune_old_tool_results(source, protect_tail_count=0, protect_tail_tokens=100)
    assert count > 0 and out != source


def check_failure_keeps_originals():
    for failure in ('summary', 'commit'):
        source = prefix_rows()
        e = engine()
        frozen = deepcopy(source)
        if failure == 'summary':
            e._tool_thread_summarize_one = lambda _: 'invalid'
            assert e._micro_compact(source) is source
        else:
            with patch('smc_checks.tool_thread.commit_tool_thread', return_value=False):
                assert e._micro_compact(source) is source
        assert source == frozen


def check_cursor_maps_boundaries_across_multiple_splices():
    source = rows() + rows()
    for cursor in range(len(source) + 1):
        e = engine()
        e._micro_compact_cursor = cursor
        out = e._micro_compact(source)
        assert out is not source
        expected, delta = cursor, 0
        for start, end in ((1, 5), (7, 11)):
            if cursor >= end:
                expected -= end - start - 1
            elif cursor > start:
                expected = start + delta + 1
            delta -= end - start - 1
        assert e._micro_compact_cursor == expected


def check_sqlite_finalizer_repair_wire_prefix():
    with TemporaryDirectory(dir='/root/.hermes/cache/scratch') as directory:
        db = SessionDB(db_path=Path(directory) / 'skills.db')
        sid = 'skills-offline'
        db.create_session(sid, source='cli', model='offline')
        source = prefix_rows()
        source[0]['api_content'] = source[0]['content'] + '\n<memory-context>memoria</memory-context>'
        for i, row in enumerate(source):
            row['display_kind'] = 'fixture'
            row['display_metadata'] = {'index': i}
            db.append_message(sid, row['role'], content=row.get('content'), api_content=row.get('api_content'),
                              tool_calls=row.get('tool_calls'), tool_call_id=row.get('tool_call_id'),
                              display_kind=row['display_kind'], display_metadata=row['display_metadata'])
        history = db.get_messages_as_conversation(sid)
        before = deepcopy(history)
        e = engine()
        e.on_session_start(sid, session_db=db)
        agent = SimpleNamespace(context_compressor=e, compression_checkpoint_required=False,
                                _persist_disabled=False, session_id=sid, _inflight_turn_session_id=sid)
        with patch('agent.conversation_compression._reset_read_dedup_caches'):
            _micro_compact_after_turn(agent, history, history[-1]['content'], logging.getLogger('offline'), 'test')
        loaded = db.get_messages_as_conversation(sid)
        for original, kept in zip(before[:3], loaded[:3]):
            for key in ('content', 'api_content', 'tool_calls', 'message_uid', 'display_metadata', '_tool_call_uids', '_tool_call_uid'):
                assert kept.get(key) == original.get(key), key
        assert is_tool_summary(loaded[3])
        for _ in range(12):
            repair_message_sequence(None, loaded)
            assert e._micro_compact(loaded) is loaded
            assert_pairs(loaded)
        req = SimpleNamespace(_current_turn_timestamp=time.time(), _copy_reasoning_content_for_api=lambda *a: None,
                              _should_sanitize_tool_calls=lambda: False, ephemeral_system_prompt='')
        def wire(messages):
            replay = messages + [{'role': 'user', 'content': 'siguiente'}]
            return build_api_messages(req, replay, current_turn_user_idx=len(replay) - 1,
                                      ext_prefetch_cache='', plugin_user_context='', moa_config=None,
                                      active_system_prompt='')[0]
        original_wire, compact_wire = wire(before), wire(loaded)
        assert original_wire[:3] == compact_wire[:3]
        assert_pairs(compact_wire)
        assert next(m for m in compact_wire if m.get('tool_call_id') == 'skill')['content'] == source[2]['content']
        db.close()


if __name__ == '__main__':
    checks = [v for k, v in sorted(globals().items()) if k.startswith('check_')]
    for check in checks:
        check()
        print('PASS', check.__name__)
    print(len(checks), 'passed, 0 failed')
