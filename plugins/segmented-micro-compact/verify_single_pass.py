"""Regression checks through the real engine: one call, no output feedback."""
from verify_tool_thread import engine, rows, plugin
from copy import deepcopy
from unittest.mock import patch
import json


def capture(e, calls):
    del e._tool_thread_summarize_one
    del e._tool_thread_summarize_batch
    def summarize(prompt, **kwargs):
        text = '\n'.join(m['content'] for m in prompt)
        calls.append(text)
        if 'Tool threads to summarize (JSON)' in text:
            items = json.loads(prompt[-1]['content'].split('\n', 1)[1])
            return json.dumps([{'id': item['id'], 'summary': '- ran: RAW_SUMMARY_OUTPUT -> passed'} for item in items])
        return '- ran: RAW_SUMMARY_OUTPUT -> passed'
    e._summarize_prompt = summarize


def check_one_call_no_cascade():
    e = engine()
    e._micro_compact_every_n_turns = 1
    e.protect_last_n = 2
    calls = []
    capture(e, calls)
    source = rows() + rows() + [{'role': 'user', 'content': 'tail'}, {'role': 'assistant', 'content': 'tail answer'}]
    before = deepcopy(source)
    out = e._micro_compact(source)
    assert len(calls) == 1, f'{len(calls)} calls, expected one'
    assert sum(bool(m.get(plugin.TOOL_THREAD_MARKER)) for m in out) == 2
    assert source == before
    calls.clear()
    # SQLite reload strips plugin metadata but preserves content.
    out = [{k: v for k, v in m.items() if not k.startswith('_hermes')} for m in out]
    for _ in range(12):
        out = e._micro_compact(out)
    assert not calls, 'already summarized history was sent back to the API'


def check_oob_one_call_and_verbatim():
    e = engine()
    e._micro_compact_every_n_turns = 1
    source = rows()
    source.insert(3, {'role': 'user', 'content': 'steer: change task'})
    calls = []
    capture(e, calls)
    out = e._micro_compact(source)
    assert len(calls) == 1
    assert [m for m in out if m['role'] == 'user' and not e._is_context_summary_message(m)] == [m for m in source if m['role'] == 'user']
    assert out[-1] is source[-1]
    assert sum(bool(m.get(plugin.TOOL_THREAD_MARKER)) for m in out) == 2


def check_failure_no_second_call():
    for response in [None, '', 'invalid', '[]', '[{"id":0,"summary":"not bullets"}]']:
        e = engine()
        e._micro_compact_every_n_turns = 1
        calls = []
        del e._tool_thread_summarize_one
        del e._tool_thread_summarize_batch
        e._summarize_prompt = lambda prompt, **kw: calls.append(prompt) or response
        source = rows() + rows()
        before = deepcopy(source)
        assert e._micro_compact(source) is source
        assert source == before
        assert len(calls) == 1


def check_db_failure_is_closed():
    e = engine()
    calls = []
    capture(e, calls)
    source = rows()
    before = deepcopy(source)
    e._session_db = object()
    e._session_id = 'offline'
    with patch('smc_checks.tool_thread.commit_tool_thread', return_value=False):
        assert e._micro_compact(source) is source
    assert len(calls) == 1 and source == before
    assert e._tool_thread_passes == 0


def check_budget_and_deferred_originals():
    e = engine()
    e.max_summary_tokens = 1500
    calls = []
    capture(e, calls)
    source = rows() + rows()
    once = e._micro_compact(source)
    assert len(calls) == 1
    assert sum(m['role'] == 'tool' for m in once) == 2
    twice = e._micro_compact(once)
    assert len(calls) == 2
    assert 'RAW_SUMMARY_OUTPUT' not in calls[1]
    assert not any(m['role'] == 'tool' for m in twice)
    calls.clear()
    e = engine()
    e.context_length = 6000
    capture(e, calls)
    source = rows()
    assert e._micro_compact(source) is source
    assert not calls


def check_aux_truncation_no_feedback():
    from types import SimpleNamespace
    e = engine()
    del e._tool_thread_summarize_one
    del e._tool_thread_summarize_batch
    source = rows() + rows()
    before = deepcopy(source)
    response = SimpleNamespace(choices=[SimpleNamespace(finish_reason='length', message=SimpleNamespace(content='[partial'))])
    with patch('agent.auxiliary_client.call_llm', return_value=response) as llm:
        for _ in range(12):
            assert e._micro_compact(source) is source
        assert llm.call_count == 1
    assert source == before


def check_batch_contract():
    from importlib import import_module
    _batch_summaries = import_module("smc_checks.tool_thread")._batch_summaries
    valid = [{'id': 1, 'summary': '- ran: B -> done'}, {'id': 0, 'summary': '- ran: A -> done'}]
    assert _batch_summaries(json.dumps(valid), 2) == ['- ran: A -> done', '- ran: B -> done']
    for bad in [valid[:1], valid + valid[:1], [valid[0], valid[0]],
                [{'id': True, 'summary': '- ran: x'}],
                [{'id': 2, 'summary': '- ran: x'}],
                [{'id': 0, 'summary': 'prose'}],
                [{'id': 0, 'summary': '- ran: x', 'extra': 1}]]:
        assert _batch_summaries(json.dumps(bad), 2 if len(bad) > 1 else 1) is None


def check_failed_input_not_retried_and_reset():
    e = engine()
    del e._tool_thread_summarize_one
    calls = []
    e._summarize_prompt = lambda *a, **kw: calls.append(a) or None
    source = rows()
    for _ in range(12):
        assert e._micro_compact(source) is source
    assert len(calls) == 1
    e.on_session_reset()
    assert e._micro_compact(source) is source
    assert len(calls) == 2


def check_reentrant_pass():
    e = engine()
    del e._tool_thread_summarize_one
    calls = []
    source = rows()
    def summary(*a, **kw):
        calls.append(a)
        assert e._micro_compact(source) is source
        return '- ran: test -> passed'
    e._summarize_prompt = summary
    assert len(e._micro_compact(source)) == 3
    assert len(calls) == 1


def check_marker_recognition_and_mode_switch():
    from importlib import import_module
    markers = import_module("smc_checks.markers")
    render_tool_thread_summary, is_tool_summary = markers.render_tool_thread_summary, markers.is_tool_summary
    for text in [render_tool_thread_summary('- ran: OLD_OUTPUT -> passed'),
                 '[TOOL WORK SUMMARY]\n- ran: OLD_OUTPUT -> passed\n[END TOOL WORK SUMMARY]']:
        marker = {'role': 'assistant', 'content': text}
        assert is_tool_summary(marker)
        assert not is_tool_summary({'role': 'user', 'content': text})
        e = engine(False)
        e._micro_compact_every_n_turns = 1
        e.protect_last_n = 2
        calls = []
        e._segment_summarize_one = lambda t: calls.append(t) or '- did: new work'
        source = [{'role': 'system', 'content': 'sys'}, {'role': 'user', 'content': 'old'}, marker,
                  {'role': 'user', 'content': 'new'}, {'role': 'assistant', 'content': 'fresh work'},
                  {'role': 'user', 'content': 'tail'}, {'role': 'assistant', 'content': 'tail'}]
        out = e._micro_compact(source)
        assert all('OLD_OUTPUT' not in t for t in calls)
        assert marker in out


def check_real_aux_and_sqlite_batch():
    from hermes_state import SessionDB
    from agent.turn_finalizer import _micro_compact_after_turn
    from types import SimpleNamespace
    import tempfile
    import pathlib
    import logging
    with tempfile.TemporaryDirectory(dir='/root/.hermes/cache/scratch') as d:
        db = SessionDB(db_path=pathlib.Path(d) / 'batch.db')
        db.create_session('batch', source='cli', model='offline')
        e = engine()
        del e._tool_thread_summarize_one
        del e._tool_thread_summarize_batch
        e._micro_compact_every_n_turns = 1
        e.on_session_start('batch', session_db=db)
        source = rows() + rows()
        source[7]['tool_calls'][0]['id'] = 'a2'
        source[8]['tool_call_id'] = 'a2'
        source[9]['tool_calls'][0]['id'] = 'b2'
        source[10]['tool_call_id'] = 'b2'
        for row in source:
            db.append_message('batch', row['role'], content=row.get('content'), tool_calls=row.get('tool_calls'), tool_call_id=row.get('tool_call_id'))
        source = db.get_messages_as_conversation('batch')
        body = json.dumps([{'id': i, 'summary': f'- ran: OUTPUT_{i} -> passed'} for i in range(2)])
        response = SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content=body))])
        agent = SimpleNamespace(context_compressor=e, compression_checkpoint_required=False, _persist_disabled=False, session_id='batch', _inflight_turn_session_id='batch')
        with patch('agent.auxiliary_client.call_llm', return_value=response) as llm:
            _micro_compact_after_turn(agent, source, source[-1]['content'], logging.getLogger('offline'), 'batch-task')
            assert llm.call_count == 1
            assert llm.call_args.kwargs['max_tokens'] == 3000
            assert llm.call_args.kwargs['task'] == 'compression'
            loaded = db.get_messages_as_conversation('batch')
            assert len(loaded) == 6
            assert len([m for m in loaded if e._is_context_summary_message(m)]) == 2
            for _ in range(12):
                assert e._micro_compact(loaded) is loaded
            assert llm.call_count == 1
        # A real DB write failure is not swallowed by the inherited core helper.
        from importlib import import_module
        commit_tool_thread = import_module("smc_checks.persistence").commit_tool_thread
        with patch.object(db, 'archive_and_compact', side_effect=RuntimeError('forced DB failure')):
            assert not commit_tool_thread(e, loaded, loaded, None)
        db.close()


if __name__ == '__main__':
    import traceback
    failed = 0
    checks = [v for k, v in sorted(globals().items()) if k.startswith('check_')]
    for check in checks:
        try:
            check()
            print('PASS', check.__name__)
        except Exception:
            failed += 1
            traceback.print_exc()
    print(len(checks) - failed, 'passed,', failed, 'failed')
    raise SystemExit(bool(failed))
