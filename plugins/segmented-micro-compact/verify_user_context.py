"""Regresiones offline: user interno separado en finalizer, SQLite y replay real."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import logging
import time

from verify_tool_thread import engine, rows
from smc_checks.markers import is_tool_summary, tool_summary_message, TOOL_CONTEXT_SOURCE
from agent.agent_runtime_helpers import repair_message_sequence
from agent.turn_context import build_api_messages
from agent.turn_finalizer import _micro_compact_after_turn
from hermes_state import SessionDB


def check_recognition_requires_durable_provenance():
    marker = tool_summary_message('- ran: test -> passed')
    assert is_tool_summary(marker)
    human = {'role': 'user', 'content': marker['content']}
    assert not is_tool_summary(human)
    assert not is_tool_summary({**human, '_hermes_tool_thread_summary': True})
    assert not is_tool_summary({**marker, 'display_metadata': {'context_source': 'other-plugin'}})
    assert is_tool_summary({'role': 'assistant', 'content': marker['content']})


def check_boundaries_and_human_merge():
    e = engine()
    human = {'role': 'user', 'content': 'Pregunta', 'api_content': 'Pregunta\n<memory-context>memoria</memory-context>'}
    marker = tool_summary_message('- ran: test -> passed')
    next_user = {'role': 'user', 'content': 'Otra pregunta'}
    source = [human, marker, next_user]
    before = deepcopy(source)
    assert repair_message_sequence(None, source) == 0
    assert source == before
    assert e._merge_adjacent_user_turns(source) == before
    assert e._is_synthetic_compression_user_turn(marker)
    assert not e._is_synthetic_compression_user_turn(human)
    assert not e._transcript_has_real_user_turn([marker])
    plain = [{'role': 'user', 'content': 'uno'}, {'role': 'user', 'content': 'dos'}]
    assert repair_message_sequence(None, plain) == 1
    assert plain[0]['content'] == 'uno\n\ndos'


def check_finalizer_sqlite_repair_request_repeated():
    with TemporaryDirectory(dir='/root/.hermes/cache/scratch') as d:
        db = SessionDB(db_path=Path(d) / 'context.db')
        sid = 'user-context-offline'
        db.create_session(sid, source='cli', model='offline')
        original = rows()
        original[0]['api_content'] = original[0]['content'] + '\n<memory-context>memoria</memory-context>'
        for row in original:
            db.append_message(sid, row['role'], content=row.get('content'),
                              api_content=row.get('api_content'), tool_calls=row.get('tool_calls'),
                              tool_call_id=row.get('tool_call_id'))
        history = db.get_messages_as_conversation(sid)
        before = deepcopy(history)
        e = engine()
        e.on_session_start(sid, session_db=db)
        agent = SimpleNamespace(context_compressor=e, compression_checkpoint_required=False,
                                _persist_disabled=False, session_id=sid, _inflight_turn_session_id=sid)
        with patch('agent.conversation_compression._reset_read_dedup_caches'):
            _micro_compact_after_turn(agent, history, history[-1]['content'], logging.getLogger('offline'), 'test')
        loaded = db.get_messages_as_conversation(sid)
        assert [m['role'] for m in loaded] == ['user', 'user', 'assistant']
        assert loaded[0]['content'] == before[0]['content']
        assert loaded[0]['api_content'] == before[0]['api_content']
        assert loaded[-1]['content'] == before[-1]['content']
        assert loaded[1]['display_metadata']['context_source'] == TOOL_CONTEXT_SOURCE
        assert '_hermes_tool_thread_summary' not in loaded[1]
        assert is_tool_summary(loaded[1])
        frozen = deepcopy(loaded)
        for _ in range(12):
            assert repair_message_sequence(None, loaded) == 0
            assert e._micro_compact(loaded) is loaded
            assert loaded == frozen
        request_agent = SimpleNamespace(_current_turn_timestamp=time.time(),
                                        _copy_reasoning_content_for_api=lambda *args: None,
                                        _should_sanitize_tool_calls=lambda: False, ephemeral_system_prompt='')
        replay = loaded + [{'role': 'user', 'content': 'Siguiente'}]
        wire, _ = build_api_messages(request_agent, replay, current_turn_user_idx=3,
                                     ext_prefetch_cache='', plugin_user_context='', moa_config=None,
                                     active_system_prompt='')
        assert [m['role'] for m in wire] == ['user', 'user', 'assistant', 'user']
        assert wire[0]['content'] == before[0]['api_content']
        assert wire[1]['content'] == loaded[1]['content']
        assert wire[2]['content'] == before[-1]['content']
        assert all(not any(k in m for k in ('display_kind', 'display_metadata', 'api_content')) for m in wire)
        assert loaded == frozen
        db.close()


if __name__ == '__main__':
    checks = [v for k, v in sorted(globals().items()) if k.startswith('check_')]
    for check in checks:
        check()
        print('PASS', check.__name__)
    print(len(checks), 'passed, 0 failed')
