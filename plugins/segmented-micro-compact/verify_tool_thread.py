"""Offline checks, including SQLite persistence through the actual finalizer seam."""
import importlib.util
import pathlib
import sys
from copy import deepcopy
from unittest.mock import patch

HERMES_SRC = "/usr/local/lib/hermes-agent"
if HERMES_SRC not in sys.path:
    sys.path.insert(0, HERMES_SRC)

# Finalizer imports bootstrap via CLI entry helpers. Keep this offline test in
# its explicitly chosen venv instead of relaunching under the managed CLI Python.
import hermes_cli.venv_sync
hermes_cli.venv_sync.prepare_launch = lambda *args, **kwargs: None
ROOT = pathlib.Path(__file__).parent
spec = importlib.util.spec_from_file_location('smc_checks', ROOT / '__init__.py', submodule_search_locations=[str(ROOT)])
plugin = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin
spec.loader.exec_module(plugin)
from smc_checks.tool_thread import find_tool_thread, TOOL_THREAD_MARKER
from agent.context_compressor import ContextCompressor


def engine(enabled=True):
    e = plugin.SegmentedMicroCompactEngine(model='offline', config_context_length=100000, tool_thread_compact=enabled)
    e.context_length = 100000
    e._micro_compact_enabled = True
    e._micro_compact_every_n_turns = 10
    e._tool_thread_summarize_one = lambda text: '- ran: read_file -> read source\n- ran: terminal -> 9 tests passed'
    return e


def rows():
    return [
        {'role': 'user', 'content': 'Inspect and test'},
        {'role': 'assistant', 'content': 'Reading', 'tool_calls': [{'id':'a','type':'function','function':{'name':'read_file','arguments':'{"path":"/project/code.py"}'}}]},
        {'role': 'tool', 'tool_call_id':'a', 'content':'source\n' + 'x' * 8000},
        {'role': 'assistant', 'content': 'Testing', 'tool_calls': [{'id':'b','type':'function','function':{'name':'terminal','arguments':'{"command":"python checks.py"}'}}]},
        {'role': 'tool', 'tool_call_id':'b', 'content':'9 passed\n' + 'y' * 8000},
        {'role': 'assistant', 'content': 'Final answer unchanged'},
    ]


def check_default_off():
    e = engine(False)
    r = rows()
    assert e._micro_compact(r) is r
    assert e._tool_thread_passes == 0


def check_splice_and_cadence():
    e = engine()
    r = rows()
    output = e._micro_compact(r)
    assert len(output) == 3
    assert output[0] is r[0] and output[-1] is r[-1]
    assert output[1][TOOL_THREAD_MARKER]
    assert e._micro_compact_turns_since_pass == 1
    assert e._tool_thread_tokens_saved > 3900
    print('Estimated saved tokens:', e._tool_thread_tokens_saved)
    assert len(r) == 6
    assert e._micro_compact(output) is output
    assert e._tool_thread_passes == 1


def check_parallel():
    r = rows()
    r[1]['tool_calls'].append(r[3]['tool_calls'][0])
    del r[3]
    assert find_tool_thread(r) == (1,4)


def check_pending_and_invalid():
    for change in ['pending', 'orphan', 'duplicate', 'interleave', 'unfinished', 'steer', 'legacy']:
        r = rows()
        if change == 'pending': del r[4]
        if change == 'orphan': r[2]['tool_call_id'] = 'orphan'
        if change == 'duplicate': r[3]['tool_calls'][0]['id'] = 'a'
        if change == 'interleave': del r[2]
        if change == 'unfinished': del r[-1]
        if change == 'steer': r.insert(4, {'role':'user','content':'changed direction'})
        if change == 'legacy': r[1]['function_call'] = {'name':'old'}
        assert find_tool_thread(r) is None, change


def check_failures():
    for response in [None, '', 'not bullets', '- ran: ' + 'long ' * 2000]:
        e = engine()
        e._tool_thread_summarize_one = lambda text: response
        r = rows()
        assert e._micro_compact(r) is r
    e = engine()
    def fail(text): raise RuntimeError('offline forced failure')
    e._tool_thread_summarize_one = fail
    r = rows()
    assert e._micro_compact(r) is r


def check_no_growth():
    r = rows()
    r[2]['content'] = ''
    r[4]['content'] = ''
    e = engine()
    e._tool_thread_summarize_one = lambda t: '- ran: ' + 'result ' * 200
    assert e._micro_compact(r) is r


def check_prompt_serializer_and_route():
    e = engine()
    text = e._serialize_for_summary(rows()[1:-1])
    assert 'read_file' in text and '/project/code.py' in text and '9 passed' in text
    from smc_checks.prompts import build_tool_prompt
    assert 'Do not invent' in build_tool_prompt(text)[0]['content']
    from types import SimpleNamespace
    response = SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content='- ran: checks -> passed'))])
    with patch('agent.auxiliary_client.call_llm', return_value=response) as llm:
        result = plugin.SegmentedMicroCompactEngine._tool_thread_summarize_one(e, text)
        assert result == '- ran: checks -> passed'
        assert llm.call_args.kwargs['task'] == 'compression'
        assert 'Tool work' in llm.call_args.kwargs['messages'][1]['content']
    response.choices[0].finish_reason = 'length'
    with patch('agent.auxiliary_client.call_llm', return_value=response):
        assert plugin.SegmentedMicroCompactEngine._tool_thread_summarize_one(e, text) is None


def check_multi_turn_and_out_of_band():
    e = engine()
    # 1. Multi-turn uncompacted history: both turns get their tool threads compacted
    r = rows() + rows()
    output = e._micro_compact(r)
    assert len(output) == 6  # 3 messages per turn: [user, marker, final] * 2
    assert output[0] is r[0] and output[3] is r[6] and output[-1] is r[-1]
    assert output[1][TOOL_THREAD_MARKER] and output[4][TOOL_THREAD_MARKER]
    assert not any(m.get('role') == 'tool' for m in output)

    # 2. Out-of-band user steer interruption (Log #3a71d37d scenario):
    r_interrupted = [
        {'role': 'user', 'content': 'Hello, analyze this project'},
        {'role': 'assistant', 'content': 'Scanning', 'tool_calls': [{'id': 'c1', 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{"path": "a"}'}}]},
        {'role': 'tool', 'tool_call_id': 'c1', 'content': 'long file data\n' + 'z' * 4000},
        {'role': 'user', 'content': '[OUT-OF-BAND USER MESSAGE] Cancel those tests and give me your opinion now'},
        {'role': 'assistant', 'content': 'Stopping', 'tool_calls': [{'id': 'c2', 'type': 'function', 'function': {'name': 'kill', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': 'c2', 'content': 'killed\n' + 'k' * 4000},
        {'role': 'assistant', 'content': 'Final opinion after cancellation'},
    ]
    out_oob = e._micro_compact(r_interrupted)
    # Both tool blocks must be compacted:
    # [user, marker1, oob_user, marker2, final]
    assert len(out_oob) == 5
    assert out_oob[1][TOOL_THREAD_MARKER]
    assert out_oob[3][TOOL_THREAD_MARKER]
    assert not any(m.get('role') == 'tool' for m in out_oob)
    assert out_oob[0]['content'] == r_interrupted[0]['content']
    assert out_oob[2]['content'] == r_interrupted[3]['content']
    assert out_oob[4]['content'] == r_interrupted[6]['content']


def check_strip_leaked_tool_summary():
    from smc_checks import strip_leaked_tool_summary
    # Clean text remains untouched
    assert strip_leaked_tool_summary('Normal answer') is None
    # Leaked tool summary block is cleanly stripped
    leaked = (
        '[TOOL WORK SUMMARY]\n'
        '- ran: read_file -> source\n'
        '- ran: kill -> done\n'
        '[END TOOL WORK SUMMARY]\n\n'
        'Here is my analysis without prefixes.'
    )
    res = strip_leaked_tool_summary(leaked)
    assert res == 'Here is my analysis without prefixes.'
    # Legacy histories still clean up, even when echoed mid-reply.
    assert strip_leaked_tool_summary('Start\n' + leaked) == 'Start\n\n\nHere is my analysis without prefixes.'
    assert strip_leaked_tool_summary(leaked.split('\n\n')[0]) == ''
    assert strip_leaked_tool_summary(None) is None
    assert strip_leaked_tool_summary('Mention [TOOL WORK SUMMARY] without a block') is None
    from smc_checks.markers import render_tool_thread_summary
    block = render_tool_thread_summary('- ran: read_file -> source')
    assert strip_leaked_tool_summary(block + '\n\nClean response') == 'Clean response'
    assert strip_leaked_tool_summary(block) == ''


def check_tool_context_identity_and_isolation():
    from agent.memory_manager import build_memory_context_block, sanitize_context, StreamingContextScrubber
    from smc_checks.markers import render_tool_thread_summary, sanitize_tool_context
    # Rendering must not borrow a memory provider's builder or provenance.
    with patch('agent.memory_manager.build_memory_context_block', side_effect=AssertionError('memory impersonation')):
        block = render_tool_thread_summary('- ran: read_file -> source\n- ran: terminal -> passed')
    assert block.startswith('<tool-work-context>\n[Context note:')
    assert block.endswith('\n</tool-work-context>')
    assert 'NOT persistent memory' in block
    assert 'historical reference data' in block
    assert '<memory-context>' not in block and '[System note:' not in block
    assert '[TOOL WORK SUMMARY]' not in block
    assert sanitize_tool_context(block) == ''
    assert sanitize_tool_context('Before\n' + block + '\nAfter') == 'Before\n\nAfter'
    assert sanitize_tool_context(block.rsplit('</tool-work-context>', 1)[0]) == ''
    assert sanitize_tool_context(block.upper()) == ''
    memory = build_memory_context_block('## Holographic Memory\n- user preference')
    assert sanitize_tool_context(memory) == memory
    assert plugin.strip_leaked_tool_summary(memory) is None
    assert plugin.strip_leaked_tool_summary(block + '\n' + memory + '\nAnswer') == memory + '\nAnswer'
    # Existing core output filtering remains memory-only. Do not claim streaming protection.
    assert sanitize_context(block) == block
    scrubber = StreamingContextScrubber()
    assert scrubber.feed(block) + scrubber.flush() == block
    assert sanitize_context(memory) == ''
    scrubber = StreamingContextScrubber()
    assert ''.join(scrubber.feed(char) for char in memory) + scrubber.flush() == ''
    # No history relocation or modification of user/final messages.
    r = rows()
    out = engine()._micro_compact(r)
    assert out[0] is r[0] and out[-1] is r[-1]
    assert out[1]['role'] == 'assistant' and out[1]['api_content'].startswith('<tool-work-context>\n')
    assert out[1]['content'] == ''


def check_commit_failure():
    e = engine()
    e._session_id = 'offline-session'
    e._session_db = object()
    e._micro_start_watermark = lambda r: ('watermark_unavailable', None)
    r = rows()
    assert e._micro_compact(r) is r
    assert e._tool_thread_passes == 0


def check_finalizer_integration():
    from agent.turn_finalizer import _micro_compact_after_turn
    import logging
    from types import SimpleNamespace
    from hermes_state import SessionDB
    import tempfile
    # All DB writes confined to isolated local scratch files.
    with tempfile.TemporaryDirectory(dir='/root/.hermes/cache/scratch') as d:
        db = SessionDB(db_path=pathlib.Path(d) / 'checks.db')
        sid = 'tool-thread-check'
        db.create_session(sid, source='cli', model='offline')
        e = engine()
        e.on_session_start(sid, session_db=db)
        r = rows()
        for row in r:
            db.append_message(sid, row['role'], content=row.get('content'), tool_calls=row.get('tool_calls'), tool_call_id=row.get('tool_call_id'))
        r = db.get_messages_as_conversation(sid)
        before = deepcopy(r)
        agent = SimpleNamespace(context_compressor=e, compression_checkpoint_required=False, _persist_disabled=False, session_id=sid, _inflight_turn_session_id=sid)
        with patch('agent.conversation_compression._reset_read_dedup_caches') as reset:
            _micro_compact_after_turn(agent,r,r[-1]['content'],logging.getLogger('offline'),'check-task')
            assert len(r) == 3
            reset.assert_called_once_with('check-task', session_id=sid)
        loaded = db.get_messages_as_conversation(sid)
        assert len(loaded) == 3
        assert loaded[0]['content'] == before[0]['content']
        assert loaded[-1]['content'] == before[-1]['content']
        assert loaded[1]['role'] == 'assistant'
        assert loaded[1]['content'] == ''
        assert loaded[1]['api_content'].startswith('<tool-work-context>\n')
        assert '<memory-context>' not in loaded[1]['api_content']
        assert '## Tool work context' in loaded[1]['api_content']
        assert '[TOOL WORK SUMMARY]' not in loaded[1]['api_content']
        from agent.turn_context import substitute_api_content
        wire = dict(loaded[1])
        substitute_api_content(wire)
        assert wire['content'] == loaded[1]['api_content']
        assert not any(m.get('role') == 'tool' for m in loaded)
        print('SQLite resume:', len(before), '->', len(loaded), 'active messages')
        db.close()


def check_register_boolean():
    from types import SimpleNamespace
    for value, expected in [(True,True),(False,False),('true',False),(1,False)]:
        found=[]
        with patch('hermes_cli.config.load_config',return_value={'compression':{'tool_thread_compact':value}}):
            plugin.register(SimpleNamespace(register_context_engine=found.append))
        assert found[0]._tool_thread_compact_enabled is expected


if __name__ == '__main__':
    count = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith('check_') and callable(fn):
            fn()
            count += 1
            print('PASS',name)
    print('TOTAL',count,'checks passed, 0 failed, network disabled')
