# Segmented micro-compaction

## Modules

- `__init__.py`: plugin registration and compatible exports
- `engine.py`: turn-based cadence and independent segment splicing
- `markers.py`: segment metadata and delimiters
- `prompts.py`: concise summary contracts
- `summarizer.py`: shared auxiliary route and response validation
- `tool_thread.py`: complete thread detection and replacement with summaries
- `persistence.py`: fail-closed transaction with watermark and coverage
- `verify_tool_thread.py`: offline checks and finalizer/SQLite integration

## Optional tool-thread compaction

```yaml
compression:
  micro_compact: true
  micro_compact_every_n_turns: 10
  tool_thread_compact: false
```

`tool_thread_compact` accepts only the actual boolean `true`, not strings or numbers.
It is disabled by default and requires `micro_compact: true` because it reuses the finalizer seam.

When enabled, it summarizes completed assistant tool-call/result threads at turn finalization, preserving user messages and the final response unchanged.
It does not wait for the 10-turn cadence to summarize tool threads.

The final response is generated using the complete data.
The summary is produced at turn finalization, before persistence, and replaces the tool thread before the final response in conversation history.
The context stays at the beginning of the assistant span, without moving to the user message or system prompt.
It uses a plugin-owned `<tool-work-context>` wrapper and a context note identifying conversation-local tool history, not persistent memory or instructions.
It does not call `build_memory_context_block` or use `<memory-context>`; that fence remains reserved for memory-provider context.
The block is stored in visible `content` and mirrored in `api_content`, so history retains the reference above the final response and request replay uses identical bytes.
An empty `content` would hide the summary from the UI even when model injection still works. Do not use a model-only sidecar for this history-visible feature.
The core replays the block at the same assistant position, including after resuming from SQLite; persistence may omit a redundant sidecar when it equals `content`.
The final-output hook removes echoed tool-context blocks, including unterminated trailing blocks, and complete legacy `[TOOL WORK SUMMARY]` blocks when visible response text remains. It leaves Holographic memory blocks untouched.
The current core ignores empty-string output transformations, so a response consisting entirely of internal context is not suppressed by this hook alone.
Streaming observers cannot transform deltas. The core streaming scrubber recognizes memory fences only, so a model-echoed tool-context block can appear during live streaming before the final-output hook removes it. No core files or global scrubbers are patched by this plugin.
Restart the CLI or gateway to load the change. It applies to new summaries and does not rewrite previously persisted summaries.
It does not guarantee that the model will never paraphrase internal data without delimiters.
It does not compact between tool rounds or reduce the context sent to generate the current response.

Pending calls, inconsistent IDs, summary/persistence failures, non-bullet output, or no token savings leave the original thread intact.
Original messages are archived through Hermes' existing mechanism rather than permanently deleted.
The existing integration invalidates read/skill deduplication so removed data can be read again.
Summarization adds auxiliary cost for eligible turns and may invalidate the prefix cache.
Semantic fidelity depends on the model and prompt and is not guaranteed by the plugin.

## Testing without API calls

Run `python3 run_tests.py` to execute both `test_segmented_micro_compact.py` and `verify_tool_thread.py`.
The tests run entirely offline with deterministic mocks, without accessing the network or changing the kernel namespace.
