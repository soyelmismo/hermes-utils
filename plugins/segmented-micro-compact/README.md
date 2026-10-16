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

Tool-thread and segment compaction are **alternative policies**, never two stages
in a pipeline. With `tool_thread_compact: true`, finalization returns directly
from the tool policy and does not run segment summarization over its output.
Each pass makes at most **one auxiliary `call_llm` invocation**. One thread uses
the bullet contract; multiple independent threads share one JSON request/response,
validated by unique integer IDs before any splice. There are no repair calls.
User messages, OOB steers, and final answers remain verbatim and in position.

The batch is bounded by the engine's input window and output-summary budget
(1500 output tokens reserved per thread, at most 25 threads). Oversized or excess
threads remain raw for later passes or normal threshold compression; the plugin
never truncates raw evidence just to fit. Input selection uses original rows only.
Existing tool markers are recognized by metadata, canonical wrapper, or legacy
prefix even after SQLite reload, and are excluded from micro-summary selection.

Invalid/empty/truncated responses and non-shrinking spans are remembered by
SHA-256 of their serialized input in a per-engine bounded cache (128 entries),
so index changes do not cause immediate retries on identical input. Changed input
or a session reset can be attempted again. This cache is not persisted across
process restarts. Reentrant tool passes are rejected. The batch uses one atomic
watermark/coverage-aware DB commit, and DB failures leave the original history
intact instead of accepting the core helper's best-effort success.

The one-call guarantee covers this plugin's micro-compaction pass, not provider
transport retries/fallbacks or Hermes' separate threshold context compression.

The final response is generated using the complete data.
The summary is produced at turn finalization, before persistence, and replaces the tool thread before the final response in conversation history.
El resumen ocupa una fila sintética nueva con `role: user`, en la posición del
thread eliminado. Nunca se concatena al mensaje humano ni a la respuesta final.
La fila usa `display_kind: internal_context` y
`display_metadata.context_source: segmented-micro-compact/tool-thread`.
La procedencia durable permite reconocerla tras SQLite sin clasificar citas
humanas del marcador como resúmenes. Las filas assistant antiguas siguen siendo
reconocibles, pero no se migran automáticamente.
It uses a plugin-owned `<tool-work-context>` wrapper and a context note identifying conversation-local tool history, not persistent memory or instructions.
It does not call `build_memory_context_block` or use `<memory-context>`; that fence remains reserved for memory-provider context.
The block is stored in visible `content` and mirrored in `api_content`, so history retains the reference above the final response and request replay uses identical bytes.
An empty `content` would hide the summary from the UI even when model injection still works. Do not use a model-only sidecar for this history-visible feature.
El replay conserva `user humano → user contexto → assistant respuesta`, incluida
la memoria del `api_content` humano. La metadata local no se envía al provider.
La persistencia puede omitir un sidecar redundante si coincide con `content`.

### Compatibilidad del core

Requiere el soporte de frontera interna de
`compat/internal-context-boundary.patch`, que incluye sus regresiones.
El parche protege el repair, el merge de micro-compaction y el merge final del
request, y excluye estas filas de las citas atribuidas al usuario en summaries.
Sin ese soporte, el creador del marcador falla cerrado y conserva los originales.
No instala monkeypatches globales ni cambia servicios por su cuenta.

Para otra instalación, revisar y aplicar desde el checkout de Hermes:

```bash
git apply --check /ruta/al/plugin/compat/internal-context-boundary.patch
git apply /ruta/al/plugin/compat/internal-context-boundary.patch
python -m pytest tests/agent/test_internal_context_boundary.py -o 'addopts=' -q
```

El test del plugin atraviesa finalizer real, SQLite, repair repetido y request
builder, con memoria inyectada en el mensaje humano. Los tests del core también
atraviesan el ensamblado final del request y sus sanitizers.
Los adapters nativos que imponen alternancia pueden agrupar bloques user en su
wire específico. Eso no convierte el resumen en contenido assistant ni reescribe
el historial canónico. El test de separación exacta cubre Chat Completions.

The final-output hook removes echoed tool-context blocks and complete legacy `[TOOL WORK SUMMARY]` blocks when visible response text remains. It leaves Holographic memory blocks untouched.
A closed block is removed wherever it appears. An unterminated block is removed ONLY when it is verifiably our own echo, i.e. the canonical note or the `## Tool work context` heading directly follows the opening tag. A bare opening tag in model prose (a reply that mentions or quotes the wrapper) is never truncated, so the hook cannot cut an answer short or trigger the CLI's post-stream transform notice on a reply it did not change.
The hook returns `None` whenever the text is unchanged, so no transform is recorded for untouched replies.
The current core ignores empty-string output transformations, so a response consisting entirely of internal context is not suppressed by this hook alone.
Streaming observers cannot transform deltas. The core streaming scrubber recognizes memory fences only, so a model-echoed tool-context block can appear during live streaming before the final-output hook removes it. El plugin no parchea scrubbers globales. El soporte de fronteras del core se aplica por separado.
Restart the CLI or gateway to load the change. It applies to new summaries and does not rewrite previously persisted summaries.
It does not guarantee that the model will never paraphrase internal data without delimiters.
It does not compact between tool rounds or reduce the context sent to generate the current response.

Pending calls, inconsistent IDs, summary/persistence failures, non-bullet output, or no token savings leave the original thread intact.
Original messages are archived through Hermes' existing mechanism rather than permanently deleted.
The existing integration invalidates read/skill deduplication so removed data can be read again.
Summarization adds auxiliary cost for eligible turns and may invalidate the prefix cache.
Semantic fidelity depends on the model and prompt and is not guaranteed by the plugin.

## Conservación del prefijo inicial de skills

Tras cada mensaje humano se conservan completos los batches iniciales que solo
leen `skill_view`, con sus calls, results, texto, IDs y orden exactos. También se
conservan varias lecturas iniciales consecutivas. No se proyectan filas ni se
mueve su contenido al resumen, por lo que ese prefijo no cambia en este pase.

```text
user → skill_view → tools  => user → skill_view → resumen
user → tools → skill_view  => user → resumen
```

La primera tool no-skill cierra la excepción durante el resto del turno. Las
skills posteriores se resumen normalmente junto con las otras tools. Un batch
mixto se resume entero, aunque su primera call lea una skill: conservar solo una
parte cambiaría la fila inicial y ya no mantendría el prefijo original.
Los wrappers paralelos son atómicos y solo se conservan si todos sus hijos son
lecturas de skills. Se reconocen el nombre dotted y el nombre coaccionado.
Un resumen interno nunca cuenta como nuevo mensaje humano ni reabre la excepción.

El engine restaura estos resultados iniciales tras el prune/dedup ordinario del
core. Las skills tardías siguen la política normal del core, sin protección extra.
No restaura skills resumidas antes de cargar este cambio. Mantener el prefijo
permite reutilizar caché, pero su disponibilidad y duración dependen del provider.
La compresión completa por umbral sigue siendo distinta y puede retirar historia
al agotar la ventana. Esta protección no crea contexto infinito.

Regresiones: skills iniciales cortas/largas, lecturas iniciales consecutivas,
skills tardías, lotes y wrappers mixtos, límites internos, nuevo turno humano,
fallos de resumen/commit, prune, calls sin ID y cursor en múltiples splices.
SQLite → finalizer → repair → request comprueba la identidad y el prefijo wire
idénticos para los pares conservados, incluidos memoria y metadata durable.
Los mensajes user internos consecutivos se conservan deliberadamente separados,
no se fusionan con preguntas humanas ni se alternan mediante respuestas ficticias.

## Testing without API calls

Run `python3 run_tests.py` to execute `test_segmented_micro_compact.py`, `verify_tool_thread.py`, `verify_single_pass.py`, `verify_user_context.py`, and `verify_skill_retention.py`.
The tests run entirely offline with deterministic mocks, without accessing the network or changing the kernel namespace.
