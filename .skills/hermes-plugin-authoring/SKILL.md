---
name: hermes-plugin-authoring
description: "Build any type of Hermes Agent plugin from scratch."
---

# Hermes Plugin Authoring Skill

You are building or modifying a Hermes Agent plugin. This skill is your **complete reference**. Do not guess APIs — everything documented here is derived from the official Hermes source.

---

## 1. Plugin Map — Pick the Right Type

Before writing code, identify which plugin type you need:

| You want to add… | Type | Location | Selection |
|---|---|---|---|
| Custom tools, hooks, slash commands, CLI commands, bundled skills | **General plugin** | `~/.hermes/plugins/<name>/` | Multi-select via `plugins.enabled` |
| An LLM inference backend | **Model provider** | `plugins/model-providers/<name>/` | `--provider` / `model.provider` |
| A gateway channel (Discord, Telegram, IRC) | **Platform plugin** | `plugins/platforms/<name>/` | `gateway.platforms.<name>.enabled` |
| Persistent cross-session memory | **Memory provider** | `plugins/memory/<name>/` | Single-select via `memory.provider` |
| Context compression strategy | **Context engine** | `plugins/context_engine/<name>/` | Single-select via `context.engine` |
| Image generation backend | **Image-gen backend** | `plugins/image_gen/<name>/` | `image_gen.provider` |
| Video generation backend | **Video-gen backend** | `plugins/video_gen/<name>/` | `video_gen.provider` |
| Web search / extract backend | **Web search provider** | `plugins/web/<name>/` | `web.search_backend` / `web.extract_backend` |
| Cloud browser backend (CDP) | **Browser provider** | `plugins/browser/<name>/` | `browser.cloud_provider` |
| Secret manager (vault, keystore) | **Secret source** | `~/.hermes/plugins/<name>/` | `secrets.<name>.enabled` |
| Desktop app UI (panes, pages, themes) | **Desktop plugin** | `~/.hermes/desktop-plugins/<name>/plugin.js` | Settings → Plugins |

**Override rule:** User plugins at `~/.hermes/plugins/` override bundled plugins of the same name (last-writer-wins).

**Opt-in rule:** General plugins and user-installed backends are **disabled by default**. Add the plugin name to `plugins.enabled` in `config.yaml`, or run `hermes plugins enable <name>`.

---

## 2. General Plugins (Tools, Hooks, Commands, Skills)

### Directory Structure

```text
~/.hermes/plugins/my-plugin/
├── plugin.yaml      # Manifest — declares identity + capabilities
├── __init__.py      # register(ctx) — wires schemas to handlers
├── schemas.py       # Tool schemas (what the LLM sees)
└── tools.py         # Handler functions (what runs)
```

### A. Manifest (`plugin.yaml`)

```yaml
name: my-plugin
version: 1.0.0
description: Short description of what the plugin does
provides_tools:
  - my_tool
provides_hooks:
  - post_tool_call
author: Your Name
requires_env:
  - name: MY_API_KEY
    description: "API key for the service"
    url: "https://example.com/keys"
    secret: true
```

Fields: `name` (required), `version`, `description`, `provides_tools`, `provides_hooks`, `author`, `requires_env`. If `requires_env` vars are missing, the plugin is disabled cleanly. During `hermes plugins install`, users are prompted interactively for missing vars.

### B. Tool Schemas (`schemas.py`)

```python
MY_TOOL = {
    "name": "my_tool",
    "description": (
        "Specific description of what this tool does and when the LLM should use it. "
        "Be precise — this is how the model decides to call your tool."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "param1": {
                "type": "string",
                "description": "What this parameter is for",
            },
        },
        "required": ["param1"],
    },
}
```

**Schema description is critical.** Vague descriptions ("Does stuff") mean the model won't know when to use it. Be specific about capabilities and when to use them.

### C. Tool Handlers (`tools.py`)

**Rules for every handler:**
1. **Signature:** `def handler(args: dict, **kwargs) -> str`
2. **Return:** Always a JSON string — success AND errors alike
3. **Never raise:** Catch all exceptions, return error JSON
4. **Accept `**kwargs`:** Hermes may pass additional context in the future

```python
import json

def handle_my_tool(args: dict, **kwargs) -> str:
    try:
        param1 = args.get("param1", "").strip()
        if not param1:
            return json.dumps({"error": "param1 is required"})
        # ... do work ...
        return json.dumps({"result": "success", "data": param1})
    except Exception as e:
        return json.dumps({"error": str(e)})
```

### D. Registration (`__init__.py`)

```python
import logging
from . import schemas, tools

logger = logging.getLogger(__name__)

def register(ctx):
    """Called exactly once at startup. If this crashes, the plugin is disabled but Hermes continues."""

    # Register Tools
    ctx.register_tool(
        name="my_tool",
        toolset="my_plugin",        # namespace for grouping
        schema=schemas.MY_TOOL,
        handler=tools.handle_my_tool,
    )

    # Register Hooks (see §5 for the full hook catalog)
    ctx.register_hook("post_tool_call", _on_post_tool)

    # Register Slash Command (/mycmd in CLI and gateway)
    ctx.register_command("mycmd", lambda raw: _handle_cmd(ctx, raw), description="Command description")

    # Register CLI Subcommand (hermes my-plugin <subcommand>)
    ctx.register_cli_command(name="my-plugin", help="Manage my plugin",
                             setup_fn=_setup_argparse, handler_fn=_my_command)

    # Bundle Skills (namespaced as my-plugin:skill-name)
    # from pathlib import Path
    # ctx.register_skill("workflow", Path(__file__).parent / "skills" / "workflow" / "SKILL.md")


def _on_post_tool(tool_name, args, result, task_id, **kwargs):
    logger.debug("Tool called: %s (session %s)", tool_name, task_id)


def _handle_cmd(ctx, raw_args: str):
    """Slash command handler. Can dispatch tools as if the model called them."""
    result = ctx.dispatch_tool("terminal", {"command": f"echo {raw_args}"})
    return result


def _my_command(args):
    sub = getattr(args, "my_command", None)
    if sub == "status":
        print("All good!")

def _setup_argparse(subparser):
    subs = subparser.add_subparsers(dest="my_command")
    subs.add_parser("status", help="Show plugin status")
    subparser.set_defaults(func=_my_command)
```

### E. Full `ctx.*` API Reference

| Method | What it does |
|---|---|
| `ctx.register_tool(name, toolset, schema, handler, check_fn=None, override=False)` | Add a tool the LLM can call |
| `ctx.register_hook(event, callback)` | Subscribe to lifecycle events |
| `ctx.register_command(name, handler, description)` | Add `/name` slash command (CLI + gateway) |
| `ctx.register_cli_command(name, help, setup_fn, handler_fn)` | Add `hermes <name>` CLI subcommand |
| `ctx.register_skill(name, path)` | Bundle a skill (namespaced `plugin:name`) |
| `ctx.dispatch_tool(name, args)` | Call any tool with parent-agent context wired |
| `ctx.inject_message(content, role="user")` | Queue a message into the conversation (CLI only) |
| `ctx.register_platform(name, label, adapter_factory, ...)` | Add gateway channel |
| `ctx.register_image_gen_provider(provider)` | Add image-gen backend |
| `ctx.register_video_gen_provider(provider)` | Add video-gen backend |
| `ctx.register_web_search_provider(provider)` | Add web search backend |
| `ctx.register_browser_provider(provider)` | Add cloud browser backend |
| `ctx.register_memory_provider(provider)` | Add memory backend |
| `ctx.register_context_engine(engine)` | Replace context compressor |
| `ctx.register_secret_source(source)` | Add secret manager backend |
| `ctx.register_slack_action_handler(action_id, callback)` | Handle Slack Block Kit clicks |
| `ctx.llm.complete(messages=..., ...)` | Run an out-of-band LLM call (see §8) |
| `ctx.llm.complete_structured(instructions=..., input=..., json_schema=..., ...)` | Structured JSON LLM call |
| `ctx.profile_name` | Active profile name (works in all contexts) |

---

## 3. Model Provider Plugins

**Location:** `plugins/model-providers/<name>/`
Declares an inference backend (OpenAI-compat, Anthropic, Codex, Bedrock).

### Minimal Implementation

```python
# __init__.py
from providers import register_provider
from providers.base import ProviderProfile

register_provider(ProviderProfile(
    name="my-provider",
    aliases=("myprov",),
    display_name="My Provider",
    description="Custom inference API",
    signup_url="https://example.com/keys",
    env_vars=("MY_API_KEY", "MY_BASE_URL"),
    base_url="https://api.example.com/v1",
    auth_type="api_key",           # api_key | oauth_external | oauth_device_code | aws_sdk
    api_mode="chat_completions",   # chat_completions | codex_responses | anthropic_messages | bedrock_converse
    default_aux_model="my-small-model",
    fallback_models=("my-large", "my-small"),
))
```

```yaml
# plugin.yaml
name: my-provider
kind: model-provider
version: 1.0.0
description: Custom inference API
```

**Auto-wires:** credential resolution, `--provider` CLI, `hermes model` picker, `hermes doctor` health check, `hermes setup` wizard, URL reverse-mapping, auxiliary model, runtime resolution, transport.

**Advanced:** Subclass `ProviderProfile` and override `prepare_messages()`, `build_extra_body()`, `build_api_kwargs_extras()`, `fetch_models()` for provider-specific quirks.

**ProviderProfile key fields:** `name`, `aliases`, `api_mode`, `display_name`, `description`, `signup_url`, `env_vars`, `base_url`, `models_url`, `auth_type`, `fallback_models`, `default_headers`, `fixed_temperature`, `default_max_tokens`, `default_aux_model`.

---

## 4. Memory Provider Plugins

**Location:** `plugins/memory/<name>/`
Only ONE can be active at a time. Selected via `memory.provider` in config.

### Implementation

Subclass `agent.memory_provider.MemoryProvider`:

```python
import os, threading, json
from agent.memory_provider import MemoryProvider

class MyMemory(MemoryProvider):
    @property
    def name(self) -> str:
        return "my-memory"

    def is_available(self) -> bool:
        """NO network calls. Just check env vars / deps."""
        return bool(os.environ.get("MY_API_KEY"))

    def initialize(self, session_id: str, **kwargs) -> None:
        """kwargs includes hermes_home (str). Use it for storage — never hardcode ~/.hermes."""
        self.session_id = session_id

    def get_config_schema(self) -> list:
        """Prompted during `hermes memory setup`."""
        return [{"key": "api_key", "description": "API Key", "secret": True,
                 "required": True, "env_var": "MY_API_KEY"}]

    def save_config(self, values: dict, hermes_home: str) -> None:
        pass  # Write non-secret config to your native location

    def get_tool_schemas(self) -> list:
        return []  # Return memory tool schemas if any

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def sync_turn(self, user_content, assistant_content, *, session_id="", messages=None):
        """MUST BE NON-BLOCKING. Use threading for network calls."""
        def _sync():
            pass  # Network request
        threading.Thread(target=_sync, daemon=True).start()

    # Optional: prefetch(), system_prompt_block(), on_session_end(),
    # on_pre_compress(), on_memory_write(), shutdown()

def register(ctx) -> None:
    ctx.register_memory_provider(MyMemory())
```

**CLI commands:** Add a `cli.py` with `register_cli(subparser)` — auto-discovered, only shows when your provider is active.

---

## 5. Event Hooks — Complete Catalog

Register with `ctx.register_hook("event_name", callback)`. All callbacks should accept `**kwargs`.

### Directive / Control Hooks

| Hook | Signature | Return |
|---|---|---|
| `pre_tool_call` | `tool_name, args, task_id, **kw` | `{"action": "block", "message": "..."}` vetoes; `{"action": "approve", "message": "..."}` escalates to human-approval |
| `pre_llm_call` | `session_id, user_message, conversation_history, is_first_turn, model, platform, **kw` | Return `{"context": "text"}` or plain string to **inject into user message** (see below) |
| `pre_verify` | ... | ... |
| `pre_gateway_dispatch` | ... | ... |

### Transform Hooks

| Hook | Description |
|---|---|
| `transform_tool_result` | Modify tool output before the LLM sees it |
| `transform_terminal_output` | Modify terminal output |
| `transform_llm_output` | Modify LLM response |

### Observer Hooks (return value ignored)

| Hook | Signature |
|---|---|
| `post_tool_call` | `tool_name, args, result, task_id, duration_ms, **kw` |
| `post_llm_call` | `session_id, user_message, assistant_response, conversation_history, model, platform, **kw` |
| `on_session_start` | `session_id, model, platform, **kw` |
| `on_session_end` | `session_id, completed, interrupted, model, platform, **kw` |
| `on_session_finalize` | `session_id, platform, **kw` |
| `on_session_reset` | `session_id, platform, **kw` |
| `on_skill_lifecycle` | ... |
| `subagent_start` / `subagent_stop` | ... |
| `pre_approval_request` / `post_approval_response` | ... |
| `pre_api_request` / `post_api_request` / `api_request_error` | ... |
| `kanban_task_claimed` | `task_id, board, assignee, run_id, profile_name, **kw` |
| `kanban_task_completed` | `task_id, board, assignee, run_id, profile_name, summary, **kw` |
| `kanban_task_blocked` | `task_id, board, assignee, run_id, profile_name, reason, **kw` |

### `pre_llm_call` Context Injection

This is the **only hook whose return value matters** (besides `pre_tool_call`). Return context to inject into the user message:

```python
def recall_context(session_id, user_message, is_first_turn, **kwargs):
    memories = fetch_memories(user_message)
    if not memories:
        return None  # no injection
    return {"context": "Recalled:\n" + "\n".join(f"- {m}" for m in memories)}

def register(ctx):
    ctx.register_hook("pre_llm_call", recall_context)
```

**Rules:** Context is appended to the user message (not system prompt — preserves prompt cache). Per-hook context capped at 10,000 chars; overflow spills to disk. Multiple plugins' contexts are joined with double newlines.

---

## 6. Image Generation Provider Plugins

**Location:** `plugins/image_gen/<name>/`

Subclass `agent.image_gen_provider.ImageGenProvider`:

```python
from agent.image_gen_provider import (
    ImageGenProvider, success_response, error_response,
    resolve_aspect_ratio, save_b64_image, normalize_reference_images,
    DEFAULT_ASPECT_RATIO,
)

class MyImageGen(ImageGenProvider):
    @property
    def name(self) -> str:
        return "my-imggen"

    def is_available(self) -> bool:
        return bool(os.environ.get("MY_API_KEY"))

    def list_models(self) -> list:
        return [{"id": "fast", "display": "Fast", "speed": "~5s"}]

    def default_model(self):
        return "fast"

    def capabilities(self) -> dict:
        return {"modalities": ["text", "image"], "max_reference_images": 4}

    def get_setup_schema(self) -> dict:
        return {"name": "My Backend", "badge": "paid",
                "env_vars": [{"key": "MY_API_KEY", "prompt": "API key"}]}

    def generate(self, prompt, aspect_ratio=DEFAULT_ASPECT_RATIO, *,
                 image_url=None, reference_image_urls=None, **kwargs):
        aspect_ratio = resolve_aspect_ratio(aspect_ratio)
        if not prompt:
            return error_response(error="Prompt required", error_type="invalid_input",
                                  provider=self.name, prompt="", aspect_ratio=aspect_ratio)
        try:
            # ... call API ...
            # For base64 output: path = save_b64_image(b64_data, prefix=self.name)
            return success_response(image=url_or_path, model="fast", prompt=prompt,
                                    aspect_ratio=aspect_ratio, provider=self.name)
        except Exception as e:
            return error_response(error=str(e), error_type=type(e).__name__,
                                  provider=self.name, prompt=prompt, aspect_ratio=aspect_ratio)

def register(ctx):
    ctx.register_image_gen_provider(MyImageGen())
```

```yaml
# plugin.yaml
name: my-imggen
kind: backend
version: 1.0.0
requires_env: [MY_API_KEY]
```

---

## 7. Video Generation Provider Plugins

**Location:** `plugins/video_gen/<name>/`

Mirrors image-gen almost line-for-line. Subclass `agent.video_gen_provider.VideoGenProvider`:

```python
from agent.video_gen_provider import VideoGenProvider, success_response, error_response

class MyVideoGen(VideoGenProvider):
    @property
    def name(self) -> str:
        return "my-videogen"

    def is_available(self) -> bool:
        return bool(os.environ.get("MY_API_KEY"))

    def capabilities(self) -> dict:
        return {"modalities": ["text", "image"], "aspect_ratios": ["16:9", "9:16"],
                "min_duration": 1, "max_duration": 10}

    def generate(self, prompt, *, model=None, image_url=None, duration=None,
                 aspect_ratio="16:9", resolution="720p", **kwargs):
        modality = "image" if image_url else "text"
        # ... call API ...
        return success_response(video=url, model=model or "fast", prompt=prompt,
                                modality=modality, aspect_ratio=aspect_ratio,
                                duration=duration or 5, provider=self.name)

def register(ctx):
    ctx.register_video_gen_provider(MyVideoGen())
```

**Routing:** `image_url` present → image-to-video; absent → text-to-video.

---

## 8. Web Search Provider Plugins

**Location:** `plugins/web/<name>/`

Subclass `agent.web_search_provider.WebSearchProvider`:

```python
from agent.web_search_provider import WebSearchProvider

class MySearch(WebSearchProvider):
    @property
    def name(self) -> str:
        return "my-search"

    def is_available(self) -> bool:
        return bool(os.getenv("MY_SEARCH_KEY"))

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> dict:
        # ... call API ...
        return {"success": True, "data": {"web": [
            {"title": "...", "url": "...", "description": "...", "position": 1}
        ]}}

def register(ctx):
    ctx.register_web_search_provider(MySearch())
```

```yaml
name: web-my-search
kind: backend
provides_web_providers: [my-search]
requires_env: [MY_SEARCH_KEY]
```

**Response shapes:** Search: `{"success": True, "data": {"web": [...]}}`. Extract: `{"success": True, "data": [{"url", "title", "content", "raw_content"}]}`. Error: `{"success": False, "error": "message"}`.

---

## 9. Browser Provider Plugins

**Location:** `plugins/browser/<name>/`

Implements **session lifecycle only** (create/close CDP sessions). Hermes drives the browser via CDP.

Subclass `agent.browser_provider.BrowserProvider`:

```python
from agent.browser_provider import BrowserProvider

class MyBrowser(BrowserProvider):
    @property
    def name(self) -> str:
        return "my-browser"

    def is_available(self) -> bool:
        return bool(os.environ.get("MY_BROWSER_KEY"))

    def create_session(self, task_id: str) -> dict:
        session = my_api.create(...)
        return {
            "session_name": f"my-browser-{task_id}",
            "bb_session_id": session.id,    # legacy key name — keep verbatim
            "cdp_url": session.cdp_ws_url,
            "features": {"stealth": True},
        }

    def close_session(self, session_id: str) -> bool:
        """Never raise — return False on error."""
        ...

    def emergency_cleanup(self, session_id: str) -> None:
        """Best-effort teardown from atexit/signal. Must not raise."""
        ...

def register(ctx):
    ctx.register_browser_provider(MyBrowser())
```

---

## 10. Context Engine Plugins

**Location:** `plugins/context_engine/<name>/`
Single-select via `context.engine` in config. Replaces the built-in `ContextCompressor`.

Subclass `agent.context_engine.ContextEngine`:

```python
from agent.context_engine import ContextEngine

class MyEngine(ContextEngine):
    @property
    def name(self) -> str:
        return "my-engine"

    def update_from_response(self, usage: dict) -> None:
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        # ...

    def should_compress(self, prompt_tokens=None) -> bool:
        return self.last_prompt_tokens > self.threshold_tokens

    def compress(self, messages, current_tokens=None, focus_topic=None) -> list:
        """Return a valid OpenAI-format message list."""
        # ... compaction logic ...
        return messages

    # Optional: on_session_start(), on_session_end(), get_tool_schemas(),
    # handle_tool_call(), select_context(), on_turn_complete()

def register(ctx):
    ctx.register_context_engine(MyEngine())
```

**Class attributes to maintain:** `last_prompt_tokens`, `last_completion_tokens`, `last_total_tokens`, `threshold_tokens`, `context_length`, `compression_count`.

---

## 11. Secret Source Plugins

Resolves credentials from external secret managers at startup.

Subclass `agent.secret_sources.base.SecretSource`:

```python
from agent.secret_sources.base import SecretSource, FetchResult, ErrorKind, run_secret_cli

class MyVault(SecretSource):
    name = "myvault"
    label = "My Vault"
    shape = "mapped"  # "mapped" (explicit VAR→ref) or "bulk" (project dump)

    def fetch(self, cfg: dict, home_path) -> FetchResult:
        """MUST NOT raise. MUST NOT prompt."""
        result = FetchResult()
        token = os.environ.get("MYVAULT_TOKEN", "").strip()
        if not token:
            result.error = "MYVAULT_TOKEN not set"
            result.error_kind = ErrorKind.NOT_CONFIGURED
            return result
        try:
            proc = run_secret_cli(["myvault-cli", "export", "--json"],
                                  allow_env=["MYVAULT_TOKEN"], timeout=30)
        except RuntimeError as exc:
            result.error = str(exc)
            result.error_kind = ErrorKind.BINARY_MISSING
            return result
        result.secrets = parse_output(proc.stdout)
        return result

    def protected_env_vars(self, cfg):
        return frozenset({"MYVAULT_TOKEN"})

def register(ctx):
    ctx.register_secret_source(MyVault())
```

**Contract:** `fetch()` never raises, never prompts, never writes `os.environ`. Use `run_secret_cli()` instead of `subprocess.run` (minimal env, no `shell=True`).

---

## 12. Plugin LLM Access (`ctx.llm`)

Plugins can make out-of-band LLM calls using the user's active provider:

```python
# Chat completion
result = ctx.llm.complete(
    messages=[{"role": "user", "content": "Summarize this"}],
    max_tokens=256, temperature=0.3, purpose="summary",
)
text = result.text  # result also has .provider, .model, .usage

# Structured extraction
result = ctx.llm.complete_structured(
    instructions="Extract tasks from meeting notes.",
    input=[{"type": "text", "text": notes}],
    json_schema={"type": "object", "properties": {"tasks": {"type": "array", ...}}},
    purpose="extract-tasks",
)
parsed = result.parsed  # Python dict, or None if parsing failed

# Async variants
result = await ctx.llm.acomplete(messages=...)
result = await ctx.llm.acomplete_structured(instructions=..., input=...)
```

**Trust gate:** By default, plugins use the user's active provider/model. To override provider or model, the operator must opt-in via `plugins.entries.<id>.llm.allow_provider_override: true` in config.

---

## 13. Desktop Plugins (Native App UI)

**Location:** `~/.hermes/desktop-plugins/<id>/plugin.js`
Plain ESM, loaded uncompiled. **No JSX** — use `jsx()` / `jsxs()` from `react/jsx-runtime`.
Only importable: `@hermes/plugin-sdk`, `react`, `react/jsx-runtime`.

```javascript
import { host, haptic, useValue, cn, Tip } from '@hermes/plugin-sdk'
import { jsx, jsxs } from 'react/jsx-runtime'

const ID = 'my-plugin'

function MyPane() {
  const gateway = useValue(host.state.gateway)
  return jsxs('div', {
    className: 'flex h-full flex-col gap-2 p-3 text-sm',
    children: [
      jsx('div', { className: 'font-medium', children: 'My Plugin' }),
      jsx('div', { className: 'text-(--ui-text-tertiary)', children: `gateway: ${gateway}` })
    ]
  })
}

export default {
  id: ID,
  name: 'My Plugin',
  register(ctx) {
    ctx.register({ id: 'pane', area: 'panes', title: 'my plugin',
                   data: { placement: 'right', width: '260px' },
                   render: () => jsx(MyPane, {}) })
  }
}
```

**Areas:** `panes`, `statusBar.left`/`.right`, routes (`ROUTES_AREA`), sidebar nav, palette (⌘K), keybinds, themes, composer slots.

**Backend:** Ship `~/.hermes/plugins/<id>/dashboard/manifest.json` + `plugin_api.py` (FastAPI `router`). Reach it via `ctx.rest('/path')` / `ctx.socket('/path', onMessage)`.

**Rules:** Never hardcode colors — use `var(--ui-*)` theme variables. Use `host.state.*` with `useValue()` in components, `.get()` in handlers.

---

## 14. Advanced Patterns

### Thread-Safe Singletons

```python
from plugins.plugin_utils import lazy_singleton

@lazy_singleton
def get_client():
    return ExpensiveClient()

client = get_client()       # safe across threads
get_client.reset()          # drop for tests / teardown
```

### Lazy Dependencies

```python
from tools.lazy_deps import ensure, FeatureUnavailable

def handler(args, **kwargs):
    try:
        ensure("my-plugin.my-backend")  # must be in LAZY_DEPS allowlist
    except FeatureUnavailable as e:
        return json.dumps({"error": str(e)})
    import heavy_sdk
    # ...
```

### Overriding Built-in Tools

```python
ctx.register_tool(name="browser_navigate", toolset="my_browser",
                  schema={...}, handler=my_handler, override=True)
```
Requires operator opt-in: `plugins.entries.<id>.allow_tool_override: true` in config.

### Conditional Tool Availability

```python
ctx.register_tool(name="my_tool", schema={...}, handler=my_handler,
                  check_fn=lambda: _has_optional_lib())  # False = hidden from model
```

### Hooks from Kanban Workers

Use `ctx.profile_name` and `ctx.dispatch_tool()` — they work everywhere (CLI, gateway, kanban workers). Do NOT use `ctx._cli_ref` (it's None in gateway/kanban).

---

## 15. Distribution

### Via pip (entry point)

```toml
# pyproject.toml
[project.entry-points."hermes_agent.plugins"]
my-plugin = "my_plugin_package"
```

### Via git

```bash
hermes plugins install user/repo --enable
```

### Plugin management

```bash
hermes plugins                    # interactive toggle UI
hermes plugins list               # table view
hermes plugins enable <name>      # add to allow-list
hermes plugins disable <name>     # remove + add to disabled
hermes plugins install user/repo  # install from Git
hermes plugins update <name>      # pull latest
hermes plugins remove <name>      # uninstall
```

---

## 16. Debugging

```bash
HERMES_PLUGINS_DEBUG=1 hermes plugins list    # verbose discovery logs
hermes logs --level WARNING | grep -i plugin  # check for failures
```

**Common issues:**
- Not in `plugins.enabled` → `hermes plugins enable <name>`
- Missing `__init__.py` or `plugin.yaml`
- `kind` wrong (platform adapters need `kind: platform`)
- Handler returns dict instead of JSON string
- Handler missing `**kwargs`
- Handler raises instead of returning error JSON
- Schema description too vague for the model to know when to use it

---

## 17. Common Mistakes

```python
# WRONG — returns dict
def handler(args, **kwargs):
    return {"result": 42}
# RIGHT — returns JSON string
def handler(args, **kwargs):
    return json.dumps({"result": 42})

# WRONG — missing **kwargs
def handler(args):
    ...
# RIGHT
def handler(args, **kwargs):
    ...

# WRONG — raises exception
def handler(args, **kwargs):
    result = 1 / int(args["value"])  # ZeroDivisionError!
# RIGHT — catches and returns error
def handler(args, **kwargs):
    try:
        result = 1 / int(args.get("value", 0))
        return json.dumps({"result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})
```
