"""Shared auxiliary compression route and response validation."""
from __future__ import annotations
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class SummarizerMixin:
    def _segment_summarize_one(self, exchange_text: str) -> Optional[str]:
        return self._summarize_prompt(self._build_segment_summary_prompt(exchange_text))

    def _tool_thread_summarize_one(self, exchange_text: str) -> Optional[str]:
        try:
            from .prompts import build_tool_prompt
        except (ImportError, ValueError):
            from prompts import build_tool_prompt
        return self._summarize_prompt(build_tool_prompt(exchange_text))

    def _summarize_prompt(self, prompt: list[dict[str, str]]) -> Optional[str]:
        """Call the existing auxiliary compression route and reject partial output."""
        from agent.auxiliary_client import aux_interrupt_protection, call_llm

        call_kwargs = {
            "task": "compression",
            "messages": prompt,
            "max_tokens": min(1500, self.max_summary_tokens or 1500),
            "temperature": 0.1,
        }
        self._apply_summary_route(call_kwargs)
        if self.model:
            call_kwargs.setdefault("main_runtime", {
                "model": self.model, "provider": self.provider or "",
                "base_url": self.base_url or "", "api_key": self.api_key or "",
                "api_mode": getattr(self, "api_mode", "") or "",
            })
        try:
            with aux_interrupt_protection():
                response = call_llm(**call_kwargs)
        except Exception as exc:
            logger.info("segment summarization call failed: %s", exc)
            return None

        # Same length guard as the built-in: a partial merge is discarded so the
        # exchange stays unabsorbed and a later pass retries. _response_finish_reason
        # is a module-level helper on the compressor module (context_compressor.py),
        # not a method on the instance.
        try:
            from agent.context_compressor import _response_finish_reason
            if _response_finish_reason(response) == "length":
                logger.warning("segment summarization hit token cap — discarding partial")
                return None
        except Exception:
            pass

        message = response.choices[0].message
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", message)
        content = (content if isinstance(content, str) else str(content) if content else "").strip()
        try:
            from agent.agent_runtime_helpers import strip_think_blocks
            content = strip_think_blocks(None, content).strip()
        except Exception:
            pass
        return content or None
