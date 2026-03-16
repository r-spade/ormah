"""LiteLLM adapter — supports any model via litellm.completion()."""

from __future__ import annotations

import logging

from ormah.background.llm.base import LLMAdapter

logger = logging.getLogger(__name__)


class LiteLLMAdapter(LLMAdapter):
    def __init__(self, model: str, timeout: int = 60) -> None:
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str, json_mode: bool = True) -> str | None:
        try:
            import litellm  # lazy import
        except ImportError:
            logger.error(
                "litellm is not installed. Install it with: pip install 'ormah[litellm]'"
            )
            return None

        messages = [{"role": "user", "content": prompt}]
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "timeout": self.timeout,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = litellm.completion(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            logger.warning("LiteLLM call failed: %s", e)
            return None
