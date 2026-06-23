"""LiteLLM adapter — supports any model via litellm.completion()."""

from __future__ import annotations

import logging

from ormah.background.llm.base import LLMAdapter

logger = logging.getLogger(__name__)


class LiteLLMAdapter(LLMAdapter):
    def __init__(self, model: str, timeout: int = 60) -> None:
        self.model = model
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        json_mode: bool = True,
        *,
        response_format: dict | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str | None:
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
        if response_format is not None:
            kwargs["response_format"] = response_format
        elif json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        try:
            response = litellm.completion(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            logger.warning("LiteLLM call failed: %s", e)
            return None
