"""Ollama LLM adapter — HTTP calls to a local Ollama instance."""

from __future__ import annotations

import logging

from ormah.background.llm.base import LLMAdapter

logger = logging.getLogger(__name__)


class OllamaAdapter(LLMAdapter):
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

    def generate(self, prompt: str, json_mode: bool = True) -> str | None:
        import httpx

        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if json_mode:
            payload["format"] = "json"

        try:
            resp = httpx.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json().get("response")
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            logger.warning("Ollama unavailable: %s", e)
            return None
        except Exception as e:
            logger.warning("Ollama call failed: %s", e)
            return None
