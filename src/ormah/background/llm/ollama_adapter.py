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
        num_predict: int = 4096,
        num_ctx: int | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.num_predict = num_predict
        # INPUT window. num_predict bounds the OUTPUT only; leaving num_ctx unset inherits the
        # server's default, which we neither control nor version. A default below the payload
        # truncates the prompt SILENTLY -- the HTTP call still returns 200 with a plausible
        # answer, so nothing surfaces.
        #
        # None means OMIT the key, NOT "substitute a default of our own": a number invented here
        # would silently NARROW every caller that today leaves the operator's server/Modelfile
        # setting in charge. Only the consolidation route opts in (#192), because it is the one
        # route whose prompt carries full source content.
        self.num_ctx = num_ctx

    def generate(
        self,
        prompt: str,
        json_mode: bool = True,
        *,
        response_format: dict | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str | None:
        import httpx

        options: dict = {"num_predict": max_tokens or self.num_predict}
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        if temperature is not None:
            options["temperature"] = temperature

        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            # Disable thinking: reasoning tokens consume the num_predict budget
            # and on large transcripts starve the JSON, yielding empty/truncated
            # extractions. Non-thinking models ignore this flag.
            "think": False,
            "options": options,
        }
        if response_format and response_format.get("type") == "json_schema":
            payload["format"] = response_format.get("json_schema", {}).get("schema", "json")
        elif json_mode:
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
