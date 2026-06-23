"""Abstract base class for LLM adapters."""

from __future__ import annotations

import abc


class LLMAdapter(abc.ABC):
    """Interface that all LLM backends must implement."""

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        json_mode: bool = True,
        *,
        response_format: dict | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str | None:
        """Send *prompt* to the LLM and return the raw response text.

        Returns ``None`` on any failure (timeout, connection error, etc.).
        When *json_mode* is True the adapter should request structured JSON
        output from the backend (if supported). *response_format* lets callers
        provide a provider-specific structured-output schema.
        """
