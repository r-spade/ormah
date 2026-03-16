"""Abstract base class for LLM adapters."""

from __future__ import annotations

import abc


class LLMAdapter(abc.ABC):
    """Interface that all LLM backends must implement."""

    @abc.abstractmethod
    def generate(self, prompt: str, json_mode: bool = True) -> str | None:
        """Send *prompt* to the LLM and return the raw response text.

        Returns ``None`` on any failure (timeout, connection error, etc.).
        When *json_mode* is True the adapter should request structured JSON
        output from the backend (if supported).
        """
