"""Plain-text providers. Subprocesses receive no benchmark files or tools."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import tomllib
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path

from eval.bench.cost import Budget, BudgetExceeded, token_cost



def _default_codex() -> str:
    """Codex binary: ORMAH_BENCH_CODEX, else the pinned ai-agents build, else PATH."""
    override = os.environ.get("ORMAH_BENCH_CODEX")
    if override:
        return override
    pinned = "/root/agent-tools/codex-0.154.0/node_modules/.bin/codex"
    if Path(pinned).exists():
        return pinned
    return shutil.which("codex") or "codex"


CODEX = _default_codex()

# Neutral system prompt for subscription CLI calls. It replaces Claude Code's
# agentic coding prompt so answers come from the benchmark prompt alone.
CLAUDE_CLI_SYSTEM_PROMPT = "You are a careful assistant. Follow the task instructions exactly."


@dataclass
class TextResult:
    text: str
    model: str
    usage: dict | None = None
    latency_s: float = 0
    usd: float = 0
    estimated_usd: float | None = None


class ProviderError(RuntimeError):
    def __init__(self, message, result=None):
        super().__init__(message)
        self.result = result


class TextProvider(ABC):
    name: str

    def __init__(self, model: str, *, ledger=None, phase="answer", budget=None, timeout=300):
        self.model, self.ledger, self.phase = model, ledger, phase
        self.budget = budget or Budget(5)
        self.timeout = timeout

    def complete(self, prompt: str, *, item_id="", max_tokens=1024) -> TextResult:
        for attempt in range(3):
            # Each ledger row describes one attempt; retry backoff belongs to
            # phase wall time, not the next provider call's latency.
            start = time.perf_counter()
            try:
                result = self._call(prompt, max_tokens)
                result.latency_s = time.perf_counter() - start
                self._record(item_id, prompt, result, attempt + 1)
                return result
            except BudgetExceeded:
                raise
            except Exception as exc:
                message = str(exc)
                failed_result = getattr(exc, "result", None)
                if failed_result is not None:
                    failed_result.latency_s = time.perf_counter() - start
                transient = any(
                    s in message.lower()
                    for s in ("429", "overloaded", "503", "502", "rate limit", "timed out")
                )
                if transient and attempt < 2:
                    self._record(
                        item_id,
                        prompt,
                        failed_result,
                        attempt + 1,
                        message,
                        time.perf_counter() - start,
                    )
                    time.sleep(2**attempt)
                    continue
                self._record(
                    item_id,
                    prompt,
                    failed_result,
                    attempt + 1,
                    message,
                    time.perf_counter() - start,
                )
                raise ProviderError(message) from exc
        raise AssertionError("unreachable")

    def _record(self, item_id, prompt, result, attempts, error=None, latency_s=0):
        if self.ledger:
            self.ledger.append(
                {
                    "item_id": item_id,
                    "phase": self.phase,
                    "provider": self.name,
                    "model_requested": self.model,
                    "attempt": attempts,
                    "prompt_tokens_estimate": (len(prompt) + 3) // 4,
                    "error": error,
                    "latency_s": latency_s,
                    **(asdict(result) if result else {}),
                }
            )

    @abstractmethod
    def _call(self, prompt: str, max_tokens: int) -> TextResult: ...

    def version(self) -> str:
        return "unknown"


class CLIProvider(TextProvider):
    executable: str

    def version(self) -> str:
        try:
            return subprocess.run(
                [self.executable, "--version"],
                capture_output=True,
                text=True,
                timeout=15,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return "unavailable"

    def run(self, args, prompt, cwd):
        env = os.environ.copy()
        # Subscription providers must never silently switch to API billing.
        for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
            env.pop(key, None)
        result = subprocess.run(
            args,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=self.timeout,
            cwd=cwd,
            env=env,
        )
        if result.returncode:
            message = (result.stderr or result.stdout)[-1500:]
            failed_result = None
            if self.name == "claude-cli":
                try:
                    data = json.loads(result.stdout)
                    message = data.get("result", message)
                    failed_result = TextResult(
                        "", self.model, data.get("usage"), estimated_usd=data.get("total_cost_usd")
                    )
                except (json.JSONDecodeError, AttributeError):
                    pass
            raise ProviderError(f"{self.name} exit {result.returncode}: {message}", failed_result)
        return result


class ClaudeCLIProvider(CLIProvider):
    name, executable = "claude-cli", "claude"

    def _call(self, prompt, max_tokens):
        args = [
            self.executable,
            "-p",
            "--model",
            self.model,
            # --safe-mode, not --bare: --bare skips OAuth, so subscription
            # login fails with "Not logged in". Safe mode still disables
            # CLAUDE.md, hooks (e.g. an Ormah whisper hook), plugins, skills
            # and MCP, which keeps the owner's own memories out of answers.
            "--safe-mode",
            "--system-prompt",
            CLAUDE_CLI_SYSTEM_PROMPT,
            "--no-session-persistence",
            "--output-format",
            "json",
            "--tools",
            "",
            "--strict-mcp-config",
            "--mcp-config",
            '{"mcpServers":{}}',
            "--disable-slash-commands",
        ]
        with tempfile.TemporaryDirectory(prefix="ormah-bench-claude-") as cwd:
            raw = self.run(args, prompt, cwd)
        data = json.loads(raw.stdout)
        if data.get("is_error") or not isinstance(data.get("result"), str):
            raise ProviderError(f"claude-cli invalid result: {str(data)[:1000]}")
        models = list(data.get("modelUsage", {}))
        return TextResult(
            data["result"],
            models[0] if len(models) == 1 else self.model,
            data.get("usage"),
            estimated_usd=data.get("total_cost_usd"),
        )


class CodexProvider(CLIProvider):
    name, executable = "codex", CODEX

    def __init__(self, model, **kwargs):
        if model == "default":
            config = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "config.toml"
            if config.exists():
                # Carry only the configured model into --ignore-user-config, not
                # hooks, MCP servers, instructions, permissions or private context.
                model = tomllib.loads(config.read_text()).get("model", "default")
        super().__init__(model, **kwargs)

    def _call(self, prompt, max_tokens):
        with tempfile.TemporaryDirectory(prefix="ormah-bench-codex-") as cwd:
            output = Path(cwd) / "answer.txt"
            args = [
                self.executable,
                "exec",
                "--ignore-user-config",
                "--sandbox",
                "read-only",
                "--color",
                "never",
                "--ephemeral",
                "--skip-git-repo-check",
                "--json",
                "-c",
                "features.shell_tool=false",
                "--output-last-message",
                str(output),
            ]
            if self.model != "default":
                args += ["--model", self.model]
            args += ["-"]
            raw = self.run(
                args, "Answer the supplied task directly. Do not use tools.\n\n" + prompt, cwd
            )
            answer = output.read_text().strip()
            if not answer:
                raise ProviderError("codex returned an empty final-message file")
            usage = None
            for line in raw.stdout.splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "turn.completed":
                    usage = event.get("usage")
            return TextResult(answer, self.model, usage)


class AnthropicProvider(TextProvider):
    name = "anthropic"

    def __init__(self, model, *, client=None, **kwargs):
        super().__init__(model, **kwargs)
        if client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise ProviderError("anthropic requires ANTHROPIC_API_KEY")
            from anthropic import Anthropic

            client = Anthropic(max_retries=0, timeout=self.timeout)
        self.client = client

    def complete(self, prompt, *, item_id="", max_tokens=1024):
        with self.budget.lock:
            self.budget.check(
                token_cost(
                    self.model,
                    {"input_tokens": (len(prompt) + 3) // 4, "output_tokens": max_tokens},
                )
            )
            result = super().complete(prompt, item_id=item_id, max_tokens=max_tokens)
            # Journal actual charges before aborting, including the crossing call.
            self.budget.add(result.usd)
            return result

    def _call(self, prompt, max_tokens):
        response = self.client.messages.create(
            model=self.model,
            temperature=0,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        usage = response.usage.model_dump()
        return TextResult(
            "".join(b.text for b in response.content if b.type == "text"),
            response.model,
            usage,
            usd=token_cost(self.model, usage),
        )


def make_provider(name, model=None, **kwargs):
    classes = {
        "claude-cli": ClaudeCLIProvider,
        "codex": CodexProvider,
        "anthropic": AnthropicProvider,
    }
    defaults = {"claude-cli": "sonnet", "codex": "default", "anthropic": "claude-haiku-4-5"}
    return classes[name](model or defaults[name], **kwargs)
