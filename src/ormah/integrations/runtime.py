"""Bounded, silent HTTP whisper retrieval shared by host-specific adapters."""
from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path

import httpx


def base_url() -> str:
    # Lazy import: shell hooks need not load Ormah's engine or embeddings.
    if value := os.environ.get("ORMAH_URL"):
        return value.rstrip("/")
    return f"http://127.0.0.1:{os.environ.get('ORMAH_PORT', '8787')}"


def headers() -> dict[str, str]:
    token = os.environ.get("ORMAH_AUTH_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def timeout_seconds() -> float:
    try:
        return max(0.1, min(float(os.environ.get("ORMAH_WHISPER_TIMEOUT", "2")), 10.0))
    except ValueError:
        return 2.0


async def space_for(workspace: str | None) -> str | None:
    if explicit := os.environ.get("ORMAH_SPACE"):
        return explicit
    if not workspace or not Path(workspace).is_dir():
        return None
    path = Path(workspace).resolve()
    if path == Path.home() or path == Path(path.anchor):
        return None
    process = None
    try:
        process = await asyncio.create_subprocess_exec(
            "git", "rev-parse", "--show-toplevel", cwd=path,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=0.25)
        if process.returncode == 0:
            return Path(stdout.decode().strip()).name
    except (OSError, TimeoutError):
        pass
    finally:
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()
    return path.name


def session_key(host: str, session: str, workspace: str | None) -> str:
    scope = str(Path(workspace).resolve()) if workspace else ""
    digest = hashlib.sha256(scope.encode()).hexdigest()[:16]
    return f"{host}:{digest}:{session}"


def context_text(text: str) -> str:
    # The server's legacy signal names a Claude/Pi custom agent. New hosts receive
    # the same signal with a portable two-call instruction; engine unchanged.
    return text.replace(
        "run the ormah-maintenance agent in the background; continue the conversation without blocking the user.",
        "use run_maintenance() to obtain work, then run_maintenance(results=...) "
        "to submit evaluations at a safe point; do not assume a named custom agent exists.",
    )


async def whisper(host: str, prompt: str, session: str, workspace: str | None) -> str:
    if not isinstance(prompt, str) or not prompt.strip() or not session:
        return ""
    try:
        async with asyncio.timeout(timeout_seconds()):
            space = await space_for(workspace)
            async with httpx.AsyncClient(
                base_url=base_url(), headers=headers(), timeout=timeout_seconds(),
                follow_redirects=False,
            ) as client:
                response = await client.post("/agent/whisper", json={
                    "prompt": prompt, "space": space,
                    "session_id": session_key(host, session, workspace),
                })
                response.raise_for_status()
                data = response.json()
                text = data.get("text") if isinstance(data, dict) else None
                return context_text(text) if isinstance(text, str) else ""
    except (httpx.HTTPError, OSError, ValueError, TimeoutError):
        return ""
