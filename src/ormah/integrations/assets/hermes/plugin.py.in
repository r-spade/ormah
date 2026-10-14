"""Hermes 0.21.4 native hook. No dependency installation in the host interpreter."""
import json
import os
import subprocess

PYTHON = __ORMAH_PYTHON__
WORKSPACE = __ORMAH_WORKSPACE__
PROFILE = __ORMAH_PROFILE__
ENV_KEYS = __ORMAH_ENV_KEYS__


def recall(session_id=None, user_message=None, platform="", **kwargs):
    if not isinstance(session_id, str) or not session_id:
        return None
    prompt = user_message
    if isinstance(prompt, list):
        prompt = "\n".join(part["text"] for part in prompt if isinstance(part, dict)
                           and part.get("type") == "text" and isinstance(part.get("text"), str))
    if not isinstance(prompt, str) or not prompt.strip():
        return None
    try:
        # Profile-aware lookup matches Hermes MCP's ${VAR} resolver. Never take
        # another hosted profile's ORMAH credentials from the process environment.
        from agent.secret_scope import get_secret
        allowed = {"PATH", "HOME", "USER", "LANG", "LC_ALL", "SYSTEMROOT", "WINDIR", "TEMP", "TMP",
                   "TMPDIR", "PYTHONPATH", "SSL_CERT_FILE", "SSL_CERT_DIR", "HTTP_PROXY", "HTTPS_PROXY",
                   "ALL_PROXY", "NO_PROXY"}
        env = {k: v for k, v in os.environ.items() if k in allowed or k.startswith("XDG_")}
        for name in ENV_KEYS:
            value = get_secret(name)
            if value is not None:
                env[name] = value
        run = subprocess.run([PYTHON, "-m", "ormah.integrations.hermes_bridge"],
                             input=json.dumps({"prompt": prompt, "session": f"{PROFILE}:{platform}:{session_id}",
                                               "workspace": WORKSPACE}),
                             env=env, capture_output=True, text=True, timeout=12, check=False)
        if run.returncode:
            return None
        text = json.loads(run.stdout).get("text")
        # Keep below Hermes' default per-hook spill threshold; never spill memory
        # to an extra host file just because the daemon returned a large string.
        return {"context": text[:9000]} if isinstance(text, str) and text.strip() else None
    except (OSError, ValueError, TypeError, ImportError, RuntimeError, subprocess.TimeoutExpired):
        return None


def register(ctx):
    ctx.register_hook("pre_llm_call", recall)
