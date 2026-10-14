# Google Antigravity, including CLI

```sh
ormah agents connect antigravity --project /absolute/workspace
ormah agents status antigravity --project /absolute/workspace
ormah agents disconnect antigravity --project /absolute/workspace
```

Setup writes owned entries into `.agents/mcp_config.json` and `.agents/hooks.json`.
Without `--project`, it uses `~/.gemini/config/` and both MCP and whisper use global
memory (or explicit `ORMAH_SPACE`/`ORMAH_WORKSPACE`). Project setup binds both to the
same absolute workspace; ambiguous multi-workspace or unmounted contexts skip
whisper. Mount the workspace in Antigravity before using its project integration.
Configuration alone is not a live connectivity check.

The CLI's native `PreInvocation` hook retrieves memories before each model call
and returns `injectSteps: [{ephemeralMessage: ...}]`. The prompt is read from the
provided transcript, not guessed from a Gemini CLI event. Only the latest
completed `USER_EXPLICIT` / `USER_INPUT` record's `USER_REQUEST` is used. Model
messages, injected context and metadata are excluded. Reads are limited to a 1 MiB
regular-file tail tied to the supplied conversation artifact directory. Missing,
partial, malformed or unfamiliar records fail open, as do daemon outages and
bounded HTTP timeouts. No transcript or memory text is copied to a persistent
Ormah cache. Retrieval runs per invocation because the injected context is
transient, including subsequent user turns and resumed conversations.

The existing MCP tools supply deliberate recall, remember, feedback and the
two-call maintenance protocol. MCP server instructions and the installed
`ormah-instructions.md` describe their use without assuming a named custom agent.
Antigravity 1.2.9 lazy-loads the tool catalog and calls tools through
`call_mcp_tool`. Set daemon URL/auth/space variables in the host environment;
CLI inheritance was checked with the native binary. See [shared options](agent-integrations.md).
No token values, model selections, permission presets, approval wildcards or
workspace trust records are installed. Review hooks using `/hooks` in CLI or
Customizations → Hooks in the UI; host enablement and trust still apply. Existing
hooks and unrelated config/comments are preserved; disconnect retains edits you
make to owned entries. POSIX setup is supported; Windows shell quoting remains
unimplemented and setup refuses it explicitly.

## Evidence, validation and draft blocker

Checked **2026-09-23**. The official CLI release manifest selected **agy 1.2.9**;
the downloaded archive was SHA512-verified against that manifest. Its public
[repository/changelog](https://github.com/google-antigravity/antigravity-cli/blob/818089f390e240921bb597b7a22ce9c96cdf7fe6/CHANGELOG.md)
is pinned at `818089f390e240921bb597b7a22ce9c96cdf7fe6`. The proprietary CLI/desktop
hook executor's source is not published there.

- Google's [migration announcement](https://developers.googleblog.com/an-important-update-transitioning-gemini-cli-to-antigravity-cli/)
  distinguishes current Antigravity from Gemini CLI; this uses the Antigravity API.
- [Official hooks](https://antigravity.google/docs/hooks) document the named-hook
  config, camelCase metadata, invocation lifecycle and transient injection output
  for CLI, desktop 2.0 and IDE. The CLI actually supplies `transcript_full.jsonl`,
  while this page illustrates `transcript.jsonl`; both accepted names are tested.
- [MCP schema and locations](https://antigravity.google/docs/mcp),
  [permissions](https://antigravity.google/docs/permissions), and
  [CLI installation](https://antigravity.google/docs/cli/install).
- The official [Python SDK hook protocol](https://github.com/google-antigravity/antigravity-sdk-python/blob/7f19db07e7c6c5038102b45a8a7a5da7eecc8b11/google/antigravity/proto/hooks.proto)
  is a different SDK lifecycle. It is not used to invent the native file-hook
  payload or transcript schema. [Plugins](https://antigravity.google/docs/plugins)
  package the same customization hooks; [sidecars](https://antigravity.google/docs/sidecars)
  send messages asynchronously and do not supply a better pre-invocation prompt contract.

The opt-in test `ORMAH_TEST_AGY_BINARY=/path/to/agy pytest
 tests/test_integrations/test_antigravity.py` launches the **real Linux CLI** with
an isolated home, fake API credential, mock Gemini endpoint and scratch Ormah
HTTP endpoint. It verifies injected memory in the actual outgoing model request,
MCP tool discovery, HTTP auth, project space, and a resumed conversation's latest
prompt/session identity. No Google account, paid model, real memory store or
personal host profile is used. Normal contract tests cover payloads captured from
that run, fail-open parsing, offline behavior and ownership.

**Draft blocker:** desktop 2.0 and IDE have the documented injection contract,
but their proprietary transcript row format has not been captured/verified.
They are reported as `unverified`; only CLI automatic whisper is validated.
Before promoting this integration as covering all Antigravity surfaces, capture
sanitized native desktop/IDE prompt+transcript fixtures, verify injection in each
surface's outgoing context, and adjust the strict parser if necessary. A shared
harness claim is not a substitute for that evidence. No minimum desktop version
or paid GUI validation is claimed. Windows support is also outside this revision.
