# Additional agent integrations

Host packages extend `ormah agents list|connect|disconnect|status`. Connecting
only configures the requested host; it does not start a daemon or load models.
Use the host's integration guide for supported versions, configuration scope,
and trust controls. Restart/reload the host after configuration changes.

Connection status describes configuration on disk, not a live host handshake.
The additive `capabilities` object on `/agent/clients` distinguishes deliberate
MCP tools from `native_hook`, `native_extension`, and `blocked` whisper support.
The existing `wired` field remains compatible. Legacy agents retain their
existing behavior and response fields.

The packaged MCP launcher reuses Ormah's existing tool schemas and dispatcher,
including the two-call maintenance protocol. Hosts must pass an actual workspace
(or set `ORMAH_SPACE`); an unbound IDE MCP connection uses global scope instead
of guessing from the IDE's process directory. Explicit `space: null` bypasses
the project default. MCP sessions use host-prefixed IDs distinct from chat
sessions; feedback should use the whisper's explicit `whisper_log_id`.

Shared runtime environment variables:

- `ORMAH_URL`: daemon base URL (default `http://127.0.0.1:8787`).
- `ORMAH_PORT`: alternate default port when `ORMAH_URL` is absent.
- `ORMAH_AUTH_TOKEN`: optional Bearer token, inherited at runtime, never written
  into generated host files.
- `ORMAH_SPACE`: explicit project-space override for tools and whisper.
- `ORMAH_WORKSPACE`: explicit workspace for otherwise unbound MCP/hook defaults.
- `ORMAH_WHISPER_TIMEOUT`: total hook retrieval budget in seconds, default 2,
  clamped to 0.1–10. Host hooks may impose a tighter outer deadline.

Whisper failures return no memory context. Requests are bounded and do not retry
or follow redirects. Cancellation propagates. Session keys include host,
workspace, and the host-provided conversation/session identifier.

Setup preserves unrelated JSON/JSONC values and comments. It refuses malformed
configuration and collisions. Ownership receipts live in
`$XDG_CONFIG_HOME/ormah/integrations` (default `~/.config/ormah/integrations`), or
`<project>/.ormah/integrations` for project setup. Keep project receipts local;
they contain installation-specific paths. Disconnect only removes exact owned
values/files; user edits are retained and reported. Empty parent objects may
remain. Reconnect after disconnect to change the Python installation path.

Runtime assets are inside `src/ormah/integrations` and ship in both wheels and
source distributions; a source checkout is not needed after installation.

## Host guides and scope

These host modules are delivered in separate integration PRs. Availability in
`ormah agents list` depends on which modules are installed. Each guide records
host versions, exact lifecycle evidence, setup scope and validation limits.

| Host guide | Automatic context route / limitation | PR |
| --- | --- | --- |
| [Cursor](integration-cursor.md) | MCP only; per-prompt injection blocked | [#310](https://github.com/r-spade/ormah/pull/310), draft |
| [OpenCode](integration-opencode.md) | Native `chat.message` plugin | [#311](https://github.com/r-spade/ormah/pull/311) |
| [Copilot](integration-github-copilot.md) | VS Code Local `UserPromptSubmit`; CLI excluded | [#312](https://github.com/r-spade/ormah/pull/312) |
| [OpenClaw](integration-openclaw.md) | `before_prompt_build` plugin, one selected workspace | [#313](https://github.com/r-spade/ormah/pull/313) |
| [Cline](integration-cline.md) | VS Code `UserPromptSubmit`, one selected workspace | [#314](https://github.com/r-spade/ormah/pull/314) |
| [Antigravity](integration-antigravity.md) | CLI `PreInvocation`; desktop transcript validation pending | [#315](https://github.com/r-spade/ormah/pull/315), draft |
| [Kilo](integration-kilo.md) | Current shared CLI/VS Code `chat.message` plugin | [#316](https://github.com/r-spade/ormah/pull/316) |
| [Hermes](integration-hermes.md) | Profile-bound `pre_llm_call` plugin | [#317](https://github.com/r-spade/ormah/pull/317) |
| [Devin Desktop](integration-devin_desktop.md) | Local `UserPromptSubmit`; Cascade MCP only | [#318](https://github.com/r-spade/ormah/pull/318), draft |
| [Kiro](integration-kiro.md) | Current IDE / CLI V3 `UserPromptSubmit` | [#319](https://github.com/r-spade/ormah/pull/319) |

These configuration routes do not imply live authenticated host validation.
The guides distinguish native binary/component execution from synthetic tests.
