# Cursor (draft: deliberate MCP, whisper blocked)

```sh
ormah agents connect cursor --project /absolute/workspace
ormah agents status cursor --project /absolute/workspace
ormah agents disconnect cursor --project /absolute/workspace
```

Project setup installs `.cursor/mcp.json` and an always-applied
`.cursor/rules/ormah.mdc` guide. Tools are bound to that workspace even when
Cursor starts MCP elsewhere. Without `--project`, setup uses
`~/.cursor/mcp.json`; that connection is global unless `ORMAH_SPACE` is set.
Use one scope per workspace to avoid duplicate MCP connections. The Agents
panel manages user scope. Detection checks the launcher, user directory, and
macOS application bundle. Configuration is never a live connection assertion.

Start Ormah separately, then reload Cursor and inspect its MCP tools. Keep the
normal MCP approval policy; setup does not change it. The existing remember,
recall, feedback and two-call maintenance tools are available. The MCP server
also advertises portable memory guidance. No named maintenance agent is required.
See [shared runtime options and ownership](agent-integrations.md).

## Whisper investigation (2026-09-23)

This integration deliberately reports `whisper: blocked`. No hook is installed.
The official [hooks reference](https://cursor.com/docs/hooks) documents:

| Route | Result |
|---|---|
| `beforeSubmitPrompt` | Receives prompt and conversation identity; output permits submission control, not additional model context. |
| `sessionStart` | Can add initial context, but lacks the submitted prompt and does not run on every user turn. |
| `postToolUse` | Can add context after a tool result; a turn may make no tool call, and the first model request has already happened. |
| `workspaceOpen` | Registers plugin directories; it cannot inject prompt-specific memories. |

Project hooks require workspace trust. User hooks run from the Cursor config
directory, so process cwd is not reliable project identity. Plugins package
[the same hooks, rules, and tools](https://cursor.com/docs/plugins); they do not
provide an additional documented per-prompt context provider. The
[MCP extension API](https://cursor.com/docs/mcp) registers servers, not prompt
interceptors. Dynamically rewriting a rule file has no documented guarantee
that Cursor reloads it before the pending request, and risks cross-session
memory leakage. Static instructions and MCP recall are deliberate mechanisms.

Official source evidence: the [Cursor cookbook at
6733ef81](https://github.com/cursor/cookbook/blob/6733ef81a7dc3cb2a6c1f524ff586ebecc703204/hooks/README.md)
also describes prompt validation, not prompt rewriting. The proprietary agent
implementation is unavailable; no numeric minimum version is inferred from
examples. This draft targets the MCP configuration schema documented on
2026-09-23. It needs a supported pre-model context hook/provider and a live
Cursor context trace before automatic whisper can be declared ready.

Validation is synthetic: setup/disconnect, scope, idempotency, preservation,
corrupt config, capability status, plus shared MCP/global-memory tests. No paid
Cursor GUI or model session was run. Runtime code and guidance ship in wheels
and sdists; the project rule is generated from the installed package.
