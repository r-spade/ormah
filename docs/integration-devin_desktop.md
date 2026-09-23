# Devin Desktop / Windsurf

```sh
# Current Devin Local: workspace-bound tools and automatic whisper
ormah agents connect devin_desktop --project /absolute/workspace
ormah agents status devin_desktop --project /absolute/workspace
ormah agents disconnect devin_desktop --project /absolute/workspace

# User scope: Local whisper; Local and current Cascade share MCP
ormah agents connect devin_desktop
```

This integration distinguishes the current **Devin Local** agent (also used by
Devin CLI) from **legacy Cascade**. New Desktop conversations use Local; existing
Cascade conversations remain available. The target remains a **draft** because
Cascade has no documented per-prompt context injection return channel.

| Surface | Deliberate tools | Automatic whisper |
| --- | --- | --- |
| Devin Local in Desktop / CLI | MCP | `UserPromptSubmit` → `hookSpecificOutput.additionalContext` |
| Current Desktop's Cascade | Shared user MCP | Blocked: `pre_user_prompt` cannot return context |
| Older Windsurf installations | Not configured in old paths | Not supported |

Setup uses dedicated `mcp_config.json` (introduced in CLI v3000.3 / Local 3.6),
not the obsolete `mcpServers` key in `config.json`. User files are under
`$XDG_CONFIG_HOME/devin` or `~/.config/devin`. Project files are under `.devin`.
Local hooks are appended to project `hooks.v1.json`, or user `config.json`'s
`hooks` key. Cascade's differently formatted `hooks.json` is untouched.
Project setup does not configure Cascade's global MCP or affect other workspaces.
Windows setup is explicitly unavailable until its command-shell behavior is
validated; no partial installation occurs.

Project MCP uses the explicit selected directory. Its hook checks the documented
`DEVIN_PROJECT_DIR` against that directory before retrieval. User-level MCP lacks
a documented workspace placeholder, so **both** tools and whisper default to
unscoped/global memory at user scope. Set `ORMAH_SPACE` or `ORMAH_WORKSPACE` in
the launching environment when selecting a global installation's default, or
prefer project setup. Remembering with explicit `space: null` stays global.
Session identifiers include the host, workspace and native session; per-turn
`prompt_id` is deliberately not used as the session key.

Set `ORMAH_URL`, `ORMAH_PORT`, `ORMAH_AUTH_TOKEN`, `ORMAH_SPACE`,
`ORMAH_WORKSPACE` and `ORMAH_WHISPER_TIMEOUT` before setup and in the host's
launch environment. MCP stores `${env:NAME}` references only for variables set
at setup. Reconnect when adding a variable. The hook inherits the environment.
The shared runtime bounds HTTP retrieval and fails open on invalid/offline
responses. The host command has a 12-second deadline. Empty/malformed input,
wrong workspaces and Cascade events produce `{}` with exit zero.

Review the hook and reload the host. Workspace trust, Restricted Mode,
organization policies, model selection and tool approvals are unchanged. MCP
tool permission prompts still apply. Server instructions and installed
`ormah-instructions.md` explain deliberate recall/save/feedback and the existing
two-call maintenance protocol without assuming a named custom agent exists.
Setup/disconnect preserve unrelated JSONC comments, hooks and servers; edited
Ormah artifacts are retained and reported. Status describes configuration, not
an authenticated live connection.

## Evidence and remaining blocker

Official documentation inspected **2026-09-23**, against the
[CLI stable v3000.11.1 release dated September 21](https://docs.devin.ai/cli/changelog/stable):

- [Devin Local's shared CLI harness and Desktop trust behavior](https://docs.devin.ai/desktop/devin-local).
- [Local hooks, exact input/output and `DEVIN_PROJECT_DIR`](https://docs.devin.ai/cli/extensibility/hooks/overview)
  and [UserPromptSubmit lifecycle](https://docs.devin.ai/cli/extensibility/hooks/lifecycle-hooks).
- [Current MCP path migration, schema and environment references](https://docs.devin.ai/cli/extensibility/mcp/configuration),
  [JSONC configuration](https://docs.devin.ai/cli/reference/configuration/config-file),
  and [Cascade's current shared user MCP path](https://docs.devin.ai/desktop/cascade/mcp).
- [Cascade hooks](https://docs.devin.ai/desktop/cascade/hooks): `pre_user_prompt`
  has `agent_action_name` and nested `tool_info.user_prompt`, not Local's schema.
  Exit codes allow/block the action; `show_output` is a UI facility and does not
  apply to this event. Post-response hooks are too late. Intentionally returning
  a blocking error containing memories would disrupt the user and is not whisper.
- [Custom ACP agents](https://docs.devin.ai/desktop/acp-custom) can receive prompts,
  but introduce a separate agent; they cannot inject into Cascade's harness.
  [Cascade rules](https://docs.devin.ai/desktop/cascade/memories) provide static
  instructions, not a documented atomic per-prompt external context provider.
  Updating a rule file from a hook would introduce undefined reload timing and
  cross-session races. No supported Cascade context-provider extension API or
  public host implementation/types were located; no private API is assumed.

The remaining work is a supported Cascade pre-model injection API plus a native
trace confirming the output enters its model request. Local's route is documented
and implemented; a live Desktop/CLI trace has not been run (`verified_live=false`).
Tests exercise official payload fixtures against a mocked daemon, workspace and
session separation, offline behavior, the generated command, idempotent setup,
comment/approval preservation, corrupt config and user-edited cleanup. These are
synthetic contract tests, not proprietary host execution. Python runtime assets
ship in wheels and sdists. Legacy Windsurf paths are detected but never migrated
or overwritten; use the host's own migration before connecting.
