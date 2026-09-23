# Hermes Agent

```sh
# Optional: select a dedicated profile before all commands.
HERMES_HOME=/absolute/hermes-profile ormah agents connect hermes --project /absolute/workspace
HERMES_HOME=/absolute/hermes-profile ormah agents status hermes
HERMES_HOME=/absolute/hermes-profile ormah agents disconnect hermes
```

Setup installs a native `pre_llm_call` plugin under `$HERMES_HOME/plugins/ormah`
and adds its name to `plugins.enabled`. MCP lives in `mcp_servers.ormah` in that
profile's `config.yaml`. The default home is `~/.hermes` on POSIX and
`%LOCALAPPDATA%/hermes` on Windows. Select named profiles by their resolved
`HERMES_HOME`; this installer does not resolve Hermes CLI `-p` names. Ownership is
recorded in that profile's `.ormah/installation.json`, keeping profiles independent.

Here `--project` binds the **entire profile** to one default memory space; it does
not create a project-specific Hermes configuration. Without it, both tools and
whisper are global unless `ORMAH_SPACE` or `ORMAH_WORKSPACE` is explicitly set.
For profiles serving multiple repositories, use explicit spaces for deliberate
tools or separate profiles. No hosted process cwd is used to guess a workspace.

Hermes runs `pre_llm_call` once per user turn. The plugin extracts only text from
`user_message`, calls Ormah's existing whisper endpoint, and returns `context`.
Hermes appends that to the API-bound user message before the model call, preserving
the clean user text and system prompt. Multimodal turns receive an extra text
part. Session keys include the Hermes profile, platform, workspace and native
session. The plugin has no cross-session mutable state and does not install a
replacement memory provider or compression engine.

The plugin uses Ormah's own Python interpreter through a bounded subprocess;
Hermes's interpreter needs no new dependencies. MCP exposes the existing recall,
remember, feedback and two-call maintenance tools. MCP server instructions and
installed `instructions.md` provide deliberate memory guidance, without assuming
a named custom agent exists. The plugin truncates output below Hermes's default
spill threshold so it does not create an additional host file containing memory.

Set custom `ORMAH_URL`, `ORMAH_PORT`, `ORMAH_AUTH_TOKEN`, `ORMAH_SPACE`,
`ORMAH_WORKSPACE` or `ORMAH_WHISPER_TIMEOUT` **before setup and in the selected
Hermes profile's environment**. Setup writes variable-name references, never
secrets. Reconnect when adding a variable. Both the MCP resolver and plugin use
Hermes's active profile secret scope, avoiding a multiplexed gateway's launch
profile credentials. See [shared runtime options](agent-integrations.md).

Review the installed executable plugin and reload Hermes. This hook needs no
privileged plugin capability grant. Only Ormah is enabled; unrelated plugins,
comments, quoting, models, memory providers and tool approvals are preserved.
Aliased/merged mappings that would cause edits to affect other settings are
rejected explicitly. Corrupt YAML is never overwritten. `plugins.disabled`
and `HERMES_SAFE_MODE` remain authoritative. Disconnect keeps user-edited files.
Hermes suppresses this hook for its `_persist_disabled` internal tasks; those
special tasks are not advertised as receiving whisper.

## Evidence and validation

Inspected **Hermes 0.21.4**, commit
`358d50ca6dcb01b93ed226ce27f7f0de6d142111`, on **2026-09-23**. The plugin declares
`requires_hermes: ">=0.21.4"`; future compatibility still depends on the host API.

- [Native hook reference](https://hermes-agent.nousresearch.com/docs/user-guide/features/hooks#pre_llm_call)
  documents the exact payload, once-per-turn lifecycle, and string/multimodal context injection.
- [Plugin manifest/enablement/capability guide](https://hermes-agent.nousresearch.com/docs/developer-guide/plugins/)
  and [MCP configuration](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp/).
- [Actual pre-hook dispatch and API-bound context composition](https://github.com/NousResearch/hermes-agent/blob/358d50ca6dcb01b93ed226ce27f7f0de6d142111/agent/turn_context.py),
  [plugin discovery/denial](https://github.com/NousResearch/hermes-agent/blob/358d50ca6dcb01b93ed226ce27f7f0de6d142111/hermes_cli/plugins_discovery.py), and
  [manifest version gate](https://github.com/NousResearch/hermes-agent/blob/358d50ca6dcb01b93ed226ce27f7f0de6d142111/hermes_cli/plugins_manifest.py).
- [MCP environment substitution/filtering](https://github.com/NousResearch/hermes-agent/blob/358d50ca6dcb01b93ed226ce27f7f0de6d142111/tools/mcp_tool_config.py),
  [active profile secrets](https://github.com/NousResearch/hermes-agent/blob/358d50ca6dcb01b93ed226ce27f7f0de6d142111/agent/secret_scope.py), and
  [platform home resolution](https://github.com/NousResearch/hermes-agent/blob/358d50ca6dcb01b93ed226ce27f7f0de6d142111/hermes_constants.py).

Tests run the installed plugin through the real Ormah subprocess and a scratch
HTTP endpoint, checking multimodal filtering, profile-specific auth, session/space
identity, timeout, YAML comments/quotes, denial/corruption and cleanup. An opt-in
`ORMAH_TEST_HERMES_SOURCE=/path/to/pinned/source` test executes the host's actual
context-composition functions extracted by AST, without importing its full
runtime or stores. These are contract/component tests, not a live CLI/Gateway or
model session (`verified_live=false`). Shared checks cover offline retrieval,
cancellation and explicit-null memory behavior. Runtime assets ship in wheel/sdist.
