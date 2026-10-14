# Kiro IDE and CLI V3

```sh
ormah agents connect kiro --project /absolute/workspace
ormah agents status kiro --project /absolute/workspace
ormah agents disconnect kiro --project /absolute/workspace
# Omit --project for a user-wide installation with a global memory default.
```

This installs MCP in `.kiro/settings/mcp.json`, a standalone
`.kiro/hooks/ormah.json`, and `.kiro/steering/ormah.md`. At user scope these live
under `~/.kiro`. The hook uses the current `version: "v1"`, `hooks` array,
`trigger: "UserPromptSubmit"`, `action.type: "command"` format, with a 12-second
host timeout. No model, agent selection, tool approval or permission file changes.

Use the current IDE (at least **1.0.293**) or the **CLI V3 engine**. The inspected
stable CLI launcher is **2.23.1**: V3 remains opt-in with `kiro-cli --v3`.
The default V2/legacy engines and embedded agent hook configuration are not
configured. CLI builds before 2.14.2 and IDE builds before 1.0.293 had known
empty-prompt / lost-stdout bugs. Cloud/Web sessions cannot reach a local daemon
through this installation and are outside scope. Setup currently supports POSIX;
Windows command-shell integration is not installed.

The hook reads `hook_event_name`, `prompt`, `session_id` and `cwd` from stdin.
The shipped agent emits PascalCase `UserPromptSubmit`; the CLI documentation's
camelCase alias is accepted too. A successful hook prints **plain text**, which
Kiro adds to model context. It does not emit a JSON context envelope. The current
shared host adapter wraps successful stdout in a hook instruction message and
adds it to the conversation before generation. Empty prompts, missing session
identity, wrong workspaces and invalid/offline daemon responses emit no text
and exit zero. No transcript scraping, environment-prompt fallback or cached
context can accidentally retrieve for an earlier turn.

Project setup binds MCP and whisper to the selected directory; hook `cwd` must
match it. Global setup deliberately leaves **both** memory defaults global,
because the IDE MCP process cwd need not be the user's workspace. Set explicit
`ORMAH_SPACE` or `ORMAH_WORKSPACE` if needed, or prefer project setup. Session keys
include Kiro, the selected workspace and native session. Explicit `space: null`
remains global for deliberate saves.

Set `ORMAH_URL`, `ORMAH_PORT`, `ORMAH_AUTH_TOKEN`, `ORMAH_SPACE`,
`ORMAH_WORKSPACE` or `ORMAH_WHISPER_TIMEOUT` before setup and in Kiro's launch
environment. Setup records `${NAME}` MCP references, never secret values, only
for variables set at setup; reconnect when adding a variable. The hook inherits
the host environment. In the IDE, review the **Mcp Approved Env Vars** prompt
for referenced variables. Ormah does not modify that approval list.

Review the hook and reload the session. Workspace trust and hook execution
approval remain with Kiro. MCP must be enabled, and custom agents must include
workspace/user MCP (`includeMcpJson`); this installer does not rewrite custom
agent choices. Higher-priority config or host policy can disable tools/hooks.
Status reports owned configuration, not a live authenticated connection.
Disconnect preserves user-edited Ormah files, unrelated hooks, comments,
servers, models and approvals; malformed config is rejected.

The steering file and MCP server instructions explain deliberate recall, saving,
feedback and two-call maintenance. No unsupported named maintenance agent is
requested. The shared runtime uses bounded HTTP retrieval and portable maintenance
instructions; it does not replace Ormah's engine.

## Evidence and validation

Inspected **2026-09-23**:

- [Current hooks file schema](https://kiro.dev/docs/hooks/),
  [event fields](https://kiro.dev/docs/hooks/types/), and
  [stdout behavior](https://kiro.dev/docs/hooks/actions/). Some CLI tab examples
  retain legacy timeout fields; the current standalone file uses `timeout` in
  seconds, confirmed by the shipped parser/executor.
- [Unified V3 harness and opt-in launch](https://kiro.dev/docs/cli/v3/),
  [global/project scopes](https://kiro.dev/docs/configuration/), and
  [MCP schema and environment approvals](https://kiro.dev/docs/mcp/configuration/).
- Maintainer fixes for [CLI prompt data](https://github.com/kirodotdev/Kiro/issues/10308)
  and [IDE prompt/stdout](https://github.com/kirodotdev/Kiro/issues/10346), linked to
  [IDE 1.0.293, August 11, 2026](https://kiro.dev/changelog/ide/1-0-293/).
- [Official 2.23.1 Linux archive](https://prod.download.cli.kiro.dev/stable/2.23.1/kirocli-x86_64-linux.zip),
  SHA256 `7b8689de625c123e9b5357265925ebd1d81be6e6d04fa81b25a85c2fce15f9a3`,
  verified against the official manifest. `BUILD-INFO`: source build hash
  `c1c98028a070a5cab7db1645d9b4f6c0cdc0e830`, built 2026-09-22.
  Its embedded `@kiro/agent` **0.66.8** contains `dist/server/acp-server.js`
  (SHA256 `c1bc24cad8f4700013dd701f6d75e92eff82b53bfc458d80f3a14ba5bc4181b3`).
  Inspected the real stdin builder, hook-file parser, timeout conversion,
  process environment/cancellation, trust gate, and context consumer. In this
  bundle, `pvr` builds stdin; `hvr`/`ZZa` select stdout; `zeo.executeHookAction`
  adds a human context message; its caller `UOi` supplies the clean user prompt.
  The bundled `Fvr` helper also builds an effective prompt but is not the active
  adapter call path; tests cover the active adapter. The public Kiro repository
  contains docs/issues rather than this implementation, so archive hashes pin it.

Tests execute the generated Python hook against a scratch HTTP server, check
plain stdout, session/space separation, offline/malformed responses, setup
idempotency, preservation and disconnect. An opt-in test with
`ORMAH_TEST_KIRO_BUNDLE=/path/to/acp-server.js` executes the actual pinned
stdin/output/adapter functions with stubbed context storage and executor,
including the untrusted-workspace gate. Shared tests cover timeout/cancellation.
These are component/contract tests. CLI help ran in an isolated scratch profile;
an attempted V3 session required authentication, so no paid model or GUI session
was tested (`verified_live=false`). Runtime and steering assets ship in wheel/sdist.
