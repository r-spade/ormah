# GitHub Copilot — VS Code Legacy Local harness

```sh
ormah agents connect github_copilot --project /absolute/workspace
ormah agents status github_copilot --project /absolute/workspace
ormah agents disconnect github_copilot --project /absolute/workspace
```

This integration targets the **Legacy Local** harness, not the newer shared
Copilot runtime.

Select **Local** as the VS Code Session Target. The newer **Copilot Agent Host**
and Copilot CLI have different hook contracts and are not covered by this
integration. MCP alone may be forwarded to other harnesses; that does not
establish automatic whisper there.

Project setup adds `.vscode/mcp.json`, `.github/hooks/ormah-vscode.json`, and
`.github/instructions/ormah.instructions.md`. User setup (without `--project`)
uses the default VS Code profile's `mcp.json` and `~/.copilot/hooks`. Set
`ORMAH_VSCODE_USER_DIR` to the directory opened by **MCP: Open User Configuration**
for a named, portable, Insiders, or remote profile. Detection recognizes the
Copilot Chat extension directory or its profile storage. Use one setup scope.

Start Ormah separately. Reload VS Code, inspect MCP tools, and review/trust the
workspace and generated hook. Local `chat.useHooks` must be enabled (the documented
default); policy or disabled hook locations can prevent execution. Setup preserves
those settings, model choices, approvals, and unrelated configuration. Windows
commands are quoted for the default PowerShell hook executor; Linux/macOS use
the host shell. The macOS Local path has live validation below; Windows execution
remains untested.

The handler reads `prompt`, `session_id`, and `cwd` from `UserPromptSubmit`.
Project setup pins the workspace for both MCP and the hook; user MCP uses VS
Code's `${workspaceFolder}` substitution and the hook's cwd is repository-relative
`.`. For multi-root ambiguity prefer project setup. Missing identity/cwd, another
harness's payload, an unavailable daemon, or invalid JSON produces `{}` with
exit code 0. Success returns `hookSpecificOutput.additionalContext` for the Local
model request, never an approval decision. Cancellation is handled by the host
terminating the hook process; retrieval also has a bounded deadline.

The daemon port follows Ormah's normal configuration (`~/.config/ormah/.env`,
then the process working directory's `.env`, then the process environment). An explicit
`ORMAH_URL` takes precedence.

Set custom `ORMAH_*` runtime variables before setup and in VS Code's launch
environment. MCP configuration records `${env:NAME}` references only for names
present during setup; no secret values are written. Disconnect/connect when
adding variables later. The hook inherits its host environment.

See [shared options](agent-integrations.md) for URL/auth/space/deadlines and
ownership. MCP exposes deliberate recall, saving, feedback, and maintenance.
Instructions use the two-call maintenance protocol without an assumed custom
agent. The hook keeps no disk transcript/cache. Host diagnostics may capture
hook input/output; their logging policy is controlled by VS Code.

## Evidence and validation

Checked 2026-09-23 against Copilot Chat **0.44.0**, VS Code engine **^1.115.0**,
source `5863f5a7088958050792b5dccbe8b46c6e13eccc` (development snapshot; not a claim
that every installed release provides this contract):

- [Official hooks guide](https://code.visualstudio.com/docs/agent-customization/hooks)
  (updated 2026-09-16) distinguishes harnesses, user/project discovery and trust.
- [Local hook reference](https://code.visualstudio.com/docs/agents/reference/hooks-reference)
  defines command configuration and common input. Its UserPromptSubmit section
  omits some output details; the following official types **and consumer** confirm
  the nested context field.
- [Input/output types](https://github.com/microsoft/vscode-copilot-chat/blob/5863f5a7088958050792b5dccbe8b46c6e13eccc/src/platform/chat/common/chatHookService.ts)
  define `UserPromptSubmitHookOutput.hookSpecificOutput.additionalContext`.
- [Request handler](https://github.com/microsoft/vscode-copilot-chat/blob/5863f5a7088958050792b5dccbe8b46c6e13eccc/src/extension/prompt/node/defaultIntentRequestHandler.ts)
  awaits the hook and calls `loop.appendAdditionalHookContext` before `loop.run`.
- [Hook executor](https://github.com/microsoft/vscode-copilot-chat/blob/5863f5a7088958050792b5dccbe8b46c6e13eccc/src/platform/chat/node/hookExecutor.ts)
  enforces timeout/cancellation and selects PowerShell on standard Windows hosts.
- [MCP documentation](https://code.visualstudio.com/docs/agent-customization/mcp-servers)
  defines profile/workspace configuration and normal server trust controls.

Synthetic tests cover real documented payload/output shapes, workspace scope,
missing identity, other harnesses, offline behavior, ownership, idempotency, and
shared runtime/MCP regressions. All runtime assets ship in wheels and sdists.

### Live validation — 2026-10-29

On macOS, VS Code **1.144.0** with bundled Copilot Chat **0.72.0**, an
authenticated **Local** session retrieved a unique synthetic fact from an isolated
Ormah daemon. The fact was absent from workspace files and the user prompt.
Native hook logs recorded the actual `UserPromptSubmit` and returned
`hookSpecificOutput.additionalContext`; the chat transcript showed the correct
answer with an empty `toolRequests` list. This establishes automatic whisper in
that tested Local session.

The newer **Copilot** session target ran the hook but did not produce the fact in
its answer. It remains unsupported; hook discovery alone is not proof of context
injection. In a new chat, select **Local** using the **Copilot/Local session
selector**, not the dropdown next to the new-chat button. Reloading or starting a
new chat can choose the newer target, so check the target before testing.

Real subprocess checks also verified an unavailable daemon returns `{}` with
exit code 0 from the hook, and a clear unavailable-server message from MCP.
CLI connect/disconnect checks in disposable projects preserved personal MCP
entries, comments, symlinks, and user-edited hooks while removing untouched owned
artifacts. These are process-level checks, not native GUI outage tests.

Live MCP save/recall also passed in the same Local session. The transcript
recorded successful `remember` and `recall` calls, and the isolated daemon
confirmed both requests with `default_space=workspace`. The stored memory had
`space: workspace`; an implicit feedback call also succeeded.

Windows, remote profiles, and other installed versions have not been live-tested. `verified_live` remains false
in status because setup checks configuration integrity, not whether that user
has completed a successful live session.
