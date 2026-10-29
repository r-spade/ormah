# GitHub Copilot — VS Code Local and Copilot modes

## Enable from the web app

1. Install VS Code and GitHub Copilot, and sign in to GitHub.
2. Keep Ormah running. Open its web app, choose **agents**, and find
   **GitHub Copilot (VS Code)**. Click **connect**.
3. Reload VS Code and open your project. Review and approve the workspace,
   Ormah MCP server, and hooks when prompted; hooks must be enabled.
4. Start a **Local** or **Copilot** chat. Ask: “Use Ormah to remember that this
   project's test launch day is Friday, then recall it.” Check that the chat
   actually calls Ormah's tools and retrieves the fact.

The web app configures the **default VS Code profile on the machine running
Ormah**, not necessarily the computer displaying the browser. Its connection
status reflects user-level setup. For project-only setup, use the commands
below instead. Do not configure both scopes for the same workspace. A connected
status confirms configuration; it does not certify a successful chat session.

Both session targets have been tested on macOS with VS Code 1.144.0 and Copilot
Chat 0.72.0. See the validation details below for other platform limitations.

## Enable for one project

Run these commands on the computer where VS Code is installed:

```sh
ormah agents connect github_copilot --project /absolute/workspace
ormah agents status github_copilot --project /absolute/workspace
ormah agents disconnect github_copilot --project /absolute/workspace
```

Setup installs separate prompt hooks for **Local** and the newer **Copilot**
session target in VS Code. Local uses `UserPromptSubmit`; the shared Copilot
runtime uses `userPromptTransformed` and must support that event. Choose the
session target using the Copilot/Local selector in a new chat. CLI and remote
hosts have not been validated and are not claimed as supported by this setup.

Project setup adds `.vscode/mcp.json`, `.github/hooks/ormah-vscode.json`, `.github/hooks/ormah-copilot.json`, and
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

The Local handler reads `prompt`, `session_id`, and `cwd` from `UserPromptSubmit`.
The shared-runtime handler reads `prompt`, `sessionId`, `cwd`, and
`transformedPrompt` from `userPromptTransformed`. It preserves the original
transformed prompt verbatim and appends memory in `modifiedTransformedPrompt`.
It never returns a permission or turn-handling decision.
Project setup pins the workspace for both MCP and the hook; user MCP uses VS
Code's `${workspaceFolder}` substitution and the hook's cwd is repository-relative
`.`. For multi-root ambiguity prefer project setup. Missing identity/cwd, another
harness's payload, an unavailable daemon, or invalid JSON produces `{}` with
exit code 0. Local success returns `hookSpecificOutput.additionalContext` for the Local
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
its answer with the original Local-only implementation. The new transformed
hook fixes that contract difference and passed the live test below. Hook
discovery alone is not proof of context injection. In a new chat, select **Local** using the **Copilot/Local session
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

### Shared-runtime extension

The installed Copilot0.72.0 SDK includes `userPromptTransformed` and consumes
`modifiedTransformedPrompt` before emitting the model-facing user message. See
the [official event contract](https://docs.github.com/en/copilot/reference/hooks-reference#userprompttransformed).

Both hook files can be discovered by the newer runtime. It ignores the legacy
`UserPromptSubmit` command output, so the transformed hook uses a separate
session namespace: a discarded legacy retrieval must not suppress the real
injection as already served. This can still cause an extra retrieval and a
legacy exposure record on a prompt; it does not append that legacy output to
the new runtime's prompt. Live tests must verify both event coexistence and the
actual model-facing result.

### Dual-mode live results — 2026-10-29

Tested runtime commit `961417d` on the same macOS / VS Code1.144.0 /
Copilot Chat0.72.0 installation:

- **New Copilot automatic whisper passed.** Native SDK events recorded a
  successful `userPromptTransformed` output; the following `user.message`
  contained that exact memory in `transformedContent`, preserving the original
  datetime and prompt. The assistant returned the fresh synthetic verification
  phrase with no tool requests.
- **New Copilot MCP save/recall passed.** Native `ormah-remember` and
  `ormah-recall` calls succeeded. The isolated daemon received both with
  `default_space=workspace`; the resulting memory had `space: workspace`.
- **Local coexistence passed after installing both hook files.** A fresh Local
  session ran one `UserPromptSubmit` hook, received the synthetic fact through
  `additionalContext`, and the user confirmed the correct answer. The Local
  MCP save/recall/feedback round-trip had already passed before the additional
  hook was introduced; the MCP configuration was unchanged.

The full suite passed **2,573 tests** (one deselected, one expected failure),
463 focused tests passed, Ruff passed, and GitHub test/lint/desktop checks
passed on the runtime commit. Independent review also passed.

These results cover both VS Code session targets on the tested macOS versions.
They do not establish Windows, remote/SSH, standalone CLI, user-profile setup,
or native GUI outage behavior.
