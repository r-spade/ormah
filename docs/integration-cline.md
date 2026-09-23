# Cline (VS Code)

```sh
ormah agents connect cline --project /absolute/workspace
ormah agents status cline
ormah agents disconnect cline
```

This integration targets the VS Code extension, with one selected workspace per
Cline configuration. Setup installs an executable `.clinerules/hooks/UserPromptSubmit`
(or `.ps1` on Windows), `.clinerules/ormah.md` guidance, and a workspace-bound
stdio MCP entry. The MCP configuration is global:
`~/.cline/data/settings/cline_mcp_settings.json`. `CLINE_DIR`, `CLINE_DATA_DIR`,
and `CLINE_MCP_SETTINGS_PATH` follow the current host's overrides. Existing legacy
profiles should first use Cline's own migration; this installer does not rewrite
old VS Code globalStorage files. Status/disconnect use the user ownership receipt
even when setup receives `--project`.

Enable **Hooks** in Cline settings, review the installed hook, and reload the
extension if discovery was already cached. Setup leaves hook enablement, tool
approval, models and sandbox settings under your control. Existing event scripts
are a collision, not overwritten or wrapped. Convert a single `.clinerules` file
to a directory yourself before installing. Rules toggles can disable guidance;
the hook remains independent of whether the model follows those instructions.

The hook reads `hookName`, `taskId`, `workspaceRoots`, and
`userPromptSubmit.prompt`. It returns `cancel: false` with `contextModification`.
Cline's VS Code adapter maps that to `beforeRun.appendContext`; the SDK runtime
flushes it into the first model request as a hook-context message. Ormah uses its
existing whisper endpoint, with a host/workspace/task namespace and bounded,
fail-open retrieval. Ambiguous multi-root windows and mismatched workspaces skip
retrieval. Use a single-root window for this integration. The global MCP default
continues to refer to the selected workspace in other windows; use explicit
spaces there or disconnect/rebind. No cross-window workspace inference is made.

For custom daemon URL, port, auth or space, set `ORMAH_*` **before setup and in the
VS Code extension-host environment**. Cline's MCP transport filters inherited
environment; setup writes `${env:NAME}` references only for variables present at
installation and never writes their secret values. Reconnect when adding a new
variable. Hooks inherit the extension environment. See [shared options and
maintenance guidance](agent-integrations.md). Recall, remember, feedback and
maintenance use the existing MCP tools; no named custom agent is assumed.

## Evidence and limitations

Verified against source snapshot **Cline 4.1.20**, commit
`9c0e4aaee09f6593eb8d06ec4a35bf19b7dc33f1`, on **2026-09-23**. The package declares
VS Code `^1.101.0`. This is the inspected current source contract, not a claim
that all older marketplace builds use it.

- [Official MCP guide](https://docs.cline.bot/mcp/mcp-overview) and
  [current plugin scope](https://docs.cline.bot/customization/plugins).
- [HookFactory input/output, discovery, timeout and executable rules](https://github.com/cline/cline/blob/9c0e4aaee09f6593eb8d06ec4a35bf19b7dc33f1/apps/vscode/src/core/hooks/hook-factory.ts)
  and [fixture](https://github.com/cline/cline/blob/9c0e4aaee09f6593eb8d06ec4a35bf19b7dc33f1/apps/vscode/src/core/hooks/__tests__/fixtures/hooks/userpromptsubmit/context-injection/UserPromptSubmit).
- [Active VS Code adapter](https://github.com/cline/cline/blob/9c0e4aaee09f6593eb8d06ec4a35bf19b7dc33f1/apps/vscode/src/sdk/hooks-adapter.ts),
  [session config wiring](https://github.com/cline/cline/blob/9c0e4aaee09f6593eb8d06ec4a35bf19b7dc33f1/apps/vscode/src/sdk/sdk-session-config-builder.ts), and
  [actual model-context consumer](https://github.com/cline/cline/blob/9c0e4aaee09f6593eb8d06ec4a35bf19b7dc33f1/sdk/packages/agents/src/agent-runtime.ts).
- [Current MCP path wiring](https://github.com/cline/cline/blob/9c0e4aaee09f6593eb8d06ec4a35bf19b7dc33f1/apps/vscode/src/sdk/SdkController.ts),
  [migration/overrides](https://github.com/cline/cline/blob/9c0e4aaee09f6593eb8d06ec4a35bf19b7dc33f1/apps/vscode/src/hosts/vscode/mcp-settings-legacy-migration.ts),
  [stdio environment](https://github.com/cline/cline/blob/9c0e4aaee09f6593eb8d06ec4a35bf19b7dc33f1/apps/vscode/src/services/mcp/McpHub.ts), and
  [environment expansion](https://github.com/cline/cline/blob/9c0e4aaee09f6593eb8d06ec4a35bf19b7dc33f1/apps/vscode/src/utils/envExpansion.ts).

The CLI's `prompt_submit` file hook has **no context return channel** in
[its source](https://github.com/cline/cline/blob/9c0e4aaee09f6593eb8d06ec4a35bf19b7dc33f1/sdk/packages/core/src/hooks/hook-file-hooks.ts).
Its separate SDK plugin API can append context, but plugins are currently
unavailable in VS Code according to the official plugin guide. Neither CLI nor
JetBrains automatic whisper is advertised by this implementation.

Tests use the pinned nested hook fixture, exercise installed POSIX script
execution and mocked HTTP context responses, and cover offline behavior,
multi-root isolation, environment references, collisions and cleanup. Shared
checks cover deadlines/cancellation and explicit-null memories. No live VS Code
model session or Windows hook execution was run (`verified_live=false`). No
host profile or running service is changed by tests. Assets ship in wheel/sdist.
