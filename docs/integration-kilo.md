# Kilo Code

```sh
ormah agents connect kilo --project /absolute/workspace
ormah agents status kilo --project /absolute/workspace
ormah agents disconnect kilo --project /absolute/workspace
```

The current Kilo CLI and VS Code extension share a backend with native plugins.
Setup adds a local Ormah plugin, MCP server and memory guidance to project
`kilo.jsonc` (or existing `kilo.json`). User scope uses
`$XDG_CONFIG_HOME/kilo`, defaulting to `~/.config/kilo`. Plugin assets live in
`.kilo/ormah` or the user config directory's `ormah` child; they ship inside the
Ormah wheel/sdist and need no npm or marketplace publication.

The module exports `{id: "ormah", server}` as required by the current Kilo API.
Its `chat.message` hook awaits automatic retrieval and appends a synthetic,
non-ignored text part **before the backend persists and sends the message**.
Only non-synthetic, non-ignored user text is used for retrieval. Message/session
IDs are attached to the injected part, and repeated handling of the same output
does not append it twice. Continuation prompts made entirely of synthetic parts
are skipped. The model does not need to request recall for whisper to occur.

MCP provides deliberate recall, remember, feedback and the existing two-call
maintenance protocol. The installed instructions describe appropriate saving and
maintenance without requiring a named custom agent. Both the plugin and MCP use
the backend's workspace directory; the MCP launcher explicitly sets its cwd to
that instance directory. A project installation binds MCP to the project root;
a global installation uses the instance cwd, never the extension-host cwd.
Session keys include host, workspace and native session ID. Explicit `space=null`
remains available for global memories. See [shared options](agent-integrations.md)
for daemon URL, auth, space and timeouts. The Kilo backend forwards `ORMAH_*` to
local MCP children; its own backend credentials are filtered by Kilo.

Reload the CLI/extension backend after connecting. Review the local plugin as
executable code. `KILO_PURE=1` disables external plugins; setup refuses that mode
and status reports whisper inactive when it sees the flag. Other host/managed
configuration can also override local entries. Setup preserves unrelated
plugins, JSONC comments, models, MCP entries and permission rules. It does not
change tool approvals or make untrusted workspace config trusted. Disconnect
removes only exact owned entries/assets, preserving your edits.

## Evidence and validation

Checked **2026-09-23**, Kilo CLI and VS Code **7.7.9**, source
`1ad0e23e78be8df9044998c84fe1ca793130e9df`; VS Code engine `^1.105.1`.
Older Cline-derived Kilo extension builds are not covered by this plugin contract.

- [Official plugin guide](https://kilo.ai/docs/automate/extending/plugins) documents
  both supported surfaces, descriptor exports, config scope, hook events and pure mode.
- [CLI configuration](https://kilo.ai/docs/code-with-ai/platforms/cli) and
  [MCP overview](https://kilo.ai/docs/automate/mcp/overview).
- [Hook types](https://github.com/Kilo-Org/kilocode/blob/1ad0e23e78be8df9044998c84fe1ca793130e9df/packages/plugin/src/index.ts),
  [loader](https://github.com/Kilo-Org/kilocode/blob/1ad0e23e78be8df9044998c84fe1ca793130e9df/packages/opencode/src/plugin/index.ts),
  [pre-model trigger](https://github.com/Kilo-Org/kilocode/blob/1ad0e23e78be8df9044998c84fe1ca793130e9df/packages/opencode/src/session/prompt.ts), and
  [model conversion](https://github.com/Kilo-Org/kilocode/blob/1ad0e23e78be8df9044998c84fe1ca793130e9df/packages/opencode/src/session/message-v2.ts)
  show that synthetic text reaches the model unless `ignored` is set.
- [VS Code backend launch](https://github.com/Kilo-Org/kilocode/blob/1ad0e23e78be8df9044998c84fe1ca793130e9df/packages/kilo-vscode/src/services/cli-backend/server-manager.ts),
  [MCP cwd](https://github.com/Kilo-Org/kilocode/blob/1ad0e23e78be8df9044998c84fe1ca793130e9df/packages/opencode/src/mcp/index.ts), and
  [environment filtering](https://github.com/Kilo-Org/kilocode/blob/1ad0e23e78be8df9044998c84fe1ca793130e9df/packages/opencode/src/kilocode/process/env.ts).
- [Config loader](https://github.com/Kilo-Org/kilocode/blob/1ad0e23e78be8df9044998c84fe1ca793130e9df/packages/opencode/src/config/config.ts)
  distinguishes user-owned and project configuration trust.

Tests execute the generated descriptor/module with Node, the real Python bridge
and a scratch HTTP endpoint. They assert actual output-part injection, duplicate
suppression, continuation filtering and scoped request identities. Setup tests
cover both scopes, idempotency, preserved permissions/comments and pure mode.
Shared tests cover cancellation, bounded timeouts/offline behavior, explicit
null memories, corrupted configs and existing-agent regressions. The host hook
exposes no abort signal; the subprocess/HTTP deadline bounds late work.
No authenticated CLI/model or live VS Code GUI session was run
(`verified_live=false`); these are source-grounded contract tests.
