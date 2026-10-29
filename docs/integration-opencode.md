# OpenCode — V1 and V2

## Enable Ormah

Install OpenCode and connect a model provider. Keep Ormah running with
`ormah server start -d`. In the Ormah web app, open **agents → OpenCode → connect**
to configure the user running the Ormah server. This changes configuration on
the server machine, which may differ from the machine displaying the browser.
Restart OpenCode in your project after setup.

For project-only setup, run these commands on the machine running OpenCode:

```sh
ormah agents connect opencode --project /absolute/workspace
ormah agents status opencode --project /absolute/workspace
ormah agents disconnect opencode --project /absolute/workspace
```

Omit `--project` for user scope (`$XDG_CONFIG_HOME/opencode`, normally
`~/.config/opencode`). Setup updates an existing `opencode.jsonc` or
`opencode.json`, preserving comments, model choices, permissions, MCP servers,
plugins, and instructions. It adds one local plugin package, an instruction file, and
Ormah's local MCP entry. Use a single scope to avoid duplicate extensions.
Restart OpenCode after connecting; for V2, restart or reload its background service
as well. Ask: “Use Ormah to remember that this project's test launch day is
Friday, then recall it.” Verify that actual Ormah tool calls succeed.

The Agents panel reports user-level configuration; use the project-scoped status
command above for a project installation. Connection status verifies owned
configuration, not whether a live model session has succeeded.

The package supports both V1 and V2. V1 loads its `main` entry; V2 loads the
package's `exports["."]` entry. Fresh setup uses V1-compatible configuration that
V2 normalizes. Existing V2 `plugins` and `mcp.servers` keys are respected.
Setup does not migrate or overwrite personal settings. If OpenCode moves Ormah
entries between the V1 and native V2 keys, status and disconnect follow those
entries using the ownership receipt. Edited values remain protected. If you tried the earlier,
unreleased single-file plugin, disconnect and reconnect before upgrading.

Set `ORMAH_URL` in the OpenCode process environment for a custom daemon address;
otherwise normal Ormah port configuration applies. Secrets are not written into
the generated files. See the shared guide below for authentication and timeouts.

## How memory reaches the model

The V1 extension awaits retrieval in `chat.message` and appends a synthetic text
part before OpenCode saves the message and builds the model request. The host's
serializer includes non-ignored synthetic text, so this is automatic context
injection on real text prompts. Synthetic continuation messages and empty prompts
are skipped. Memories remain in the normal host conversation history and follow
its compaction lifecycle; nothing is cached across sessions by this plugin.

The V2 extension uses the native `prompt` admission hook and appends the same
memory block to the existing prompt text. Session metadata supplies the current
workspace. Original text, attachments, mention offsets, and other metadata stay
intact. Metadata lookup is bounded to two seconds and fails open. V2 stores the
appended text in the conversation, just as V1 retains its synthetic text part.
That context can also reach the title-generation model and can be sent again
when a saved message is edited or replayed.

Both MCP and whisper use the active workspace; OpenCode explicitly launches
local MCP processes from that directory. IDs separate host, workspace, and
session. `space: null` stays global. See [runtime environment and
ownership](agent-integrations.md) for URL, auth, space, and deadline settings.
The HTTP budget defaults to two seconds; a twelve-second child-process ceiling
covers the maximum configured budget and interpreter startup. Failures yield no
extra part. The bridge has no host cancellation signal: timeout bounds work
already in progress. No prompts, memories, or tokens are logged by the adapter.

Memory instructions cover deliberate saving/recall, event-specific feedback,
and the two-call maintenance protocol without assuming a named custom agent.
Plugins execute local code; review/trust the generated plugin as required by
your OpenCode deployment. Setup does not change any tool permission or policy.

## Contract evidence

Inspected 2026-09-23, OpenCode plugin **1.18.32**, source commit
`7cb044ee892fa8116610ba31a82922c656eaf86c`:

- [Plugin docs](https://opencode.ai/docs/plugins/) describe local plugin loading
  and the plugin interface; [MCP docs](https://opencode.ai/docs/mcp-servers/)
  define local command/environment configuration.
- [Hooks types](https://github.com/anomalyco/opencode/blob/7cb044ee892fa8116610ba31a82922c656eaf86c/packages/plugin/src/index.ts)
  define `chat.message(input, {message, parts})` and `PluginInput.directory`.
- [Prompt lifecycle](https://github.com/anomalyco/opencode/blob/7cb044ee892fa8116610ba31a82922c656eaf86c/packages/opencode/src/session/prompt.ts)
  awaits that hook before persisting the parts.
- [Model message conversion](https://github.com/anomalyco/opencode/blob/7cb044ee892fa8116610ba31a82922c656eaf86c/packages/opencode/src/session/message-v2.ts)
  passes non-ignored text parts to the model, including synthetic parts.
- [MCP transport](https://github.com/anomalyco/opencode/blob/7cb044ee892fa8116610ba31a82922c656eaf86c/packages/opencode/src/mcp/index.ts)
  specifies the workspace process cwd.

This is the checked contract, not a claim about every older/newer version.
Tests run the actual installed JavaScript/Python bridge against a local mock
HTTP daemon and assert the injected part, scope, and idempotency. Shared tests
cover timeouts, cancellation, malformed responses, offline behavior, and global
memories. Configuration status retains `verified_live: false`: it cannot certify a user's
model session. Runtime assets ship in the Python wheel/sdist and require no npm
publication.

### V2 contract

Checked against published `@opencode/plugin`, `@opencode/schema`, and
`@opencode/client` **2.2.1**:

- [V1 migration guide](https://opencode.ai/v2/docs/migrate-v1/) describes legacy
  config normalization and the incompatible plugin API.
- [V2 plugin loading](https://opencode.ai/v2/docs/plugins/) requires configured
  local package directories; the V2 loader resolves the package export.
- [Prompt admission hook](https://opencode.ai/v2/docs/build/plugins/server/#prompt-admission)
  permits changing prompt text before the model request.
- [V1 1.19.5 package loader](https://github.com/anomalyco/opencode/blob/v1.19.5/packages/opencode/src/plugin/shared.ts)
  resolves `main` when the package does not export `./server`.

### Live validation — 2026-10-29

On macOS, OpenCode **1.19.5** and **2.2.1** ran real model sessions with a free
OpenCode provider in an isolated project and Ormah store:

- V1 and V2 answered a synthetic verification phrase absent from the user prompt
  and workspace files, without calling tools. V1's saved transcript contains the
  synthetic memory part; V2's saved user message contains the appended context.
- Both versions successfully invoked MCP remember and recall. Stored facts had
  the expected project space; implicit feedback also succeeded.
- The shared package was rerun in V1 after adding V2 support.
- User-profile setup passed in both versions using a second isolated project;
  V2 used native `plugins` and `mcp.servers` configuration.
- Real process checks verified offline MCP/bridge behavior and CLI
  connect/idempotency/disconnect preservation of personal configuration,
  symlinks, and user-edited plugin files. These are not native GUI outage tests.

Full Python validation: **2,618 passed**, one expected failure, one deselected.
All **57 UI tests**, Ruff, production UI build, wheel/sdist builds, and package
content checks passed. An independent review found no blocking findings.

Tests cover configuration preservation, disconnect, package routing, native V2
configuration, metadata failures, empty/malformed responses, and unavailable
daemons. Windows, Linux, desktop GUI, and remote deployments have not been
live-validated by these macOS CLI sessions. Use one setup scope per workspace.
