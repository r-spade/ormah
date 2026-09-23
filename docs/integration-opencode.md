# OpenCode

```sh
ormah agents connect opencode --project /absolute/workspace
ormah agents status opencode --project /absolute/workspace
ormah agents disconnect opencode --project /absolute/workspace
```

Omit `--project` for user scope (`$XDG_CONFIG_HOME/opencode`, normally
`~/.config/opencode`). Setup updates an existing `opencode.jsonc` or
`opencode.json`, preserving comments, model choices, permissions, MCP servers,
plugins, and instructions. It adds one local plugin, an instruction file, and
Ormah's local MCP entry. Use a single scope to avoid duplicate extensions.
Reload OpenCode after connecting. Start the Ormah daemon separately.

The extension awaits retrieval in `chat.message` and appends a synthetic text
part before OpenCode saves the message and builds the model request. The host's
serializer includes non-ignored synthetic text, so this is automatic context
injection on real text prompts. Synthetic continuation messages and empty prompts
are skipped. Memories remain in the normal host conversation history and follow
its compaction lifecycle; nothing is cached across sessions by this plugin.

Both MCP and whisper use the active workspace; OpenCode explicitly launches
local MCP processes from that directory. IDs separate host, workspace, and
session. `space: null` stays global. See [runtime environment and
ownership](agent-integrations.md) for URL, auth, space, and deadline settings.
The HTTP budget defaults to two seconds; a twelve-second child-process ceiling
covers the maximum configured budget and interpreter startup. Failures yield no
extra part. The hook API exposes no cancellation signal: timeout bounds work
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
memories. No authenticated OpenCode model session has been run; configuration
status therefore retains `verified_live: false`. Runtime assets ship in the
Python wheel/sdist and require no npm publication.
