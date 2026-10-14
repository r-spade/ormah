# OpenClaw

```sh
ormah agents connect openclaw --project /absolute/openclaw-workspace
ormah agents status openclaw
ormah agents disconnect openclaw
```

OpenClaw uses a Gateway-wide configuration. Here `--project` selects the workspace
to bind, not a separate project configuration file. Without it, setup uses
`agents.defaults.workspace` or the existing `~/.openclaw/workspace`. The same
workspace is passed to MCP and checked by the native whisper extension. Other
agent workspaces receive no automatic whisper. The Gateway-wide MCP entry retains
this selected default space; agents using it elsewhere must pass explicit spaces.
Use a separate OpenClaw profile/config for independently scoped workspaces.

Setup edits `~/.openclaw/openclaw.json` (or `OPENCLAW_CONFIG_PATH`), installs
`ormah-plugin` under `OPENCLAW_STATE_DIR` (default `~/.openclaw`), and records
ownership at user scope. JSON5 comments, unquoted keys, unrelated plugin settings,
models and memory slots are preserved. It grants only Ormah's required
`hooks.allowConversationAccess` and adds Ormah to an existing plugin allowlist.
An authored Ormah entry, denial, or globally disabled plugins causes a clear
error instead of overriding the user's choice. No tool auto-approval is added.

Review the local plugin and use OpenClaw's plugin inspection/reload controls.
Directly discovered `plugins.load.paths` plugins have a different consent model
from managed installations; setup does not forge managed-install acceptance.
`hooks.allowPromptInjection: false` blocks prompt hooks in OpenClaw. Existing
policy and operator trust remain authoritative. No running Gateway is modified
by tests or restarted by setup; users apply configuration on their own instance.

The native `before_prompt_build` handler uses `currentUserMessage` when supplied,
including respecting an explicitly empty message, and otherwise uses `prompt`.
It returns `prependContext`, which the embedded runner prepends to the actual
model prompt. Session identity includes host, workspace, agent and native
session. It checks `hookInvocation.assertActive` before/after retrieval, discarding
late results. Underlying HTTP work has its own deadline because that host
capability does not cancel I/O. No persistent prompt cache is created.

Deliberate tools use Ormah's existing MCP server and two-call maintenance
protocol. Instructions are advertised by MCP and included in the installed
package. See [shared options](agent-integrations.md). If configuring URL, port,
space or auth, set the relevant `ORMAH_*` variables **before setup and in the
Gateway environment**: MCP configuration references their names using OpenClaw's
`${VAR}` expansion, never their secret values. Disconnect/reconnect if adding a
new variable. The native hook inherits the same Gateway environment.

## Evidence and verification

Checked 2026-09-23 against OpenClaw **2026.9.5**, source
`4ed6847ae875f16c8d23572e1bf7e53ac5962760`. OpenClaw documents its plugin APIs as
experimental; pin the host and recheck upgrades. Its documented host requirement
is Node 24.16+ or 26.1+.

- [Building plugins](https://docs.openclaw.ai/plugins/building-plugins),
  [hook policy](https://docs.openclaw.ai/plugins/hooks), and
  [installation/trust](https://docs.openclaw.ai/plugins/manage-plugins).
- [Prompt lifecycle](https://docs.openclaw.ai/plugins/hooks/prompt-and-session)
  and [source input/output types](https://github.com/openclaw/openclaw/blob/4ed6847ae875f16c8d23572e1bf7e53ac5962760/src/plugins/hook-before-agent-start.types.ts).
- [Actual model prompt composition](https://github.com/openclaw/openclaw/blob/4ed6847ae875f16c8d23572e1bf7e53ac5962760/src/agents/embedded-agent-runner/run/attempt-prompt-build.ts)
  applies the returned context to `effectivePrompt`.
- [MCP configuration](https://docs.openclaw.ai/tools/mcp) and
  [stdio schema](https://github.com/openclaw/openclaw/blob/4ed6847ae875f16c8d23572e1bf7e53ac5962760/src/config/zod-schema.mcp-server.ts).
- [JSON5 configuration](https://docs.openclaw.ai/gateway/configuration).

Tests execute the generated JS plugin through Python against a scratch HTTP
daemon, including current-message precedence, empty messages, other workspaces,
and expired invocations. Setup tests verify native JSON5 preservation, auth-name
references, policy conflicts and idempotent cleanup. Shared tests cover offline,
timeout, explicit-global memories and existing-agent regressions. No live
OpenClaw Gateway/model session was run (`verified_live=false`); additional native
harnesses beyond the embedded runner are not claimed as validated. Assets ship in
the wheel/sdist, with no marketplace publication required.
