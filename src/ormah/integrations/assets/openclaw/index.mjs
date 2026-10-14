// OpenClaw 2026.9.5 before_prompt_build contract; no replacement memory slot.
import { execFile } from 'node:child_process';
import { realpathSync } from 'node:fs';
const python = __ORMAH_PYTHON__;
const workspace = __ORMAH_WORKSPACE__;

export default {
  id: 'ormah', name: 'Ormah',
  register(api) {
    api.on('before_prompt_build', async (event, ctx) => {
      try {
        if (!ctx.workspaceDir || realpathSync(ctx.workspaceDir) !== realpathSync(workspace)) return;
        const session = ctx.sessionId || ctx.sessionKey;
        const prompt = event.currentUserMessage ?? event.prompt;
        if (!session || typeof prompt !== 'string' || !prompt.trim()) return;
        ctx.hookInvocation?.assertActive();
        const text = await new Promise((resolve) => {
          const child = execFile(python, ['-m', 'ormah.integrations.openclaw_bridge'], {
            timeout: 12000, maxBuffer: 1024 * 1024, windowsHide: true,
          }, (error, stdout) => {
            if (error) return resolve('');
            try {
              const data = JSON.parse(stdout);
              resolve(typeof data.text === 'string' ? data.text : '');
            } catch { resolve(''); }
          });
          child.stdin.on('error', () => {});
          child.stdin.end(JSON.stringify({prompt, session: (ctx.agentId || '') + ':' + session, workspace}));
        });
        ctx.hookInvocation?.assertActive();
        if (text) return { prependContext: text };
      } catch { /* stale invocation, invalid workspace, daemon/child failure: fail open */ }
    });
  },
};
