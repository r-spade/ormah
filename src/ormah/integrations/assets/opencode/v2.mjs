// Contract: @opencode/plugin 2.2.1 SessionPrompt hook.
import { execFile } from 'node:child_process';

const python = __ORMAH_PYTHON__;
const nonempty = (value) => typeof value === 'string' && value.trim().length > 0;

function retrieve(payload) {
  return new Promise((resolve) => {
    const child = execFile(python, ['-m', 'ormah.integrations.opencode_bridge'], {
      timeout: 12000, maxBuffer: 1024 * 1024, windowsHide: true,
    }, (error, stdout) => {
      if (error) return resolve('');
      try {
        const data = JSON.parse(stdout);
        resolve(typeof data?.text === 'string' ? data.text : '');
      } catch { resolve(''); }
    });
    child.stdin.on('error', () => {});
    child.stdin.end(JSON.stringify(payload));
  });
}

async function sessionLocation(ctx, sessionID) {
  const controller = new AbortController();
  let timer;
  try {
    const session = await Promise.race([
      ctx.session.get({ sessionID }, { signal: controller.signal }),
      new Promise((resolve) => {
        timer = setTimeout(() => { controller.abort(); resolve(undefined); }, 2000);
      }),
    ]);
    return session?.location?.directory;
  } finally {
    clearTimeout(timer);
  }
}

// Plugin.define is an identity helper; a plain definition needs no SDK dependency.
export default {
  id: 'ormah',
  async setup(ctx) {
    await ctx.session.hook('prompt', async (event) => {
      try {
        const prompt = event?.prompt?.text;
        if (!nonempty(event?.sessionID) || !nonempty(prompt)) return;
        if (prompt.includes('\n\n<ormah-memory>\n') && prompt.endsWith('\n</ormah-memory>')) return;
        const workspace = await sessionLocation(ctx, event.sessionID);
        if (!nonempty(workspace)) return;
        const text = await retrieve({ prompt, session: event.sessionID, workspace });
        if (!nonempty(text)) return;
        // Appending keeps attachment mention offsets and all admission metadata intact.
        event.prompt.text = prompt + '\n\n<ormah-memory>\n' + text + '\n</ormah-memory>';
      } catch {
        // Memory is optional; metadata or bridge failures must not reject the prompt.
      }
    });
  },
};
