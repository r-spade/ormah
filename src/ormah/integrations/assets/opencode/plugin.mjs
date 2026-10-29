// Contract: @opencode-ai/plugin 1.18.32 Hooks['chat.message'].
import { execFile } from 'node:child_process';
import { randomUUID } from 'node:crypto';

const python = __ORMAH_PYTHON__;

function retrieve(payload) {
  return new Promise((resolve) => {
    const child = execFile(python, ['-m', 'ormah.integrations.opencode_bridge'], {
      timeout: 12000, maxBuffer: 1024 * 1024, windowsHide: true,
    }, (error, stdout) => {
      if (error) return resolve('');
      try {
        const data = JSON.parse(stdout);
        resolve(typeof data.text === 'string' ? data.text : '');
      } catch { resolve(''); }
    });
    child.stdin.on('error', () => {});
    child.stdin.end(JSON.stringify(payload));
  });
}

export default async function Ormah({ directory }) {
  return {
    'chat.message': async (input, output) => {
      if (!input.sessionID || !output.message?.id || !Array.isArray(output.parts)) return;
      if (output.parts.some((p) => p.type === 'text' && p.synthetic &&
          p.text.startsWith('<ormah-memory>'))) return;
      const prompt = output.parts.filter((p) => p.type === 'text' && !p.synthetic && !p.ignored)
        .map((p) => p.text).join('\n');
      if (!prompt.trim()) return;
      const text = await retrieve({ prompt, session: input.sessionID, workspace: directory });
      if (!text.trim()) return;
      output.parts.push({
        id: 'prt_' + randomUUID().replaceAll('-', ''),
        sessionID: input.sessionID, messageID: output.message.id,
        type: 'text', synthetic: true,
        text: '<ormah-memory>\n' + text + '\n</ormah-memory>',
      });
    },
  };
}
