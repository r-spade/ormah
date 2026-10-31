import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';

const output = mkdtempSync(join(tmpdir(), 'ormah-ui-tests-'));
try {
  const compile = spawnSync(process.execPath, [
    'node_modules/typescript/bin/tsc', 'src/lib/setup.ts', 'src/lib/bridge.ts',
    '--target', 'ES2020', '--module', 'ES2020', '--moduleResolution', 'bundler',
    '--strict', '--skipLibCheck', '--outDir', output,
  ], { stdio: 'inherit' });
  if (compile.status !== 0) process.exitCode = compile.status ?? 1;
  else {
    const tests = spawnSync(process.execPath, ['--test', 'tests/setup.test.mjs', 'tests/bridge.test.mjs'], {
      stdio: 'inherit', env: { ...process.env, ORMAH_UI_TEST_OUTPUT: output },
    });
    process.exitCode = tests.status ?? 1;
  }
} finally {
  rmSync(output, { recursive: true, force: true });
}
