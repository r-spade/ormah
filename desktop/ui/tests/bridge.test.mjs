import { test } from 'node:test';
import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
const { onServerStatus, waitForServer, graphUrl, waitForGraphTransition } = await import(pathToFileURL(`${process.env.ORMAH_UI_TEST_OUTPUT}/bridge.js`));
const flush = () => new Promise(resolve => setImmediate(resolve));

test('status listener is installed before snapshot and events during snapshot survive', async () => {
  let emit;
  let resolveSnapshot;
  const seen = [];
  globalThis.window = { __TAURI__: {
    event: { listen: async (_name, callback) => { emit = callback; return () => {}; } },
    core: { invoke: async cmd => { assert.equal(cmd, 'runtime_setup_status'); return new Promise(resolve => { resolveSnapshot = resolve; }); } },
  } };
  const stop = onServerStatus(status => seen.push(status.sequence), error => assert.fail(error));
  await flush();
  emit({ payload: { sequence: 9 } });
  resolveSnapshot({ sequence: 8 });
  await flush();
  assert.deepEqual(seen, [9, 8]); // Model's monotonic guard rejects the stale snapshot.
  stop();
});

test('teardown before listen resolves immediately removes late listener and skips snapshot', async () => {
  let resolveListen;
  let removed = 0;
  let fetched = 0;
  globalThis.window = { __TAURI__: {
    event: { listen: () => new Promise(resolve => { resolveListen = resolve; }) },
    core: { invoke: async () => { fetched++; } },
  } };
  const stop = onServerStatus(() => assert.fail('late callback'), error => assert.fail(error));
  stop();
  resolveListen(() => { removed++; });
  await flush();
  assert.equal(removed, 1);
  assert.equal(fetched, 0);
});

test('subscription and snapshot failures are reported to the UI', async () => {
  for (const failListen of [true, false]) {
    const errors = [];
    globalThis.window = { __TAURI__: {
      event: { listen: async () => { if (failListen) throw Error('listen blocked'); return () => {}; } },
      core: { invoke: async () => { throw Error('snapshot blocked'); } },
    } };
    const stop = onServerStatus(() => {}, error => errors.push(error));
    await flush();
    assert.match(errors.join(' '), failListen ? /listen blocked/ : /snapshot blocked/);
    stop();
  }
});

test('health polling survives failures until late readiness and honors cancellation', async () => {
  let checks = 0;
  const errors = [];
  globalThis.window = { __TAURI__: { core: { invoke: async cmd => {
    assert.equal(cmd, 'server_status');
    checks++;
    if (checks === 1) throw Error('health permission error');
    return checks >= 4;
  } } } };
  await waitForServer(new AbortController().signal, error => errors.push(error), 1);
  assert.equal(checks, 4);
  assert.match(errors.join(' '), /health permission error/);
  const controller = new AbortController();
  globalThis.window.__TAURI__.core.invoke = async () => false;
  const pending = waitForServer(controller.signal, () => {}, 10000);
  controller.abort();
  await assert.rejects(pending, { name: 'AbortError' });
});

test('graph URL command failure propagates rather than silently navigating to a guess', async () => {
  globalThis.window = { __TAURI__: { core: { invoke: async () => { throw Error('URL blocked'); } } } };
  await assert.rejects(graphUrl(), /URL blocked/);
});


test('healthy old server keeps polling through initialization and active upgrade until failed or starting', async () => {
  for (const terminal of ['failed', 'starting']) {
    let checks = 0;
    let phase = 'uninitialized';
    globalThis.window = { __TAURI__: { core: { invoke: async () => {
      phase = ['uninitialized', 'installing', 'retrying', terminal][checks++];
      return true;
    } } } };
    await waitForServer(new AbortController().signal, error => assert.fail(error), 1,
      () => !['uninitialized', 'installing', 'retrying'].includes(phase));
    assert.equal(checks, 4);
  }
});

test('navigation transition rechecks live phase after splash and fade delays', async () => {
  for (const changeDuring of ['splash', 'fade']) {
    let allowed = true;
    const fades = [];
    const transition = waitForGraphTransition(new AbortController().signal, () => allowed,
      fading => { fades.push(fading); if (changeDuring === 'fade' && fading) allowed = false; }, 5, 1);
    if (changeDuring === 'splash') allowed = false;
    assert.equal(await transition, false);
    assert.deepEqual(fades, changeDuring === 'splash' ? [] : [true, false]);
  }
  const fades = [];
  assert.equal(await waitForGraphTransition(new AbortController().signal, () => true, value => fades.push(value), 0, 1), true);
  assert.deepEqual(fades, [true]);
});

test('status initialization settles after null snapshot or failure and never after teardown', async () => {
  for (const result of ['null', 'error']) {
    const errors = [];
    let initialized = 0;
    globalThis.window = { __TAURI__: {
      event: { listen: async () => () => {} },
      core: { invoke: async () => { if (result === 'error') throw Error('snapshot unavailable'); return null; } },
    } };
    const stop = onServerStatus(() => {}, error => errors.push(error), () => { initialized++; });
    await flush();
    assert.equal(initialized, 1);
    assert.equal(errors.length, result === 'error' ? 1 : 0);
    stop();
  }
  let resolveListen;
  let initialized = 0;
  globalThis.window = { __TAURI__: { event: { listen: () => new Promise(resolve => { resolveListen = resolve; }) } } };
  const stop = onServerStatus(() => assert.fail('disposed'), () => assert.fail('disposed'), () => { initialized++; });
  stop();
  resolveListen(() => {});
  await flush();
  assert.equal(initialized, 0);
});

test('one health loop refreshes snapshots and recovers from installing without lifecycle events', async () => {
  const { initialSetup, receiveStatus, completeStatusInitialization, canOpenGraph } =
    await import(pathToFileURL(`${process.env.ORMAH_UI_TEST_OUTPUT}/setup.js`));
  for (const terminal of ['starting', 'failed']) {
    let state = initialSetup(0);
    let snapshots = 0;
    let healthChecks = 0;
    const commands = [];
    globalThis.window = { __TAURI__: { core: { invoke: async cmd => {
      commands.push(cmd);
      if (cmd === 'runtime_setup_status') {
        snapshots++;
        return snapshots === 1
          ? { sequence: 1, updated_at_ms: 0, phase: 'installing', version: '2.0', attempt: 1, attempts: 5 }
          : terminal === 'starting'
            ? { sequence: 2, updated_at_ms: 0, phase: 'starting' }
            : { sequence: 2, updated_at_ms: 0, phase: 'failed', reason: 'Install failed', can_retry: true };
      }
      assert.equal(cmd, 'server_status');
      healthChecks++;
      if (healthChecks > 4) throw Error('snapshot callback was never invoked');
      return true;
    } } } };
    await waitForServer(new AbortController().signal, error => { throw Error(error); }, 1,
      () => snapshots > 0 && canOpenGraph(state), async () => {
        const snapshot = await globalThis.window.__TAURI__.core.invoke('runtime_setup_status');
        state = completeStatusInitialization(receiveStatus(state, snapshot, 0));
      });
    assert.equal(snapshots, 2);
    assert.equal(healthChecks, 2);
    assert.deepEqual(commands, ['runtime_setup_status', 'server_status', 'runtime_setup_status', 'server_status']);
    assert.equal(state.status.phase, terminal);
  }
});
