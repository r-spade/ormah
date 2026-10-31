import { test } from 'node:test';
import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
const { initialSetup, receiveStatus, beginRetry, finishRetry, setupView, markHealthy, canOpenGraph, completeStatusInitialization } =
  await import(pathToFileURL(`${process.env.ORMAH_UI_TEST_OUTPUT}/setup.js`));
const failed = (sequence, reason = 'Download failed') => ({ sequence, updated_at_ms: 1000, phase: 'failed', reason, can_retry: true });

test('a failure received on first launch immediately enables recovery', () => {
  const state = receiveStatus(initialSetup(1000), failed(1), 1000);
  assert.equal(setupView(state, 1000).canRetry, true);
  assert.equal(setupView(state, 1000).message, 'Download failed');
});

test('a failed retry releases busy immediately even while acceptance is pending', () => {
  let state = receiveStatus(initialSetup(1000), failed(1), 1000);
  state = beginRetry(state);
  assert.equal(state.retryBusy, true);
  state = receiveStatus(state, failed(2, 'Still offline'), 1100);
  assert.equal(state.retryBusy, false);
  assert.equal(setupView(state, 1100).canRetry, true);
  state = finishRetry(state, true, 1200);
  assert.equal(setupView(state, 1200).message, 'Still offline');
});

test('accepted retry releases busy before installation completes', () => {
  let state = beginRetry(receiveStatus(initialSetup(0), failed(1), 0));
  state = finishRetry(state, true, 1000);
  assert.equal(state.retryBusy, false);
  assert.equal(setupView(state, 1000).canRetry, false);
});

test('rejected and errored retry commands always explain recovery', () => {
  const state = beginRetry(receiveStatus(initialSetup(0), failed(1), 0));
  assert.match(setupView(finishRetry(state, false, 1), 1).error, /already running/i);
  assert.match(setupView(finishRetry(state, false, 1, 'Permission denied'), 1).error, /Permission denied/);
});

test('installation has no UI timeout, but starting eventually exposes recovery', () => {
  const installing = receiveStatus(initialSetup(0), { sequence: 1, updated_at_ms: 0, phase: 'installing', version: '2.0', attempt: 1, attempts: 5 }, 0);
  const installView = setupView(installing, 24 * 60 * 60 * 1000);
  assert.equal(installView.canRetry, false);
  assert.match(installView.message, /^Installing Ormah 2.0…/);
  assert.doesNotMatch(installView.message, /attempt|try 1/);
  const starting = receiveStatus(installing, { sequence: 2, updated_at_ms: 1000, phase: 'starting' }, 1000);
  assert.equal(setupView(starting, 600999).canRetry, false);
  assert.equal(setupView(starting, 601000).canRetry, true);
  assert.match(setupView(starting, 601000).message, /longer than expected/);
});

test('late health succeeds after a starting deadline or terminal install failure', () => {
  for (const state of [initialSetup(0), receiveStatus(initialSetup(0), failed(1), 0)]) {
    assert.equal(setupView(state, 700000).canRetry, true);
    assert.equal(markHealthy(state).healthy, true);
    assert.equal(setupView(markHealthy(state), 700000).canRetry, false);
  }
});

test('out-of-order and duplicate snapshots cannot replace a newer failure', () => {
  const newest = receiveStatus(initialSetup(0), failed(8), 1000);
  for (const sequence of [7, 8]) {
    const stale = receiveStatus(newest, { sequence, updated_at_ms: 0, phase: 'starting' }, 2000);
    assert.equal(stale, newest);
  }
});

test('retry countdown uses next attempt and elapsed time since backend timestamp', () => {
  const state = receiveStatus(initialSetup(0), { sequence: 2, updated_at_ms: 1000, phase: 'retrying', version: '2.0', attempt: 2, attempts: 5, delay_seconds: 10 }, 3000);
  assert.equal(setupView(state, 3000).message, "Couldn't connect. Trying again in 8s (try 2 of 5)");
  assert.match(setupView(state, 10500).message, /in 1s/);
  assert.match(setupView(state, 13000).message, /in 0s/);
});


test('old healthy server cannot bypass pending initialization or active upgrade phases', () => {
  let state = initialSetup(0);
  assert.equal(canOpenGraph(state), false);
  state = completeStatusInitialization(state);
  assert.equal(canOpenGraph(state), true);
  state = receiveStatus(state, { sequence: 1, updated_at_ms: 0, phase: 'installing', version: '2.0', attempt: 1, attempts: 5 }, 0);
  assert.equal(canOpenGraph(state), false);
  state = receiveStatus(state, { sequence: 2, updated_at_ms: 0, phase: 'retrying', version: '2.0', attempt: 2, attempts: 5, delay_seconds: 10 }, 0);
  assert.equal(canOpenGraph(state), false);
  state = receiveStatus(state, failed(3), 1000);
  assert.equal(canOpenGraph(state), true);
  state = beginRetry(state);
  assert.equal(canOpenGraph(state), false);
  state = receiveStatus(state, { sequence: 4, updated_at_ms: 1000, phase: 'starting' }, 1000);
  state = finishRetry(state, true, 1000);
  assert.equal(canOpenGraph(state), true);
});

test('failed status initialization preserves visible recovery and permits late health', () => {
  const state = completeStatusInitialization({ ...initialSetup(0), error: 'Snapshot unavailable' });
  assert.equal(canOpenGraph(state), true);
  assert.equal(setupView(state, 1).canRetry, true);
  assert.equal(setupView(state, 1).error, 'Snapshot unavailable');
});

test('accepted retry stays gated until a fresh status or explicit snapshot result arrives', () => {
  let state = finishRetry(beginRetry(receiveStatus(initialSetup(0), failed(1), 0)), true, 1000);
  assert.equal(state.retryBusy, false);
  assert.equal(canOpenGraph(state), false);
  // An old delayed failure snapshot cannot reopen navigation during the retry.
  state = receiveStatus(state, failed(1), 1100);
  assert.equal(canOpenGraph(state), false);
  state = receiveStatus(state, { sequence: 2, updated_at_ms: 1000, phase: 'installing', version: '2.0', attempt: 1, attempts: 5 }, 1200);
  state = completeStatusInitialization(state);
  assert.equal(canOpenGraph(state), false);
  state = receiveStatus(state, { sequence: 3, updated_at_ms: 2000, phase: 'starting' }, 2000);
  assert.equal(canOpenGraph(state), true);
});

test('unavailable retry snapshot preserves visible recovery while settling degraded initialization', () => {
  const accepted = finishRetry(beginRetry(receiveStatus(initialSetup(0), failed(1), 0)), true, 1000);
  assert.equal(canOpenGraph(accepted), false);
  const recovered = completeStatusInitialization({ ...accepted, error: 'Retry snapshot unavailable' });
  assert.equal(canOpenGraph(recovered), true);
  assert.equal(setupView(recovered, 1001).error, 'Retry snapshot unavailable');
  assert.equal(setupView(recovered, 1001).canRetry, true);
});
