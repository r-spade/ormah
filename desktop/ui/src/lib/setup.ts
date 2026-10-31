import type { ServerStatus } from "./bridge";

const STARTING_TIMEOUT_MS = 10 * 60 * 1000;

export interface SetupState {
  status: ServerStatus | null;
  sequence: number;
  statusInitialized: boolean;
  waitingSince: number;
  retryBusy: boolean;
  retrySequence: number;
  error: string | null;
  healthy: boolean;
}

export function initialSetup(now: number): SetupState {
  return { status: null, sequence: -1, statusInitialized: false, waitingSince: now, retryBusy: false, retrySequence: -1, error: null, healthy: false };
}

export function receiveStatus(state: SetupState, status: ServerStatus, now: number): SetupState {
  if (status.sequence <= state.sequence) return state;
  return {
    ...state, status, sequence: status.sequence, statusInitialized: true,
    waitingSince: status.phase === "starting" ? Math.min(now, status.updated_at_ms) : now,
    retryBusy: status.phase === "failed" ? false : state.retryBusy,
    error: null,
  };
}

export function completeStatusInitialization(state: SetupState): SetupState {
  return { ...state, statusInitialized: true };
}

// An already-running older daemon must not bypass an in-progress upgrade.
export function canOpenGraph(state: SetupState): boolean {
  return state.statusInitialized && !state.retryBusy
    && state.status?.phase !== "installing" && state.status?.phase !== "retrying";
}

export function beginRetry(state: SetupState): SetupState {
  return { ...state, retryBusy: true, retrySequence: state.sequence, error: null };
}

export function finishRetry(state: SetupState, accepted: boolean, now: number, error?: string): SetupState {
  const next = { ...state, retryBusy: false };
  if (error) return { ...next, error: `Couldn't retry setup: ${error}` };
  if (!accepted) return { ...next, error: "Setup is already running. Waiting for its current status; you can try again if it fails." };
  // A new failure may arrive while the command is still being accepted.
  if (state.sequence !== state.retrySequence) return next;
  return { ...next, status: null, statusInitialized: false, waitingSince: now, error: null };
}

export function markHealthy(state: SetupState): SetupState {
  return { ...state, healthy: true, error: null };
}

export function setupView(state: SetupState, now: number) {
  const status = state.status;
  const timedOut = !state.healthy && (!status || status.phase === "starting")
    && now - state.waitingSince >= STARTING_TIMEOUT_MS;
  let message = "Starting Ormah…";
  let hint = "";
  if (status?.phase === "installing") {
    message = `Installing Ormah ${status.version}…`;
    hint = "The first download can take a few minutes.";
    if (status.attempt > 1) hint += ` Try ${status.attempt} of ${status.attempts}.`;
  } else if (status?.phase === "retrying") {
    const remaining = Math.max(0, Math.ceil((status.updated_at_ms + status.delay_seconds * 1000 - now) / 1000));
    message = `Couldn't connect. Trying again in ${remaining}s (try ${status.attempt} of ${status.attempts})`;
  } else if (status?.phase === "failed") {
    message = status.reason;
  }
  if (timedOut) message = "Ormah is taking longer than expected to start. You can try again or share the setup details. We'll keep checking for it.";
  const failed = status?.phase === "failed" || timedOut || !!state.error;
  return {
    message, hint, error: state.error, failed: !state.healthy && failed,
    canRetry: !state.healthy && (timedOut || !!state.error || (status?.phase === "failed" && status.can_retry)),
  };
}
