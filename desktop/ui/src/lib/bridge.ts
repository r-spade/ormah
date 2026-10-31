// Thin bridge to the Tauri commands (works in-app; no-ops in a plain browser).
type Invoke = (cmd: string, args?: Record<string, unknown>) => Promise<unknown>;

function getInvoke(): Invoke | null {
  const t = (window as unknown as { __TAURI__?: { core?: { invoke?: Invoke } } }).__TAURI__;
  return t?.core?.invoke ?? null;
}

export async function invoke<T = unknown>(cmd: string, args?: Record<string, unknown>): Promise<T> {
  const fn = getInvoke();
  if (!fn) throw new Error("tauri unavailable");
  return fn(cmd, args) as Promise<T>;
}

export async function graphUrl(): Promise<string> {
  return invoke<string>("graph_url");
}

export const errorText = (error: unknown) => error instanceof Error ? error.message : String(error);

export function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) { reject(new DOMException("Cancelled", "AbortError")); return; }
    const abort = () => { clearTimeout(timer); reject(new DOMException("Cancelled", "AbortError")); };
    const timer = setTimeout(() => { signal?.removeEventListener("abort", abort); resolve(); }, ms);
    signal?.addEventListener("abort", abort, { once: true });
  });
}

// A single caller owns this loop for the lifetime of the splash. Setup phases
// never stop health checks, so a slow or late server can always recover.
export async function waitForServer(signal: AbortSignal, onError: (message: string) => void, interval = 2000, canOpen: () => boolean = () => true, onPoll?: () => Promise<void>): Promise<void> {
  while (!signal.aborted) {
    try {
      await onPoll?.();
      if (signal.aborted) throw new DOMException("Cancelled", "AbortError");
      const ready = await invoke<boolean>("server_status");
      if (signal.aborted) throw new DOMException("Cancelled", "AbortError");
      if (ready && canOpen()) return;
    } catch (error) {
      if (signal.aborted) throw new DOMException("Cancelled", "AbortError");
      onError(`Couldn't check Ormah's status: ${errorText(error)}`);
    }
    await sleep(interval, signal);
  }
  throw new DOMException("Cancelled", "AbortError");
}

// A setup retry can start during either animation delay. Check the current
// phase at each boundary, and restore the splash if a fade was interrupted.
export async function waitForGraphTransition(
  signal: AbortSignal, canOpen: () => boolean, fade: (active: boolean) => void,
  splashDelay: number, fadeDelay = 620,
): Promise<boolean> {
  await sleep(splashDelay, signal);
  if (!canOpen()) return false;
  fade(true);
  await sleep(fadeDelay, signal);
  if (!canOpen()) { fade(false); return false; }
  return true;
}

// Sequence numbers order lifecycle events and the startup snapshot together.
export type ServerStatus = { sequence: number; updated_at_ms: number } & (
  | { phase: "installing"; version: string; attempt: number; attempts: number }
  | { phase: "retrying"; version: string; attempt: number; attempts: number; delay_seconds: number }
  | { phase: "starting" }
  | { phase: "failed"; reason: string; can_retry: boolean }
);

type ListenFn = (event: string, cb: (e: { payload: unknown }) => void) => Promise<() => void>;

export function onServerStatus(cb: (s: ServerStatus) => void, onError: (message: string) => void, onInitialized: () => void = () => {}): () => void {
  const t = (window as unknown as { __TAURI__?: { event?: { listen?: ListenFn } } }).__TAURI__;
  let disposed = false;
  let unlisten: (() => void) | null = null;
  void (async () => {
    try {
      if (!t?.event?.listen) throw new Error("Setup status events are unavailable");
      const stop = await t.event.listen("ormah://status", (event) => {
        if (!disposed) cb(event.payload as ServerStatus);
      });
      if (disposed) { stop(); return; }
      unlisten = stop;
      // Subscribe before fetching: an intervening event is never missed.
      // receiveStatus rejects a stale snapshot using its sequence number.
      const snapshot = await invoke<ServerStatus | null>("runtime_setup_status");
      if (!disposed && snapshot) cb(snapshot);
    } catch (error) {
      if (!disposed) onError(`Couldn't read setup progress: ${errorText(error)}`);
    } finally {
      if (!disposed) onInitialized();
    }
  })();
  return () => { disposed = true; unlisten?.(); };
}

// Window controls for the custom (decorationless) title bar.
function currentWindow(): { minimize?: () => Promise<void>; close?: () => Promise<void>; toggleMaximize?: () => Promise<void>; startDragging?: () => Promise<void> } | null {
  const w = window as unknown as { __TAURI__?: { window?: { getCurrentWindow?: () => unknown } } };
  return (w.__TAURI__?.window?.getCurrentWindow?.() as never) ?? null;
}
export const winMinimize = () => currentWindow()?.minimize?.();
export const winClose = () => currentWindow()?.close?.();
export const winToggleMaximize = () => currentWindow()?.toggleMaximize?.();
export const winStartDragging = () => currentWindow()?.startDragging?.();
