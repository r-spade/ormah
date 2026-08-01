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
  try { return await invoke<string>("graph_url"); }
  catch { return "http://127.0.0.1:8787"; }
}

export async function waitForServer(onSlow?: () => void): Promise<boolean> {
  for (let i = 0; i < 150; i++) {
    try { await invoke("fetch_stats"); return true; }
    catch { if (i === 6) onSlow?.(); await new Promise((r) => setTimeout(r, 2000)); }
  }
  return false;
}

export const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

// Listen for server lifecycle events emitted by sidecar.rs.
export type ServerStatus =
  | { phase: "installing"; version: string; attempt: number; attempts: number }
  | { phase: "retrying"; version: string; attempt: number; attempts: number; delay_seconds: number }
  | { phase: "starting" }
  | { phase: "failed"; reason: string; can_retry: boolean };

type ListenFn = (event: string, cb: (e: { payload: unknown }) => void) => Promise<() => void>;

export function onServerStatus(cb: (s: ServerStatus) => void): () => void {
  const t = (window as unknown as { __TAURI__?: { event?: { listen?: ListenFn } } }).__TAURI__;
  const listen = t?.event?.listen;
  if (!listen) return () => {};
  let unlisten: (() => void) | null = null;
  listen("ormah://status", (e) => cb(e.payload as ServerStatus)).then((fn) => { unlisten = fn; });
  return () => { unlisten?.(); };
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
