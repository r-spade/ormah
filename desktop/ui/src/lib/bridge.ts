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

export const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

// Setup state owned by sidecar.rs. Only "ready" means the server is up.
export type SetupStatus =
  | { phase: "installing"; version: string; attempt: number; attempts: number }
  | { phase: "retrying"; attempt: number; attempts: number }
  | { phase: "starting" }
  | { phase: "ready" }
  | { phase: "failed"; reason: string; can_open: boolean };

// Window controls for the custom (decorationless) title bar.
function currentWindow(): { minimize?: () => Promise<void>; close?: () => Promise<void>; toggleMaximize?: () => Promise<void>; startDragging?: () => Promise<void> } | null {
  const w = window as unknown as { __TAURI__?: { window?: { getCurrentWindow?: () => unknown } } };
  return (w.__TAURI__?.window?.getCurrentWindow?.() as never) ?? null;
}
export const winMinimize = () => currentWindow()?.minimize?.();
export const winClose = () => currentWindow()?.close?.();
export const winToggleMaximize = () => currentWindow()?.toggleMaximize?.();
export const winStartDragging = () => currentWindow()?.startDragging?.();
