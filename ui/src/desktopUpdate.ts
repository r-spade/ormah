import { invoke } from "@tauri-apps/api/core";

export type DesktopUpdatePhase =
  | "unknown"
  | "checking"
  | "current"
  | "available"
  | "downloading"
  | "installing"
  | "failed";

export interface DesktopUpdateStatus {
  phase: DesktopUpdatePhase;
  version: string | null;
  notes: string;
  progress_percent: number | null;
  message: string | null;
}

export function desktopUpdateProgress(status: DesktopUpdateStatus): string {
  if (status.phase === "installing") return "Installing signed update";
  if (status.phase === "downloading") {
    return status.progress_percent === null
      ? "Downloading signed update"
      : `Downloading signed update · ${status.progress_percent}%`;
  }
  return "";
}

export const desktopUpdater = {
  status: () => invoke<DesktopUpdateStatus>("desktop_update_status"),
  check: () => invoke<DesktopUpdateStatus>("check_desktop_update"),
  install: () => invoke<DesktopUpdateStatus>("install_desktop_update"),
};
