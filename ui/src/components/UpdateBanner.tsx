import { useCallback, useEffect, useState } from "react";
import { AlertTriangle, Download, LoaderCircle, RefreshCw, X } from "lucide-react";

import {
  desktopUpdateProgress,
  desktopUpdater,
  type DesktopUpdateStatus,
} from "../desktopUpdate";
import { isDesktopApp } from "../productBridge";

const POLL_MS = 750;

export default function UpdateBanner() {
  const [status, setStatus] = useState<DesktopUpdateStatus | null>(null);
  const [dismissedVersion, setDismissedVersion] = useState<string | null>(null);
  const [installAttempted, setInstallAttempted] = useState(false);

  const refresh = useCallback(async () => {
    try {
      setStatus(await desktopUpdater.status());
    } catch {
      // An older desktop bridge cannot update in-app. Its existing tray
      // updater remains available, so keep the product UI quiet.
      setStatus(null);
    }
  }, []);

  useEffect(() => {
    if (!isDesktopApp()) return;
    void refresh();
  }, [refresh]);

  useEffect(() => {
    if (!status || !["checking", "downloading", "installing"].includes(status.phase)) return;
    const timer = window.setInterval(() => void refresh(), POLL_MS);
    return () => window.clearInterval(timer);
  }, [refresh, status]);

  const install = useCallback(() => {
    setInstallAttempted(true);
    setStatus((current) => current ? { ...current, phase: "downloading", progress_percent: 0 } : current);
    void desktopUpdater.install().catch((error: unknown) => {
      const message = error instanceof Error ? error.message : String(error);
      setStatus((current) => ({
        phase: "failed",
        version: current?.version ?? null,
        notes: current?.notes ?? "",
        progress_percent: null,
        message: message || "The update could not be installed. Your current version is unchanged.",
      }));
    });
  }, []);

  const retry = useCallback(async () => {
    setInstallAttempted(false);
    setStatus((current) => current ? { ...current, phase: "checking", message: null } : current);
    try {
      setStatus(await desktopUpdater.check());
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setStatus({
        phase: "failed",
        version: null,
        notes: "",
        progress_percent: null,
        message: message || "Ormah could not check for updates.",
      });
    }
  }, []);

  if (!status || status.phase === "checking" || status.phase === "current") return null;
  if (status.phase === "available" && dismissedVersion === status.version) return null;
  if (status.phase === "failed" && dismissedVersion === "failed") return null;

  const working = status.phase === "downloading" || status.phase === "installing";
  const failed = status.phase === "failed";
  const title = failed
    ? installAttempted ? "Update interrupted" : "Update check unavailable"
    : working
      ? desktopUpdateProgress(status)
      : `Ormah ${status.version} is available`;

  return (
    <section className={`update-banner ${failed ? "update-failed" : ""}`} aria-live="polite">
      <div className="update-banner-icon" aria-hidden="true">
        {failed ? <AlertTriangle size={17} />
          : working ? <LoaderCircle className="spin" size={17} />
            : <Download size={17} />}
      </div>
      <div className="update-banner-copy">
        <strong>{title}</strong>
        <span>{failed
          ? status.message || "Your current version is unchanged."
          : working
            ? "Ormah will restart only after the signed update is installed."
            : status.notes || "A new signed desktop version is ready."}</span>
        {status.phase === "downloading" && status.progress_percent !== null && (
          <div className="update-progress" aria-label={`${status.progress_percent}% downloaded`}>
            <span style={{ width: `${status.progress_percent}%` }} />
          </div>
        )}
      </div>
      <div className="update-banner-actions">
        {status.phase === "available" && (
          <>
            <button className="update-now" onClick={install}>
              <Download size={13} /> Update now
            </button>
            <button
              className="update-later"
              onClick={() => setDismissedVersion(status.version)}
            >Later</button>
          </>
        )}
        {failed && (
          <button className="update-now" onClick={() => void retry()}>
            <RefreshCw size={13} /> Try again
          </button>
        )}
      </div>
      {!working && (
        <button
          className="update-dismiss"
          aria-label="Dismiss update notice"
          onClick={() => setDismissedVersion(status.version || "failed")}
        ><X size={14} /></button>
      )}
    </section>
  );
}
