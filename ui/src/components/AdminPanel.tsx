import { useCallback, useEffect, useState } from "react";
import {
  createBackup,
  fetchBackupStatus,
  fetchCloudStatus,
  fetchAdminTasks,
  runAdminTask,
  runAllTasks,
  pauseTask,
  resumeTask,
  pauseAllTasks,
  resumeAllTasks,
  updateBackupSettings,
} from "../api";
import type { AdminTask, BackupStatus, CloudStatus } from "../api";

interface Props {
  open: boolean;
  onClose: () => void;
  onToast: (message: string, type: "success" | "error" | "info") => void;
}

export default function AdminPanel({ open, onClose, onToast }: Props) {
  const [tasks, setTasks] = useState<AdminTask[]>([]);
  const [backupStatus, setBackupStatus] = useState<BackupStatus | null>(null);
  const [cloudStatus, setCloudStatus] = useState<CloudStatus | null>(null);
  const [backupStatusLoaded, setBackupStatusLoaded] = useState(false);
  const [backupSettingsOpen, setBackupSettingsOpen] = useState(false);
  const [backupDirInput, setBackupDirInput] = useState("");
  const [retentionInput, setRetentionInput] = useState("10");
  const [running, setRunning] = useState<string | null>(null);
  const [runningAll, setRunningAll] = useState(false);
  const [creatingBackup, setCreatingBackup] = useState(false);
  const [savingBackupSettings, setSavingBackupSettings] = useState(false);
  const [toggling, setToggling] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<{ task: string; ok: boolean } | null>(null);

  useEffect(() => {
    if (!open) return;
    fetchAdminTasks().then((data) => setTasks(data.tasks)).catch(() => {});
    setBackupStatusLoaded(false);
    fetchBackupStatus()
      .then(setBackupStatus)
      .catch(() => setBackupStatus(null))
      .finally(() => setBackupStatusLoaded(true));
    fetchCloudStatus().then(setCloudStatus).catch(() => setCloudStatus(null));
  }, [open]);

  useEffect(() => {
    if (!backupStatus) return;
    setBackupDirInput(backupStatus.backup_dir);
    setRetentionInput(String(backupStatus.retention_count));
  }, [backupStatus]);

  const refreshBackupStatus = useCallback(() => {
    fetchBackupStatus()
      .then(setBackupStatus)
      .catch(() => setBackupStatus(null))
      .finally(() => setBackupStatusLoaded(true));
  }, []);

  const handleCreateBackup = useCallback(async () => {
    setCreatingBackup(true);
    try {
      const result = await createBackup();
      setBackupStatus(result.backup_status);
      onToast(`Backup created: ${result.backup.name}`, "success");
    } catch {
      onToast("Backup failed", "error");
      refreshBackupStatus();
    } finally {
      setCreatingBackup(false);
    }
  }, [onToast, refreshBackupStatus]);

  const handleSaveBackupSettings = useCallback(async () => {
    const retentionCount = Number.parseInt(retentionInput, 10);
    if (!backupDirInput.trim() || Number.isNaN(retentionCount) || retentionCount < 1) {
      onToast("Backup settings are invalid", "error");
      return;
    }

    setSavingBackupSettings(true);
    try {
      const result = await updateBackupSettings({
        backup_dir: backupDirInput.trim(),
        retention_count: retentionCount,
      });
      setBackupStatus(result.backup_status);
      onToast("Backup settings saved", "success");
    } catch {
      onToast("Failed to save backup settings", "error");
      refreshBackupStatus();
    } finally {
      setSavingBackupSettings(false);
    }
  }, [backupDirInput, retentionInput, onToast, refreshBackupStatus]);

  const handleRun = useCallback(async (taskId: string) => {
    setRunning(taskId);
    setLastResult(null);
    try {
      await runAdminTask(taskId);
      setLastResult({ task: taskId, ok: true });
      onToast(`Task "${taskId}" completed`, "success");
    } catch {
      setLastResult({ task: taskId, ok: false });
      onToast(`Task "${taskId}" failed`, "error");
    } finally {
      setRunning(null);
    }
  }, [onToast]);

  const handleRunAll = useCallback(async () => {
    setRunningAll(true);
    setLastResult(null);
    try {
      await runAllTasks();
      setLastResult({ task: "__all__", ok: true });
      onToast("Sleep cycle complete", "success");
    } catch {
      setLastResult({ task: "__all__", ok: false });
      onToast("Sleep cycle failed", "error");
    } finally {
      setRunningAll(false);
    }
  }, [onToast]);

  const refreshTasks = useCallback(() => {
    fetchAdminTasks().then((data) => setTasks(data.tasks)).catch(() => {});
  }, []);

  const handleToggle = useCallback(async (taskId: string, paused: boolean) => {
    setToggling(taskId);
    try {
      if (paused) {
        await resumeTask(taskId);
        onToast(`Task "${taskId}" resumed`, "success");
      } else {
        await pauseTask(taskId);
        onToast(`Task "${taskId}" paused`, "info");
      }
      refreshTasks();
    } catch {
      onToast(`Failed to toggle "${taskId}"`, "error");
    } finally {
      setToggling(null);
    }
  }, [onToast, refreshTasks]);

  const allPaused = tasks.length > 0 && tasks.every((t) => t.paused);

  const handleToggleAll = useCallback(async () => {
    setToggling("__all__");
    try {
      if (allPaused) {
        await resumeAllTasks();
        onToast("All tasks resumed", "success");
      } else {
        await pauseAllTasks();
        onToast("All tasks paused", "info");
      }
      refreshTasks();
    } catch {
      onToast("Failed to toggle all tasks", "error");
    } finally {
      setToggling(null);
    }
  }, [onToast, refreshTasks, allPaused]);

  const busy = running !== null || runningAll || creatingBackup || savingBackupSettings;

  return (
    <div className={`side-panel admin-panel ${open ? "open" : ""}`}>
      <div className="side-panel-header">
        <div className="review-title">admin</div>
        <button className="node-detail-close" onClick={onClose}>×</button>
      </div>

      <div className="admin-section">
        <div className="admin-section-heading admin-backup-heading">
          <span>local backups</span>
          {backupStatus && (
            <div className="admin-backup-heading-meta">
              <span className="admin-backup-automatic">{backupAutomaticLabel(backupStatus)}</span>
              <button
                className="admin-backup-settings-toggle"
                type="button"
                aria-label="backup settings"
                aria-expanded={backupSettingsOpen}
                title="backup settings"
                onClick={() => setBackupSettingsOpen((value) => !value)}
              />
            </div>
          )}
        </div>
        {backupStatus ? (
          <div className="admin-backup-card">
            <div className="admin-backup-row">
              <span>last</span>
              <strong>
                {backupStatus.latest ? formatDate(backupStatus.latest.created_at) : "none"}
              </strong>
            </div>
            {!backupStatus.has_backupable_memory && (
              <div className="admin-backup-note">no memory nodes to auto-backup yet</div>
            )}
            {cloudStatus && (
              <div
                className={`admin-cloud-verification ${
                  cloudStatus.last_verify_ok === true
                    ? "ok"
                    : cloudStatus.last_verify_ok === false
                      ? "failed"
                      : "unknown"
                }`}
                title={cloudStatus.last_verify_error || undefined}
              >
                <span>Last verified restorable:</span>
                <strong>
                  {cloudStatus.last_verify_ok === true
                    ? `✓ ${cloudStatus.last_verify_at ? formatDate(cloudStatus.last_verify_at) : "date unavailable"}`
                    : cloudStatus.last_verify_ok === false
                      ? `✗ ${cloudStatus.last_verify_at ? formatDate(cloudStatus.last_verify_at) : "date unavailable"}`
                      : "not yet verified"}
                </strong>
              </div>
            )}
            <button
              className="review-btn approve admin-backup-button"
              disabled={busy}
              onClick={handleCreateBackup}
            >
              {creatingBackup ? "creating..." : "create backup"}
            </button>
            {backupSettingsOpen && (
              <div className="admin-backup-settings">
                <label className="admin-backup-field">
                  <span>folder</span>
                  <input
                    value={backupDirInput}
                    onChange={(event) => setBackupDirInput(event.target.value)}
                  />
                </label>
                <label className="admin-backup-field compact">
                  <span>keep last</span>
                  <input
                    inputMode="numeric"
                    pattern="[0-9]*"
                    type="text"
                    value={retentionInput}
                    onChange={(event) => setRetentionInput(event.target.value)}
                  />
                </label>
                <button
                  className="review-btn approve admin-backup-save"
                  disabled={busy}
                  onClick={handleSaveBackupSettings}
                >
                  {savingBackupSettings ? "saving..." : "save settings"}
                </button>
              </div>
            )}
          </div>
        ) : (
          <div className="review-empty">
            {backupStatusLoaded ? "backup status unavailable" : "loading backup status..."}
          </div>
        )}
      </div>

      <div className="admin-section-heading">background tasks</div>
      <div style={{ marginBottom: 14, display: "flex", gap: 8 }}>
        <button
          className="review-btn approve"
          disabled={busy}
          onClick={handleRunAll}
          style={{ flex: 1 }}
        >
          {runningAll ? "running sleep cycle..." : "run sleep cycle"}
        </button>
        <button
          className={`review-btn ${allPaused ? "approve" : "reject"}`}
          disabled={toggling === "__all__"}
          onClick={handleToggleAll}
          style={{ whiteSpace: "nowrap" }}
        >
          {toggling === "__all__" ? "..." : allPaused ? "resume all" : "pause all"}
        </button>
      </div>
      {lastResult && lastResult.task === "__all__" && (
        <div style={{ marginBottom: 10 }}>
          <span className={`admin-task-result ${lastResult.ok ? "ok" : "err"}`}>
            {lastResult.ok ? "cycle complete" : "cycle failed"}
          </span>
        </div>
      )}
      {tasks.length === 0 && (
        <div className="review-empty">no tasks registered</div>
      )}
      {tasks.map((t) => (
        <div className={`admin-task-card${t.paused ? " paused" : ""}`} key={t.id}>
          <div className="admin-task-header">
            <span className="admin-task-name">{t.name}</span>
            <span className="admin-task-id">{t.id}</span>
            {t.description && (
              <div className="admin-task-desc">{t.description}</div>
            )}
          </div>
          {t.next_run && (
            <div className="admin-task-next">next: {t.next_run}</div>
          )}
          <label className="toggle-switch" title={t.paused ? "Resume task" : "Pause task"}>
            <input
              type="checkbox"
              checked={!t.paused}
              disabled={toggling === t.id}
              onChange={() => handleToggle(t.id, t.paused)}
            />
            <span className="toggle-slider" />
          </label>
          <button
            className="review-btn approve"
            disabled={busy || t.paused}
            onClick={() => handleRun(t.id)}
          >
            {running === t.id ? "running..." : "run now"}
          </button>
          {lastResult && lastResult.task === t.id && (
            <span className={`admin-task-result ${lastResult.ok ? "ok" : "err"}`}>
              {lastResult.ok ? "done" : "failed"}
            </span>
          )}
        </div>
      ))}
    </div>
  );
}

function formatDate(value: string): string {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(parsed);
}

function backupAutomaticLabel(status: BackupStatus): string {
  if (!status.enabled) return "automatic backups off";
  return `automatic backups every ${status.interval_hours}h`;
}
