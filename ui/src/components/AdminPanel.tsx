import { useCallback, useEffect, useState } from "react";
import {
  fetchAdminTasks,
  runAdminTask,
  runAllTasks,
  pauseTask,
  resumeTask,
  pauseAllTasks,
  resumeAllTasks,
} from "../api";
import type { AdminTask } from "../api";

interface Props {
  open: boolean;
  onClose: () => void;
  onToast: (message: string, type: "success" | "error" | "info") => void;
}

export default function AdminPanel({ open, onClose, onToast }: Props) {
  const [tasks, setTasks] = useState<AdminTask[]>([]);
  const [running, setRunning] = useState<string | null>(null);
  const [runningAll, setRunningAll] = useState(false);
  const [toggling, setToggling] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<{ task: string; ok: boolean } | null>(null);

  useEffect(() => {
    if (!open) return;
    fetchAdminTasks().then((data) => setTasks(data.tasks)).catch(() => {});
  }, [open]);

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

  const busy = running !== null || runningAll;

  return (
    <div className={`side-panel admin-panel ${open ? "open" : ""}`}>
      <div className="side-panel-header">
        <div className="review-title">background tasks</div>
        <button className="node-detail-close" onClick={onClose}>×</button>
      </div>
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
