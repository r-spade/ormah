import { useEffect, useState } from "react";
import { fetchAgentClients, runAgentSetup, wireAgent, unwireAgent } from "../api";
import type { AgentInfo } from "../api";

interface Props {
  open: boolean;
  onClose: () => void;
}

function statusLabel(agent: AgentInfo): { text: string; cls: string } {
  if (!agent.available_on_current_os) return { text: "n/a on this OS", cls: "agent-status--na" };
  if (!agent.detected)               return { text: "not installed",    cls: "agent-status--absent" };
  if (agent.capabilities && agent.wired) {
    const whisper = agent.capabilities.whisper;
    const text = whisper === "native_hook" || whisper === "native_extension"
      ? "whisper configured" : "tools only";
    return { text, cls: "agent-status--wired" };
  }
  if (agent.wired)                   return { text: "connected",        cls: "agent-status--wired" };
  return                                    { text: "not wired",        cls: "agent-status--detected" };
}

export default function AgentsPanel({ open, onClose }: Props) {
  const [agents, setAgents]       = useState<AgentInfo[]>([]);
  const [loading, setLoading]     = useState(false);
  const [busy, setBusy]           = useState<string | null>(null);
  const [errors, setErrors]       = useState<Record<string, string>>({});
  const [fetchError, setFetchError] = useState<string | null>(null);

  const load = () => {
    setLoading(true);
    setFetchError(null);
    fetchAgentClients()
      .then(setAgents)
      .catch(() => setFetchError("Could not reach server."))
      .finally(() => setLoading(false));
  };

  useEffect(() => { if (open) { load(); setErrors({}); } }, [open]);

  const act = async (
    agentId: string,
    fn: () => Promise<{ errors: Record<string, string> }>,
  ) => {
    setBusy(agentId);
    setErrors(e => { const n = { ...e }; delete n[agentId]; return n; });
    try {
      const res = await fn();
      if (res.errors[agentId]) setErrors(e => ({ ...e, [agentId]: res.errors[agentId] }));
    } catch {
      setErrors(e => ({ ...e, [agentId]: "Request failed." }));
    } finally {
      setBusy(null);
      load();
    }
  };

  const handleWireAll = async () => {
    setBusy("__all__");
    setErrors({});
    try {
      const res = await runAgentSetup();
      if (Object.keys(res.errors).length) setErrors(res.errors);
    } catch {
      setFetchError("Setup failed — is ormah running?");
    } finally {
      setBusy(null);
      load();
    }
  };

  const hasUnwired = agents.some(a => a.available_on_current_os && a.detected && !a.wired);

  return (
    <div className={`side-panel admin-panel ${open ? "open" : ""}`}>
      <div className="side-panel-header">
        <div className="review-title">agents</div>
        <button className="node-detail-close" onClick={onClose}>×</button>
      </div>

      <div className="admin-section">
        <div className="admin-section-heading">
          integrations
          {hasUnwired && (
            <button
              className="review-btn approve"
              onClick={handleWireAll}
              disabled={busy !== null}
            >
              {busy === "__all__" ? "wiring…" : "connect all"}
            </button>
          )}
        </div>

        {loading && <div className="review-empty">loading…</div>}

        {!loading && agents.map(agent => {
          const { text, cls } = statusLabel(agent);
          return (
            <div key={agent.id} className="admin-task-card">
              <div className="admin-task-header">
                <span className="admin-task-name">{agent.name}</span>
                <span className={`agent-status ${cls}`}>{text}</span>
                {agent.capabilities && <div className="admin-task-desc">{agent.capabilities.detail}</div>}
                {errors[agent.id] && (
                  <div className="admin-task-desc" style={{ color: "var(--edge-contradicts)", marginTop: 4 }}>
                    {errors[agent.id]}
                  </div>
                )}
              </div>

              {agent.available_on_current_os && agent.detected && !agent.wired && (
                <button
                  className="review-btn approve"
                  onClick={() => act(agent.id, () => wireAgent(agent.id))}
                  disabled={busy !== null}
                >
                  {busy === agent.id ? "…" : "connect"}
                </button>
              )}
              {agent.wired && (
                <button
                  className="review-btn"
                  onClick={() => act(agent.id, () => unwireAgent(agent.id) as Promise<{ errors: Record<string, string> }>)}
                  disabled={busy !== null}
                >
                  {busy === agent.id ? "…" : "disconnect"}
                </button>
              )}
            </div>
          );
        })}

        {!loading && !hasUnwired && !fetchError && agents.length > 0 && (
          <div className="admin-backup-note">All detected agents are configured.</div>
        )}

        {fetchError && (
          <div className="admin-task-desc" style={{ color: "var(--edge-contradicts)" }}>
            {fetchError}
          </div>
        )}
      </div>
    </div>
  );
}
