import { useEffect, useState } from "react";
import { fetchBlindSpots, fetchInsights, fetchRecallDebug } from "../api";
import type { BlindSpotsData, InsightsData, InsightNode, RecallDebugData } from "../types";

interface Props {
  open: boolean;
  onClose: () => void;
  onNodeClick: (nodeId: string) => void;
  onPairHover: (ids: string[]) => void;
  onPairHoverEnd: () => void;
}

function formatDate(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleDateString("en-US", { month: "short", year: "numeric" });
  } catch {
    return iso;
  }
}

function NodeSummary({
  node,
  marker,
  onClick,
}: {
  node: InsightNode;
  marker: string;
  onClick: () => void;
}) {
  return (
    <div className="insight-node" onClick={onClick}>
      <span className="insight-node-marker">{marker}</span>
      <div>
        <div className="insight-node-title">
          {node.title || node.content.slice(0, 50)}
        </div>
        <div className="insight-node-meta">
          {node.type} &middot; {node.tier}
        </div>
        <div className="insight-node-content">
          {node.content.length > 120
            ? node.content.slice(0, 120) + "..."
            : node.content}
        </div>
      </div>
    </div>
  );
}

function formatDateTime(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  } catch {
    return iso;
  }
}

export default function InsightsPanel({
  open,
  onClose,
  onNodeClick,
  onPairHover,
  onPairHoverEnd,
}: Props) {
  const [data, setData] = useState<InsightsData | null>(null);
  const [blindSpots, setBlindSpots] = useState<BlindSpotsData | null>(null);
  const [recallDebug, setRecallDebug] = useState<RecallDebugData | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!open) return;
    setLoading(true);
    Promise.all([
      fetchInsights().catch(() => ({ evolutions: [], tensions: [] })),
      fetchBlindSpots().catch(() => ({ isolated_nodes: [], sparse_spaces: [], uncovered_types: [] })),
      fetchRecallDebug(30).catch(() => ({ entries: [] })),
    ]).then(([ins, bs, rd]) => {
      setData(ins as InsightsData);
      setBlindSpots(bs as BlindSpotsData);
      setRecallDebug(rd as RecallDebugData);
    }).finally(() => setLoading(false));
  }, [open]);

  return (
    <div className={`side-panel insights-panel ${open ? "open" : ""}`}>
      <div className="insights-inner">
        <div className="node-detail-header">
          <span className="node-detail-title">insights</span>
          <button className="node-detail-close" onClick={onClose}>
            &times;
          </button>
        </div>

        {loading && <div className="insights-empty">loading...</div>}

        {!loading && data && (
          <>
            <div className="insights-section">
              <div className="insights-section-title">belief evolution</div>
              {data.evolutions.length === 0 && (
                <div className="insights-empty">no evolutions detected</div>
              )}
              {data.evolutions.map((evo, i) => (
                <div
                  className="insights-card evolution-card"
                  key={i}
                  onMouseEnter={() =>
                    onPairHover([evo.older.id, evo.newer.id])
                  }
                  onMouseLeave={onPairHoverEnd}
                >
                  <div className="evolution-date">
                    {formatDate(evo.older.created)}
                  </div>
                  <NodeSummary
                    node={evo.older}
                    marker="&#9675;"
                    onClick={() => onNodeClick(evo.older.id)}
                  />
                  <div className="evolution-arrow">&darr; evolved into</div>
                  <NodeSummary
                    node={evo.newer}
                    marker="&#9679;"
                    onClick={() => onNodeClick(evo.newer.id)}
                  />
                  {evo.explanation && (
                    <div className="insights-explanation">{evo.explanation}</div>
                  )}
                </div>
              ))}
            </div>

            <div className="insights-section">
              <div className="insights-section-title">conflicting ideas</div>
              {data.tensions.length === 0 && (
                <div className="insights-empty">no conflicting ideas detected</div>
              )}
              {data.tensions.map((t, i) => (
                <div
                  className="insights-card tension-card"
                  key={i}
                  onMouseEnter={() =>
                    onPairHover([t.node_a.id, t.node_b.id])
                  }
                  onMouseLeave={onPairHoverEnd}
                >
                  <div className="tension-pair">
                    <span
                      className="tension-title clickable"
                      onClick={() => onNodeClick(t.node_a.id)}
                    >
                      {t.node_a.title || t.node_a.content.slice(0, 30)}
                    </span>
                    <span className="tension-separator">&harr;</span>
                    <span
                      className="tension-title clickable"
                      onClick={() => onNodeClick(t.node_b.id)}
                    >
                      {t.node_b.title || t.node_b.content.slice(0, 30)}
                    </span>
                  </div>
                  {t.explanation && (
                    <div className="insights-explanation">{t.explanation}</div>
                  )}
                </div>
              ))}
            </div>
          </>
        )}

        {!loading && blindSpots && (
          <>
            <div className="insights-section">
              <div className="insights-section-title">blind spots</div>

              {blindSpots.uncovered_types.length > 0 && (
                <div className="insights-card">
                  <div className="insights-subsection-title">types with no core memories</div>
                  <div className="blindspot-tag-list">
                    {blindSpots.uncovered_types.map((t) => (
                      <span key={t.type} className="blindspot-type-badge">
                        {t.type} <span className="blindspot-count">({t.total})</span>
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {blindSpots.sparse_spaces.length > 0 && (
                <div className="insights-card">
                  <div className="insights-subsection-title">sparse spaces</div>
                  {blindSpots.sparse_spaces.map((s, i) => (
                    <div key={i} className="blindspot-row">
                      <span className="blindspot-space">{s.space || "(global)"}</span>
                      <span className="blindspot-meta">{s.node_count} node{s.node_count !== 1 ? "s" : ""} &middot; {s.edge_count} edge{s.edge_count !== 1 ? "s" : ""}</span>
                    </div>
                  ))}
                </div>
              )}

              {blindSpots.isolated_nodes.length > 0 && (
                <div className="insights-card">
                  <div className="insights-subsection-title">unconnected nodes</div>
                  {blindSpots.isolated_nodes.map((n) => (
                    <div
                      key={n.id}
                      className="blindspot-row clickable"
                      onClick={() => onNodeClick(n.id)}
                    >
                      <span className="blindspot-title">{n.title || n.id.slice(0, 12) + "…"}</span>
                      <span className="blindspot-meta">{n.type} &middot; {n.tier}</span>
                    </div>
                  ))}
                </div>
              )}

              {blindSpots.uncovered_types.length === 0 &&
                blindSpots.sparse_spaces.length === 0 &&
                blindSpots.isolated_nodes.length === 0 && (
                  <div className="insights-empty">no blind spots detected</div>
                )}
            </div>
          </>
        )}

        {!loading && recallDebug && (
          <div className="insights-section">
            <div className="insights-section-title">recall debug</div>
            {recallDebug.entries.length === 0 && (
              <div className="insights-empty">no whisper log entries yet</div>
            )}
            {recallDebug.entries.length > 0 && (
              <div className="insights-card recall-debug-card">
                <table className="recall-debug-table">
                  <thead>
                    <tr>
                      <th>time</th>
                      <th>node</th>
                      <th>score</th>
                      <th>injected</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recallDebug.entries.map((e) => (
                      <tr
                        key={e.id}
                        className={`recall-row ${e.was_injected ? "recall-injected" : "recall-skipped"}`}
                        onClick={() => onNodeClick(e.node_id)}
                        title={e.prompt_text ?? undefined}
                      >
                        <td className="recall-time">{formatDateTime(e.logged_at)}</td>
                        <td className="recall-node">
                          {e.node_title || e.node_id.slice(0, 10) + "…"}
                          {e.node_type && <span className="recall-type"> {e.node_type}</span>}
                        </td>
                        <td className="recall-score">{e.score.toFixed(2)}</td>
                        <td className="recall-injected-cell">{e.was_injected ? "✓" : "–"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
