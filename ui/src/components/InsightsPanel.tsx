import { useEffect, useState } from "react";
import { fetchInsights } from "../api";
import type { InsightsData, InsightNode } from "../types";

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

export default function InsightsPanel({
  open,
  onClose,
  onNodeClick,
  onPairHover,
  onPairHoverEnd,
}: Props) {
  const [data, setData] = useState<InsightsData | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!open) return;
    setLoading(true);
    fetchInsights()
      .then(setData)
      .catch(() => setData({ evolutions: [], tensions: [] }))
      .finally(() => setLoading(false));
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
      </div>
    </div>
  );
}
