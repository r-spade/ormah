import { useCallback, useEffect, useState } from "react";
import { fetchProposals, resolveProposal } from "../api";
import type { Proposal } from "../types";

interface Props {
  open: boolean;
  onClose: () => void;
  onToast: (message: string, type: "success" | "error" | "info") => void;
  onNodeClick: (nodeId: string) => void;
  onPairHover: (ids: string[]) => void;
  onPairHoverEnd: () => void;
}

export default function ReviewQueue({
  open,
  onClose,
  onToast,
  onNodeClick,
  onPairHover,
  onPairHoverEnd,
}: Props) {
  const [proposals, setProposals] = useState<Proposal[]>([]);
  const [loading, setLoading] = useState(true);
  const [mergeResults, setMergeResults] = useState<Record<string, string>>({});

  useEffect(() => {
    if (!open) return;
    setLoading(true);
    fetchProposals()
      .then(setProposals)
      .catch(() => setProposals([]))
      .finally(() => setLoading(false));
  }, [open]);

  const handleAction = useCallback(
    async (id: string, action: "approved" | "rejected") => {
      try {
        const result = await resolveProposal(id, action);
        onToast(`Proposal ${action}`, "success");
        if (result.merge_result) {
          setMergeResults((prev) => ({ ...prev, [id]: result.merge_result! }));
          setTimeout(() => {
            setMergeResults((prev) => {
              const next = { ...prev };
              delete next[id];
              return next;
            });
            setProposals((p) => p.filter((x) => x.id !== id));
          }, 3000);
        } else {
          setProposals((p) => p.filter((x) => x.id !== id));
        }
      } catch {
        onToast(`Failed to ${action.replace("ed", "")} proposal`, "error");
      }
    },
    [onToast]
  );

  return (
    <div className={`side-panel review-panel ${open ? "open" : ""}`}>
      <div className="side-panel-header">
        <div className="review-title">review queue</div>
        <button className="node-detail-close" onClick={onClose}>
          &times;
        </button>
      </div>
      {loading ? (
        <div className="review-empty">loading...</div>
      ) : proposals.length === 0 ? (
        <div className="review-empty">no pending proposals</div>
      ) : (
        proposals.map((p) => {
          const nodeIds = p.nodes.map((n) => n.id);
          return (
            <div
              key={p.id}
              className="review-card"
              onMouseEnter={() => onPairHover(nodeIds)}
              onMouseLeave={onPairHoverEnd}
            >
              <div className="review-card-type">{p.type}</div>
              <div className="review-card-action">{p.action_summary}</div>

              {/* Source nodes */}
              {p.nodes.length > 0 && (
                <div className="review-source-nodes">
                  <div className="review-source-label">nodes involved</div>
                  {p.nodes.map((node) => (
                    <div
                      key={node.id}
                      className="review-source-node clickable"
                      onClick={() => onNodeClick(node.id)}
                    >
                      <div className="review-source-node-title">
                        {node.title || node.content.slice(0, 50)}
                      </div>
                      <div className="review-source-node-meta">
                        {node.type} &middot; {node.tier}
                        {node.space && ` · ${node.space}`}
                      </div>
                      <div className="review-source-node-content">
                        {node.content.length > 120
                          ? node.content.slice(0, 120) + "..."
                          : node.content}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* Merged preview */}
              {p.merged_preview && (
                <div className="review-merged-preview">
                  <div className="review-source-label">merged result</div>
                  <div className="review-merged-content">
                    {p.merged_preview}
                  </div>
                </div>
              )}

              {p.reason && (
                <div className="review-card-reason">{p.reason}</div>
              )}

              {mergeResults[p.id] ? (
                <div className="review-card-result">{mergeResults[p.id]}</div>
              ) : (
                <div className="review-card-actions">
                  <button
                    className="review-btn approve"
                    onClick={() => handleAction(p.id, "approved")}
                  >
                    approve
                  </button>
                  <button
                    className="review-btn reject"
                    onClick={() => handleAction(p.id, "rejected")}
                  >
                    reject
                  </button>
                </div>
              )}
            </div>
          );
        })
      )}
    </div>
  );
}
