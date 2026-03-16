import type { NodeDetail } from "../types";

function timeAgo(iso: string): string {
  const sec = Math.floor((Date.now() - Date.parse(iso)) / 1000);
  if (sec < 60) return "just now";
  if (sec < 3600) return `${Math.floor(sec / 60)}m ago`;
  if (sec < 86400) return `${Math.floor(sec / 3600)}h ago`;
  return `${Math.floor(sec / 86400)}d ago`;
}

interface Props {
  detail: NodeDetail | null;
  onClose: () => void;
  onConnectionClick: (nodeId: string) => void;
}

export default function NodeDetailPanel({
  detail,
  onClose,
  onConnectionClick,
}: Props) {
  const open = detail !== null;

  return (
    <div className={`node-detail ${open ? "open" : ""} ${detail ? `tier-${detail.node.tier}` : ""}`}>
      {detail && (
        <div className="node-detail-inner">
          <div className="node-detail-header">
            <div className="node-detail-title">
              {detail.node.title || detail.node.content.slice(0, 50)}
              <span className="node-detail-id">
                #{detail.node.id.split("-")[0]}
              </span>
            </div>
            <button className="node-detail-close" onClick={onClose}>
              &times;
            </button>
          </div>
          <div className="node-detail-fields">
            <span className={`badge-tier tier-${detail.node.tier}`}>
              {detail.node.tier}
            </span>
            <div className="field-row">
              <span className="field-label">type</span>
              <span className="field-value">{detail.node.type}</span>
            </div>
            {detail.node.space && (
              <div className="field-row">
                <span className="field-label">space</span>
                <span className="field-value">{detail.node.space}</span>
              </div>
            )}
            {detail.tags.length > 0 && (
              <div className="field-row field-row-tags">
                <span className="field-label">tags</span>
                <div className="tag-list">
                  {detail.tags.map((t) => (
                    <span key={t} className="tag">
                      {t}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
          <div className="node-detail-access">
            accessed {detail.node.access_count}x · last{" "}
            {timeAgo(detail.node.last_accessed)}
          </div>
          <div className="node-detail-content">{detail.node.content}</div>

          {detail.edges.length > 0 && (
            <div className="node-detail-section">
              <div className="node-detail-section-title">
                connections ({detail.edges.length})
              </div>
              <div className="connections-list">
                {detail.edges.map((e) => {
                  const targetId =
                    e.source_id === detail.node.id ? e.target_id : e.source_id;
                  const neighbor = detail.neighbors.find(
                    (n) => n.id === targetId
                  );
                  return (
                    <div
                      key={`${e.source_id}-${e.edge_type}-${e.target_id}`}
                      className="connection-item"
                      onClick={() => onConnectionClick(targetId)}
                    >
                      <span className="connection-edge-type">
                        {e.edge_type}
                      </span>
                      <span>
                        {neighbor?.title ||
                          neighbor?.content.slice(0, 40) ||
                          targetId.split("-")[0]}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
