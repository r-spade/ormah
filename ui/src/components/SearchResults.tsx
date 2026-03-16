import type { MemoryNode } from "../types";

interface Props {
  results: MemoryNode[];
  selectedIndex: number;
  onSelect: (nodeId: string) => void;
  onHover: (nodeId: string) => void;
  onHoverEnd: () => void;
}

export default function SearchResults({ results, selectedIndex, onSelect, onHover, onHoverEnd }: Props) {
  return (
    <div className="search-results" onMouseLeave={onHoverEnd}>
      {results.map((node, i) => (
        <div
          key={node.id}
          className={`search-result-item ${i === selectedIndex ? "selected" : ""}`}
          onMouseDown={() => onSelect(node.id)}
          onMouseEnter={() => onHover(node.id)}
        >
          <div className="search-result-title">
            <span className={`search-result-type ${node.tier}`}>
              {node.type}
            </span>
            {node.title || node.content.slice(0, 40)}
          </div>
          <div className="search-result-meta">
            {node.tier} &middot; {node.space || "—"} &middot;{" "}
            {node.id.split("-")[0]}
          </div>
        </div>
      ))}
    </div>
  );
}
