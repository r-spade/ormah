import type { Filters } from "../App";
import type { Edge, EdgeType, MemoryNode, NodeType, Tier } from "../types";

interface Props {
  open: boolean;
  filters: Filters;
  allSpaces: string[];
  nodes: MemoryNode[];
  edges: Edge[];
  onToggle: <K extends keyof Filters>(key: K, value: string) => void;
  clusterBySpace: boolean;
  onToggleCluster: () => void;
}

const TIERS: Tier[] = ["core", "working", "archival"];
const TYPES: NodeType[] = [
  "fact", "decision", "preference", "event", "person",
  "project", "concept", "procedure", "goal", "observation",
];
const EDGE_TYPES: EdgeType[] = [
  "supports", "contradicts", "part_of", "defines",
  "evolved_from", "depends_on", "derived_from",
  "preceded_by", "caused_by", "instance_of", "related_to",
];

export default function FilterDrawer({
  open,
  filters,
  allSpaces,
  nodes,
  edges,
  onToggle,
  clusterBySpace,
  onToggleCluster,
}: Props) {
  const countByTier = (t: Tier) => nodes.filter((n) => n.tier === t).length;
  const countByType = (t: NodeType) => nodes.filter((n) => n.type === t).length;
  const countBySpace = (s: string) =>
    nodes.filter((n) => n.space === s).length;
  const countByEdgeType = (t: EdgeType) => edges.filter((e) => e.edge_type === t).length;

  return (
    <div className={`side-panel filter-drawer ${open ? "open" : ""}`}>
      <div className="filter-section">
        <div className="filter-section-title">layout</div>
        <div className="filter-option" onClick={onToggleCluster}>
          <div className={`filter-checkbox ${clusterBySpace ? "checked" : ""}`} />
          <span>group by space</span>
        </div>
      </div>
      <div className="filter-section">
        <div className="filter-section-title">tier</div>
        {TIERS.map((t) => (
          <div
            key={t}
            className="filter-option"
            onClick={() => onToggle("tiers", t)}
          >
            <div
              className={`filter-checkbox ${
                filters.tiers.has(t) ? "checked" : ""
              }`}
            />
            <span>{t}</span>
            <span className="filter-count">{countByTier(t)}</span>
          </div>
        ))}
      </div>
      <div className="filter-section">
        <div className="filter-section-title">type</div>
        {TYPES.filter((t) => countByType(t) > 0).map((t) => (
          <div
            key={t}
            className="filter-option"
            onClick={() => onToggle("types", t)}
          >
            <div
              className={`filter-checkbox ${
                filters.types.has(t) ? "checked" : ""
              }`}
            />
            <span>{t}</span>
            <span className="filter-count">{countByType(t)}</span>
          </div>
        ))}
      </div>
      {allSpaces.length > 0 && (
        <div className="filter-section">
          <div className="filter-section-title">space</div>
          {allSpaces.map((s) => (
            <div
              key={s}
              className="filter-option"
              onClick={() => onToggle("spaces", s)}
            >
              <div
                className={`filter-checkbox ${
                  filters.spaces.has(s) ? "checked" : ""
                }`}
              />
              <span>{s}</span>
              <span className="filter-count">{countBySpace(s)}</span>
            </div>
          ))}
        </div>
      )}
      <div className="filter-section">
        <div className="filter-section-title">edge types</div>
        {EDGE_TYPES.filter((t) => countByEdgeType(t) > 0).map((t) => (
          <div
            key={t}
            className="filter-option"
            onClick={() => onToggle("edgeTypes", t)}
          >
            <div
              className={`filter-checkbox ${
                filters.edgeTypes.has(t) ? "checked" : ""
              }`}
            />
            <span>{t.replace(/_/g, " ")}</span>
            <span className="filter-count">{countByEdgeType(t)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
