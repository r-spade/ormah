import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { fetchGraph, fetchNodeDetail } from "./api";
import type { Edge, GraphData, MemoryNode, NodeDetail, Tier, NodeType, EdgeType } from "./types";
import GraphView from "./components/GraphView";
import TopBar from "./components/TopBar";
import NodeDetailPanel from "./components/NodeDetail";
import FilterDrawer from "./components/FilterDrawer";
import ReviewQueue from "./components/ReviewQueue";
import InsightsPanel from "./components/InsightsPanel";
import AdminPanel from "./components/AdminPanel";
import ToastContainer from "./components/Toast";
import type { ToastData } from "./components/Toast";
import useKeyboardShortcuts from "./hooks/useKeyboardShortcuts";

export interface Filters {
  tiers: Set<Tier>;
  types: Set<NodeType>;
  spaces: Set<string>;
  edgeTypes: Set<EdgeType>;
  clusterBySpace: boolean;
}

const ALL_TIERS: Tier[] = ["core", "working", "archival"];
const ALL_TYPES: NodeType[] = [
  "fact", "decision", "preference", "event", "person",
  "project", "concept", "procedure", "goal", "observation",
];
const ALL_EDGE_TYPES: EdgeType[] = [
  "supports", "contradicts", "part_of", "defines",
  "evolved_from", "depends_on", "derived_from",
  "preceded_by", "caused_by", "instance_of", "related_to",
];
const DEFAULT_EDGE_TYPES = new Set<EdgeType>(ALL_EDGE_TYPES);

type PanelId = "filter" | "review" | "insights" | "admin" | null;

export default function App() {
  const [graph, setGraph] = useState<GraphData | null>(null);
  const [selectedDetail, setSelectedDetail] = useState<NodeDetail | null>(null);
  const [activePanel, setActivePanel] = useState<PanelId>(null);
  const [focusNodeId, setFocusNodeId] = useState<string | null>(null);
  const [filters, setFilters] = useState<Filters>({
    tiers: new Set(ALL_TIERS),
    types: new Set(ALL_TYPES),
    spaces: new Set<string>(),
    edgeTypes: new Set(DEFAULT_EDGE_TYPES),
    clusterBySpace: true,
  });
  const [allSpaces, setAllSpaces] = useState<string[]>([]);
  const [userNodeId, setUserNodeId] = useState<string | null>(null);
  const [toasts, setToasts] = useState<ToastData[]>([]);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const graphViewRef = useRef<{
    focusNode: (id: string) => void;
    highlightNode: (id: string) => void;
    highlightNodes: (ids: string[]) => void;
    clearHighlight: () => void;
  }>(null);

  const addToast = useCallback((message: string, type: ToastData["type"] = "info") => {
    const id = Date.now();
    setToasts(t => [...t, { id, message, type }]);
    setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), 3000);
  }, []);

  const togglePanel = useCallback((id: PanelId) => {
    setActivePanel((p) => (p === id ? null : id));
  }, []);

  useKeyboardShortcuts({
    onTogglePanel: togglePanel as (id: "filter" | "review" | "insights" | "admin") => void,
    onClosePanel: useCallback(() => setActivePanel(null), []),
    onCloseDetail: useCallback(() => setSelectedDetail(null), []),
    onFocusSearch: useCallback(() => searchInputRef.current?.focus(), []),
    activePanel: activePanel as "filter" | "review" | "insights" | "admin" | null,
    hasDetail: selectedDetail !== null,
  });

  useEffect(() => {
    fetchGraph().then((data) => {
      setGraph(data);
      setUserNodeId(data.user_node_id);
      const spaces = new Set<string>();
      data.nodes.forEach((n) => {
        if (n.space) spaces.add(n.space);
      });
      const spaceList = Array.from(spaces).sort();
      setAllSpaces(spaceList);
      setFilters((f) => ({ ...f, spaces: new Set(spaceList) }));
    });
  }, []);

  const handleNodeSelect = useCallback(async (nodeId: string) => {
    const detail = await fetchNodeDetail(nodeId);
    setSelectedDetail(detail);
  }, []);

  const handleSearchSelect = useCallback(
    (nodeId: string) => {
      setFocusNodeId(nodeId);
      graphViewRef.current?.focusNode(nodeId);
      handleNodeSelect(nodeId);
    },
    [handleNodeSelect]
  );

  const handleConnectionClick = useCallback(
    (nodeId: string) => {
      setFocusNodeId(nodeId);
      graphViewRef.current?.focusNode(nodeId);
      handleNodeSelect(nodeId);
    },
    [handleNodeSelect]
  );

  const filteredNodes = useMemo(() => {
    if (!graph) return [];
    return graph.nodes.filter(
      (n) =>
        filters.tiers.has(n.tier) &&
        filters.types.has(n.type) &&
        (filters.spaces.size === 0 ||
          !n.space ||
          filters.spaces.has(n.space))
    );
  }, [graph, filters]);

  const filteredEdges = useMemo(() => {
    if (!graph) return [];
    return graph.edges.filter((e) => filters.edgeTypes.has(e.edge_type));
  }, [graph, filters.edgeTypes]);

  const toggleFilter = useCallback(
    <K extends keyof Filters>(key: K, value: string) => {
      setFilters((f) => {
        const next = new Set(f[key] as Set<string>);
        if (next.has(value)) next.delete(value);
        else next.add(value);
        return { ...f, [key]: next };
      });
    },
    []
  );

  const toggleCluster = useCallback(() => {
    setFilters((f) => ({ ...f, clusterBySpace: !f.clusterBySpace }));
  }, []);

  if (!graph) {
    return <div className="loading">ormahing...</div>;
  }

  return (
    <>
      <TopBar
        nodeCount={filteredNodes.length}
        activePanel={activePanel as "filter" | "review" | "insights" | "admin" | null}
        onTogglePanel={togglePanel as (id: "filter" | "review" | "insights" | "admin") => void}
        onSearchSelect={handleSearchSelect}
        onSearchHover={(id) => graphViewRef.current?.highlightNode(id)}
        onSearchHoverEnd={() => graphViewRef.current?.clearHighlight()}
        searchInputRef={searchInputRef}
      />
      <div className="graph-container">
        {graph && (
          <GraphView
            ref={graphViewRef}
            nodes={filteredNodes}
            edges={filteredEdges}
            onNodeSelect={handleNodeSelect}
            focusNodeId={focusNodeId}
            userNodeId={userNodeId}
            clusterBySpace={filters.clusterBySpace}
          />
        )}
      </div>
      <NodeDetailPanel
        detail={selectedDetail}
        onClose={() => setSelectedDetail(null)}
        onConnectionClick={handleConnectionClick}
      />
      <FilterDrawer
        open={activePanel === "filter"}
        filters={filters}
        allSpaces={allSpaces}
        nodes={graph.nodes}
        edges={graph.edges}
        onToggle={toggleFilter}
        clusterBySpace={filters.clusterBySpace}
        onToggleCluster={toggleCluster}
      />
      <InsightsPanel
        open={activePanel === "insights"}
        onClose={() => setActivePanel(null)}
        onNodeClick={handleSearchSelect}
        onPairHover={(ids) => graphViewRef.current?.highlightNodes(ids)}
        onPairHoverEnd={() => graphViewRef.current?.clearHighlight()}
      />
      <ReviewQueue
        open={activePanel === "review"}
        onClose={() => setActivePanel(null)}
        onToast={addToast}
        onNodeClick={handleSearchSelect}
        onPairHover={(ids) => graphViewRef.current?.highlightNodes(ids)}
        onPairHoverEnd={() => graphViewRef.current?.clearHighlight()}
      />
      <AdminPanel
        open={activePanel === "admin"}
        onClose={() => setActivePanel(null)}
        onToast={addToast}
      />
      <ToastContainer toasts={toasts} />
    </>
  );
}
