export type NodeType =
  | "fact"
  | "decision"
  | "preference"
  | "event"
  | "person"
  | "project"
  | "concept"
  | "procedure"
  | "goal"
  | "observation";

export type Tier = "core" | "working" | "archival";

export type EdgeType =
  | "related_to"
  | "supports"
  | "contradicts"
  | "part_of"
  | "derived_from"
  | "depends_on"
  | "defines"
  | "evolved_from";

export interface MemoryNode {
  id: string;
  type: NodeType;
  tier: Tier;
  source: string;
  space: string | null;
  title: string | null;
  content: string;
  created: string;
  updated: string;
  last_accessed: string;
  access_count: number;
  file_path: string;
  file_hash: string;
}

export interface Edge {
  source_id: string;
  target_id: string;
  edge_type: EdgeType;
  weight: number;
  created: string;
}

export interface GraphData {
  nodes: MemoryNode[];
  edges: Edge[];
  user_node_id: string | null;
}

export interface NodeDetail {
  node: MemoryNode;
  edges: Edge[];
  neighbors: MemoryNode[];
  tags: string[];
}

export type ProposalType = "merge" | "conflict" | "decay";
export type ProposalStatus = "pending" | "approved" | "rejected";

export interface ProposalNode {
  id: string;
  title: string | null;
  content: string;
  type: NodeType;
  tier: Tier;
  space: string | null;
  created: string;
}

export interface Proposal {
  id: string;
  type: ProposalType;
  status: ProposalStatus;
  source_nodes: string;
  proposed_action: string;
  action_summary: string;
  merged_preview: string | null;
  nodes: ProposalNode[];
  reason: string | null;
  created: string;
  resolved: string | null;
}

export interface InsightNode {
  id: string;
  title: string | null;
  type: NodeType;
  tier: Tier;
  content: string;
  created: string;
}

export interface Evolution {
  newer: InsightNode;
  older: InsightNode;
  explanation: string;
}

export interface Tension {
  node_a: InsightNode;
  node_b: InsightNode;
  explanation: string;
}

export interface InsightsData {
  evolutions: Evolution[];
  tensions: Tension[];
}

export interface SearchResult {
  node: MemoryNode;
  score: number;
  source: string;
  formatted: string;
}

// Blind spots
export interface IsolatedNode {
  id: string;
  title: string | null;
  type: NodeType;
  tier: Tier;
  space: string | null;
  created: string;
}

export interface SparseSpace {
  space: string;
  node_count: number;
  edge_count: number;
}

export interface UncoveredType {
  type: NodeType;
  total: number;
  core_count: number;
}

export interface BlindSpotsData {
  isolated_nodes: IsolatedNode[];
  sparse_spaces: SparseSpace[];
  uncovered_types: UncoveredType[];
}

// Recall debug
export interface RecallDebugEntry {
  id: number;
  session_id: string;
  space: string | null;
  prompt_text: string | null;
  node_id: string;
  node_title: string | null;
  node_type: NodeType | null;
  score: number;
  was_injected: number;
  logged_at: string;
}

export interface RecallDebugData {
  entries: RecallDebugEntry[];
}
