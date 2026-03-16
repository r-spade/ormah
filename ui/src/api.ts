import type { GraphData, InsightsData, MemoryNode, NodeDetail, Proposal } from "./types";

const BASE = "";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`GET ${path}: ${res.status}`);
  return res.json();
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`POST ${path}: ${res.status}`);
  return res.json();
}

export function fetchGraph(): Promise<GraphData> {
  return get("/ui/graph");
}

export function fetchNodeDetail(nodeId: string): Promise<NodeDetail> {
  return get(`/ui/graph/node/${nodeId}`);
}

export function searchNodes(query: string): Promise<MemoryNode[]> {
  return get(`/ui/search?q=${encodeURIComponent(query)}&limit=20`);
}

export function fetchProposals(): Promise<Proposal[]> {
  return get("/agent/proposals");
}

export function resolveProposal(
  proposalId: string,
  action: "approved" | "rejected"
): Promise<{ status: string; proposal_id: string; merge_result: string | null }> {
  return post(`/agent/proposals/${proposalId}`, { action });
}

export function fetchInsights(): Promise<InsightsData> {
  return get("/ui/insights");
}

export function fetchStats(): Promise<Record<string, unknown>> {
  return get("/admin/stats");
}

export interface AdminTask {
  id: string;
  name: string;
  next_run: string | null;
  description: string | null;
  paused: boolean;
}

export function fetchAdminTasks(): Promise<{ tasks: AdminTask[] }> {
  return get("/admin/tasks");
}

export function runAdminTask(taskId: string): Promise<{ status: string; task: string }> {
  return post(`/admin/tasks/${taskId}/run`);
}

export function runAllTasks(): Promise<{ status: string; results: Record<string, string> }> {
  return post("/admin/tasks/run-all");
}

export function pauseTask(taskId: string): Promise<{ status: string; task: string }> {
  return post(`/admin/tasks/${taskId}/pause`);
}

export function resumeTask(taskId: string): Promise<{ status: string; task: string }> {
  return post(`/admin/tasks/${taskId}/resume`);
}

export function pauseAllTasks(): Promise<{ status: string }> {
  return post("/admin/tasks/pause-all");
}

export function resumeAllTasks(): Promise<{ status: string }> {
  return post("/admin/tasks/resume-all");
}
