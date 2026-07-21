import type { RunRecord, ThreadDetail, ThreadRecord } from "./types";

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
export const API_KEY = process.env.NEXT_PUBLIC_API_KEY ?? "";

export function authHeaders(): Record<string, string> {
  return API_KEY ? { Authorization: `Bearer ${API_KEY}` } : {};
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...authHeaders(), ...init?.headers }
  });

  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? `Request failed with status ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export function listThreads(): Promise<ThreadRecord[]> {
  return request("/api/v1/threads");
}

export function createThread(title: string): Promise<ThreadRecord> {
  return request("/api/v1/threads", {
    method: "POST",
    body: JSON.stringify({ title })
  });
}

export function getThread(threadId: string): Promise<ThreadDetail> {
  return request(`/api/v1/threads/${threadId}`);
}

export function createRun(threadId: string, query: string): Promise<RunRecord> {
  return request(`/api/v1/threads/${threadId}/runs`, {
    method: "POST",
    body: JSON.stringify({ query, with_rag: true })
  });
}

export function cancelRun(runId: string): Promise<RunRecord> {
  return request(`/api/v1/runs/${runId}/cancel`, { method: "POST" });
}
