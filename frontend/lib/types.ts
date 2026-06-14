export type RunStatus = "queued" | "running" | "completed" | "failed" | "cancelled";

export type AgentStage =
  | "planning"
  | "stock_price"
  | "stock_history"
  | "news_search"
  | "sentiment"
  | "private_rag"
  | "synthesis";

export interface ThreadRecord {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface RunRecord {
  id: string;
  thread_id: string;
  query: string;
  with_rag: boolean;
  status: RunStatus;
  report: string | null;
  error: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export interface ThreadDetail extends ThreadRecord {
  runs: RunRecord[];
}

export interface RunEvent {
  id: string;
  run_id: string;
  sequence: number;
  type: string;
  timestamp: string;
  stage: AgentStage | null;
  payload: Record<string, unknown>;
}
