"use client";

import {
  Activity,
  BarChart3,
  BookOpenText,
  Check,
  ChevronRight,
  CircleStop,
  Clock3,
  Database,
  FileText,
  LineChart,
  Menu,
  MessageSquareText,
  Newspaper,
  Plus,
  Search,
  Send,
  ShieldCheck,
  Sparkles,
  TrendingUp,
  X
} from "lucide-react";
import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import ReactMarkdown from "react-markdown";

import { API_URL, cancelRun, createRun, createThread, getThread, listThreads } from "@/lib/api";
import type { AgentStage, RunEvent, RunStatus, ThreadRecord } from "@/lib/types";

const suggestions = [
  "Analyze NVIDIA's investment outlook and AI research initiatives",
  "Compare Microsoft and Google across AI strategy and market sentiment",
  "Assess Amazon's three-year performance and current AI opportunities"
];

const stageMeta: Record<AgentStage, { label: string; icon: typeof Activity }> = {
  planning: { label: "Planning research", icon: Sparkles },
  stock_price: { label: "Current market data", icon: TrendingUp },
  stock_history: { label: "Historical performance", icon: LineChart },
  news_search: { label: "Financial news", icon: Newspaper },
  sentiment: { label: "Market sentiment", icon: BarChart3 },
  private_rag: { label: "Private analyst reports", icon: Database },
  synthesis: { label: "Synthesizing report", icon: FileText }
};

function statusLabel(status: RunStatus) {
  return status.charAt(0).toUpperCase() + status.slice(1);
}

const subscribeToHydration = () => () => undefined;

export default function WorkspacePage() {
  const [threads, setThreads] = useState<ThreadRecord[]>([]);
  const [query, setQuery] = useState("");
  const [activeThread, setActiveThread] = useState<ThreadRecord | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [status, setStatus] = useState<RunStatus>("completed");
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [report, setReport] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const hydrated = useSyncExternalStore(
    subscribeToHydration,
    () => true,
    () => false
  );
  const eventSourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    listThreads().then(setThreads).catch(() => setThreads([]));
    return () => eventSourceRef.current?.close();
  }, []);

  const activeStages = useMemo(() => {
    const latest = new Map<AgentStage, RunEvent>();
    for (const event of events) {
      if (event.stage) latest.set(event.stage, event);
    }
    return [...latest.values()];
  }, [events]);

  async function startResearch(input?: string) {
    const researchQuery = (input ?? query).trim();
    if (!researchQuery || status === "running" || status === "queued") return;

    setError(null);
    setReport("");
    setEvents([]);
    setStatus("queued");
    setQuery(researchQuery);

    try {
      const thread = await createThread(researchQuery.slice(0, 70));
      setActiveThread(thread);
      setThreads((current) => [thread, ...current]);
      const run = await createRun(thread.id, researchQuery);
      setRunId(run.id);
      connectToRun(run.id);
    } catch (cause) {
      setStatus("failed");
      setError(cause instanceof Error ? cause.message : "Unable to start research");
    }
  }

  async function openThread(thread: ThreadRecord) {
    setError(null);
    setActiveThread(thread);
    setSidebarOpen(false);

    try {
      const detail = await getThread(thread.id);
      const latestRun = detail.runs.at(-1);
      setEvents([]);
      setRunId(latestRun?.id ?? null);
      setQuery(latestRun?.query ?? "");
      setReport(latestRun?.report ?? "");
      setStatus(latestRun?.status ?? "completed");
      setError(latestRun?.error ?? null);

      if (latestRun?.status === "queued" || latestRun?.status === "running") {
        connectToRun(latestRun.id);
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Unable to load research");
    }
  }

  function connectToRun(id: string) {
    eventSourceRef.current?.close();
    const source = new EventSource(`${API_URL}/api/v1/runs/${id}/events`);
    eventSourceRef.current = source;

    source.onmessage = (message) => handleEvent(JSON.parse(message.data) as RunEvent);
    const eventNames = [
      "run.created",
      "run.started",
      "tool.started",
      "tool.completed",
      "report.delta",
      "run.completed",
      "run.failed",
      "run.cancelled"
    ];
    eventNames.forEach((name) => {
      source.addEventListener(name, (message) => {
        handleEvent(JSON.parse((message as MessageEvent).data) as RunEvent);
      });
    });
    source.onerror = () => {
      if (status === "running") setError("Live updates interrupted. Reconnecting...");
    };
  }

  function handleEvent(event: RunEvent) {
    if (event.type !== "heartbeat") {
      setEvents((current) =>
        current.some((item) => item.id === event.id) ? current : [...current, event]
      );
    }
    if (event.type === "run.started") setStatus("running");
    if (event.type === "report.delta" || event.type === "run.completed") {
      const content = event.payload.content ?? event.payload.report;
      if (typeof content === "string") setReport(content);
    }
    if (event.type === "run.completed") {
      setStatus("completed");
      eventSourceRef.current?.close();
    }
    if (event.type === "run.failed") {
      setStatus("failed");
      setError(String(event.payload.message ?? "Research failed"));
      eventSourceRef.current?.close();
    }
    if (event.type === "run.cancelled") {
      setStatus("cancelled");
      eventSourceRef.current?.close();
    }
  }

  async function stopResearch() {
    if (!runId) return;
    await cancelRun(runId).catch(() => undefined);
  }

  const isActive = status === "queued" || status === "running";

  return (
    <main className="shell">
      <aside className={`sidebar ${sidebarOpen ? "sidebar-open" : ""}`}>
        <div className="brand-row">
          <div className="brand-mark"><TrendingUp size={20} /></div>
          <div><strong>Argent</strong><span>Research intelligence</span></div>
          <button className="mobile-close" onClick={() => setSidebarOpen(false)}><X /></button>
        </div>
        <button className="new-research" onClick={() => { setQuery(""); setReport(""); setEvents([]); }}>
          <Plus size={17} /> New research
        </button>
        <div className="sidebar-search"><Search size={15} /><input placeholder="Search research" /></div>
        <p className="eyebrow">Recent research</p>
        <nav className="thread-list">
          {threads.length === 0 && <p className="empty-list">Your saved research will appear here.</p>}
          {threads.map((thread) => (
            <button key={thread.id} className={activeThread?.id === thread.id ? "active" : ""} onClick={() => openThread(thread)}>
              <MessageSquareText size={16} /><span>{thread.title}</span><ChevronRight size={14} />
            </button>
          ))}
        </nav>
        <div className="sidebar-footer">
          <div className="avatar">SR</div>
          <div><strong>Senior Researcher</strong><span>Internal workspace</span></div>
          <ShieldCheck size={17} />
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <button className="menu-button" onClick={() => setSidebarOpen(true)}><Menu /></button>
          <div><p className="eyebrow">Financial research agent</p><h1>{activeThread?.title ?? "New company analysis"}</h1></div>
          <div className={`status-pill status-${status}`}><span />{statusLabel(status)}</div>
        </header>

        <div className="research-canvas">
          {!report && !isActive ? (
            <section className="welcome-panel">
              <div className="hero-icon"><Sparkles size={28} /></div>
              <p className="eyebrow">Evidence, not noise</p>
              <h2>What company should we investigate?</h2>
              <p>Combine live market data, financial news, sentiment, and private analyst reports in one transparent research workflow.</p>
              <div className="suggestion-grid">
                {suggestions.map((suggestion) => (
                  <button key={suggestion} onClick={() => startResearch(suggestion)}>
                    <BookOpenText size={18} /><span>{suggestion}</span><ChevronRight size={16} />
                  </button>
                ))}
              </div>
            </section>
          ) : (
            <section className="report-panel">
              <div className="query-card"><span>Your request</span><p>{query}</p></div>
              {isActive && !report && <ReportSkeleton />}
              {report && <article className="report"><ReactMarkdown>{report}</ReactMarkdown></article>}
              {error && <div className="error-banner">{error}</div>}
            </section>
          )}
        </div>

        <div className="composer-wrap">
          <div className="composer">
            <textarea
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  startResearch();
                }
              }}
              placeholder="Ask for a company analysis, comparison, or investment research brief..."
              disabled={isActive}
            />
            {isActive ? (
              <button className="stop-button" onClick={stopResearch}><CircleStop size={18} /> Stop</button>
            ) : (
              <button
                className="send-button"
                onClick={() => startResearch()}
                disabled={hydrated && !query.trim()}
              >
                <Send size={18} />
              </button>
            )}
          </div>
          <p>AI-generated research can be inaccurate. Verify material decisions independently.</p>
        </div>
      </section>

      <aside className="activity-panel">
        <div className="activity-header"><div><p className="eyebrow">Live activity</p><h2>Research process</h2></div><Activity size={20} /></div>
        <div className="run-summary">
          <div className={`pulse-orb ${isActive ? "is-running" : ""}`}><Sparkles size={20} /></div>
          <div><strong>{isActive ? "Agent is researching" : status === "completed" ? "Research ready" : "Waiting to begin"}</strong><span>{isActive ? "Live execution updates" : "Stages and sources appear here"}</span></div>
        </div>
        <div className="timeline">
          {activeStages.length === 0 && <div className="timeline-empty"><Clock3 size={22} /><p>No active run</p><span>Start a query to watch the agent work.</span></div>}
          {activeStages.map((event, index) => {
            const stage: AgentStage = event.stage ?? "planning";
            const meta = stageMeta[stage];
            const Icon = meta.icon;
            const done = event.type.includes("completed") || event.type === "run.completed";
            return (
              <div className="timeline-item" key={`${event.stage}-${index}`}>
                <div className={`timeline-icon ${done ? "done" : ""}`}>{done ? <Check size={15} /> : <Icon size={15} />}</div>
                <div><strong>{meta.label}</strong><span>{event.type === "tool.started" ? "In progress" : done ? "Completed" : "Processing"}</span>
                  {typeof event.payload.tool === "string" && <code>{event.payload.tool}</code>}
                </div>
              </div>
            );
          })}
        </div>
        <div className="operator-card"><ShieldCheck size={18} /><div><strong>Observability connected</strong><span>Metrics and traces are available to operators.</span></div></div>
      </aside>
    </main>
  );
}

function ReportSkeleton() {
  return <div className="report-skeleton"><div className="skeleton-title" /><div /><div /><div className="short" /><div className="skeleton-title second" /><div /><div className="short" /></div>;
}
