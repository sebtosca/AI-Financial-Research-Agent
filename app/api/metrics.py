from prometheus_client import Counter, Gauge, Histogram


RUNS_TOTAL = Counter(
    "financial_agent_runs_total",
    "Agent runs by terminal status.",
    ("status",),
)
ACTIVE_RUNS = Gauge(
    "financial_agent_active_runs",
    "Agent runs currently executing.",
)
RUN_DURATION = Histogram(
    "financial_agent_run_duration_seconds",
    "End-to-end agent run duration.",
)
TOOL_CALLS = Counter(
    "financial_agent_tool_calls_total",
    "Tool calls observed in agent execution.",
    ("tool", "status"),
)
SSE_CONNECTIONS = Gauge(
    "financial_agent_sse_connections",
    "Current SSE client connections.",
)
