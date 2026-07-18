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
LLM_TOKENS_TOTAL = Counter(
    "financial_agent_llm_tokens_total",
    "LLM tokens consumed, by model and direction.",
    ("model", "direction"),
)
LLM_COST_USD_TOTAL = Counter(
    "financial_agent_llm_cost_usd_total",
    "Estimated LLM cost in USD, by model. Approximate -- see app/providers/pricing.py.",
    ("model",),
)
ROUTING_DECISIONS_TOTAL = Counter(
    "financial_agent_routing_decisions_total",
    "Routing decisions by selected model tier.",
    ("model_tier",),
)
TOOL_CALL_DURATION = Histogram(
    "financial_agent_tool_call_duration_seconds",
    "Tool call duration, by tool.",
    ("tool",),
)
