import os
from pathlib import Path
from uuid import uuid4

import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from app.agent.graph import create_enhanced_financial_agent
from app.config import (
    CHROMA_DB_DIR,
    OPENAI_API_KEY,
    TAVILY_API_KEY,
)


pytestmark = [pytest.mark.integration, pytest.mark.live]

REQUIRED_TOOL_NAMES = {
    "get_stock_price",
    "get_stock_history",
    "search_financial_news",
    "analyze_sentiment",
    "query_private_database",
}


def _live_tests_enabled() -> bool:
    return os.getenv("RUN_LIVE_AGENT_TESTS", "").lower() in {
        "1",
        "true",
        "yes",
    }


def _require_live_environment() -> None:
    if not _live_tests_enabled():
        pytest.skip("Set RUN_LIVE_AGENT_TESTS=true to run live agent tests")

    if not Path(CHROMA_DB_DIR).expanduser().exists():
        pytest.skip(f"Chroma database is unavailable at {CHROMA_DB_DIR}")

    if not OPENAI_API_KEY:
        pytest.skip("OPENAI_API_KEY is required for live agent tests")

    if not TAVILY_API_KEY:
        pytest.skip("TAVILY_API_KEY is required for live agent tests")


def _run_live_query(query: str) -> tuple[str, set[str]]:
    _require_live_environment()
    agent = create_enhanced_financial_agent(with_rag=True, with_memory=True)
    result = agent.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"configurable": {"thread_id": f"live-{uuid4().hex}"}},
    )
    messages = result["messages"]
    tool_names = {
        message.name
        for message in messages
        if isinstance(message, ToolMessage) and message.name
    }
    answer = messages[-1].content

    assert isinstance(answer, str) and answer.strip()
    return answer, tool_names


def test_nvidia_analysis_uses_private_ai_research():
    answer, tool_names = _run_live_query(
        "Provide a comprehensive investment analysis for NVIDIA (NVDA), "
        "including its AI research initiatives."
    )

    assert "query_private_database" in tool_names
    assert {"get_stock_price", "get_stock_history"} <= tool_names
    assert "NVIDIA" in answer or "NVDA" in answer


def test_microsoft_analysis_uses_synergistic_tools():
    _, tool_names = _run_live_query(
        "Analyze Microsoft's position in the AI market. Include recent news "
        "sentiment and its strategic AI initiatives."
    )

    assert REQUIRED_TOOL_NAMES <= tool_names


@pytest.mark.slow
def test_multi_company_ai_investment_ranking():
    answer, tool_names = _run_live_query(
        "Rank MSFT, GOOGL, NVDA, AMZN, and IBM by AI-focused investment "
        "attractiveness. Use all available tools for every company and provide "
        "financial, AI innovation, and sentiment scores plus Buy/Hold/Sell "
        "views with confidence."
    )

    assert REQUIRED_TOOL_NAMES <= tool_names
    for company in ("MSFT", "GOOGL", "NVDA", "AMZN", "IBM"):
        assert company in answer
