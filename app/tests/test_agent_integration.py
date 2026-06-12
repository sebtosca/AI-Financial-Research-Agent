import pytest
from typing import Dict
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

from app.agent.graph import create_financial_agent
from app.tools import get_stock_history, search_financial_news, analyze_sentiment


def _make_failing_stock_price():
    @tool
    def get_stock_price_failing(ticker: str) -> Dict:
        """Returns current stock price for a given ticker symbol."""
        return {
            "ticker": ticker.upper(),
            "status": "error",
            "error": "API connection timeout - service temporarily unavailable",
        }
    return get_stock_price_failing


@pytest.mark.integration
def test_traditional_agent_does_not_call_tools():
    """Traditional LLM should respond without invoking any tools."""
    agent = create_financial_agent(agent_type="traditional", with_memory=False)
    result = agent.invoke({"messages": [HumanMessage(content="Tell me about Apple stock")]})

    last_message = result["messages"][-1]
    assert isinstance(last_message.content, str) and len(last_message.content) > 0
    tool_calls = getattr(last_message, "tool_calls", [])
    assert not tool_calls, "Traditional agent should not call tools"


@pytest.mark.integration
def test_basic_agent_calls_tools():
    """Basic goal-oriented agent should use tools to gather data."""
    agent = create_financial_agent(agent_type="basic", with_memory=False)
    result = agent.invoke({"messages": [HumanMessage(content="Tell me about Apple stock")]})

    messages = result["messages"]
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) > 0, "Basic agent should call at least one tool"

    last_message = messages[-1]
    assert isinstance(last_message.content, str) and len(last_message.content) > 0


@pytest.mark.integration
def test_full_agent_comprehensive_analysis():
    """Full agent should use multiple tools to produce a comprehensive analysis."""
    agent = create_financial_agent(agent_type="full", with_memory=False)
    query = "Provide a comprehensive investment analysis for Microsoft (MSFT) including 3-year performance and AI research activity"
    result = agent.invoke({"messages": [HumanMessage(content=query)]})

    messages = result["messages"]
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) > 1, "Full agent should use multiple tools"

    last_message = messages[-1]
    assert isinstance(last_message.content, str) and len(last_message.content) > 0


@pytest.mark.integration
def test_full_agent_handles_tool_failure():
    """Agent should recover when get_stock_price fails and continue with alternative tools."""
    failing_tools = [
        _make_failing_stock_price(),
        get_stock_history,
        search_financial_news,
        analyze_sentiment,
    ]
    agent = create_financial_agent(agent_type="full", with_memory=False, tools=failing_tools)
    result = agent.invoke({"messages": [HumanMessage(content="Analyze Apple stock (AAPL)")]})

    messages = result["messages"]
    last_message = messages[-1]
    assert isinstance(last_message.content, str) and len(last_message.content) > 0

    # Agent should have continued with other tools despite the failure
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) > 0, "Agent should attempt tool calls even when one fails"
