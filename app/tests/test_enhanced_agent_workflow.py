from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from app.agent import graph as graph_module


class ScriptedModel:
    def __init__(self):
        self.call_count = 0

    def invoke(self, messages):
        self.call_count += 1

        if self.call_count == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_stock_history_test",
                        "args": {"ticker": "MSFT", "period": "3y"},
                        "id": "history-call",
                        "type": "tool_call",
                    },
                    {
                        "name": "search_financial_news_test",
                        "args": {"query": "Microsoft AI news"},
                        "id": "news-call",
                        "type": "tool_call",
                    },
                    {
                        "name": "analyze_sentiment_test",
                        "args": {"text": "Microsoft expands AI investment"},
                        "id": "sentiment-call",
                        "type": "tool_call",
                    },
                    {
                        "name": "query_private_database_test",
                        "args": {"query": "Microsoft AI initiatives"},
                        "id": "rag-call",
                        "type": "tool_call",
                    },
                ],
            )

        return AIMessage(
            content=(
                "Microsoft has positive AI momentum supported by market, "
                "news, sentiment, and private-report evidence."
            )
        )


@tool
def get_stock_history_test(ticker: str, period: str) -> dict:
    """Return deterministic historical stock data."""
    return {"ticker": ticker, "period": period, "return_pct": 25.0}


@tool
def search_financial_news_test(query: str) -> list[dict]:
    """Return deterministic financial news."""
    return [{"title": "Microsoft expands AI", "url": "https://example.com/ai"}]


@tool
def analyze_sentiment_test(text: str) -> dict:
    """Return deterministic sentiment data."""
    return {"sentiment": "positive", "score": 0.8, "text": text}


@tool
def query_private_database_test(query: str) -> str:
    """Return deterministic private analyst-report evidence."""
    return "Microsoft is investing in Copilot and Azure AI."


def test_enhanced_agent_combines_financial_news_sentiment_and_rag(monkeypatch):
    model = ScriptedModel()
    monkeypatch.setattr(graph_module, "build_model", lambda tools, **kwargs: model)

    agent = graph_module.create_financial_agent(
        agent_type="full",
        with_memory=True,
        tools=[
            get_stock_history_test,
            search_financial_news_test,
            analyze_sentiment_test,
            query_private_database_test,
        ],
    )
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Analyze Microsoft's financial and AI position."
                )
            ]
        },
        config={"configurable": {"thread_id": "deterministic-synergy-test"}},
    )

    tool_names = {
        message.name
        for message in result["messages"]
        if isinstance(message, ToolMessage)
    }

    assert tool_names == {
        "get_stock_history_test",
        "search_financial_news_test",
        "analyze_sentiment_test",
        "query_private_database_test",
    }
    assert "private-report evidence" in result["messages"][-1].content
    assert model.call_count == 2
