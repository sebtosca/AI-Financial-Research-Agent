"""Versioned golden evaluation dataset for the agent/routing eval harness.

`expected_tools` lists the general-purpose tools (excludes
query_private_database, tracked separately via expected_rag_engaged) a
correctly-routed run should call for this query.
"""

from dataclasses import dataclass, field

EVAL_DATASET_VERSION = "v1"


@dataclass(frozen=True)
class EvalCase:
    id: str
    query: str
    expected_tools: tuple[str, ...]
    expected_rag_engaged: bool
    with_rag_requested: bool = True
    reference_answer: str | None = None
    expected_sources: tuple[str, ...] = field(default_factory=tuple)
    category: str = "general"


GOLDEN_EVAL_CASES: tuple[EvalCase, ...] = (
    EvalCase(
        id="price-nvda",
        query="What is the current price of NVDA?",
        expected_tools=("get_stock_price",),
        expected_rag_engaged=False,
        category="price_lookup",
    ),
    EvalCase(
        id="price-msft-worth",
        query="What is Microsoft stock worth right now?",
        expected_tools=("get_stock_price",),
        expected_rag_engaged=False,
        category="price_lookup",
    ),
    EvalCase(
        id="history-msft",
        query="What has Microsoft's stock performance trend looked like over the past year?",
        expected_tools=("get_stock_price", "get_stock_history"),
        expected_rag_engaged=False,
        category="history_lookup",
    ),
    EvalCase(
        id="news-amzn",
        query="What's the latest news on Amazon?",
        expected_tools=("search_financial_news",),
        expected_rag_engaged=False,
        category="news_lookup",
    ),
    EvalCase(
        id="sentiment-ibm",
        query="What is the market sentiment around IBM lately?",
        expected_tools=("search_financial_news", "analyze_sentiment"),
        expected_rag_engaged=False,
        category="sentiment_lookup",
    ),
    EvalCase(
        id="rag-nvda-ai-initiatives",
        query="What do the private analyst reports say about NVIDIA's AI initiatives?",
        expected_tools=(),
        expected_rag_engaged=True,
        reference_answer=(
            "NVIDIA Blackwell GPU platforms accelerate model training and "
            "inference for AI infrastructure."
        ),
        expected_sources=("NVDA",),
        category="private_rag",
    ),
    EvalCase(
        id="rag-ibm-governance",
        query="What internal analyst reports exist about IBM's AI governance roadmap?",
        expected_tools=(),
        expected_rag_engaged=True,
        reference_answer=(
            "IBM watsonx supports enterprise AI governance, model lifecycle "
            "management, and trusted deployment."
        ),
        expected_sources=("IBM",),
        category="private_rag",
    ),
    EvalCase(
        id="full-analysis-nvda",
        query=(
            "Provide a comprehensive investment analysis for NVIDIA (NVDA), "
            "including its financial metrics, recent news, market sentiment, "
            "and AI research initiatives, with a buy/hold/sell view."
        ),
        expected_tools=(
            "get_stock_price",
            "get_stock_history",
            "search_financial_news",
            "analyze_sentiment",
        ),
        expected_rag_engaged=True,
        expected_sources=("NVDA",),
        category="full_analysis",
    ),
    EvalCase(
        id="comparison-ai-investment",
        query=(
            "Compare and rank MSFT, GOOGL, NVDA, and AMZN by AI-focused "
            "investment attractiveness, using financial, news, and sentiment "
            "evidence."
        ),
        expected_tools=(
            "get_stock_price",
            "get_stock_history",
            "search_financial_news",
            "analyze_sentiment",
        ),
        expected_rag_engaged=True,
        category="full_analysis",
    ),
    EvalCase(
        id="no-rag-when-disabled",
        query="What do the private analyst reports say about NVIDIA's AI initiatives?",
        expected_tools=(),
        expected_rag_engaged=False,
        with_rag_requested=False,
        category="rag_override",
    ),
)
