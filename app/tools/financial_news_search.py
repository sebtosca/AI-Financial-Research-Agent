import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

load_dotenv()

logger = logging.getLogger(__name__)

MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "5"))
SEARCH_DEPTH = os.getenv("TAVILY_SEARCH_DEPTH", "advanced")


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_query(query: str) -> str:
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty")

    return query.strip()


def _validate_environment() -> None:
    if not os.getenv("TAVILY_API_KEY"):
        raise EnvironmentError("TAVILY_API_KEY is missing from environment")


def _build_tavily_client() -> TavilySearch:
    _validate_environment()

    return TavilySearch(
        max_results=MAX_RESULTS,
        search_depth=SEARCH_DEPTH,
        include_answer=True,
        include_raw_content=False,
        include_images=False,
    )


@tool
def search_financial_news(query: str) -> List[Dict[str, Any]]:
    """
    Search recent financial news using Tavily.

    Args:
        query: Search query, for example: "Apple AI initiatives 2026".

    Returns:
        A list of search results or an error payload.
    """

    timestamp = _utc_timestamp()

    try:
        cleaned_query = _validate_query(query)

        logger.info(
            "Searching financial news | query=%s | max_results=%d | search_depth=%s",
            cleaned_query,
            MAX_RESULTS,
            SEARCH_DEPTH,
        )

        tavily_client = _build_tavily_client()

        result = tavily_client.invoke(cleaned_query)

        logger.info(
            "Financial news search completed | query=%s",
            cleaned_query,
        )

        if isinstance(result, dict):
            results = result.get("results", [])

            if not results:
                return [
                    {
                        "status": "empty",
                        "query": cleaned_query,
                        "message": "No financial news results found.",
                        "timestamp": timestamp,
                        "source": "tavily",
                    }
                ]

            formatted_results = []

            for item in results:
                formatted_results.append(
                    {
                        "title": item.get("title"),
                        "url": item.get("url"),
                        "content": item.get("content"),
                        "score": item.get("score"),
                        "status": "success",
                        "query": cleaned_query,
                        "timestamp": timestamp,
                        "source": "tavily",
                    }
                )

            return formatted_results

        if isinstance(result, list):
            return result

        logger.warning(
            "Unexpected Tavily response format | query=%s | response_type=%s",
            cleaned_query,
            type(result).__name__,
        )

        return [
            {
                "status": "error",
                "query": cleaned_query,
                "error": "Unexpected Tavily response format.",
                "timestamp": timestamp,
                "source": "tavily",
            }
        ]

    except Exception as e:
        logger.exception(
            "Financial news search failed | query=%s",
            query,
        )

        return [
            {
                "status": "error",
                "query": query,
                "error": f"Error searching financial news: {e}",
                "timestamp": timestamp,
                "source": "tavily",
            }
        ]