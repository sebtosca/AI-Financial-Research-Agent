from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from typing import List, Dict
from dotenv import load_dotenv
import os

load_dotenv()

tavily_api_key = os.getenv("TAVILY_API_KEY")

# Initialize tavily search news 
tavily_tools = TavilySearchResults(
    max_results=5, 
    search_depth="advanced",
    include_answer=True, 
    include_raw_content=False,
    include_images=False
)

@tool
def search_financial_news(query: str) -> List[Dict]:
    """
    Searches real-time financial news using Tavily search API.

    This tool searches the web for recent financial news articles related to your query.
    Use this to find market sentiment, recent developement, and news about companies.

    Args: 
        query: Search query string (e.g, "Apple AI initiatives 2024")

    Returns: 
        list: List of news articles with: 
        - title: Article title
        - url: Article URL
        - content: Article snippet/summary
        - score: Relevance score

    Example: 
        >>> results = search_financial_news("Microsoft AI research")
        >>> for article in results:
        >>>     print(f"{article['title']}: {article['url']})
    """
    try: 
        result = tavily_tools.invoke(query)
        return result
    except Exception as e: 
        return [{
            'status': 'error',
            'error': f'Error searching news: {str(e)}'
        }]
