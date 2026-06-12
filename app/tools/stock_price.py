import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import yfinance as yf
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_ticker(ticker: str) -> str:
    if not ticker or not ticker.strip():
        raise ValueError("Ticker symbol cannot be empty")

    return ticker.strip().upper()


def _round_optional(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None

    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return None


@tool
def get_stock_price(ticker: str) -> Dict[str, Any]:
    """
    Return the current stock price and basic market information for a ticker.

    Args:
        ticker: Stock ticker symbol, for example: AAPL, MSFT, GOOGL.

    Returns:
        Dictionary containing stock price data or an error payload.
    """

    timestamp = _utc_timestamp()

    try:
        normalized_ticker = _normalize_ticker(ticker)

        logger.info(
            "Fetching stock price | ticker=%s",
            normalized_ticker,
        )

        stock = yf.Ticker(normalized_ticker)
        info = stock.info

        current_price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )

        if current_price is None:
            logger.warning(
                "No price data found | ticker=%s",
                normalized_ticker,
            )

            return {
                "ticker": normalized_ticker,
                "status": "error",
                "error": (
                    "Could not retrieve price data. "
                    "Ticker may be invalid or unavailable."
                ),
                "timestamp": timestamp,
                "source": "yfinance",
            }

        result = {
            "ticker": normalized_ticker,
            "company_name": info.get("longName") or info.get("shortName"),
            "current_price": _round_optional(current_price),
            "currency": info.get("currency", "USD"),
            "day_high": _round_optional(
                info.get("dayHigh")
                or info.get("regularMarketDayHigh")
            ),
            "day_low": _round_optional(
                info.get("dayLow")
                or info.get("regularMarketDayLow")
            ),
            "volume": info.get("volume") or info.get("regularMarketVolume"),
            "market_cap": info.get("marketCap"),
            "timestamp": timestamp,
            "status": "success",
            "source": "yfinance",
        }

        logger.info(
            "Stock price fetched successfully | ticker=%s | price=%s | currency=%s",
            normalized_ticker,
            result["current_price"],
            result["currency"],
        )

        return result

    except Exception as e:
        logger.exception(
            "Failed to fetch stock price | ticker=%s",
            ticker,
        )

        return {
            "ticker": ticker.strip().upper() if ticker else None,
            "status": "error",
            "error": f"Error fetching stock data: {e}",
            "timestamp": timestamp,
            "source": "yfinance",
        }