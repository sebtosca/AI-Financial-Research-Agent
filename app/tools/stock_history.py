import logging
from datetime import datetime, timezone
from typing import Any, Dict

import yfinance as yf
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

VALID_PERIODS = {
    "1d",
    "5d",
    "1mo",
    "3mo",
    "6mo",
    "1y",
    "2y",
    "3y",
    "5y",
    "10y",
    "ytd",
    "max",
}


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_ticker(ticker: str) -> str:
    if not ticker or not ticker.strip():
        raise ValueError("Ticker symbol cannot be empty")

    return ticker.strip().upper()


def _validate_period(period: str) -> str:
    if not period or not period.strip():
        return "1y"

    cleaned_period = period.strip().lower()

    if cleaned_period not in VALID_PERIODS:
        raise ValueError(
            f"Invalid period: {period}. Valid periods are: {sorted(VALID_PERIODS)}"
        )

    return cleaned_period


@tool
def get_stock_history(ticker: str, period: str = "1y") -> Dict[str, Any]:
    """
    Return historical stock price data for a ticker.

    Args:
        ticker: Stock ticker symbol, for example: AAPL, MSFT, GOOGL.
        period: Historical period. Examples: 1mo, 3mo, 6mo, 1y, 3y, 5y.

    Returns:
        Dictionary containing historical stock metrics or an error payload.
    """

    timestamp = _utc_timestamp()

    try:
        normalized_ticker = _normalize_ticker(ticker)
        validated_period = _validate_period(period)

        logger.info(
            "Fetching stock history | ticker=%s | period=%s",
            normalized_ticker,
            validated_period,
        )

        stock = yf.Ticker(normalized_ticker)
        hist = stock.history(period=validated_period)

        if hist.empty:
            logger.warning(
                "No historical data found | ticker=%s | period=%s",
                normalized_ticker,
                validated_period,
            )

            return {
                "ticker": normalized_ticker,
                "period": validated_period,
                "status": "error",
                "error": "No historical data available for this ticker and period.",
                "timestamp": timestamp,
                "source": "yfinance",
            }

        start_price = float(hist["Close"].iloc[0])
        end_price = float(hist["Close"].iloc[-1])

        if start_price == 0:
            raise ValueError("Start price is zero; cannot calculate return percentage")

        return_pct = ((end_price - start_price) / start_price) * 100

        result = {
            "ticker": normalized_ticker,
            "period": validated_period,
            "start_date": hist.index[0].strftime("%Y-%m-%d"),
            "end_date": hist.index[-1].strftime("%Y-%m-%d"),
            "start_price": round(start_price, 2),
            "end_price": round(end_price, 2),
            "return_pct": round(return_pct, 2),
            "high": round(float(hist["High"].max()), 2),
            "low": round(float(hist["Low"].min()), 2),
            "avg_volume": int(hist["Volume"].mean()),
            "data_points": int(len(hist)),
            "status": "success",
            "timestamp": timestamp,
            "source": "yfinance",
        }

        logger.info(
            "Stock history fetched successfully | ticker=%s | period=%s | return_pct=%.2f | data_points=%d",
            normalized_ticker,
            validated_period,
            result["return_pct"],
            result["data_points"],
        )

        return result

    except Exception as e:
        logger.exception(
            "Failed to fetch stock history | ticker=%s | period=%s",
            ticker,
            period,
        )

        return {
            "ticker": ticker.strip().upper() if ticker else None,
            "period": period,
            "status": "error",
            "error": f"Error fetching historical data: {e}",
            "timestamp": timestamp,
            "source": "yfinance",
        }