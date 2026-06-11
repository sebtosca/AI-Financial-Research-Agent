from langchain_core.tools import tool
from typing import Dict
from pprint import pprint
import yfinance as yf

@tool
def get_stock_history(ticker: str, period: str = "1y") -> Dict:
    """
    Returns historical stock price data for analysis of 3-year performance.

    This tool fetches historical stock data over a specified period, useful for
    analyzing trends, calculating returns, and assessing long-term performance.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'GOOGL')
        period: Time period for historical data. Options: '1mo', '3mo', '6mo',
                '1y', '2y', '3y', '5y', '10y'. Default is '1y'.

    Returns:
        dict: {
            'ticker': str,
            'period': str,
            'start_date': str,
            'end_date': str,
            'start_price': float,
            'end_price': float,
            'return_pct': float,
            'high': float,
            'low': float,
            'avg_volume': int,
            'data_points': int,
            'status': str
        }
    """
    try:
        stock = yf.Ticker(ticker.upper())
        hist = stock.history(period=period)

        if hist.empty:
            return {
                'ticker': ticker.upper(),
                'status': 'error',
                'error': f'No historical data available for {ticker} over period {period}'
            }

        # Calculate key metrics
        start_price = hist['Close'].iloc[0]
        end_price = hist['Close'].iloc[-1]
        return_pct = ((end_price - start_price) / start_price) * 100

        result = {
            'ticker': ticker.upper(),
            'period': period,
            'start_date': hist.index[0].strftime('%Y-%m-%d'),
            'end_date': hist.index[-1].strftime('%Y-%m-%d'),
            'start_price': round(start_price, 2),
            'end_price': round(end_price, 2),
            'return_pct': round(return_pct, 2),
            'high': round(hist['High'].max(), 2),
            'low': round(hist['Low'].min(), 2),
            'avg_volume': int(hist['Volume'].mean()),
            'data_points': len(hist),
            'status': 'success'
        }

        return result

    except Exception as e:
        return {
            'ticker': ticker.upper(),
            'status': 'error',
            'error': f'Error fetching historical data: {str(e)}'
        }
