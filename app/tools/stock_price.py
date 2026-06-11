from ast import Dict
import yfinance as yf
from langchain_core.tools import tool
import logging 
from pprint import pprint
from datetime import datetime

@tool
def get_stock_price(ticker: str) -> Dict:
    """
    Returns the current stock price and basic information for a given ticker symbol.

    This tool fetches real-time stock data including current price, day's range,
    volume, and market cap. Use this when you need current stock pricing information.

    Args: 
        ticker: Stock ticker symbol(e.g, 'APPL', 'MSFT', 'GOOGL')

    Returns: 
        dict: {
            'ticker': str,
            'current_price': float,
            'currency': str,
            'day_high': float,
            'day_low': float,
            'volume': int,
            'market_cap': int,
            'timestamp': str,
            'status': str,
            'error': str (optional)
        }
    
    Example: 
        >>> result = get_stock_price("AAPL")
        >>> print(f"Apple stock price: ${result['current_price']}")
    """
    try: 
        stock = yf.Ticker(ticker.upper())
        info = stock.info

        current_price = (
            info.get("currentPrice") or 
            info.get("regularMarketPrice") or
            info.get("previousClose")
        )

        if current_price == None: 
            return {
                'ticker': ticker.upper(),
                'status': str,
                'error': f'could not retieve price data for {ticker}. Ticker may be invalid'
            }
        
        result = {
            'ticker': ticker.upper(),
            'current_price': round(current_price, 2),
            'currency': info.get('currency', 'USD'),
            'day_high': info.get('day_high', info.get('regularMarketDayHigh')),
            'day_low': info.get('day_low', info.get('regularMarketDayLow')),
            'volume': info.get('volume', info.get('regularMarketVolume')),
            'market_cap': info.get('marketCap'),
            'company_name': info.get('longName', info.get('shortName')),
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }

        return result

    except Exception as e: 
        return {
            'ticker': ticker.upper(),
            'status': 'error',
            'error': f'Error fetching stock data: {str(e)}',
            'timestamp': datetime.now.isoformat()
        }