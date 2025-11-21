import yfinance as yf

def fetch_live_price(symbol: str) -> float:
    """
    Fetch live price for NSE stock using Yahoo Finance.
    Example: RELIANCE.NS, TCS.NS
    """
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d")
        if data.empty:
            return None
        return float(data["Close"].iloc[-1])
    except Exception as e:
        print("Error fetching price:", e)
        return None

def fetch_multiple_prices(symbols: list):
    result = {}
    for s in symbols:
        price = fetch_live_price(s)
        result[s] = price
    return result
