from fastapi import APIRouter, Query
import yfinance as yf
import pandas as pd

router = APIRouter()

def safe_last_value(series):
    """Extracts the last valid scalar value."""
    try:
        val = series.iloc[-1]
        if isinstance(val, pd.Series):
            val = val.iloc[0]
        if pd.isna(val):
            return None
        return float(val)
    except:
        return None


@router.get("/live")
def get_live_signals(symbol: str = Query("RELIANCE.NS")):
    
    df = yf.download(symbol, period="5d", interval="1h")

    if df is None or df.empty:
        return {"error": "No data found", "symbol": symbol}

    close = df["Close"]

    # ===== Simple Moving Averages =====
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()

    sma20_last = safe_last_value(sma20)
    sma50_last = safe_last_value(sma50)

    trend = None
    if sma20_last is not None and sma50_last is not None:
        trend = "bullish crossover" if sma20_last > sma50_last else "bearish crossover"

    # ===== Bollinger Bands =====
    bb_mid = sma20
    bb_std = close.rolling(20).std()

    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    bb_up_last = safe_last_value(bb_upper)
    bb_low_last = safe_last_value(bb_lower)
    close_last = safe_last_value(close)

    bb_signal = None
    if close_last is not None and bb_up_last is not None and bb_low_last is not None:
        if close_last > bb_up_last:
            bb_signal = "upper breakout"
        elif close_last < bb_low_last:
            bb_signal = "lower breakdown"
        else:
            bb_signal = "normal"

    return {
        "symbol": symbol,
        "last_price": close_last,
        "trend_signal": trend,
        "bollinger_signal": bb_signal
    }
