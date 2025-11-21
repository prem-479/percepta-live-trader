import pandas as pd

# ========================
# SIMPLE MOVING AVERAGE
# ========================
def sma(series: pd.Series, period: int = 14):
    return series.rolling(window=period).mean()

# ========================
# EXPONENTIAL MOVING AVERAGE
# ========================
def ema(series: pd.Series, period: int = 14):
    return series.ewm(span=period, adjust=False).mean()

# ========================
# RSI (Relative Strength Index)
# ========================
def rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / (avg_loss + 1e-10)  # avoid division by zero
    rsi_value = 100 - (100 / (1 + rs))

    return rsi_value

# ========================
# MACD (12 EMA - 26 EMA)
# ========================
def macd(series: pd.Series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

# ========================
# BOLLINGER BANDS
# ========================
def bollinger_bands(series: pd.Series, period: int = 20, std_factor: float = 2):
    middle = sma(series, period)
    std_dev = series.rolling(window=period).std()

    upper = middle + std_factor * std_dev
    lower = middle - std_factor * std_dev

    return upper, middle, lower
