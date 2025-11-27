import requests
import json
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import pandas as pd

load_dotenv()

ACCESS_TOKEN = os.getenv("GROWW_ACCESS_TOKEN")
API_KEY = os.getenv("GROWW_API_KEY")

headers = {
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "Accept": "application/json",
    "X-API-VERSION": "1.0"
}

BASE_URL = "https://api.groww.in/v1"

def get_quote(token):
    url = f"{BASE_URL}/live-data/quote"
    params = {"instrument_token": token, "segment": "CASH"}
    r = requests.get(url, headers=headers, params=params)
    print("QUOTE RESPONSE:", r.status_code)
    return r.json()

def get_ohlc(token, interval="5m"):
    import time
    end_ts = int(time.time() * 1000)
    start_ts = end_ts - (2 * 24 * 60 * 60 * 1000)  # 2 days
    url = f"{BASE_URL}/historical/candle/range"
    params = {"instrument_token": token, "interval": interval,
              "start": start_ts, "end": end_ts}
    r = requests.get(url, headers=headers, params=params)
    print("OHLC RESPONSE:", r.status_code)
    return r.json()

def plot_candles(df, title="Price Chart"):
    plt.figure(figsize=(10, 5))
    plt.plot(df['timestamp'], df['close'], label="Close Price")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    plt.show()

def main():
    token = "NSE:RELIANCE"  # Test symbol

    print("\n=== Testing QUOTE ===\n")
    quote = get_quote(token)
    print(json.dumps(quote, indent=2))

    print("\n=== Testing OHLC ===\n")
    ohlc = get_ohlc(token)

    candles = ohlc.get("candles") or ohlc.get("data") or []
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    print(df.head())

    print("\n=== Plotting Chart ===\n")
    plot_candles(df, "RELIANCE Last 2 Days (5m Candles)")

if __name__ == "__main__":
    main()
