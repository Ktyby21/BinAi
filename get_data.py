import json
import time
from datetime import datetime
from typing import List

import pandas as pd
import requests

# Load parameters from config
with open("config.json", "r") as f:
    config = json.load(f)

TOP_N = config.get("data_top_n", 100)
INTERVAL = config.get("data_interval", "1h")
START_DATE = config.get("data_start_date")
END_DATE = config.get("data_end_date")
OUTPUT_FILE = config.get("data_output_file", "top_pairs_1h.csv")


def date_to_milliseconds(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.timestamp() * 1000)


def get_top_symbols(limit: int) -> List[str]:
    url = "https://api.binance.com/api/v3/ticker/24hr"
    resp = requests.get(url)
    data = resp.json()
    df = pd.DataFrame(data)
    df['quoteVolume'] = df['quoteVolume'].astype(float)
    df.sort_values(by='quoteVolume', ascending=False, inplace=True)
    return df['symbol'].head(limit).tolist()


def fetch_binance_data(symbol: str, interval: str, start_time: int, end_time: int, limit: int = 1000):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time,
        "limit": limit,
    }
    response = requests.get(url, params=params)
    return response.json()


def fetch_symbol_history(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    all_data = []
    start = start_ms
    while start < end_ms:
        data = fetch_binance_data(symbol, INTERVAL, start, end_ms)
        if not data:
            break
        all_data.extend(data)
        start = data[-1][0] + 1
        time.sleep(0.1)
    if not all_data:
        return pd.DataFrame()
    df = pd.DataFrame(
        all_data,
        columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore",
        ],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df["symbol"] = symbol
    return df


def main():
    symbols = get_top_symbols(TOP_N)
    start_ms = date_to_milliseconds(START_DATE)
    end_ms = date_to_milliseconds(END_DATE)
    frames = []
    for sym in symbols:
        df = fetch_symbol_history(sym, start_ms, end_ms)
        if not df.empty:
            frames.append(df)
    if frames:
        result = pd.concat(frames, ignore_index=True)
        result.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved data for {len(frames)} symbols to {OUTPUT_FILE}")
    else:
        print("No data fetched")


if __name__ == "__main__":
    main()
