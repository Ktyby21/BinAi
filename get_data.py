import json
import time
from datetime import datetime
from typing import List
from math import ceil

import pandas as pd
import requests
from tqdm import tqdm  # NEW

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


def interval_to_milliseconds(interval: str) -> int:
    """Binance interval -> ms"""
    mapping = {
        "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
        "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000,
        "8h": 28_800_000, "12h": 43_200_000, "1d": 86_400_000, "3d": 259_200_000,
        "1w": 604_800_000, "1M": 2_592_000_000  # приблизительно
    }
    return mapping[interval]


def get_top_symbols(limit: int) -> List[str]:
    url = "https://api.binance.com/api/v3/ticker/24hr"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
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
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_symbol_history(symbol: str, start_ms: int, end_ms: int, interval_ms: int) -> pd.DataFrame:
    all_data = []
    start = start_ms

    # Оценим количество запросов (1 запрос ≈ 1000 свечей)
    total_klines = max(0, (end_ms - start_ms) // interval_ms)
    est_requests = max(1, ceil(total_klines / 1000))

    with tqdm(total=est_requests, desc=f"{symbol}", unit="req", leave=False) as pbar:
        while start < end_ms:
            data = fetch_binance_data(symbol, INTERVAL, start, end_ms, limit=1000)
            if not data:
                break
            all_data.extend(data)
            start = data[-1][0] + 1  # следующий миллисек после последней свечи
            pbar.update(1)           # один запрос выполнен
            time.sleep(0.1)          # не агрессим по лимитам

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
    interval_ms = interval_to_milliseconds(INTERVAL)

    frames = []
    for sym in tqdm(symbols, desc="Symbols", unit="sym"):
        try:
            df = fetch_symbol_history(sym, start_ms, end_ms, interval_ms)
        except requests.HTTPError as e:
            tqdm.write(f"[HTTPError] {sym}: {e}")
            continue
        except Exception as e:
            tqdm.write(f"[Error] {sym}: {e}")
            continue

        if not df.empty:
            frames.append(df)

    if frames:
        result = pd.concat(frames, ignore_index=True)
        result.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved data for {len(frames)} symbols, rows: {len(result):,} → {OUTPUT_FILE}")
    else:
        print("No data fetched")


if __name__ == "__main__":
    main()