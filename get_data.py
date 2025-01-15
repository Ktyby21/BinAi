import requests
import pandas as pd
from datetime import datetime, timedelta
import time

# Specify parameters
symbol = "BTCUSDT"
interval = "1h"  # 4-hour interval
start_date = "2022-01-01"  # start date for data collection
end_date = "2025-01-15"    # end date for data collection
output_file = "historical_data_1h.csv"

# Function to convert a date string to milliseconds
def date_to_milliseconds(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.timestamp() * 1000)

# Fetch historical data from Binance API
def fetch_binance_data(symbol, interval, start_time, end_time, limit=1000):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time,
        "limit": limit
    }
    response = requests.get(url, params=params)
    data = response.json()
    return data

# Convert dates to milliseconds
start_time = date_to_milliseconds(start_date)
end_time = date_to_milliseconds(end_date)
all_data = []

while start_time < end_time:
    data = fetch_binance_data(symbol, interval, start_time, end_time)
    if not data:
        break
    all_data.extend(data)
    
    # Update start_time for the next request
    start_time = data[-1][0] + (1 * 60 * 60 * 1000)  # move forward by 4 hours
    time.sleep(0.1)  # delay to avoid API rate limits

# Convert data to a DataFrame and save to CSV
df = pd.DataFrame(all_data, columns=[
    "timestamp", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "number_of_trades",
    "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
])
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")  # convert timestamps to datetime

# Keep only the required columns
df = df[["timestamp", "open", "high", "low", "close", "volume"]]
df.to_csv(output_file, index=False)
print(f"Historical 4-hour data has been saved to {output_file}")