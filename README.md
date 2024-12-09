# Binance Historical Data Fetcher

This script fetches historical candlestick (OHLCV) data for a specified cryptocurrency trading pair from the Binance API. The data is saved as a CSV file for further analysis or use.

## Features

- Fetches historical data for a specific symbol and interval.
- Handles large date ranges by splitting requests into smaller chunks.
- Saves data in a clean and readable CSV format.
- Includes necessary columns like open, high, low, close, and volume.

## Requirements

- Python 3.7 or higher
- `pandas` library
- `requests` library

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/binance-historical-fetcher.git
   cd binance-historical-fetcher
   ```
Install the required Python packages:
pip install pandas requests
Configuration

Before running the script, you can customize the following parameters directly in the script:

symbol: The trading pair to fetch data for (e.g., "BNBUSDT").
interval: The candlestick interval (e.g., "4h" for 4-hour intervals).
start_date: Start date for data collection (e.g., "2014-01-01").
end_date: End date for data collection (e.g., "2024-09-30").
output_file: Name of the output CSV file (e.g., "historical_data_4h_bnb.csv").
Usage

Run the script using Python:
python fetch_binance_data.py
After successful execution, the script saves the historical data in the specified CSV file (default: historical_data_4h_bnb.csv).
Output Format

The output CSV file includes the following columns:

timestamp: The start time of the candlestick (UTC).
open: Opening price.
high: Highest price.
low: Lowest price.
close: Closing price.
volume: Total trading volume during the interval.
Example

If you configure the script as follows:
  ```bash
  symbol = "BNBUSDT"
  interval = "4h"
  start_date = "2020-01-01"
  end_date = "2023-12-31"
  ```
The script will fetch all 4-hour candlestick data for the BNB/USDT trading pair from January 1, 2020, to December 31, 2023, and save it to historical_data_4h_bnb.csv.

Notes

The Binance API limits requests to 1000 candlesticks per call. The script automatically handles this by paginating requests.
A small delay (0.1 seconds) is added between requests to avoid exceeding Binance's rate limits.
Ensure your internet connection is stable during data fetching, as the process may take time for large date ranges.
Troubleshooting

Empty or incomplete CSV file: Check if the API response is empty or if rate limits are being hit. Adjust the time.sleep delay if necessary.
Connection errors: Ensure that you have a stable internet connection. Retry running the script if needed.
License

This project is licensed under the MIT License. See the LICENSE file for details.

Author

Developed by Ktyby21. Feel free to contact me for questions or suggestions.
