# paper_trading.py
import time
import json
import pandas as pd
import logging
import csv, os
from datetime import datetime, timedelta
from binance import Client
from stable_baselines3 import PPO
from env.hourly_trading_env import HourlyTradingEnv

# Загрузка конфигурации из JSON
with open("config.json", "r") as f:
    config = json.load(f)

# Настройка логирования
logging.basicConfig(
    level=getattr(logging, config.get("logging_level", "INFO")),
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("paper_trading.log"),
        logging.StreamHandler()
    ]
)

# Настройка подключения к Binance
client = Client(config["api_key"], config["api_secret"], testnet=config["testnet"])
symbol = config["symbol"]

# Преобразование интервала из конфигурации в формат Binance API
interval_map = {
    "1m": Client.KLINE_INTERVAL_1MINUTE,
    "5m": Client.KLINE_INTERVAL_5MINUTE,
    "1h": Client.KLINE_INTERVAL_1HOUR
}
interval = interval_map.get(config["interval"], Client.KLINE_INTERVAL_1HOUR)

def seconds_until_next_hour():
    now = datetime.utcnow()
    next_hour = (now.replace(second=0, microsecond=0, minute=0) + timedelta(hours=1))
    return (next_hour - now).total_seconds()

def load_initial_data(symbol, interval, limit=2000):
    candles = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    cols = ["timestamp", "open", "high", "low", "close", "volume", 
            "close_time", "quote_asset_volume", "number_of_trades", 
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"]
    df = pd.DataFrame(candles, columns=cols)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df['hl_range'] = df['high'] - df['low']
    df['atr'] = df['hl_range'].rolling(14).mean().bfill()
    return df

def update_market_data(df):
    last_timestamp = df["timestamp"].iloc[-1]
    candles = client.get_klines(
        symbol=symbol,
        interval=interval,
        startTime=int(last_timestamp.timestamp()*1000 + 1)
    )
    if candles:
        cols = ["timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"]
        new_df = pd.DataFrame(candles, columns=cols)
        new_df = new_df[["timestamp", "open", "high", "low", "close", "volume"]]
        new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            new_df[col] = new_df[col].astype(float)
        df = pd.concat([df, new_df]).drop_duplicates(subset="timestamp").reset_index(drop=True)
        df['hl_range'] = df['high'] - df['low']
        df['atr'] = df['hl_range'].rolling(14).mean().bfill()
    return df

def save_trade_log(trade):
    file_exists = os.path.isfile('trade_history.csv')
    with open('trade_history.csv', mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["direction", "entry_bar", "exit_bar", "entry_price", "exit_price", "size_in_coins", "pnl", "closed"])
        writer.writerow([
            trade.direction,
            trade.entry_bar,
            trade.exit_bar,
            trade.entry_price,
            trade.exit_price,
            trade.size_in_coins,
            trade.pnl,
            trade.closed
        ])

def main():
    df = load_initial_data(symbol, interval, limit=2000)

    env = HourlyTradingEnv(
        df=df,
        window_size=config["window_size"],
        initial_balance=config["initial_balance"],
        commission_rate=config["commission_rate"],
        slippage_rate=config["slippage_rate"],
        max_bars=config["max_bars"],
        reward_scaling=config["reward_scaling"],
        penalize_no_trade_steps=config["penalize_no_trade_steps"],
        no_trade_penalty=config["no_trade_penalty"],
        consecutive_no_trade_allowed=config["consecutive_no_trade_allowed"],
        ma_short_window=config["ma_short_window"],
        ma_long_window=config["ma_long_window"],
        vol_ma_window=config["vol_ma_window"]
    )

    model_path = "ppo_hourly_model.zip"
    model = PPO.load(model_path, env=env)

    # Переменная для отслеживания времени последнего дообучения
    last_retrain_time = datetime.utcnow()

    obs, _ = env.reset()

    while True:
        current_time = datetime.utcnow()

        # Проверяем, прошло ли 24 часа с последнего дообучения
        if (current_time - last_retrain_time).total_seconds() >= 24 * 3600:
            logging.info("Начало дообучения модели.")
            model.learn(total_timesteps=config["learn_timesteps"])
            model.save(model_path)  # сохраняем обновлённую модель
            logging.info("Дообучение завершено и модель сохранена.")
            last_retrain_time = current_time  # обновляем время последнего дообучения

        df = update_market_data(df)
        env.df = df

        action, _ = model.predict(obs, deterministic=True)
        logging.info(f"Predicted action: {action}")

        obs, reward, terminated, truncated, info = env.step(action)
        logging.info(f"Reward: {reward}, Balance: {env.balance}")

        for trade in env.trade_log:
            if trade.closed and not hasattr(trade, 'logged'):
                save_trade_log(trade)
                trade.logged = True

        if terminated or truncated:
            logging.info("Эпизод завершён. Сброс среды.")
            obs, _ = env.reset()

        sleep_duration = seconds_until_next_hour()
        logging.info(f"Ожидание {sleep_duration} секунд до начала следующего часа.")
        time.sleep(sleep_duration)

if __name__ == "__main__":
    main()
