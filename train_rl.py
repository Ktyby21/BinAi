"""
train_rl.py

- Loads historical 1-hour CSV data
- Creates HourlyTradingEnv
- Either loads a previously saved PPO model (if "ppo_hourly_model.zip" exists)
  or creates a new PPO model from scratch.
- Trains the PPO agent for 500,000 timesteps.
- Saves (or overwrites) the model afterwards.
- Then does a quick test *separately* on a single environment (no DummyVecEnv),
  using random starts for each test episode if you want multiple episodes in test.
"""
import os
import json
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from environment import HourlyTradingEnv

# Загрузка конфигурации из JSON
with open("config.json", "r") as f:
    config = json.load(f)

def main():
    # 1) Load CSV
    df = pd.read_csv("historical_data_1h.csv")
    required_cols = ["open", "high", "low", "close", "volume"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in CSV.")
    df.reset_index(drop=True, inplace=True)

    # 2) Create environment for TRAINING
    train_env = HourlyTradingEnv(
        df=df,
        window_size=config["window_size"],
        initial_balance=config["initial_balance"],
        commission_rate=config["commission_rate"],
        slippage_rate=config["slippage_rate"],
        max_bars=config["max_bars"],
        reward_scaling=config["reward_scaling"],
        penalize_no_trade_steps=config["penalize_no_trade_steps"],
        no_trade_penalty=config["no_trade_penalty"],
        consecutive_no_trade_allowed=config["consecutive_no_trade_allowed"]
    )
    vec_train_env = DummyVecEnv([lambda: train_env])

    model_path = "ppo_hourly_model.zip"
    if os.path.isfile(model_path):
        print("Found existing model file. Loading it...")
        model = PPO.load(model_path, env=vec_train_env, verbose=1)
    else:
        print("No existing model file found. Creating new PPO model from scratch...")
        model = PPO(
            policy="MlpPolicy",
            env=vec_train_env,
            verbose=1,
            tensorboard_log="./tensorboard_logs/"
        )

    print("Starting training...")
    model.learn(total_timesteps=config["train_timesteps"])
    print("Training finished.")

    model.save(model_path)
    print(f"Model saved to {model_path}")

    test_env = HourlyTradingEnv(
        df=df,
        window_size=config["window_size"],
        initial_balance=config["initial_balance"],
        commission_rate=config["commission_rate"],
        slippage_rate=config["slippage_rate"],
        max_bars=config["max_bars"],
        reward_scaling=config["reward_scaling"],
        penalize_no_trade_steps=config["penalize_no_trade_steps"],
        no_trade_penalty=config["no_trade_penalty"],
        consecutive_no_trade_allowed=config["consecutive_no_trade_allowed"]
    )

    n_test_episodes = 3
    for ep_i in range(n_test_episodes):
        obs, info = test_env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            total_reward += reward
            done = (terminated or truncated)
            steps += 1

        print(f"Test Episode {ep_i+1}: finished in {steps} steps.")
        print(f"  total_reward: {total_reward:.2f}")
        print(f"  final balance: {test_env.balance:.2f}")
        closed_trades = [t for t in test_env.trade_log if t.closed]
        print(f"  number of closed trades: {len(closed_trades)}")
        if len(closed_trades) > 0:
            print("  sample closed trades:")
            for i, tr in enumerate(closed_trades[:5]):
                print(f"    {i+1}) {tr.direction}, bar={tr.entry_bar}→{tr.exit_bar}, "
                      f"entry={tr.entry_price:.2f}, exit={tr.exit_price:.2f}, pnl={tr.pnl:.2f}")

if __name__ == "__main__":
    main()