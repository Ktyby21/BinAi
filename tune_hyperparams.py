"""Simple hyperparameter tuning for PPO using Optuna."""
import json

import optuna
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from env.hourly_trading_env import HourlyTradingEnv

# Load configuration
with open("config.json", "r") as f:
    config = json.load(f)

# Load data
DF = pd.read_csv("historical_data_1h.csv")

REQUIRED_COLS = ["open", "high", "low", "close", "volume"]
for col in REQUIRED_COLS:
    if col not in DF.columns:
        raise ValueError(f"Column '{col}' not found in CSV.")
DF.reset_index(drop=True, inplace=True)


def make_env():
    return HourlyTradingEnv(
        df=DF,
        window_size=config["window_size"],
        initial_balance=config["initial_balance"],
        commission_rate=config["commission_rate"],
        slippage_rate=config["slippage_rate"],
        max_bars=config["max_bars"],
        reward_scaling=config["reward_scaling"],
        penalize_no_trade_steps=config["penalize_no_trade_steps"],
        no_trade_penalty=config["no_trade_penalty"],
        consecutive_no_trade_allowed=config["consecutive_no_trade_allowed"],
    )


def objective(trial: optuna.Trial) -> float:
    env = DummyVecEnv([make_env])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.9999)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    n_steps = trial.suggest_categorical("n_steps", [128, 256, 512])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        batch_size=batch_size,
        n_steps=n_steps,
        verbose=0,
    )
    model.learn(total_timesteps=max(1000, config["train_timesteps"] // 10))
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=1)
    env.close()
    return mean_reward


def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    with open("best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    print("Best params saved to best_params.json")


if __name__ == "__main__":
    main()
