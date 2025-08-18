"""Hyperparameter tuning for PPO using Optuna with CLI overrides."""

import argparse
import json
import os
import random

import numpy as np
import optuna
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from env.hourly_trading_env import HourlyTradingEnv

# ---------------------------------------------------------------------------
# Configuration and CLI
# ---------------------------------------------------------------------------
with open("config.json", "r") as f:
    config = json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default=config.get("data_output_file", "top_pairs_1h.csv"),
        help="Path to CSV dataset",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=config.get("train_timesteps", 100000),
        help="Training timesteps per trial",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed",
    )
    return parser.parse_args()


ARGS = parse_args()

# Seeding for reproducibility
random.seed(ARGS.seed)
np.random.seed(ARGS.seed)
torch.manual_seed(ARGS.seed)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
data_file = ARGS.data
if not os.path.exists(data_file):
    raise FileNotFoundError(
        f"Data file '{data_file}' not found. Please generate it or provide --data."
    )
DF = pd.read_csv(data_file)

REQUIRED_COLS = ["open", "high", "low", "close", "volume"]
for col in REQUIRED_COLS:
    if col not in DF.columns:
        raise ValueError(f"Column '{col}' not found in CSV.")
DF.reset_index(drop=True, inplace=True)


def make_env(seed: int | None = None):
    env = HourlyTradingEnv(
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
        ma_short_window=config["ma_short_window"],
        ma_long_window=config["ma_long_window"],
        vol_ma_window=config["vol_ma_window"],
    )
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


def evaluate_multi_episode(model: PPO, episodes: int = 5) -> float:
    """Evaluate a model over multiple episodes with different seeds."""
    rewards = []
    for idx in range(episodes):
        eval_env = DummyVecEnv([lambda: make_env(ARGS.seed + 1000 + idx)])
        r, _ = evaluate_policy(model, eval_env, n_eval_episodes=1, deterministic=False)
        rewards.append(r)
        eval_env.close()
    return float(np.mean(rewards))


def objective(trial: optuna.Trial) -> float:
    env = DummyVecEnv([lambda: make_env(ARGS.seed + trial.number)])
    eval_env = DummyVecEnv([lambda: make_env(ARGS.seed + trial.number + 5000)])

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.9999)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    n_steps = trial.suggest_categorical("n_steps", [128, 256, 512])
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.1)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 1.0)
    vf_coef = trial.suggest_float("vf_coef", 0.1, 1.0)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 1.0)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        batch_size=batch_size,
        n_steps=n_steps,
        ent_coef=ent_coef,
        clip_range=clip_range,
        gae_lambda=gae_lambda,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        seed=ARGS.seed + trial.number,
        verbose=0,
    )

    model_dir = f"best_model_trial_{trial.number}"
    callback = EvalCallback(
        eval_env,
        n_eval_episodes=3,
        eval_freq=max(ARGS.timesteps // 10, 1000),
        best_model_save_path=model_dir,
        verbose=0,
    )

    model.learn(total_timesteps=ARGS.timesteps, callback=callback)
    mean_reward = evaluate_multi_episode(model)
    trial.set_user_attr("model_path", callback.best_model_path)

    env.close()
    eval_env.close()
    return mean_reward


def main():
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=10)
    best_params = study.best_params
    best_params["model_path"] = study.best_trial.user_attrs.get("model_path")
    with open("best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    print("Best params and model path saved to best_params.json")


if __name__ == "__main__":
    main()
