#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hyperparameter tuning for PPO (Optuna) adapted to the updated HourlyTradingEnv (phase-2).

Fixes:
- Avoid Optuna dynamic Categorical space by searching static `batch_size_exp` and snapping to a valid divisor of rollout size.
- Minor stability/clarity tweaks in callbacks and evaluation.
"""

# --- глушим спам до любых других импортов ---
import os
import warnings
warnings.filterwarnings("ignore", message="Protobuf gencode version", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module=r".*google\.protobuf\.runtime_version")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import json
import random
import shutil
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import optuna
import pandas as pd
import torch
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from env.hourly_trading_env import HourlyTradingEnv

# тише вывод Optuna (оставляем только WARN/ERROR)
optuna.logging.set_verbosity(optuna.logging.WARNING)

torch.set_num_threads(max(1, os.cpu_count() // 2 or 1))

# ----------------- CONFIG & CLI -----------------
with open("config.json", "r") as f:
    CONFIG = json.load(f)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=CONFIG.get("data_output_file", "top_pairs_1h.csv"),
                   help="Path to CSV dataset")
    p.add_argument("--timesteps", type=int, default=1_000_000,
                   help="Training timesteps per trial")
    p.add_argument("--seed", type=int, default=42, help="Base random seed")
    p.add_argument("--symbol", type=str, default=None,
                   help="Symbol to filter from CSV (recommended). If omitted, uses first symbol.")
    p.add_argument("--phase", type=str, default="safe",
                   choices=["safe", "aggr", "wide"],
                   help="Search space preset: safe (narrow), aggr (explorative), wide (broad).")
    p.add_argument("--n-trials", type=int, default=100, help="Number of Optuna trials")
    return p.parse_args()

ARGS = parse_args()
random.seed(ARGS.seed)
np.random.seed(ARGS.seed)
torch.manual_seed(ARGS.seed)

# ----------------- I/O paths -----------------
ARTIFACTS_DIR = "./tune"           # все артефакты тут
LOG_DIR = os.path.join(ARTIFACTS_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
RESULTS_CSV = os.path.join(LOG_DIR, "tuning_results.csv")

# ----------------- DATA -----------------
if not os.path.exists(ARGS.data):
    raise FileNotFoundError(f"Data file '{ARGS.data}' not found.")

df_raw = pd.read_csv(ARGS.data)
need_cols = {"open", "high", "low", "close", "volume"}
missing = need_cols - set(df_raw.columns)
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

if ARGS.symbol and "symbol" in df_raw.columns:
    df_raw = df_raw[df_raw["symbol"] == ARGS.symbol].copy()
    if df_raw.empty:
        raise ValueError(f"No rows for symbol {ARGS.symbol} in {ARGS.data}")
    df_raw = df_raw.drop(columns=["symbol"])
else:
    if "symbol" in df_raw.columns:
        first_sym = df_raw["symbol"].iloc[0]
        df_raw = df_raw[df_raw["symbol"] == first_sym].copy().drop(columns=["symbol"])

df_raw.reset_index(drop=True, inplace=True)

# ----------------- ENV FACTORY -----------------

def make_base_env(env_kwargs: Dict[str, Any]) -> HourlyTradingEnv:
    return HourlyTradingEnv(**env_kwargs)


def make_vec_env(training: bool,
                 seed: Optional[int],
                 env_overrides: Optional[Dict[str, Any]] = None) -> VecNormalize:
    ekw = dict(
        df=df_raw,
        window_size=CONFIG["window_size"],
        initial_balance=CONFIG["initial_balance"],
        commission_rate=CONFIG["commission_rate"],
        slippage_rate=CONFIG["slippage_rate"],
        max_bars=CONFIG["max_bars"],
        reward_scaling=CONFIG["reward_scaling"],
        penalize_no_trade_steps=CONFIG["penalize_no_trade_steps"],
        no_trade_penalty=CONFIG["no_trade_penalty"],
        consecutive_no_trade_allowed=CONFIG["consecutive_no_trade_allowed"],
        ma_short_window=CONFIG["ma_short_window"],
        ma_long_window=CONFIG["ma_long_window"],
        vol_ma_window=CONFIG["vol_ma_window"],
        atr_window=CONFIG.get("atr_window", 14),
        rsi_window=CONFIG.get("rsi_window", 14),
        risk_fraction=CONFIG.get("risk_fraction", 0.01),
        max_alloc_per_trade=CONFIG.get("max_alloc_per_trade", 0.3),
        min_notional=CONFIG.get("min_notional", 1.0),
    )
    if env_overrides:
        ekw.update(env_overrides)

    def _thunk():
        env = make_base_env(ekw)
        return Monitor(env)

    venv = DummyVecEnv([_thunk])   # 1 env
    # более жёсткий clip_obs для стабильности
    vec = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=20.0, clip_reward=1e6)
    vec.training = training
    if seed is not None:
        vec.seed(seed)
    return vec

# ---- Rollout с извлечением episode_summary.final_balance / net_pnl ----

def rollout_final_balance(model: PPO, vecenv: VecNormalize, deterministic: bool = True) -> Tuple[float, Dict[str, Any]]:
    vecenv.training = False
    vecenv.norm_reward = False
    obs = vecenv.reset()
    done = False
    total_rew = 0.0
    ep_info: Dict[str, Any] = {}
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, dones, infos = vecenv.step(action)
        total_rew += float(reward[0])
        done = bool(dones[0])
        if done:
            ep_info = infos[0].get("episode_summary", {})
    metric = float(ep_info.get("final_balance", np.nan))
    if not np.isfinite(metric):
        metric = float(ep_info.get("net_pnl", np.nan))
    if not np.isfinite(metric):
        metric = total_rew
    return metric, ep_info


def evaluate_multi(model: PPO, train_vec: VecNormalize, seeds: List[int]) -> float:
    scores = []
    for s in seeds:
        eval_vec = make_vec_env(training=False, seed=s)
        eval_vec.obs_rms = train_vec.obs_rms  # sync нормализацию наблюдений
        score, _ = rollout_final_balance(model, eval_vec, deterministic=True)
        scores.append(score)
        eval_vec.close()
    return float(np.mean(scores)) if scores else -np.inf

# ---- Callback: sync VecNorm + NaN guard ----

class SyncVecNormCallback(BaseCallback):
    def __init__(self, eval_vec: VecNormalize):
        super().__init__()
        self.eval_vec = eval_vec
    def _on_step(self) -> bool:
        if isinstance(self.training_env, VecNormalize):
            self.eval_vec.obs_rms = self.training_env.obs_rms
        return True


class NanGuardCallback(BaseCallback):
    """Прерывает обучение, если появляются NaN/Inf в буфере rollout или в статистиках VecNormalize."""
    def _on_step(self) -> bool:
        try:
            if isinstance(self.training_env, VecNormalize):
                rms = self.training_env.obs_rms
                if rms is not None:
                    if not np.all(np.isfinite(rms.mean)) or not np.all(np.isfinite(rms.var)):
                        return False
        except Exception:
            pass
        return True

    def _on_rollout_end(self) -> bool:
        rb = getattr(self.model, "rollout_buffer", None)
        if rb is None:
            return True

        def _is_bad(x) -> bool:
            try:
                arr = x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)
            except Exception:
                return False
            return not np.all(np.isfinite(arr))

        to_check = [
            getattr(rb, "observations", None),
            getattr(rb, "returns", None),
            getattr(rb, "advantages", None),
            getattr(rb, "actions", None),
            getattr(rb, "values", None),
            getattr(rb, "log_probs", None),
        ]
        for arr in to_check:
            if arr is not None and _is_bad(arr):
                return False
        return True

# ---- CSV helper ----

def append_result_row(csv_path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = pd.DataFrame([row])
    header = not os.path.isfile(csv_path)
    df.to_csv(csv_path, mode="a", index=False, header=header)

# ----------------- SEARCH SPACES -----------------

def _choose_batch_size(trial: optuna.Trial, n_steps: int, n_envs: int = 1) -> int:
    """Статичное пространство через экспоненту; затем снапим к допустимому делителю rollout.
    Избегаем dynamic value space для Optuna.
    """
    exp = trial.suggest_int("batch_size_exp", 5, 12)  # 2^5=32 … 2^12=4096 (статично)
    raw_bs = 2 ** exp
    total = int(n_steps) * int(n_envs)
    # clamp: не больше total
    bs = min(raw_bs, total)
    # привести к ближайшей степени двойки вниз, делящей total
    while bs > 1 and total % bs != 0:
        bs //= 2
    bs = max(32, bs)
    return bs


def suggest_params(trial: optuna.Trial, phase: str) -> Dict[str, Any]:
    """Возвращает и PPO-гиперы, и env-overrides, и технические настройки."""
    if phase == "safe":
        # узкий диапазон вокруг того, что дало наилучшие результаты в ваших данных
        lr        = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
        gamma     = trial.suggest_float("gamma", 0.95, 0.985)
        ent_coef  = trial.suggest_float("ent_coef", 0.02, 0.05)
        clip_rg   = trial.suggest_float("clip_range", 0.20, 0.30)
        gae_lmbd  = trial.suggest_float("gae_lambda", 0.80, 0.90)
        n_steps   = trial.suggest_categorical("n_steps", [256, 512])
        vf_coef   = trial.suggest_float("vf_coef", 0.4, 0.9)
        max_gn    = trial.suggest_float("max_grad_norm", 0.4, 0.8)
        target_kl = trial.suggest_float("target_kl", 0.02, 0.05)
        # env
        risk_fraction      = trial.suggest_float("risk_fraction", 0.003, 0.02, log=True)
        max_alloc_per_trade= trial.suggest_float("max_alloc_per_trade", 0.2, 0.6)
        no_trade_penalty   = trial.suggest_float("no_trade_penalty", 8.0, 60.0, log=True)
        arch_key  = trial.suggest_categorical("net_arch", ["128-128", "256-256", "256-128"])
    elif phase == "aggr":
        # более смелые шаги: выше энтропия и клип, шире lr
        lr        = trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True)
        gamma     = trial.suggest_float("gamma", 0.94, 0.99)
        ent_coef  = trial.suggest_float("ent_coef", 0.03, 0.06)
        clip_rg   = trial.suggest_float("clip_range", 0.25, 0.35)
        gae_lmbd  = trial.suggest_float("gae_lambda", 0.80, 0.92)
        n_steps   = trial.suggest_categorical("n_steps", [256, 512])
        vf_coef   = trial.suggest_float("vf_coef", 0.3, 1.0)
        max_gn    = trial.suggest_float("max_grad_norm", 0.3, 1.0)
        target_kl = trial.suggest_float("target_kl", 0.015, 0.035)
        # env
        risk_fraction      = trial.suggest_float("risk_fraction", 0.003, 0.03, log=True)
        max_alloc_per_trade= trial.suggest_float("max_alloc_per_trade", 0.15, 0.7)
        no_trade_penalty   = trial.suggest_float("no_trade_penalty", 6.0, 120.0, log=True)
        arch_key  = trial.suggest_categorical("net_arch", ["128-128", "256-256", "256-128"])
    else:  # "wide" — широкий, но с безопасными ограничителями
        lr        = trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True)
        gamma     = trial.suggest_float("gamma", 0.90, 0.9999)
        ent_coef  = trial.suggest_float("ent_coef", 0.0, 0.06)
        clip_rg   = trial.suggest_float("clip_range", 0.10, 0.35)
        gae_lmbd  = trial.suggest_float("gae_lambda", 0.80, 0.99)
        n_steps   = trial.suggest_categorical("n_steps", [128, 256, 512, 1024])
        vf_coef   = trial.suggest_float("vf_coef", 0.2, 1.0)
        max_gn    = trial.suggest_float("max_grad_norm", 0.3, 1.0)
        target_kl = trial.suggest_float("target_kl", 0.02, 0.05)
        # env
        risk_fraction      = trial.suggest_float("risk_fraction", 0.002, 0.05, log=True)
        max_alloc_per_trade= trial.suggest_float("max_alloc_per_trade", 0.1, 0.8)
        no_trade_penalty   = trial.suggest_float("no_trade_penalty", 5.0, 300.0, log=True)
        arch_key  = trial.suggest_categorical("net_arch", ["64-64", "128-128", "256-256", "256-128"])

    # batch_size — статичное пространство (через экспоненту) + снап к валидному делителю
    batch_size = _choose_batch_size(trial, n_steps=n_steps, n_envs=1)

    return dict(
        # PPO гиперы
        learning_rate=lr, gamma=gamma, ent_coef=ent_coef, clip_range=clip_rg,
        gae_lambda=gae_lmbd, n_steps=n_steps, vf_coef=vf_coef,
        max_grad_norm=max_gn, batch_size=batch_size, target_kl=target_kl,
        # env overrides
        risk_fraction=risk_fraction,
        max_alloc_per_trade=max_alloc_per_trade,
        no_trade_penalty=no_trade_penalty,
        # архитектура
        net_arch=arch_key,
    )

# ----------------- OPTUNA OBJECTIVE -----------------

def objective(trial: optuna.Trial) -> float:
    params = suggest_params(trial, ARGS.phase)

    # архитектура политики — строка -> tuple
    arch_map = {
        "64-64":   (64, 64),
        "128-128": (128, 128),
        "256-256": (256, 256),
        "256-128": (256, 128),
    }
    net = arch_map[params["net_arch"]]

    env_overrides = {
        "risk_fraction": params["risk_fraction"],
        "max_alloc_per_trade": params["max_alloc_per_trade"],
        "no_trade_penalty": params["no_trade_penalty"],
    }

    # envs
    seed_train = ARGS.seed + trial.number
    seed_eval  = ARGS.seed + trial.number + 10_000
    train_vec = make_vec_env(training=True, seed=seed_train, env_overrides=env_overrides)
    eval_vec  = make_vec_env(training=False, seed=seed_eval,  env_overrides=env_overrides)
    sync_cb   = SyncVecNormCallback(eval_vec)
    nan_cb    = NanGuardCallback()

    policy_kwargs = dict(
        net_arch=list(net),
        activation_fn=nn.Tanh,   # дефолт, но укажем явно
        ortho_init=False,        # помягче старт
        log_std_init=-2.0,       # менее «широкие» действия на старте
    )

    model = PPO(
        "MlpPolicy",
        train_vec,
        learning_rate=params["learning_rate"],
        gamma=params["gamma"],
        batch_size=params["batch_size"],
        n_steps=params["n_steps"],
        ent_coef=params["ent_coef"],
        clip_range=params["clip_range"],
        gae_lambda=params["gae_lambda"],
        vf_coef=params["vf_coef"],
        max_grad_norm=params["max_grad_norm"],
        seed=seed_train,
        policy_kwargs=policy_kwargs,
        use_sde=True,             # стабильная стохастика для континуума
        sde_sample_freq=64,
        target_kl=params["target_kl"],
        verbose=0,
        tensorboard_log=None,
        device="auto",
    )

    # директория трайла
    trial_dir = os.path.join(ARTIFACTS_DIR, f"tune_trial_{trial.number}")
    if os.path.isdir(trial_dir):
        shutil.rmtree(trial_dir)
    os.makedirs(trial_dir, exist_ok=True)

    eval_cb = EvalCallback(
        eval_vec,
        n_eval_episodes=1,
        eval_freq=max(ARGS.timesteps // 10, 2_000),
        deterministic=True,
        best_model_save_path=trial_dir,
        verbose=0,
    )

    status = "ok"
    mean_score = -np.inf
    try:
        model.learn(total_timesteps=ARGS.timesteps, callback=[sync_cb, nan_cb, eval_cb])
        seeds = [ARGS.seed + trial.number + k * 123 for k in range(3)]
        mean_score = evaluate_multi(model, train_vec, seeds=seeds)
    except Exception as e:
        status = "failed"
        mean_score = -np.inf
    finally:
        # сохраняем артефакты трайла
        try:
            model.save(os.path.join(trial_dir, "final_model.zip"))
            train_vec.save(os.path.join(trial_dir, "vecnorm.pkl"))
        except Exception:
            pass

    # user attrs для Optuna
    trial.set_user_attr("model_dir", trial_dir)
    trial.set_user_attr("mean_score", mean_score)
    trial.set_user_attr("status", status)
    trial.set_user_attr("phase", ARGS.phase)

    # единый CSV-отчёт по трайлу
    row = {
        "trial": trial.number,
        "value": mean_score,
        "status": status,
        "phase": ARGS.phase,
        "seed": ARGS.seed,
        "timesteps": ARGS.timesteps,
        "symbol": (ARGS.symbol if ARGS.symbol else "first_in_file"),
        # env overrides
        "risk_fraction": params["risk_fraction"],
        "max_alloc_per_trade": params["max_alloc_per_trade"],
        "no_trade_penalty": params["no_trade_penalty"],
        # ppo
        "learning_rate": params["learning_rate"],
        "gamma": params["gamma"],
        "batch_size": params["batch_size"],
        "n_steps": params["n_steps"],
        "ent_coef": params["ent_coef"],
        "clip_range": params["clip_range"],
        "gae_lambda": params["gae_lambda"],
        "vf_coef": params["vf_coef"],
        "max_grad_norm": params["max_grad_norm"],
        "target_kl": params["target_kl"],
        "net_arch": params["net_arch"],
        "model_dir": trial_dir,
    }
    append_result_row(RESULTS_CSV, row)

    # чистим
    eval_vec.close()
    train_vec.close()
    return float(mean_score)


def main():
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=0)
    sampler = optuna.samplers.TPESampler(seed=ARGS.seed)
    study = optuna.create_study(direction="maximize", pruner=pruner, sampler=sampler)
    study.optimize(objective, n_trials=ARGS.n_trials)

    best = study.best_trial
    best_params = {
        **best.params,
        "model_dir": best.user_attrs.get("model_dir"),
        "mean_score": best.user_attrs.get("mean_score"),
        "status": best.user_attrs.get("status"),
        "phase": best.user_attrs.get("phase"),
    }
    with open(os.path.join(LOG_DIR, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=2)
    print("Best params saved to", os.path.join(LOG_DIR, "best_params.json"))
    print("Best model dir:", best_params["model_dir"]) 
    print("All trials CSV:", RESULTS_CSV)


if __name__ == "__main__":
    main()
