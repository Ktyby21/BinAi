"""
train_rl_all_pairs.py

- Последовательно тренируем PPO на КАЖДОЙ паре из большого CSV.
- Читаем CSV чанками, фильтруем по symbol (чтобы не грузить весь файл в RAM).
- По каждой паре показываем «живой» прогресс в одной строке:
  train[SYMBOL] ... FPS=... ETA=... lastR=... avgR=... eps=...
- После КАЖДОЙ пары сохраняем модель и сводку в runs/logs/training_summary.csv.
- Логи для TensorBoard: runs/tensorboard  (запуск: `tensorboard --logdir runs/tensorboard`)

Требуется: pandas, tqdm, numpy, stable_baselines3, gymnasium (твоя env), (опц.) tensorboard.
"""

import warnings

# Убираем спам варнингов от protobuf (до импорта сторонних библиотек)
warnings.filterwarnings(
    "ignore", message="Protobuf gencode version", category=UserWarning
)

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import gc
import time
import json
import numpy as np
import pandas as pd
from math import ceil
from typing import List, Set, Dict, Any
from statistics import mean, median

from tqdm import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure as sb3_configure

from env.hourly_trading_env import HourlyTradingEnv  # твоя среда


# ----------------- CONFIG -----------------
with open("config.json", "r") as f:
    config = json.load(f)

ARTIFACTS_DIR = "./runs"
LOG_DIR = os.path.join(ARTIFACTS_DIR, "logs")
TB_DIR = os.path.join(ARTIFACTS_DIR, "tensorboard")
CHECKPOINT_DIR = os.path.join(ARTIFACTS_DIR, "checkpoints")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "ppo_hourly_model.zip")
VECNORM_PATH = os.path.join(ARTIFACTS_DIR, "vecnorm.pkl")

for d in (LOG_DIR, TB_DIR, CHECKPOINT_DIR):
    os.makedirs(d, exist_ok=True)

PER_SYMBOL_TIMESTEPS = int(config.get("per_symbol_timesteps", config.get("learn_timesteps", 10_000)))
SLEEP_BETWEEN = float(config.get("sleep_between_symbols_sec", 2))
CHECKPOINT_FREQ = int(config.get("checkpoint_freq", 20_000))
WHITELIST = set(config.get("symbols_whitelist", []))
BLACKLIST = set(config.get("symbols_blacklist", []))
SUMMARY_CSV = os.path.join(LOG_DIR, "training_summary.csv")


def _detect_csv_path() -> str:
    candidates = [
        config.get("train_csv_file"),
        "historical_data_1h.csv",
        config.get("data_output_file"),
        "top_pairs_1h.csv",
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    raise FileNotFoundError("Не нашёл CSV. Укажи config['train_csv_file'] или положи файл рядом со скриптом.")


CSV_PATH = _detect_csv_path()


# ----------------- HELPERS -----------------
def print_progress_legend() -> None:
    print(
        "\nПояснения к прогрессу обучения:\n"
        "  FPS  — шагов обучения/сек (чем выше, тем быстрее идёт тренировка).\n"
        "  ETA  — ориентировочное время до конца обучения НА ТЕКУЩЕЙ ПАРЕ.\n"
        "  lastR — суммарная награда последнего завершённого эпизода.\n"
        "  avgR  — средняя награда по последним N эпизодам (окно по умолчанию N=20).\n"
        "  eps   — сколько эпизодов завершено на текущей паре за эту сессию.\n"
    )


def discover_symbols(path: str, chunk_size: int = 1_000_000) -> List[str]:
    """Собираем уникальные символы из огромного CSV без загрузки всего файла."""
    syms: Set[str] = set()
    for ch in pd.read_csv(path, usecols=["symbol"], chunksize=chunk_size, dtype={"symbol": "string"}):
        syms.update(ch["symbol"].dropna().unique().tolist())
    symbols = sorted(s for s in syms if s)

    if WHITELIST:
        symbols = [s for s in symbols if s in WHITELIST]
    if BLACKLIST:
        symbols = [s for s in symbols if s not in BLACKLIST]
    if not symbols:
        raise ValueError("После фильтров не осталось символов.")
    return symbols


def load_one_symbol_csv(path: str, symbol: str, chunk_size: int = 1_000_000) -> pd.DataFrame:
    """Читаем CSV чанками и оставляем только строки нужного symbol."""
    usecols = ["timestamp", "open", "high", "low", "close", "volume", "symbol"]

    # Оценка количества чанков чисто для progress bar (необязательно)
    try:
        file_size_mb = os.path.getsize(path) / (1024 * 1024)
        avg_row_size_bytes = 45  # грубая эвристика
        est_rows = int((file_size_mb * 1024 * 1024) / avg_row_size_bytes)
        est_chunks = max(1, ceil(est_rows / chunk_size))
    except Exception:
        est_chunks = None

    parts = []
    reader = pd.read_csv(
        path,
        usecols=usecols,
        parse_dates=["timestamp"],
        chunksize=chunk_size,
        dtype={
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
            "symbol": "string",
        },
    )

    with tqdm(total=est_chunks, unit="chunk", desc=f"load[{symbol}]", leave=False) as pbar:
        for ch in reader:
            part = ch[ch["symbol"] == symbol]
            if not part.empty:
                parts.append(part)
            if est_chunks:
                pbar.update(1)

    if not parts:
        raise ValueError(f"В файле нет строк для {symbol}")

    df = pd.concat(parts, ignore_index=True)
    df.sort_values("timestamp", inplace=True)
    df.drop_duplicates(subset=["timestamp", "symbol"], inplace=True)

    # sanity-check: только один символ
    assert df["symbol"].nunique() == 1 and df["symbol"].iloc[0] == symbol

    # экономия памяти
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    # среде symbol не нужен
    df.drop(columns=["symbol"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def make_env_from_df(df: pd.DataFrame, training: bool = True) -> VecNormalize:
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"Нет столбцов: {miss}")

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
        vol_ma_window=config["vol_ma_window"],
        risk_fraction=config.get("risk_fraction", 0.01),
        max_alloc_per_trade=config.get("max_alloc_per_trade", 0.3),
        min_notional=config.get("min_notional", 1.0),
    )
    env = Monitor(env)
    venv = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=1e6, clip_reward=1e6)
    vec_env.training = training
    return vec_env


def evaluate_once(model: PPO, env: VecNormalize) -> Dict[str, Any]:
    # ненормализованные награды при оценке
    env.training = False
    env.norm_reward = False
    obs = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    summary: Dict[str, Any] = {}
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done_vec, infos = env.step(action)
        total_reward += float(reward[0])
        done = bool(done_vec[0])
        if done:
            summary = infos[0].get("episode_summary", {})
        steps += 1
    return {
        "steps": steps,
        "total_reward": total_reward,
        **summary,
    }


def append_summary_row(path: str, row: Dict[str, Any]) -> None:
    df = pd.DataFrame([row])
    header = not os.path.isfile(path)
    df.to_csv(path, mode="a", index=False, header=header)

def summarize_run(rows: List[Dict[str, Any]], initial_balance: float, save_dir: str, run_name: str) -> None:
    if not rows:
        print("\n(Сводка прогона: нет строк для агрегации)")
        return

    df = pd.DataFrame(rows)
    df["roi"] = np.clip(df["final_balance"] / float(initial_balance) - 1.0, -10.0, 10.0)

    n = len(df)
    profitable_mask = df["final_balance"] > float(initial_balance)

    metrics = {
        "n_pairs": int(n),
        "steps_total": int(df["steps"].sum()),
        "steps_mean": float(df["steps"].mean()),
        "reward_mean": float(df["total_reward"].mean()),
        "reward_median": float(df["total_reward"].median()),
        "reward_std": float(df["total_reward"].std(ddof=0)),
        "final_balance_mean": float(df["final_balance"].mean()),
        "final_balance_median": float(df["final_balance"].median()),
        "final_balance_std": float(df["final_balance"].std(ddof=0)),
        "roi_mean": float(df["roi"].mean()),
        "roi_median": float(df["roi"].median()),
        "roi_std": float(df["roi"].std(ddof=0)),
        "profitable_pairs": int(profitable_mask.sum()),
        "profitable_share": float(profitable_mask.mean()),
        "closed_trades_total": int(df["closed_trades"].sum()),
        "closed_trades_mean": float(df["closed_trades"].mean()),
    }

    # Лучшие/худшие по финальному балансу
    best_row = df.loc[df["final_balance"].idxmax()]
    worst_row = df.loc[df["final_balance"].idxmin()]
    metrics.update({
        "best_symbol": str(best_row["symbol"]),
        "best_final_balance": float(best_row["final_balance"]),
        "best_reward": float(best_row["total_reward"]),
        "best_closed_trades": int(best_row["closed_trades"]),
        "worst_symbol": str(worst_row["symbol"]),
        "worst_final_balance": float(worst_row["final_balance"]),
        "worst_reward": float(worst_row["total_reward"]),
        "worst_closed_trades": int(worst_row["closed_trades"]),
    })

    # Печать в консоль — аккуратная сводка
    print("\n==============================")
    print(" СВОДКА ПО ВСЕМ ПАРАМ ПРОГОНА ")
    print("==============================")
    print(f"Пар: {metrics['n_pairs']}, шагов (факт): {metrics['steps_total']:,}")
    print(f"Средн. награда: {metrics['reward_mean']:.2f}  | медиана: {metrics['reward_median']:.2f}  | σ: {metrics['reward_std']:.2f}")
    print(f"Средн. фин. баланс: {metrics['final_balance_mean']:.2f}  | медиана: {metrics['final_balance_median']:.2f}  | σ: {metrics['final_balance_std']:.2f}")
    print(f"Средн. ROI: {metrics['roi_mean']*100:.2f}%  | медиана ROI: {metrics['roi_median']*100:.2f}%  | σ ROI: {metrics['roi_std']*100:.2f}%")
    print(f"Профитных пар: {metrics['profitable_pairs']}/{metrics['n_pairs']}  ({metrics['profitable_share']*100:.1f}%)")
    print(f"Сделок всего: {metrics['closed_trades_total']:,}  | в среднем на пару: {metrics['closed_trades_mean']:.1f}")
    print(f"Лучшее:  {metrics['best_symbol']}  баланс={metrics['best_final_balance']:.2f}  reward={metrics['best_reward']:.2f}  trades={metrics['best_closed_trades']}")
    print(f"Худшее:  {metrics['worst_symbol']} баланс={metrics['worst_final_balance']:.2f} reward={metrics['worst_reward']:.2f} trades={metrics['worst_closed_trades']}")

    # Сохранения: подробный CSV по парам + JSON со сводкой
    os.makedirs(save_dir, exist_ok=True)
    csv_pairs = os.path.join(save_dir, f"run_{run_name}_summary_pairs.csv")
    json_path = os.path.join(save_dir, f"run_{run_name}_summary.json")

    df.sort_values("final_balance", ascending=False).to_csv(csv_pairs, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\nСводные файлы сохранены:")
    print(f"  • Пары (детально): {csv_pairs}")
    print(f"  • Агрегат (JSON):  {json_path}")

# --------- Callbacks для «живого» прогресса ---------
class EpisodeStatsCallback(BaseCallback):
    """Обновляет postfix у tqdm: lastR/avgR/eps по завершённым эпизодам."""
    def __init__(self, pbar, window: int = 20):
        super().__init__()
        self.pbar = pbar
        self.window = window
        self.ep_rewards: List[float] = []
        self.total_episodes = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")  # добавляет Monitor на конце эпизода
            if ep is not None:
                self.total_episodes += 1
                r = float(ep.get("r", 0.0))
                self.ep_rewards.append(r)
                avg = float(np.mean(self.ep_rewards[-self.window:])) if self.ep_rewards else 0.0
                # обновим подпись у прогресс-бара
                self.pbar.set_postfix(
                    lastR=f"{r:.2f}",
                    avgR=f"{avg:.2f}",
                    eps=self.total_episodes
                )
        return True


class TqdmProgressCallback(BaseCallback):
    """Рисует прогресс timesteps, FPS и ETA без спама stdout-логами SB3."""
    def __init__(self, pbar, total_timesteps: int):
        super().__init__()
        self.pbar = pbar
        self.total = total_timesteps
        self.start_n = 0
        self.start_t = 0.0

    def _on_training_start(self) -> None:
        self.start_n = self.num_timesteps
        self.start_t = time.time()
        self.pbar.reset(total=self.total)

    def _on_step(self) -> bool:
        done = self.num_timesteps - self.start_n
        if done > self.pbar.n:
            self.pbar.n = min(done, self.total)
        elapsed = max(1e-6, time.time() - self.start_t)
        fps = done / elapsed
        remaining = max(0, self.total - done)
        eta = remaining / fps if fps > 0 else float("inf")
        self.pbar.set_postfix(
            FPS=f"{fps:,.0f}",
            ETA=f"{eta:,.0f}s",  # секунд до конца текущей пары
        )
        self.pbar.refresh()
        return True

    def _on_training_end(self) -> None:
        self.pbar.n = self.total
        self.pbar.refresh()


# ----------------- MAIN -------------------
def main():
    print(f"CSV: {CSV_PATH}")
    print_progress_legend()

    symbols = discover_symbols(CSV_PATH)
    print(f"Символов для обучения: {len(symbols)}")
    print("Пример:", symbols[:10])

    # чекпоинты (по глобальному счётчику шагов)
    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_hourly",
    )

    # первая пара — чтобы инициализировать модель/логгер
    first_df = load_one_symbol_csv(CSV_PATH, symbols[0])
    vec_env = make_env_from_df(first_df, training=True)
    if os.path.isfile(VECNORM_PATH):
        vec_env = VecNormalize.load(VECNORM_PATH, vec_env)
        vec_env.training = True

    # создаём/грузим модель, без спама в консоль
    if os.path.isfile(MODEL_PATH):
        print("Загружаю сохранённую модель…")
        model = PPO.load(MODEL_PATH, env=vec_env, verbose=0, tensorboard_log=TB_DIR)
    else:
        print("Создаю новую модель…")
        model = PPO(policy="MlpPolicy", env=vec_env, verbose=0, tensorboard_log=TB_DIR)

    # аккуратный логгер: CSV + TensorBoard
    run_name = time.strftime("%Y%m%d-%H%M%S")
    per_run_logdir = os.path.join(LOG_DIR, f"run_{run_name}")
    os.makedirs(per_run_logdir, exist_ok=True)
    new_logger = sb3_configure(per_run_logdir, ["csv", "tensorboard"])
    model.set_logger(new_logger)
    run_rows: List[Dict[str, Any]] = []

    for i, sym in enumerate(tqdm(symbols, desc="Symbols", unit="sym")):
        try:
            df = first_df if i == 0 else load_one_symbol_csv(CSV_PATH, sym)

            # пропуск слишком коротких серий
            if len(df) <= config["window_size"] + 5:
                tqdm.write(f"[skip] {sym}: мало данных ({len(df)})")
                continue

            vec_env.venv.envs[0].env.update_data(df)
            vec_env.training = True
            vec_env.reset()
            model.set_env(vec_env)

            # прогресс по текущей паре
            with tqdm(total=PER_SYMBOL_TIMESTEPS, desc=f"train[{sym}]", unit="step", leave=False) as pbar:
                cb_progress = TqdmProgressCallback(pbar, PER_SYMBOL_TIMESTEPS)
                cb_episode  = EpisodeStatsCallback(pbar, window=20)
                model.learn(
                    total_timesteps=PER_SYMBOL_TIMESTEPS,
                    reset_num_timesteps=False,
                    callback=[checkpoint_cb, cb_progress, cb_episode],
                )

            # сохраняем модель после каждой пары
            model.save(MODEL_PATH)
            vec_env.save(VECNORM_PATH)
            tqdm.write(f"[saved] {MODEL_PATH} / {VECNORM_PATH}")

            # мини-оценка и сводка
            eval_env = make_env_from_df(df, training=False)
            if os.path.isfile(VECNORM_PATH):
                eval_env = VecNormalize.load(VECNORM_PATH, eval_env)
            eval_env.training = False
            eval_env.norm_reward = False
            summary = evaluate_once(model, eval_env)
            summary_row = {
                "symbol": sym,
                "timesteps": PER_SYMBOL_TIMESTEPS,
                **summary
            }
            append_summary_row(SUMMARY_CSV, summary_row)
            tqdm.write(f"[summary] {sym}: steps={summary['steps']}, "
                       f"reward={summary['total_reward']:.2f}, "
                       f"final_balance={summary['final_balance']:.2f}, "
                       f"closed_trades={summary['closed_trades']}")
            run_rows.append(summary_row)
            # «перерыв», если нужен
            if SLEEP_BETWEEN > 0:
                time.sleep(SLEEP_BETWEEN)

        except KeyboardInterrupt:
            print("\nОстановлено пользователем. Сохраняю модель и выхожу…")
            model.save(MODEL_PATH)
            break
        except Exception as e:
            tqdm.write(f"[error] {sym}: {e}")
        finally:
            # подчистка
            gc.collect()
    summarize_run(run_rows, initial_balance=config["initial_balance"], save_dir=LOG_DIR, run_name=run_name)
    print(f"\nГотово.\nСводка: {SUMMARY_CSV}\nTensorBoard: {TB_DIR}  (запуск: tensorboard --logdir={TB_DIR})")


if __name__ == "__main__":
    main()
