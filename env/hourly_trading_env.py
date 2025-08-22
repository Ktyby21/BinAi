"""
HourlyTradingEnv

A custom Gymnasium environment simulating trading on hourly candlesticks.
Observation includes equity ratio along with several log-normalized
market indicators.
"""
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import List, Optional

from trade import Trade


class HourlyTradingEnv(gym.Env):
    @staticmethod
    def _log_ratio(num: float, den: float, eps: float = 1e-12) -> float:
        """Return log(num/den) with safeguards for zeros.

        Both numerator and denominator are clamped below by ``eps`` to avoid
        divisions by zero and ``log(0)`` warnings.
        """
        return float(np.log(max(num, eps)) - np.log(max(den, eps)))

    @staticmethod
    def _finite(x: float) -> float:
        return float(x) if np.isfinite(x) else 0.0
    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 24,
        initial_balance: float = 1000.0,
        commission_rate: float = 0.0005,
        slippage_rate: float = 0.0001,
        max_bars: Optional[int] = 2000,
        reward_scaling: float = 1.0,
        penalize_no_trade_steps: bool = True,
        no_trade_penalty: float = 0.1,
        consecutive_no_trade_allowed: int = 10,
        ma_short_window: int = 24,
        ma_long_window: int = 168,
        vol_ma_window: int = 24,
        risk_fraction: float = 0.01,
        max_alloc_per_trade: float = 0.3,
        min_notional: float = 1.0,
        # NEW:
        atr_window: int = 14,
        rsi_window: int = 14,
    ):
        super().__init__()

        self.window_size = window_size
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.max_bars = max_bars
        self.reward_scaling = reward_scaling

        self.penalize_no_trade_steps = penalize_no_trade_steps
        self.no_trade_penalty = no_trade_penalty
        self.consecutive_no_trade_allowed = consecutive_no_trade_allowed

        self.ma_short_window = ma_short_window
        self.ma_long_window = ma_long_window
        self.vol_ma_window = vol_ma_window
        self.risk_fraction = risk_fraction
        self.max_alloc_per_trade = max_alloc_per_trade
        self.min_notional = min_notional

        # NEW:
        self.atr_window = atr_window
        self.rsi_window = rsi_window

        self.update_data(df)

        # Internal states
        self.current_bar: int = 0
        self.start_bar: int = 0
        self.balance: float = 0.0
        self.open_trades: List[Trade] = []
        self.trade_log: List[Trade] = []
        self.consecutive_no_trade_steps = 0

        # Observation:
        # [equity_log, ma_log, price_log, vol_log, atr_pct, rsi_c,
        #  net_exposure_ratio, open_notional_ratio]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )

        # [open_long_frac, open_short_frac, close_fraction, sl_factor, tp_factor]
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 3, 3], dtype=np.float32),
        )

    def _rsi(self, s: pd.Series, n: int) -> pd.Series:
        diff = s.diff()
        up = diff.clip(lower=0.0)
        down = -diff.clip(upper=0.0)
        # Wilder smoothing (EMA с alpha=1/n)
        roll_up = up.ewm(alpha=1.0 / n, adjust=False).mean()
        roll_down = down.ewm(alpha=1.0 / n, adjust=False).mean()
        rs = roll_up / roll_down.replace(0.0, 1e-12)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    def update_data(self, df: pd.DataFrame):
        """Recompute indicators and reset internal dataframe."""
        self.df = df.reset_index(drop=True)

        prev_close = self.df["close"].shift(1)
        tr = pd.concat(
            [
                self.df["high"] - self.df["low"],
                (self.df["high"] - prev_close).abs(),
                (self.df["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        # ATR и ATR%
        self.df["atr"] = tr.rolling(self.atr_window).mean()
        self.df["atr_pct"] = (self.df["atr"] / self.df["close"].replace(0.0, 1e-12)).clip(0.0, 1.0)

        # MA и объёмные MA как было
        self.df["ma_short"] = self.df["close"].rolling(self.ma_short_window).mean()
        self.df["ma_long"] = self.df["close"].rolling(self.ma_long_window).mean()
        self.df["vol_ma"] = self.df["volume"].rolling(self.vol_ma_window).mean()

        # RSI(14) центрированный в [-1,1] (слегка клипуем)
        rsi = self._rsi(self.df["close"], self.rsi_window)
        self.df["rsi_c"] = ((rsi - 50.0) / 50.0).clip(-1.5, 1.5)

        # очистка: убираем NaN/Inf по ключевым колонкам
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df.dropna(subset=["ma_short", "ma_long", "vol_ma", "atr", "atr_pct", "rsi_c"], inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.n_bars = len(self.df)

    def _get_total_coins(self) -> float:
        total = 0.0
        for t in self.open_trades:
            if not t.closed:
                total += t.size_in_coins
        return total

    def _get_obs(self) -> np.ndarray:
        idx = min(self.current_bar, self.n_bars - 1)
        row = self.df.iloc[idx]

        current_price = float(row["close"])
        ma_short = float(row["ma_short"])
        ma_long = float(row["ma_long"])
        vol_ma = float(row["vol_ma"])

        # Портфельные величины
        unrealized = 0.0
        total_notional = 0.0
        net_notional = 0.0
        for t in self.open_trades:
            if t.closed:
                continue
            total_notional += t.notional
            if t.direction == "long":
                unrealized += (current_price - t.entry_price) * t.size_in_coins
                net_notional += t.notional
            else:
                unrealized += (t.entry_price - current_price) * abs(t.size_in_coins)
                net_notional -= t.notional

        # equity учитывает стоимость открытых позиций (симметрично лонгам и шортам)
        equity_log = self._log_ratio(
            self.balance + unrealized + total_notional, self.initial_balance
        )

        ma_log = self._log_ratio(ma_short, ma_long)
        price_log = self._log_ratio(current_price, ma_short)
        vol_log = self._log_ratio(float(row["volume"]), vol_ma)
        # NEW фичи
        atr_pct = self._finite(float(row["atr_pct"]))
        rsi_c = self._finite(float(row["rsi_c"]))

        net_exposure_ratio = net_notional / max(self.initial_balance, 1e-8)
        open_notional_ratio = total_notional / max(self.initial_balance, 1e-8)

        obs = np.array(
            [
                equity_log,
                ma_log,
                price_log,
                vol_log,
                atr_pct,
                rsi_c,
                net_exposure_ratio,
                open_notional_ratio,
            ],
            dtype=np.float32,
        )
        return obs

    def _settle_trade(
        self,
        trade: Trade,
        exec_price: float,
        bar_idx: int,
        proportion: float = 1.0,
    ) -> float:
        size = trade.size_in_coins * proportion
        notional_part = trade.notional * proportion
        open_fee_part = trade.open_fee * proportion

        if trade.direction == "long":
            proceeds = exec_price * size
            fee_close = abs(proceeds) * self.commission_rate
            self.balance += proceeds - fee_close
            pnl_part = proceeds - notional_part - open_fee_part - fee_close
        else:
            # Маржу мы резервировали при открытии (balance -= notional + fee_open)
            # При закрытии: освобождаем резерв и выкупаем позицию.
            cost = abs(size) * exec_price
            fee_close = cost * self.commission_rate
            self.balance += (2 * notional_part) - cost - fee_close
            pnl_part = (notional_part - cost) - open_fee_part - fee_close

        trade.pnl += pnl_part
        trade.notional -= notional_part
        trade.open_fee -= open_fee_part
        trade.size_in_coins -= size
        if proportion >= 1.0 or abs(trade.size_in_coins) < 1e-8:
            trade.closed = True
            trade.exit_bar = bar_idx
            trade.exit_price = exec_price
        return pnl_part

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.open_trades.clear()
        self.trade_log.clear()
        self.consecutive_no_trade_steps = 0
        self.balance = self.initial_balance
        self.prev_equity = self.initial_balance
        self.penalty_total = 0.0

        min_start = 0

        # Guarantee training horizon away from dataset end
        horizon = int(self.max_bars) if self.max_bars is not None else 512
        horizon = max(2, min(horizon, self.n_bars - min_start - 2))

        max_start = self.n_bars - horizon - 1
        if max_start <= min_start:
            self.start_bar = min_start
        else:
            self.start_bar = int(
                self.np_random.integers(low=min_start, high=max_start + 1)
            )

        self.current_bar = self.start_bar
        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray):
        open_long_frac  = float(np.clip(action[0], 0.0, 1.0))
        open_short_frac = float(np.clip(action[1], 0.0, 1.0))
        close_fraction  = float(np.clip(action[2], 0.0, 1.0))
        sl_factor       = float(np.clip(action[3], 0.5, 3.0))
        tp_factor       = float(np.clip(action[4], 0.5, 3.0))

        if self.current_bar >= self.n_bars:
            return self._get_obs(), 0.0, True, False, {}

        terminated = truncated = False
        row = self.df.iloc[self.current_bar]
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        atr_val = float(max(row["atr"], 1e-8))
        bar_idx = self.current_bar

        # ===== 1) SL/TP =====
        any_settlement = False
        for trade in self.open_trades:
            if trade.closed:
                continue
            if trade.direction == "long":
                if l <= trade.stop_loss:
                    self._settle_trade(
                        trade,
                        trade.stop_loss * (1.0 - self.slippage_rate),
                        bar_idx,
                    )
                    any_settlement = True
                elif h >= trade.take_profit:
                    self._settle_trade(
                        trade,
                        trade.take_profit * (1.0 - self.slippage_rate),
                        bar_idx,
                    )
                    any_settlement = True
            else:
                if h >= trade.stop_loss:
                    self._settle_trade(
                        trade,
                        trade.stop_loss * (1.0 + self.slippage_rate),
                        bar_idx,
                    )
                    any_settlement = True
                elif l <= trade.take_profit:
                    self._settle_trade(
                        trade,
                        trade.take_profit * (1.0 + self.slippage_rate),
                        bar_idx,
                    )
                    any_settlement = True

        # ===== 2) Partial close =====
        if close_fraction > 1e-8:
            for trade in self.open_trades:
                if trade.closed:
                    continue
                px = (
                    c * (1.0 - self.slippage_rate)
                    if trade.direction == "long"
                    else c * (1.0 + self.slippage_rate)
                )
                self._settle_trade(trade, px, bar_idx, proportion=close_fraction)
                any_settlement = True

        # Уберём закрытые из списка активных
        if any_settlement:
            self.open_trades = [t for t in self.open_trades if not t.closed]

        # ===== 3) Possibly open trades (risk-based sizing) =====
        max_alloc = float(self.max_alloc_per_trade)

        # Текущая equity (для риска)
        unrealized_eq = 0.0
        reserve_total = 0.0
        for t in self.open_trades:
            if t.closed:
                continue
            if t.direction == "long":
                unrealized_eq += (c - t.entry_price) * t.size_in_coins
            else:
                unrealized_eq += (t.entry_price - c) * abs(t.size_in_coins)
            reserve_total += t.notional
        equity = self.balance + unrealized_eq + reserve_total

        def open_long(risk_scale: float):
            if sl_factor * atr_val <= 1e-8:
                return
            dollar_risk = max(0.0, equity * self.risk_fraction * risk_scale)
            size_in_coins = dollar_risk / (sl_factor * atr_val)
            entry_price = o * (1.0 + self.slippage_rate)
            notional = abs(size_in_coins * entry_price)
            notional = min(notional, self.balance * max_alloc)
            if notional < self.min_notional:
                return
            fee_open = notional * self.commission_rate
            if notional > 0 and self.balance >= notional + fee_open:
                self.balance -= notional + fee_open
                sl_price = entry_price - sl_factor * atr_val
                tp_price = entry_price + tp_factor * atr_val
                size_in_coins = notional / max(entry_price, 1e-8)
                new_trade = Trade(
                    direction="long",
                    entry_bar=bar_idx,
                    entry_price=entry_price,
                    size_in_coins=size_in_coins,
                    stop_loss=sl_price,
                    take_profit=tp_price,
                    notional=notional,
                    open_fee=fee_open,
                )
                self.open_trades.append(new_trade)
                self.trade_log.append(new_trade)

        def open_short(risk_scale: float):
            if sl_factor * atr_val <= 1e-8:
                return
            dollar_risk = max(0.0, equity * self.risk_fraction * risk_scale)
            size_in_coins = dollar_risk / (sl_factor * atr_val)
            entry_price = o * (1.0 - self.slippage_rate)
            notional = abs(size_in_coins * entry_price)
            notional = min(notional, self.balance * max_alloc)
            if notional < self.min_notional:
                return
            fee_open = notional * self.commission_rate
            if notional > 0 and self.balance >= notional + fee_open:
                # резервируем маржу под шорт
                self.balance -= notional + fee_open
                sl_price = entry_price + sl_factor * atr_val
                tp_price = entry_price - tp_factor * atr_val
                new_trade = Trade(
                    direction="short",
                    entry_bar=bar_idx,
                    entry_price=entry_price,
                    size_in_coins=-notional / max(entry_price, 1e-8),
                    stop_loss=sl_price,
                    take_profit=tp_price,
                    notional=notional,
                    open_fee=fee_open,
                )
                self.open_trades.append(new_trade)
                self.trade_log.append(new_trade)

        net = open_long_frac - open_short_frac
        if abs(net) > 1e-3:
            (open_long if net > 0 else open_short)(abs(net))

        # Штраф за отсутствие АКТИВНЫХ позиций
        extra_penalty = 0.0
        active_open = sum(1 for t in self.open_trades if not t.closed)
        if active_open == 0:
            self.consecutive_no_trade_steps += 1
            if self.penalize_no_trade_steps:
                extra_penalty += self.no_trade_penalty
                if self.consecutive_no_trade_steps > self.consecutive_no_trade_allowed:
                    extra_penalty += self.no_trade_penalty
        else:
            self.consecutive_no_trade_steps = 0

        self.current_bar += 1

        if self.max_bars is not None:
            used_bars = self.current_bar - self.start_bar
            if used_bars >= self.max_bars:
                truncated = True
        if self.current_bar >= self.n_bars:
            terminated = True

        unrealized = 0.0
        reserve_total = 0.0
        for t in self.open_trades:
            if t.closed:
                continue
            if t.direction == "long":
                unrealized += (c - t.entry_price) * t.size_in_coins
            else:
                unrealized += (t.entry_price - c) * abs(t.size_in_coins)
            reserve_total += t.notional
        current_equity = self.balance + unrealized + reserve_total

        if self.balance <= 0.0 or current_equity <= 0.0:
            terminated = True

        # Форс-клоуз на конце
        info = {}
        if terminated or truncated:
            forced_close_pnl = 0.0
            for trade in self.open_trades:
                if trade.closed:
                    continue
                px = c * (1.0 - self.slippage_rate) if trade.direction == "long" else c * (1.0 + self.slippage_rate)
                forced_close_pnl += self._settle_trade(trade, px, self.current_bar)
            info["forced_close_pnl"] = forced_close_pnl
            # После принудительного закрытия все позиции закрыты,
            # поэтому equity совпадает с текущим балансом
            current_equity = self.balance

        # ===== Reward: лог-доходность портфеля (equity учитывает стоимость открытых позиций)
        if not hasattr(self, "prev_equity"):
            self.prev_equity = current_equity
        delta = np.log((current_equity + 1e-6) / (self.prev_equity + 1e-6))
        reward = float(delta * 100.0 * self.reward_scaling)
        self.prev_equity = current_equity

        self.penalty_total += extra_penalty
        reward -= (extra_penalty / max(self.initial_balance, 1e-8)) * self.reward_scaling

        if terminated or truncated:
            gross_pnl = float(sum(t.pnl for t in self.trade_log))
            net_pnl = gross_pnl - self.penalty_total
            info["episode_summary"] = {
                "final_balance": self.balance,
                "trades_opened": len(self.trade_log),
                "trades_closed": sum(1 for t in self.trade_log if t.closed),
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "penalty_total": self.penalty_total,
                "forced_close_pnl": info.get("forced_close_pnl", 0.0),
            }

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
