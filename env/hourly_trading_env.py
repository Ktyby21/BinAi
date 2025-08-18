"""
HourlyTradingEnv

A custom Gymnasium environment simulating trading on hourly candlesticks.
Observation includes equity ratio along with several dynamically
normalized market indicators.
"""
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import List, Optional

from trade import Trade


class HourlyTradingEnv(gym.Env):
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

        self.update_data(df)

        # Internal states
        self.current_bar: int = 0
        self.start_bar: int = 0
        self.balance: float = 0.0
        self.open_trades: List[Trade] = []
        self.trade_log: List[Trade] = []
        self.consecutive_no_trade_steps = 0

        # Observation: [equity_ratio, ma_ratio, price_ratio, vol_ratio,
        #               net_exposure_ratio, open_notional_ratio]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        # [open_long_frac, open_short_frac, close_fraction, sl_factor, tp_factor]
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 3, 3], dtype=np.float32),
        )

    def update_data(self, df: pd.DataFrame):
        """Recompute indicators and reset internal dataframe."""
        self.df = df.reset_index(drop=True)
        self.n_bars = len(self.df)

        prev_close = self.df['close'].shift(1)
        tr = pd.concat(
            [
                self.df['high'] - self.df['low'],
                (self.df['high'] - prev_close).abs(),
                (self.df['low'] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        self.df['atr'] = tr.rolling(14).mean().bfill()

        self.df['ma_short'] = (
            self.df['close'].rolling(self.ma_short_window).mean().bfill()
        )
        self.df['ma_long'] = (
            self.df['close'].rolling(self.ma_long_window).mean().bfill()
        )
        self.df['vol_ma'] = (
            self.df['volume'].rolling(self.vol_ma_window).mean().bfill()
        )

    def _get_total_coins(self) -> float:
        total = 0.0
        for t in self.open_trades:
            if not t.closed:
                total += t.size_in_coins
        return total

    def _get_obs(self) -> np.ndarray:
        idx = min(self.current_bar, self.n_bars - 1)
        row = self.df.iloc[idx]
        ma_short = row['ma_short']
        ma_long = row['ma_long']
        vol_ma = row['vol_ma']

        current_price = row['close']
        unrealized = 0.0
        total_notional = 0.0
        net_notional = 0.0
        for t in self.open_trades:
            if t.closed:
                continue
            total_notional += t.notional
            if t.direction == 'long':
                unrealized += (current_price - t.entry_price) * t.size_in_coins
                net_notional += t.notional
            else:
                unrealized += (t.entry_price - current_price) * abs(t.size_in_coins)
                net_notional -= t.notional

        equity_ratio = (self.balance + unrealized) / self.initial_balance
        ma_ratio = row['ma_short'] / (ma_long + 1e-8)
        price_ratio = row['close'] / (ma_short + 1e-8)
        vol_ratio = row['volume'] / (vol_ma + 1e-8)
        net_exposure_ratio = net_notional / self.initial_balance
        open_notional_ratio = total_notional / self.initial_balance

        equity_ratio = np.clip(equity_ratio, 0.2, 5.0)
        ma_ratio = np.clip(ma_ratio, 0.2, 5.0)
        price_ratio = np.clip(price_ratio, 0.2, 5.0)
        vol_ratio = np.clip(vol_ratio, 0.2, 5.0)
        net_exposure_ratio = np.clip(net_exposure_ratio, -5.0, 5.0)
        open_notional_ratio = np.clip(open_notional_ratio, 0.0, 5.0)

        obs = np.array(
            [
                equity_ratio,
                ma_ratio,
                price_ratio,
                vol_ratio,
                net_exposure_ratio,
                open_notional_ratio,
            ],
            dtype=np.float32,
        )
        obs = np.nan_to_num(obs, nan=1.0, posinf=5.0, neginf=0.2)
        return obs

    def _settle_trade(
        self,
        trade: Trade,
        exec_price: float,
        bar_idx: int,
        proportion: float = 1.0,
    ) -> float:
        """Close whole or part of a trade and update balance and PnL.

        Returns the realized PnL portion."""
        size = trade.size_in_coins * proportion
        notional_part = trade.notional * proportion
        open_fee_part = trade.open_fee * proportion

        if trade.direction == 'long':
            proceeds = exec_price * size
            fee = abs(proceeds) * self.commission_rate
            self.balance += proceeds - fee
            pnl_part = proceeds - notional_part - open_fee_part - fee
        else:
            cost = abs(size) * exec_price
            fee = cost * self.commission_rate
            self.balance += (notional_part - cost - fee)
            pnl_part = (notional_part - cost) - open_fee_part - fee
        
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

        min_start = self.window_size

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
        open_long_frac = float(np.clip(action[0], 0.0, 1.0))
        open_short_frac = float(np.clip(action[1], 0.0, 1.0))
        close_fraction = float(np.clip(action[2], 0.0, 1.0))
        sl_factor = float(np.clip(action[3], 0.5, 3.0))
        tp_factor = float(np.clip(action[4], 0.5, 3.0))

        if self.current_bar >= self.n_bars:
            obs = self._get_obs()
            return obs, 0.0, True, False, {}

        terminated = False
        truncated = False

        row = self.df.iloc[self.current_bar]
        o, h, l, c = row['open'], row['high'], row['low'], row['close']
        atr_val = float(max(row['atr'], 1e-8))
        bar_idx = self.current_bar

        # ======= 1) Check SL/TP =======
        for trade in self.open_trades:
            if trade.closed:
                continue
            if trade.direction == 'long':
                if l <= trade.stop_loss:
                    exec_price = trade.stop_loss * (1.0 - self.slippage_rate)
                    self._settle_trade(trade, exec_price, bar_idx)
                elif h >= trade.take_profit:
                    exec_price = trade.take_profit * (1.0 - self.slippage_rate)
                    self._settle_trade(trade, exec_price, bar_idx)
            else:
                if h >= trade.stop_loss:
                    exec_price = trade.stop_loss * (1.0 + self.slippage_rate)
                    self._settle_trade(trade, exec_price, bar_idx)
                elif l <= trade.take_profit:
                    exec_price = trade.take_profit * (1.0 + self.slippage_rate)
                    self._settle_trade(trade, exec_price, bar_idx)

        # ======= 2) Partial close =======
        if close_fraction > 1e-8:
            for trade in self.open_trades:
                if trade.closed:
                    continue
                exec_price = (
                    c * (1.0 - self.slippage_rate)
                    if trade.direction == 'long'
                    else c * (1.0 + self.slippage_rate)
                )
                self._settle_trade(trade, exec_price, bar_idx, proportion=close_fraction)

        # ======= 3) Possibly open trades =======
        no_trade_this_step = True
        max_alloc = 0.3  # limit per-trade allocation
        if open_long_frac > 1e-8 and open_short_frac < 1e-8:
            invest_amount = min(
                self.balance * open_long_frac, self.balance * max_alloc
            )
            if invest_amount > 0:
                entry_price = o * (1.0 + self.slippage_rate)
                size_in_coins = invest_amount / entry_price
                sl_price = entry_price - sl_factor * atr_val
                tp_price = entry_price + tp_factor * atr_val
                fee = invest_amount * self.commission_rate
                self.balance -= invest_amount + fee
                new_trade = Trade(
                    direction="long",
                    entry_bar=bar_idx,
                    entry_price=entry_price,
                    size_in_coins=size_in_coins,
                    stop_loss=sl_price,
                    take_profit=tp_price,
                    notional=invest_amount,
                    open_fee=fee,
                )
                self.open_trades.append(new_trade)
                self.trade_log.append(new_trade)
                no_trade_this_step = False
        elif open_short_frac > 1e-8 and open_long_frac < 1e-8:
            invest_amount = min(
                self.balance * open_short_frac, self.balance * max_alloc
            )
            if invest_amount > 0:
                entry_price = o * (1.0 - self.slippage_rate)
                size_in_coins = -invest_amount / max(entry_price, 1e-8)
                sl_price = entry_price + sl_factor * atr_val
                tp_price = entry_price - tp_factor * atr_val
                fee = invest_amount * self.commission_rate
                self.balance += invest_amount - fee
                new_trade = Trade(
                    direction="short",
                    entry_bar=bar_idx,
                    entry_price=entry_price,
                    size_in_coins=size_in_coins,
                    stop_loss=sl_price,
                    take_profit=tp_price,
                    notional=invest_amount,
                    open_fee=fee,
                )
                self.open_trades.append(new_trade)
                self.trade_log.append(new_trade)
                no_trade_this_step = False

        # Treat holding existing positions as an action
        if no_trade_this_step and len(self.open_trades) > 0:
            no_trade_this_step = False

        extra_penalty = 0.0
        if no_trade_this_step:
            self.consecutive_no_trade_steps += 1
            if self.penalize_no_trade_steps and len(self.open_trades) == 0:
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

        unrealized_check = 0.0
        for t in self.open_trades:
            if t.closed:
                continue
            if t.direction == 'long':
                unrealized_check += (c - t.entry_price) * t.size_in_coins
            else:
                unrealized_check += (t.entry_price - c) * abs(t.size_in_coins)
        current_equity_check = self.balance + unrealized_check
        if current_equity_check <= 0.0 or self.balance <= 0.0:
            terminated = True

        info = {}
        if terminated or truncated:
            forced_close_pnl = 0.0
            c_price = c
            for trade in self.open_trades:
                if not trade.closed:
                    exec_price = (
                        c_price * (1.0 - self.slippage_rate)
                        if trade.direction == 'long'
                        else c_price * (1.0 + self.slippage_rate)
                    )
                    forced_close_pnl += self._settle_trade(trade, exec_price, self.current_bar)
            info["forced_close_pnl"] = forced_close_pnl

        # Compute reward based on equity change
        unrealized = 0.0
        for t in self.open_trades:
            if t.closed:
                continue
            if t.direction == 'long':
                unrealized += (c - t.entry_price) * t.size_in_coins
            else:
                unrealized += (t.entry_price - c) * abs(t.size_in_coins)
        current_equity = self.balance + unrealized
        if not hasattr(self, "prev_equity"):
            self.prev_equity = current_equity
        delta = np.log((current_equity + 1e-6) / (self.prev_equity + 1e-6))
        reward = float(np.clip(delta * 100.0, -1.0, 1.0) * self.reward_scaling)
        self.prev_equity = current_equity
        reward -= (extra_penalty / max(self.initial_balance, 1e-8)) * self.reward_scaling

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
