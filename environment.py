"""
environment.py

An advanced Gymnasium environment for hourly trading:
- Multi-bar holding
- Commission + slippage
- SL/TP checks each step
- Partial closes
- Penalty for no-trade
- Forced closure of all remaining trades at the end of the episode
- NOW with "random start" in reset(), so each episode may start from a random index.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import List, Optional

class Trade:
    """
    Stores info about a single trade (long or short).
    """
    def __init__(
        self,
        direction: str,       # 'long' or 'short'
        entry_bar: int,
        entry_price: float,
        size_in_coins: float,
        stop_loss: float,
        take_profit: float
    ):
        self.direction = direction
        self.entry_bar = entry_bar
        self.entry_price = entry_price
        self.size_in_coins = size_in_coins
        self.stop_loss = stop_loss
        self.take_profit = take_profit

        self.exit_bar: Optional[int] = None
        self.exit_price: Optional[float] = None
        self.pnl: float = 0.0
        self.closed: bool = False

    def close(self, exit_bar: int, exit_price: float):
        """
        Closes the trade at exit_price and computes final PnL.
        """
        self.exit_bar = exit_bar
        self.exit_price = exit_price
        self.closed = True

        if self.direction == 'long':
            self.pnl = (self.exit_price - self.entry_price) * self.size_in_coins
        else:  # short
            self.pnl = (self.entry_price - self.exit_price) * abs(self.size_in_coins)

    def __str__(self):
        """
        For easy debugging/printing.
        """
        return (f"Trade(dir={self.direction}, entry_bar={self.entry_bar}, "
                f"exit_bar={self.exit_bar}, entry_price={self.entry_price:.2f}, "
                f"exit_price={self.exit_price}, size={self.size_in_coins:.4f}, "
                f"pnl={self.pnl:.2f}, closed={self.closed})")


class HourlyTradingEnv(gym.Env):
    """
    A custom Gymnasium environment simulating trading on hourly candlesticks.

    Observations:
      - A window of size `window_size` with [close, volume, atr]
      - Current balance
      - Total coins (long positive, short negative)

    Actions (float array of length 5):
      [open_long_frac, open_short_frac, close_fraction, sl_factor, tp_factor]
    
    We do "random start" in reset(), combined with max_bars=..., so each episode
    covers a random slice of up to `max_bars` bars.
    """

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
        consecutive_no_trade_allowed: int = 10
    ):
        super(HourlyTradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.n_bars = len(self.df)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.max_bars = max_bars
        self.reward_scaling = reward_scaling

        self.penalize_no_trade_steps = penalize_no_trade_steps
        self.no_trade_penalty = no_trade_penalty
        self.consecutive_no_trade_allowed = consecutive_no_trade_allowed

        # Compute ATR for SL/TP
        self.df['hl_range'] = self.df['high'] - self.df['low']
        self.df['atr'] = self.df['hl_range'].rolling(14).mean().bfill()

        # Internal states
        self.current_bar: int = 0
        self.start_bar: int = 0  # <=== We'll store random start for each episode
        self.balance: float = 0.0
        self.open_trades: List[Trade] = []
        self.trade_log: List[Trade] = []
        self.consecutive_no_trade_steps = 0

        # We'll use [close, volume, atr] as features
        self.indicator_names = ['close', 'volume', 'atr']
        self.num_features = len(self.indicator_names)

        obs_dim = self.window_size * self.num_features + 2  # + balance, + total_coins
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # [open_long_frac, open_short_frac, close_fraction, sl_factor, tp_factor]
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 3, 3], dtype=np.float32),
        )

    def _get_total_coins(self) -> float:
        """
        Returns net coins from all open trades (long positive, short negative).
        """
        total = 0.0
        for t in self.open_trades:
            if not t.closed:
                total += t.size_in_coins
        return total

    def _get_obs(self) -> np.ndarray:
        """
        Builds the observation from the last `window_size` bars + [balance, total_coins].
        """
        # current_bar goes from start_bar to (start_bar + max_bars)
        # But for slicing features we just do: [current_bar - window_size : current_bar]
        start_idx = max(self.current_bar - self.window_size, 0)
        window_df = self.df.iloc[start_idx:self.current_bar]

        if len(window_df) < self.window_size:
            diff = self.window_size - len(window_df)
            if len(window_df) == 0:
                first_row = self.df.iloc[0]
            else:
                first_row = window_df.iloc[0]
            pad_df = pd.DataFrame([first_row]*diff)
            window_df = pd.concat([pad_df, window_df], ignore_index=True)

        data_list = []
        for col in self.indicator_names:
            data_list.append(window_df[col].values)
        data_array = np.array(data_list).flatten()

        total_coins = self._get_total_coins()
        obs = np.concatenate([data_array, [self.balance], [total_coins]])
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        """
        Random start approach:
          - pick random start_bar in [window_size, n_bars - max_bars], if possible
          - set current_bar = start_bar
        """
        super().reset(seed=seed)
        self.open_trades.clear()
        self.trade_log.clear()
        self.consecutive_no_trade_steps = 0
        self.balance = self.initial_balance

        # 1) Calculate how many bars we can shift
        if self.max_bars is not None:
            # we want start_bar + max_bars <= n_bars
            # => start_bar <= n_bars - max_bars
            max_start = self.n_bars - self.max_bars
        else:
            # if max_bars is None, we can in theory start anywhere
            max_start = self.n_bars - 1  # or so

        min_start = self.window_size

        if max_start < min_start:
            # if dataset is too small or max_bars is too large
            # fallback to just start = window_size
            self.start_bar = self.window_size
        else:
            # pick random start (high is exclusive, so add 1 to include max_start)
            self.start_bar = self.np_random.integers(low=min_start, high=max_start + 1)

        # 2) Now set current_bar at start_bar
        self.current_bar = self.start_bar

        # 3) Build first observation
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action: np.ndarray):
        """
        Step method:
        action = [open_long_frac, open_short_frac, close_fraction, sl_factor, tp_factor]
        """
        open_long_frac  = float(action[0])
        open_short_frac = float(action[1])
        close_fraction  = float(action[2])
        sl_factor       = float(action[3])
        tp_factor       = float(action[4])

        # Check if out of dataset
        if self.current_bar >= self.n_bars:
            obs = self._get_obs()
            reward = 0.0
            terminated = True
            truncated = False
            return obs, reward, terminated, truncated, {}

        terminated = False
        truncated = False
        reward = 0.0

        row = self.df.iloc[self.current_bar]
        o, h, l, c = row['open'], row['high'], row['low'], row['close']
        atr_val = row['atr']
        bar_idx = self.current_bar

        closed_trades = []

        # =============== 1) Check SL/TP ===============
        for trade in self.open_trades:
            if trade.closed:
                continue

            if trade.direction == 'long':
                if l <= trade.stop_loss:
                    exec_price = trade.stop_loss * (1.0 - self.slippage_rate)
                    trade.close(bar_idx, exec_price)
                    fee = (abs(trade.entry_price * trade.size_in_coins) +
                           abs(exec_price * trade.size_in_coins)) * self.commission_rate
                    trade.pnl -= fee
                    self.balance += trade.pnl
                    closed_trades.append(trade)
                elif h >= trade.take_profit:
                    exec_price = trade.take_profit * (1.0 - self.slippage_rate)
                    trade.close(bar_idx, exec_price)
                    fee = (abs(trade.entry_price * trade.size_in_coins) +
                           abs(exec_price * trade.size_in_coins)) * self.commission_rate
                    trade.pnl -= fee
                    self.balance += trade.pnl
                    closed_trades.append(trade)

            else:  # short
                if h >= trade.stop_loss:
                    exec_price = trade.stop_loss * (1.0 + self.slippage_rate)
                    trade.close(bar_idx, exec_price)
                    fee = (abs(trade.entry_price * trade.size_in_coins) +
                           abs(exec_price * trade.size_in_coins)) * self.commission_rate
                    trade.pnl -= fee
                    self.balance += trade.pnl
                    closed_trades.append(trade)
                elif l <= trade.take_profit:
                    exec_price = trade.take_profit * (1.0 + self.slippage_rate)
                    trade.close(bar_idx, exec_price)
                    fee = (abs(trade.entry_price * trade.size_in_coins) +
                           abs(exec_price * trade.size_in_coins)) * self.commission_rate
                    trade.pnl -= fee
                    self.balance += trade.pnl
                    closed_trades.append(trade)

        # =============== 2) Partial close ===============
        if close_fraction > 1e-8:
            for trade in self.open_trades:
                if trade.closed:
                    continue
                coins_to_close = trade.size_in_coins * close_fraction
                if abs(coins_to_close) < 1e-8:
                    continue

                if trade.direction == 'long':
                    exec_price = c * (1.0 - self.slippage_rate)
                    partial_pnl = (exec_price - trade.entry_price) * coins_to_close
                else:
                    exec_price = c * (1.0 + self.slippage_rate)
                    partial_pnl = (trade.entry_price - exec_price) * abs(coins_to_close)

                fee = (abs(trade.entry_price * coins_to_close) +
                       abs(exec_price * coins_to_close)) * self.commission_rate
                partial_pnl -= fee

                self.balance += partial_pnl
                trade.pnl += partial_pnl
                trade.size_in_coins -= coins_to_close

                if abs(trade.size_in_coins) < 1e-8:
                    trade.closed = True
                    trade.exit_bar = bar_idx
                    trade.exit_price = exec_price
                    closed_trades.append(trade)

        # =============== 3) Possibly open trades ===============
        no_trade_this_step = True

        if open_long_frac > 1e-8 and open_short_frac < 1e-8:
            invest_amount = self.balance * open_long_frac
            if invest_amount > 0:
                entry_price = o * (1.0 + self.slippage_rate)
                size_in_coins = invest_amount / entry_price
                sl_price = entry_price - sl_factor * atr_val
                tp_price = entry_price + tp_factor * atr_val

                new_trade = Trade(
                    direction='long',
                    entry_bar=bar_idx,
                    entry_price=entry_price,
                    size_in_coins=size_in_coins,
                    stop_loss=sl_price,
                    take_profit=tp_price
                )
                self.open_trades.append(new_trade)
                self.trade_log.append(new_trade)
                fee = invest_amount * self.commission_rate
                self.balance -= fee

                no_trade_this_step = False

        elif open_short_frac > 1e-8 and open_long_frac < 1e-8:
            invest_amount = self.balance * open_short_frac
            if invest_amount > 0:
                entry_price = o * (1.0 - self.slippage_rate)
                size_in_coins = - invest_amount / entry_price
                sl_price = entry_price + sl_factor * atr_val
                tp_price = entry_price - tp_factor * atr_val

                new_trade = Trade(
                    direction='short',
                    entry_bar=bar_idx,
                    entry_price=entry_price,
                    size_in_coins=size_in_coins,
                    stop_loss=sl_price,
                    take_profit=tp_price
                )
                self.open_trades.append(new_trade)
                self.trade_log.append(new_trade)
                fee = invest_amount * self.commission_rate
                self.balance -= fee

                no_trade_this_step = False

        step_closed_pnl = sum(t.pnl for t in closed_trades)
        reward = step_closed_pnl * self.reward_scaling

        # penalty for no-trade
        if no_trade_this_step:
            self.consecutive_no_trade_steps += 1
            if (self.penalize_no_trade_steps and
                self.consecutive_no_trade_steps > self.consecutive_no_trade_allowed):
                reward -= self.no_trade_penalty
        else:
            self.consecutive_no_trade_steps = 0

        # advance bar
        self.current_bar += 1

        # check truncated if max_bars
        truncated = False
        if self.max_bars is not None:
            # how many bars have we used in this episode so far?
            # used_bars = self.current_bar - self.start_bar
            used_bars = self.current_bar - self.start_bar
            if used_bars >= self.max_bars:
                truncated = True

        # check terminated if out of dataset
        terminated = (self.current_bar >= self.n_bars)

        # bankrupt?
        if self.balance <= 0.0:
            terminated = True
            reward -= 1000.0 * self.reward_scaling

        obs = self._get_obs()
        info = {}

        # force close if done
        if terminated or truncated:
            print("[DEBUG] End of episode. Forcing closure of open trades.")
            additional_pnl = 0.0
            forced_close_count = 0
            c_price = c  # bar's close
            for trade in self.open_trades:
                if not trade.closed:
                    if trade.direction == 'long':
                        exec_price = c_price * (1.0 - self.slippage_rate)
                    else:
                        exec_price = c_price * (1.0 + self.slippage_rate)
                    trade.close(self.current_bar, exec_price)
                    fee = (abs(trade.entry_price * trade.size_in_coins) +
                           abs(exec_price * trade.size_in_coins)) * self.commission_rate
                    trade.pnl -= fee
                    additional_pnl += trade.pnl
                    trade.closed = True
                    forced_close_count += 1
                    print(f"[DEBUG] ForceClosed trade: {trade}")

            self.balance += additional_pnl
            reward += additional_pnl * self.reward_scaling
            print(f"[DEBUG] Forced close {forced_close_count} trades at end. additional_pnl={additional_pnl:.2f}")
            print(f"[DEBUG] final balance after forced close = {self.balance:.2f}")

        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
