from typing import Optional


class Trade:
    """Stores info about a single trade (long or short)."""

    def __init__(
        self,
        direction: str,       # 'long' or 'short'
        entry_bar: int,
        entry_price: float,
        size_in_coins: float,
        stop_loss: float,
        take_profit: float,
        notional: float,
        open_fee: float = 0.0,
    ):
        self.direction = direction
        self.entry_bar = entry_bar
        self.entry_price = entry_price
        self.size_in_coins = size_in_coins
        self.stop_loss = stop_loss
        self.take_profit = take_profit

        self.notional = notional
        self.open_fee = open_fee

        self.exit_bar: Optional[int] = None
        self.exit_price: Optional[float] = None
        self.pnl: float = 0.0
        self.closed: bool = False

    def close(self, exit_bar: int, exit_price: float):
        """Closes the trade at exit_price and computes final PnL."""
        self.exit_bar = exit_bar
        self.exit_price = exit_price
        self.closed = True

        if self.direction == 'long':
            self.pnl = (self.exit_price - self.entry_price) * self.size_in_coins
        else:  # short
            self.pnl = (self.entry_price - self.exit_price) * abs(self.size_in_coins)

    def __str__(self):
        """For easy debugging/printing."""
        return (
            f"Trade(dir={self.direction}, entry_bar={self.entry_bar}, "
            f"exit_bar={self.exit_bar}, entry_price={self.entry_price:.2f}, "
            f"exit_price={self.exit_price}, size={self.size_in_coins:.4f}, "
            f"pnl={self.pnl:.2f}, closed={self.closed})"
        )
