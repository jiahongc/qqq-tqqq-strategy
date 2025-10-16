import os
import datetime as dt
from typing import Optional, Dict, Any

import pandas as pd
from alpaca_trade_api import REST


def _get_env(name: str, default: Optional[str] = None) -> str:
    val = os.getenv(name, default)
    if val is None:
        raise RuntimeError(f"Missing required env var: {name}")
    return val


class AlpacaClient:
    """Thin wrapper around Alpaca REST for paper trading and daily bar fetches."""

    def __init__(self) -> None:
        base_url = _get_env("ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2")
        key_id = _get_env("ALPACA_API_KEY_ID")
        secret = _get_env("ALPACA_API_SECRET_KEY")
        self.data_feed = os.getenv("ALPACA_DATA_FEED", "iex")
        self.api = REST(key_id, secret, base_url)

    # ------------- Data -------------
    def get_daily_opens(self, symbols: list[str], limit: int = 260) -> Dict[str, pd.Series]:
        """Return dict of symbol -> Series of daily open prices (indexed by date)."""
        out: Dict[str, pd.Series] = {}
        for sym in symbols:
            bars = self.api.get_bars(sym, timeframe="1Day", limit=limit, feed=self.data_feed)
            # Convert to DataFrame for consistency
            df = bars.df
            if df.empty:
                out[sym] = pd.Series(dtype=float)
                continue
            # df index is multi; ensure date index in local tz-neutral
            s = df["open"].copy()
            s.index = pd.to_datetime(s.index)
            s.index = s.index.tz_convert(None) if s.index.tz is not None else s.index
            # Collapse to plain date
            s.index = s.index.date
            out[sym] = s
        return out

    # ------------- Trading -------------
    def get_position_qty(self, symbol: str) -> float:
        try:
            pos = self.api.get_position(symbol)
            return float(pos.qty)
        except Exception:
            return 0.0

    def get_position_avg_entry_price(self, symbol: str) -> Optional[float]:
        try:
            pos = self.api.get_position(symbol)
            return float(pos.avg_entry_price)
        except Exception:
            return None

    def get_account_cash(self) -> float:
        acct = self.api.get_account()
        return float(acct.cash)

    def market_buy(self, symbol: str, notional: Optional[float] = None, qty: Optional[float] = None) -> Dict[str, Any]:
        if notional is None and qty is None:
            raise ValueError("Provide notional or qty")
        order = self.api.submit_order(
            symbol=symbol,
            side="buy",
            type="market",
            time_in_force="day",
            notional=notional,
            qty=qty,
        )
        return order._raw

    def market_sell(self, symbol: str, qty: Optional[float] = None) -> Dict[str, Any]:
        if qty is None:
            qty = self.get_position_qty(symbol)
        if qty <= 0:
            return {"status": "no_position"}
        order = self.api.submit_order(
            symbol=symbol,
            side="sell",
            type="market",
            time_in_force="day",
            qty=qty,
        )
        return order._raw


