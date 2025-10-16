from typing import Dict, Tuple
import pandas as pd
import numpy as np

from backtest_params import (
    RULE_DROP_THRESHOLD,
    VIX_ABS_THRESHOLD,
    VIX_REL_MULTIPLIER,
    RSI_PERIOD,
    RSI_THRESHOLD,
    PROFIT_TARGET_PCT,
)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    for i in range(period + 1, len(series)):
        pg = avg_gain.iat[i - 1]
        pl = avg_loss.iat[i - 1]
        if pd.notna(pg) and pd.notna(pl):
            avg_gain.iat[i] = (pg * (period - 1) + gain.iat[i]) / period
            avg_loss.iat[i] = (pl * (period - 1) + loss.iat[i]) / period
    rs = avg_gain / avg_loss.replace(0, np.nan)
    r = 100 - (100 / (1 + rs))
    return r.fillna(50)


def compute_today_signals(opens: Dict[str, pd.Series]) -> Tuple[bool, Dict[str, bool], Dict[str, float]]:
    """Compute today's buy signals using daily open series.

    Returns: any_buy, rule_flags, metrics
    metrics keys: qqq_ret1, vix, vix_ma20, rsi14
    """
    qqq = opens.get("QQQ")
    vix = opens.get("^VIX") or opens.get("VIXY")  # fallback to VIXY if ^VIX missing

    if qqq is None or qqq.empty:
        return False, {"rule1": False, "rule2": False, "rule3": False, "rule4": False}, {}

    ret1 = qqq.pct_change()
    r1 = (qqq < qqq.shift(1)) & (qqq.shift(1) < qqq.shift(2))
    r2 = ret1 <= RULE_DROP_THRESHOLD
    if vix is not None and not vix.empty:
        vma20 = vix.rolling(20, min_periods=1).mean()
        r3 = (vix >= VIX_ABS_THRESHOLD) | (vix >= VIX_REL_MULTIPLIER * vma20)
        v_today = float(vix.iloc[-1])
        vma_today = float(vma20.iloc[-1])
    else:
        r3 = pd.Series(False, index=qqq.index)
        v_today, vma_today = np.nan, np.nan

    rsi14_series = rsi(qqq, period=RSI_PERIOD)
    r4 = rsi14_series < RSI_THRESHOLD

    any_buy = bool((r1 | r2 | r3 | r4).iloc[-1])
    rule_flags = {
        "rule1_two_red": bool(r1.iloc[-1]),
        "rule2_big_drop": bool(r2.iloc[-1]),
        "rule3_vix_spike": bool(r3.iloc[-1]),
        "rule4_rsi_lt_45": bool(r4.iloc[-1]),
    }
    metrics = {
        "qqq_ret1": float(ret1.iloc[-1]) if pd.notna(ret1.iloc[-1]) else np.nan,
        "vix": v_today,
        "vix_ma20": vma_today,
        "rsi14": float(rsi14_series.iloc[-1]) if pd.notna(rsi14_series.iloc[-1]) else np.nan,
    }
    return any_buy, rule_flags, metrics


