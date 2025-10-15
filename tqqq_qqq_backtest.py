#!/usr/bin/env python3
"""
TQQQ Strategy Backtest (2015–2025) — Year-by-Year Report

Signals (all indicators computed using OPEN prices unless noted):
  BUY (from QQQ + VIX; execute in TQQQ at next open) if ANY:
    1) Two consecutive red opens on QQQ (open[t] < open[t-1] < open[t-2])
    2) QQQ open-to-open drop <= -1.5%
    3) VIX spike: VIX(Open) >= 25 OR VIX(Open) >= 1.25 * VIX(Open)_20d_avg
    4) QQQ RSI-14 computed on QQQ(Open) < 45

Exit:
  SELL only using OPEN prices:
    - Profit target: sell when today's TQQQ OPEN >= entry_price * 1.10
    - Year-end forced exit: sell at the last trading day's OPEN of the year if still holding

Report (per year 2015–2025 inclusive):
  - Number of buys / sells
  - List each trade with entry date/price, exit date/price, P&L (% and $ for $10k lot)
  - Aggregate performance per year
  - Buy & hold benchmarks for that year (QQQ, TQQQ): Jan-1 open to Dec-31 close (or nearest trading days)

Usage:
  python tqqq_qqq_backtest_2020_2024.py
  (Optionally edit the YEAR_RANGE below.)
"""

import sys
import math
import os
import io
import contextlib
from dataclasses import dataclass
from typing import List, Optional, Dict
import pandas as pd
import numpy as np

try:
    import yfinance as yf
except Exception as e:
    print("Please install yfinance: pip install yfinance")
    raise

# ------------------ Config ------------------
from backtest_params import (
    TICK_QQQ,
    TICK_TQQQ,
    TICK_VIX,
    TICK_SPY,
    START_DATE,
    END_DATE,
    INIT_CAPITAL_PER_YEAR,
    RULE_DROP_THRESHOLD,
    VIX_ABS_THRESHOLD,
    VIX_REL_MULTIPLIER,
    RSI_PERIOD,
    RSI_THRESHOLD,
    PROFIT_TARGET_PCT,
)
YEAR_RANGE = list(range(pd.to_datetime(START_DATE).year, pd.to_datetime(END_DATE).year + 1))
CACHE_DIR = "data_cache"

# ------------------ Helpers ------------------

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI on a price series."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Initial average gains/losses: simple mean of first 'period' values
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Wilder smoothing for subsequent points
    for i in range(period + 1, len(series)):
        prev_gain = avg_gain.iat[i - 1]
        prev_loss = avg_loss.iat[i - 1]
        if pd.notna(prev_gain) and pd.notna(prev_loss):
            avg_gain.iat[i] = (prev_gain * (period - 1) + gain.iat[i]) / period
            avg_loss.iat[i] = (prev_loss * (period - 1) + loss.iat[i]) / period

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

@dataclass
class Trade:
    entry_date: pd.Timestamp
    entry_price: float
    capital_at_entry: float  # capital used for this trade
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    reason: str = ""  # which buy rule triggered
    reason_detailed: str = ""  # descriptive context for reason
    exit_reason: str = ""  # why we exited (e.g., target, year_end)

    def pnl_pct(self) -> Optional[float]:
        if self.exit_price is None:
            return None
        return (self.exit_price / self.entry_price) - 1.0

    def pnl_usd(self) -> Optional[float]:
        p = self.pnl_pct()
        if p is None:
            return None
        return self.capital_at_entry * p
    
    def final_capital(self) -> Optional[float]:
        if self.exit_price is None:
            return None
        return self.capital_at_entry * (1 + self.pnl_pct())

# ------------------ Data ------------------

def _cache_path(ticker: str) -> str:
    safe = ticker.replace("^", "_")
    return os.path.join(CACHE_DIR, f"{safe}.csv")

def _fix_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).capitalize() for c in df.columns]
    return df

def _read_cached(ticker: str) -> Optional[pd.DataFrame]:
    path = _cache_path(ticker)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        return _fix_columns(df)
    except Exception:
        return None

def _write_cached(ticker: str, df: pd.DataFrame) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    df_sorted = df.sort_index()
    df_sorted.to_csv(_cache_path(ticker), index=True, date_format="%Y-%m-%d")

def _yf_download(ticker: str, start: str, end: str) -> pd.DataFrame:
    # Suppress yfinance stdout/stderr noise
    buf_out, buf_err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    except Exception:
        # On error, return empty frame and continue; cache merge will still work
        df = pd.DataFrame()
    return _fix_columns(df)

def _ensure_cached(ticker: str, start: str, end: str) -> pd.DataFrame:
    cached = _read_cached(ticker)
    if cached is None or cached.empty:
        df = _yf_download(ticker, start=start, end=end)
        _write_cached(ticker, df)
        return df

    need_start = pd.to_datetime(start)
    need_end = pd.to_datetime(end)
    have_start = cached.index.min()
    have_end = cached.index.max()

    parts = [cached]

    # Fetch earlier missing segment
    if pd.notna(have_start) and need_start < have_start:
        df_early = _yf_download(ticker, start=start, end=have_start.strftime("%Y-%m-%d"))
        parts.append(df_early)

    # Fetch later missing segment (end in yfinance is exclusive-like; still safe to request to need_end)
    if pd.notna(have_end) and need_end > have_end:
        start_late = (have_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        df_late = _yf_download(ticker, start=start_late, end=end)
        parts.append(df_late)

    if len(parts) > 1:
        merged = pd.concat(parts).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]  # de-dup on index
        _write_cached(ticker, merged)
        return merged
    else:
        return cached

def fetch_data():
    # Use cache for each ticker; update only missing ranges
    qqq = _ensure_cached(TICK_QQQ, START_DATE, END_DATE)
    tqqq = _ensure_cached(TICK_TQQQ, START_DATE, END_DATE)
    vix = _ensure_cached(TICK_VIX, START_DATE, END_DATE)
    spy = _ensure_cached(TICK_SPY, START_DATE, END_DATE)

    # Align to business days, forward fill for VIX missing
    idx = qqq.index.union(tqqq.index).union(vix.index).union(spy.index).sort_values()
    qqq = qqq.reindex(idx).ffill()
    tqqq = tqqq.reindex(idx).ffill()
    vix = vix.reindex(idx).ffill()
    spy = spy.reindex(idx).ffill()

    return qqq, tqqq, vix, spy

# ------------------ Signals ------------------

def compute_buy_signals(qqq: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(index=qqq.index)
    qopen = qqq["Open"]
    qclose = qqq["Close"]
    vopen = vix["Open"]

    # Rule 1: two consecutive red opens (open[t] < open[t-1] < open[t-2])
    red1 = qopen < qopen.shift(1)
    red2 = qopen.shift(1) < qopen.shift(2)
    rule1 = red1 & red2

    # Rule 2: open-to-open drop <= configured threshold
    ret1 = qopen.pct_change()
    rule2 = ret1 <= RULE_DROP_THRESHOLD

    # Rule 3: VIX spike using Open and Open-based 20d MA
    vma20 = vopen.rolling(20, min_periods=1).mean()
    rule3 = (vopen >= VIX_ABS_THRESHOLD) | (vopen >= VIX_REL_MULTIPLIER * vma20)

    # Rule 4: RSI(Open)
    rsi14 = rsi(qopen, period=RSI_PERIOD)
    rule4 = rsi14 < RSI_THRESHOLD

    # SMA200(Open) retained for sell rule context only
    sma200_open = qopen.rolling(200, min_periods=200).mean()

    df["rule1_two_red"] = rule1
    df["rule2_big_drop"] = rule2
    df["rule3_vix_spike"] = rule3
    df["rule4_rsi_lt_45"] = rule4
    # any_buy is OR of boolean rule columns only (without SMA200 close rule)
    df["any_buy"] = df[[
        "rule1_two_red",
        "rule2_big_drop",
        "rule3_vix_spike",
        "rule4_rsi_lt_45",
    ]].any(axis=1)
    df["rsi14"] = rsi14
    df["qqq_ret1"] = ret1
    df["vix"] = vopen
    df["vix_ma20"] = vma20
    df["qqq_sma200_open"] = sma200_open

    # For detailed reasons: capture prior-day stats for signal day (based on Open)
    df["prev_close"] = qopen.shift(1)
    df["prev2_close"] = qopen.shift(2)
    df["prev_ret"] = qopen.pct_change().shift(0)
    df["prev_ret1"] = qopen.pct_change().shift(1)
    df["prev_vix"] = vopen

    return df

# ------------------ Backtest ------------------

def backtest(qqq: pd.DataFrame, tqqq: pd.DataFrame, vix: pd.DataFrame, spy: pd.DataFrame) -> Dict[int, Dict]:
    sig = compute_buy_signals(qqq, vix)
    results_per_year: Dict[int, Dict] = {}

    holding: Optional[Trade] = None
    trades: List[Trade] = []
    current_capital = INIT_CAPITAL_PER_YEAR  # Start with $10k
    current_year = None  # Track year changes
    # Exits are evaluated at the OPEN only (no intraday logic)

    # Iterate over dates
    for i, dt in enumerate(sig.index):
        # Skip if outside our full data intersection
        if dt not in tqqq.index or dt not in qqq.index:
            continue

        # Check if we're in a new year - before resetting, force exit any open position at last trading day's OPEN of previous year
        if current_year is None or dt.year != current_year:
            if current_year is not None and holding is not None:
                # Use the previous index date (last session of the prior year) for exit
                prev_dt = sig.index[i - 1] if i > 0 else None
                if prev_dt is not None and prev_dt in tqqq.index:
                    exit_dt = prev_dt
                    exit_price = float(tqqq.loc[exit_dt, "Open"])
                else:
                    # Fallback to current dt open if prev_dt missing (should be rare)
                    exit_dt = dt
                    exit_price = float(tqqq.loc[dt, "Open"]) if dt in tqqq.index else float(holding.entry_price)
                holding.exit_date = exit_dt
                holding.exit_price = exit_price
                holding.exit_reason = "year_end"
                current_capital = holding.final_capital()
                trades.append(holding)
                holding = None
            current_year = dt.year
            current_capital = INIT_CAPITAL_PER_YEAR
            # If we had an open position from previous year, close it (shouldn't happen with our logic)
            if holding is not None:
                holding = None

        # If we have an open trade, check profit target using today's OPEN
        if holding is not None:
            open_today = float(tqqq.loc[dt, "Open"])
            if open_today >= holding.entry_price * (1.0 + PROFIT_TARGET_PCT):
                holding.exit_date = dt
                holding.exit_price = open_today
                holding.exit_reason = "target"
                # Update capital with the gains/losses
                current_capital = holding.final_capital()
                trades.append(holding)
                holding = None
                # After exit, we do NOT open new trade same day; signals evaluated next day
                continue

        # If not holding, check if any buy signal fired YESTERDAY to buy today (at open)
        # (We generate signal on t-1 close, buy at today's open)
        if holding is None:
            prev_idx = i - 1
            if prev_idx >= 0:
                prev = sig.index[prev_idx]
                if sig.loc[prev, "any_buy"]:
                    # Determine which rule triggered (priority order 1..4)
                    reason = None
                    for rule_col, label in [
                        ("rule1_two_red", "two_red"),
                        ("rule2_big_drop", "big_drop"),
                        ("rule3_vix_spike", "vix_spike"),
                        ("rule4_rsi_lt_45", "rsi_lt_45"),
                    ]:
                        if bool(sig.loc[prev, rule_col]):
                            reason = label
                            break
                    # Build reason_detailed from prev day context
                    detailed = ""
                    if reason == "two_red":
                        d1 = (sig.loc[prev, "prev_close"] / sig.loc[prev, "prev2_close"] - 1.0) if pd.notna(sig.loc[prev, "prev2_close"]) else None
                        d2 = sig.loc[prev, "qqq_ret1"]
                        if d1 is not None and d2 is not None:
                            detailed = f"two_red: day-2 {d1*100:.2f}%, day-1 {d2*100:.2f}%"
                        else:
                            detailed = "two_red"
                    elif reason == "big_drop":
                        drop = sig.loc[prev, "qqq_ret1"]
                        detailed = f"drop: {drop*100:.2f}%" if pd.notna(drop) else "drop"
                    elif reason == "vix_spike":
                        v = sig.loc[prev, "vix"]
                        ma = sig.loc[prev, "vix_ma20"]
                        if pd.notna(v) and pd.notna(ma):
                            detailed = f"vix: {v:.2f} (ma20 {ma:.2f})"
                        elif pd.notna(v):
                            detailed = f"vix: {v:.2f}"
                        else:
                            detailed = "vix_spike"
                    elif reason == "rsi_lt_45":
                        r = sig.loc[prev, "rsi14"]
                        detailed = f"rsi14: {r:.2f}" if pd.notna(r) else "rsi_lt_45"
                    entry_price = float(tqqq.loc[dt, "Open"])
                    holding = Trade(
                        entry_date=dt, 
                        entry_price=entry_price, 
                        capital_at_entry=current_capital,
                        reason=reason,
                        reason_detailed=detailed
                    )

    # If a trade is still open at the very end, force exit on the last available open
    if holding is not None:
        last_dt = sig.index[-1]
        exit_price = float(tqqq.loc[last_dt, "Open"]) if last_dt in tqqq.index else float(holding.entry_price)
        holding.exit_date = last_dt
        holding.exit_price = exit_price
        current_capital = holding.final_capital()
        trades.append(holding)
        holding = None

    # Build per-year stats
    # Collate trades by year (by EXIT date if exists else by ENTRY year)
    trades_df = pd.DataFrame([
        {
            "entry_date": tr.entry_date,
            "entry_price": tr.entry_price,
            "capital_at_entry": tr.capital_at_entry,
            "exit_date": tr.exit_date,
            "exit_price": tr.exit_price,
            "reason": tr.reason,
            "reason_detailed": tr.reason_detailed,
            "exit_reason": tr.exit_reason,
            "pnl_pct": tr.pnl_pct(),
            "pnl_usd": tr.pnl_usd() if tr.pnl_usd() is not None else None,
            "final_capital": tr.final_capital() if tr.final_capital() is not None else None,
            "holding_days": (tr.exit_date - tr.entry_date).days if tr.exit_date is not None else None,
        }
        for tr in trades
    ])

    # Benchmarks per year
    bench = {}
    for y in YEAR_RANGE:
        ymask = (qqq.index.year == y)
        qqq_y = qqq.loc[ymask]
        tqqq_y = tqqq.loc[ymask]
        spy_y = spy.loc[ymask]
        if len(qqq_y) == 0 or len(tqqq_y) == 0:
            bench[y] = {"qqq": None, "tqqq": None, "spy": None}
            continue

        # Buy & hold from first trading day OPEN to last trading day OPEN of the year
        qqq_ret = (qqq_y["Open"].iloc[-1] / qqq_y["Open"].iloc[0]) - 1.0
        tqqq_ret = (tqqq_y["Open"].iloc[-1] / tqqq_y["Open"].iloc[0]) - 1.0
        spy_ret = (spy_y["Open"].iloc[-1] / spy_y["Open"].iloc[0]) - 1.0 if len(spy_y) > 0 else None
        bench[y] = {"qqq": qqq_ret, "tqqq": tqqq_ret, "spy": spy_ret}

    # Per-year aggregation of trades
    for y in YEAR_RANGE:
        # Trades that EXITED in year y
        td_y = trades_df[(trades_df["exit_date"].notna()) & (trades_df["exit_date"].dt.year == y)].copy()

        # Count buys/sells (sells == number of exited trades in that year)
        n_sells = len(td_y)
        # Buys that led to those sells are same count; but we also report buys that happened in year y
        n_buys_in_y = sum(1 for tr in trades if tr.entry_date.year == y)

        # Get starting and ending capital for the year
        # Year always starts with $10k
        start_capital = INIT_CAPITAL_PER_YEAR
        end_capital = td_y["final_capital"].iloc[-1] if len(td_y) > 0 else start_capital
        
        pnl_usd_year = end_capital - start_capital if n_sells > 0 else 0.0
        pnl_pct_year = (end_capital / start_capital) - 1.0 if n_sells > 0 and start_capital > 0 else 0.0

        results_per_year[y] = {
            "year": y,
            "num_buys": n_buys_in_y,
            "num_sells": n_sells,
            "pnl_pct_year": pnl_pct_year,
            "pnl_usd_year": pnl_usd_year,
            "start_capital": start_capital,
            "end_capital": end_capital,
            "trades_exited": td_y.sort_values("exit_date").to_dict(orient="records"),
            "benchmarks": bench.get(y, {"qqq": None, "tqqq": None}),
        }

    return results_per_year, trades_df, bench

# ------------------ Main ------------------

def build_yearly_summary(results_per_year: Dict[int, Dict]) -> pd.DataFrame:
    """Construct yearly summary DataFrame and benchmark diffs from backtest results.

    The returned frame has numeric rates in decimal form (e.g., 0.1234 == 12.34%).
    """
    rows = []
    for y, res in results_per_year.items():
        rows.append({
            "year": y,
            "num_buys": res["num_buys"],
            "num_sells": res["num_sells"],
            "year_return_pct": res["pnl_pct_year"],
            "year_pnl_usd": res["pnl_usd_year"],
            "start_capital": res["start_capital"],
            "end_capital": res["end_capital"],
            "bench_SPY_BH_ret": res["benchmarks"].get("spy"),
            "bench_QQQ_BH_ret": res["benchmarks"].get("qqq"),
            "bench_TQQQ_BH_ret": res["benchmarks"].get("tqqq"),
        })
    summary_df = pd.DataFrame(rows).sort_values("year")
    # Add differences vs benchmarks
    if "bench_TQQQ_BH_ret" in summary_df.columns:
        summary_df["diff_vs_TQQQ"] = summary_df["year_return_pct"] - summary_df["bench_TQQQ_BH_ret"]
    if "bench_SPY_BH_ret" in summary_df.columns:
        summary_df["diff_vs_SPY"] = summary_df["year_return_pct"] - summary_df["bench_SPY_BH_ret"]
    if "bench_QQQ_BH_ret" in summary_df.columns:
        summary_df["diff_vs_QQQ"] = summary_df["year_return_pct"] - summary_df["bench_QQQ_BH_ret"]
    return summary_df

def write_markdown_report(
    summary_df: pd.DataFrame,
    results_per_year: Dict[int, Dict],
    last_dt: pd.Timestamp,
    any_buy: bool,
    r1: bool,
    r2: bool,
    r3: bool,
    r4: bool,
    ret_str: str,
    vix_str: str,
    vma_str: str,
    rsi_str: str,
    avg_annual_return: float,
    avg_annual_spy: float,
    avg_annual_qqq: float,
    avg_annual_tqqq: float,
) -> None:
    """Write Markdown report to report.md using current results and insights.

    This mirrors the console output and includes a Today's Insight section for the last date.
    """
    lines: List[str] = []
    lines.append("# TQQQ Strategy Backtest Report\n")
    lines.append("## Strategy\n\n")
    lines.append(f"- Start Date: {START_DATE}\n")
    lines.append(f"- End Date: {END_DATE}\n")
    lines.append(f"- Capital per Year: ${INIT_CAPITAL_PER_YEAR:,.0f}\n")
    lines.append("- Buy Rules:\n")
    lines.append("  - Two Red Opens (QQQ)\n")
    lines.append(f"  - QQQ Open-to-Open drop <= {RULE_DROP_THRESHOLD*100:.2f}%\n")
    lines.append(f"  - VIX spike: VIX(Open) >= {VIX_ABS_THRESHOLD:.0f} or >= {VIX_REL_MULTIPLIER:.2f} × MA20(Open)\n")
    lines.append(f"  - RSI(Open, {RSI_PERIOD}) < {RSI_THRESHOLD:.0f}\n")
    lines.append("- Sell Rules:\n")
    lines.append(f"  - Profit target at +{PROFIT_TARGET_PCT*100:.0f}% on TQQQ Open\n")
    lines.append("  - Year-end forced exit\n\n")
    # Averages (dynamic)
    lines.append("## Averages (Open-Open)\n\n")
    lines.append(f"- Strategy avg annual return: {avg_annual_return:.2f}%\n")
    lines.append(f"- SPY avg annual return: {avg_annual_spy:.2f}%\n")
    lines.append(f"- QQQ avg annual return: {avg_annual_qqq:.2f}%\n")
    lines.append(f"- TQQQ avg annual return: {avg_annual_tqqq:.2f}%\n\n")

    # Today's Insight (mirrors console output)
    lines.append("## Today's Insight\n\n")
    lines.append(f"- Date: {last_dt.date()}\n")
    lines.append(f"- any_buy: {any_buy}\n")
    lines.append(f"- rule1_two_red: {r1}\n")
    lines.append(f"- rule2_big_drop: {r2} (qqq open->open ret1: {ret_str})\n")
    lines.append(f"- rule3_vix_spike: {r3} (vix: {vix_str}  ma20: {vma_str})\n")
    lines.append(f"- rule4_rsi_lt_45: {r4} (rsi14: {rsi_str})\n\n")

    # Yearly summary
    lines.append("## Yearly Summary\n\n")
    md = summary_df.copy()
    for col in [
        "year_return_pct",
        "bench_SPY_BH_ret",
        "bench_QQQ_BH_ret",
        "bench_TQQQ_BH_ret",
        "diff_vs_SPY",
        "diff_vs_QQQ",
        "diff_vs_TQQQ",
    ]:
        if col in md.columns:
            md[col] = (md[col] * 100).map(lambda x: f"{x:.2f}%")
    md = md[[
        "year", "num_buys", "num_sells", "year_return_pct", "year_pnl_usd", "end_capital",
        "bench_SPY_BH_ret", "bench_QQQ_BH_ret", "bench_TQQQ_BH_ret", "diff_vs_SPY", "diff_vs_QQQ", "diff_vs_TQQQ"
    ]]
    lines.append(md.to_markdown(index=False))

    # Detailed trades by year
    lines.append("\n\n## Detailed Trades by Year\n")
    for y in sorted(results_per_year.keys()):
        res = results_per_year[y]
        trades_in_year = res.get("trades_exited", [])
        lines.append(f"\n### {y}\n")
        if len(trades_in_year) == 0:
            lines.append("No completed trades this year.\n")
        else:
            td = pd.DataFrame(trades_in_year)
            def _fmt_money(x):
                try:
                    return f"${x:,.2f}"
                except Exception:
                    return ""
            def _fmt_pct(x):
                try:
                    return f"{x*100:.2f}%"
                except Exception:
                    return ""
            def _fmt_date(x):
                try:
                    return pd.to_datetime(x).strftime('%Y-%m-%d')
                except Exception:
                    return ""
            # Derive sell reason more
            exit_more = []
            for rr in td.get("exit_reason", []).fillna(""):
                if rr == "target":
                    exit_more.append(f"target +{int(PROFIT_TARGET_PCT*100)}%")
                elif rr == "year_end":
                    exit_more.append("year-end forced")
                else:
                    exit_more.append("")
            td["exit_reason_more"] = exit_more

            table = pd.DataFrame({
                "Entry Date": td["entry_date"].map(_fmt_date),
                "Entry $": td["entry_price"],
                "Exit Date": td["exit_date"].map(_fmt_date),
                "Exit $": td["exit_price"],
                "Hold": td.get("holding_days", None),
                "P&L %": td["pnl_pct"].map(_fmt_pct),
                "P&L $": td["pnl_usd"].map(_fmt_money),
                "Start Cap": td["capital_at_entry"].map(_fmt_money),
                "End Cap": td["final_capital"].map(_fmt_money),
                "Buy reason": td["reason"].fillna(""),
                "Buy reason more": td["reason_detailed"].fillna(""),
                "Sell reason": td["exit_reason"].fillna(""),
                "Sell reason more": td["exit_reason_more"].fillna(""),
            })
            lines.append(table.to_markdown(index=False))
    with open("report.md", "w") as f:
        f.write("\n".join(lines) + "\n")


def print_separator(char="=", length=100):
    print(char * length)

def print_header(text):
    print_separator()
    print(f"  {text}")
    print_separator()

def main():
    print("Downloading data...")
    qqq, tqqq, vix, spy = fetch_data()
    print("Computing signals and backtest...")
    results_per_year, trades_df, bench = backtest(qqq, tqqq, vix, spy)

    # Today's Insight: show buy rules booleans and metrics for the latest date
    sig = compute_buy_signals(qqq, vix)
    last_dt = sig.index[-1]
    r1 = bool(sig.loc[last_dt, "rule1_two_red"]) if "rule1_two_red" in sig.columns else False
    r2 = bool(sig.loc[last_dt, "rule2_big_drop"]) if "rule2_big_drop" in sig.columns else False
    r3 = bool(sig.loc[last_dt, "rule3_vix_spike"]) if "rule3_vix_spike" in sig.columns else False
    r4 = bool(sig.loc[last_dt, "rule4_rsi_lt_45"]) if "rule4_rsi_lt_45" in sig.columns else False
    any_buy = bool(sig.loc[last_dt, "any_buy"]) if "any_buy" in sig.columns else False
    # Metrics
    m_ret1 = float(sig.loc[last_dt, "qqq_ret1"]) if not pd.isna(sig.loc[last_dt, "qqq_ret1"]) else None
    m_rsi = float(sig.loc[last_dt, "rsi14"]) if not pd.isna(sig.loc[last_dt, "rsi14"]) else None
    m_vix = float(sig.loc[last_dt, "vix"]) if not pd.isna(sig.loc[last_dt, "vix"]) else None
    m_vma = float(sig.loc[last_dt, "vix_ma20"]) if not pd.isna(sig.loc[last_dt, "vix_ma20"]) else None

    print("\n📌 TODAY'S INSIGHT")
    print_separator("-")
    print(f"Date: {last_dt.date()}")
    print(f"  any_buy: {any_buy}")
    print(f"  rule1_two_red: {r1}  (prev2_open -> prev_open -> today_open)")
    if m_ret1 is None:
        ret_str = "NA"
    else:
        ret_str = f"{m_ret1*100:.2f}%"
    vix_str = f"{m_vix:.2f}" if m_vix is not None else "NA"
    vma_str = f"{m_vma:.2f}" if m_vma is not None else "NA"
    rsi_str = f"{m_rsi:.2f}" if m_rsi is not None else "NA"
    print(f"  rule2_big_drop: {r2}  (qqq open->open ret1: {ret_str})")
    print(f"  rule3_vix_spike: {r3}  (vix: {vix_str}  ma20: {vma_str})")
    print(f"  rule4_rsi_lt_45: {r4}  (rsi14: {rsi_str})")

    # Strategy summary (dynamic)
    print("\n🎯 STRATEGY (Open-only)")
    print_separator("-")
    print(f"  - Buy: two red opens on QQQ")
    print(f"  - Buy: QQQ open->open <= {RULE_DROP_THRESHOLD*100:.2f}%")
    print(f"  - Buy: VIX(Open) >= {VIX_ABS_THRESHOLD:.0f} or >= {VIX_REL_MULTIPLIER:.2f}× MA20(Open)")
    print(f"  - Buy: RSI(Open, {RSI_PERIOD}) < {RSI_THRESHOLD:.0f}")
    print(f"  - Sell: TQQQ +{PROFIT_TARGET_PCT*100:.0f}% target at open; Year-end forced exit")

    # Summary across years (used for console + markdown)
    summary_df = build_yearly_summary(results_per_year)
    # summary_df.to_csv("yearly_summary_2018_2025.csv", index=False)

    # Print nicely formatted results
    print("\n")
    hdr_start = pd.to_datetime(START_DATE).year
    hdr_end = pd.to_datetime(END_DATE).year
    print_header(f"TQQQ STRATEGY BACKTEST RESULTS ({hdr_start}-{hdr_end})")
    
    # Print yearly summary
    print("\n📊 YEARLY SUMMARY (Each year starts with $10,000)")
    print_separator("-")
    print(f"{'Year':<6} {'Buys':<6} {'Sells':<6} {'Return %':<12} {'P&L $':<14} {'End $':<14} {'SPY %':<10} {'QQQ %':<10} {'TQQQ %':<10} {'Diff vs SPY':<14} {'Diff vs QQQ':<14} {'Diff vs TQQQ':<14}")
    print_separator("-")
    
    for _, row in summary_df.iterrows():
        year = int(row['year'])
        buys = int(row['num_buys'])
        sells = int(row['num_sells'])
        pnl_pct = row['year_return_pct'] * 100
        pnl_usd = row['year_pnl_usd']
        end_cap = row['end_capital']
        spy_bh = row['bench_SPY_BH_ret'] * 100 if row['bench_SPY_BH_ret'] is not None else 0
        qqq_bh = row['bench_QQQ_BH_ret'] * 100 if row['bench_QQQ_BH_ret'] is not None else 0
        tqqq_bh = row['bench_TQQQ_BH_ret'] * 100 if row['bench_TQQQ_BH_ret'] is not None else 0
        diff_vs_spy = (row['diff_vs_SPY'] * 100) if 'diff_vs_SPY' in row and row['diff_vs_SPY'] is not None else 0
        diff_vs_qqq = (row['diff_vs_QQQ'] * 100) if 'diff_vs_QQQ' in row and row['diff_vs_QQQ'] is not None else 0
        diff_vs_tqqq = (row['diff_vs_TQQQ'] * 100) if 'diff_vs_TQQQ' in row and row['diff_vs_TQQQ'] is not None else 0
        
        print(f"{year:<6} {buys:<6} {sells:<6} {pnl_pct:>10.2f}%  ${pnl_usd:>11,.2f}  ${end_cap:>11,.2f}  {spy_bh:>8.2f}%  {qqq_bh:>8.2f}%  {tqqq_bh:>8.2f}%  {diff_vs_spy:>12.2f}%  {diff_vs_qqq:>12.2f}%  {diff_vs_tqqq:>12.2f}%")
    
    print_separator("-")
    
    # Print detailed trades per year
    print("\n\n📈 DETAILED TRADES BY YEAR")
    
    for y in sorted(results_per_year.keys()):
        res = results_per_year[y]
        trades_in_year = res["trades_exited"]
        
        print(f"\n  ▸ {y}")
        print_separator("-", 80)
        
        if len(trades_in_year) == 0:
            print("    No completed trades this year")
        else:
            print(f"    {'Entry Date':<12} {'Entry $':<10} {'Exit Date':<12} {'Exit $':<10} {'Hold':<6} {'P&L %':<10} {'P&L $':<12} {'Start Cap':<13} {'End Cap':<13} {'Buy reason':<15} {'Buy reason more':<40} {'Sell reason':<12} {'Sell reason more':<18}")
            print(f"    {'-'*12} {'-'*10} {'-'*12} {'-'*10} {'-'*6} {'-'*10} {'-'*12} {'-'*13} {'-'*13} {'-'*15} {'-'*40} {'-'*12} {'-'*18}")
            
            for trade in trades_in_year:
                entry_date = pd.to_datetime(trade['entry_date']).strftime('%Y-%m-%d')
                exit_date = pd.to_datetime(trade['exit_date']).strftime('%Y-%m-%d')
                entry_price = trade['entry_price']
                exit_price = trade['exit_price']
                pnl_pct = trade['pnl_pct'] * 100
                pnl_usd = trade['pnl_usd']
                start_cap = trade['capital_at_entry']
                end_cap = trade['final_capital']
                reason = trade['reason']
                reason_detailed = trade.get('reason_detailed', '')
                exit_reason = trade.get('exit_reason', '')
                # Derive a simple sell reason more string
                if exit_reason == 'target':
                    exit_reason_more = f"target +{int(PROFIT_TARGET_PCT*100)}%"
                elif exit_reason == 'year_end':
                    exit_reason_more = 'year-end forced'
                else:
                    exit_reason_more = ''
                hold_days = trade.get('holding_days', None)
                
                print(f"    {entry_date:<12} ${entry_price:<9.2f} {exit_date:<12} ${exit_price:<9.2f} {hold_days!s:<6} {pnl_pct:>8.2f}%  ${pnl_usd:>10,.2f}  ${start_cap:>11,.2f}  ${end_cap:>11,.2f}  {reason:<15} {reason_detailed:<40} {exit_reason:<12} {exit_reason_more:<18}")
    
    # Print overall statistics
    print("\n\n")
    print_header("OVERALL STATISTICS")
    
    total_trades = len(trades_df[trades_df['exit_date'].notna()])
    avg_pnl_pct = trades_df['pnl_pct'].mean() * 100 if total_trades > 0 else 0
    win_rate = (trades_df['pnl_pct'] > 0).sum() / total_trades * 100 if total_trades > 0 else 0
    
    # Calculate average annual returns
    yearly_returns = [row['year_return_pct'] for _, row in summary_df.iterrows() if row['num_sells'] > 0]
    avg_annual_return = np.mean(yearly_returns) * 100 if len(yearly_returns) > 0 else 0
    qqq_returns = [row['bench_QQQ_BH_ret'] for _, row in summary_df.iterrows() if row['bench_QQQ_BH_ret'] is not None]
    tqqq_returns = [row['bench_TQQQ_BH_ret'] for _, row in summary_df.iterrows() if row['bench_TQQQ_BH_ret'] is not None]
    avg_annual_qqq = (np.mean(qqq_returns) * 100) if len(qqq_returns) > 0 else 0
    avg_annual_tqqq = (np.mean(tqqq_returns) * 100) if len(tqqq_returns) > 0 else 0
    spy_returns = [row['bench_SPY_BH_ret'] for _, row in summary_df.iterrows() if row['bench_SPY_BH_ret'] is not None]
    avg_annual_spy = (np.mean(spy_returns) * 100) if len(spy_returns) > 0 else 0
    
    print(f"\n  Total Completed Trades: {total_trades}")
    print(f"  Average Annual Return: {avg_annual_return:.2f}%")
    print(f"  Average P&L per Trade: {avg_pnl_pct:.2f}%")
    print(f"  Average Annual QQQ (Open-Open): {avg_annual_qqq:.2f}%")
    print(f"  Average Annual TQQQ (Open-Open): {avg_annual_tqqq:.2f}%")
    print(f"  Average Annual SPY (Open-Open): {avg_annual_spy:.2f}%")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"\n  Note: Each year starts with $10,000 (not compounded across years)")
    
    # print("\n  📁 Files Saved:")
    # print(f"     - yearly_summary_2018_2025.csv")
    # print(f"     - trades_all_2018_2025.csv")
    
    print("\n  ℹ️  Note: Benchmarks calculated from first trading day's OPEN to last trading day's OPEN of each year.")
    print("\n" + "="*100 + "\n")

    # Write Markdown report
    try:
        write_markdown_report(
            summary_df=summary_df,
            results_per_year=results_per_year,
            last_dt=last_dt,
            any_buy=any_buy,
            r1=r1,
            r2=r2,
            r3=r3,
            r4=r4,
            ret_str=ret_str,
            vix_str=vix_str,
            vma_str=vma_str,
            rsi_str=rsi_str,
            avg_annual_return=avg_annual_return,
            avg_annual_spy=avg_annual_spy,
            avg_annual_qqq=avg_annual_qqq,
            avg_annual_tqqq=avg_annual_tqqq,
        )
        print("\n  📄 Markdown report written to report.md")
    except Exception as e:
        print(f"\n  ⚠️ Failed to write report.md: {e}")

if __name__ == "__main__":
    main()
