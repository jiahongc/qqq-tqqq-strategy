# qqq-tqqq-strategy

This repo contains a reproducible, open-only backtest for a TQQQ strategy using QQQ/VIX-based buy rules and simple sell rules. It prints a year-by-year report to the terminal and also generates a Markdown report (report.md).

## Features
- Open-only signals and execution to avoid lookahead bias
- Buy rules (all based on QQQ Open unless noted):
  - Two red opens in a row
  - Open-to-open drop ≤ -1.5%
  - VIX spike: VIX(Open) ≥ 25 or VIX(Open) ≥ 1.25 × MA20(Open)
  - RSI(Open, 14) < 45
- Sell rules:
  - Profit target: +10% (sell at TQQQ next open if today open ≥ entry × 1.10)
  - Year-end forced exit
- Yearly summary with SPY/QQQ/TQQQ open-to-open benchmarks and “Diff vs TQQQ”
- Detailed trades include: capital compounding per year, holding days, buy/sell reasons
- Local CSV-style cache for quotes to avoid Yahoo rate limits (data_cache/)
- Today’s Insight section showing rule booleans and metrics for the latest date

## Project Structure
- tqqq_qqq_backtest.py — main backtest script and CLI output
- backtest_params.py — parameters (dates, thresholds, tickers, profit target)
- data_cache/ — cached quotes (auto-created/updated)
- requirements.txt — Python dependencies
- report.md — generated Markdown report of the yearly summary

## Requirements
- Python 3.9+
- See requirements.txt for packages (pandas, numpy, yfinance, tabulate, etc.)

Install dependencies:
```bash
python -m pip install -r requirements.txt
```

## Configure Parameters
Edit backtest_params.py to tune dates and rules:
- START_DATE, END_DATE
- INIT_CAPITAL_PER_YEAR
- RULE_DROP_THRESHOLD, VIX_ABS_THRESHOLD, VIX_REL_MULTIPLIER
- RSI_PERIOD, RSI_THRESHOLD
- PROFIT_TARGET_PCT

## Quickstart
```bash
# 1) Install dependencies
python -m pip install -r requirements.txt

# 2) (Optional) Edit parameters
# open backtest_params.py and adjust START_DATE, END_DATE, thresholds, etc.

# 3) Run the backtest
python tqqq_qqq_backtest.py
```
This will:
- Load/update local cached data in data_cache/
- Print a formatted yearly summary and detailed trades
- Generate report.md with a readable Markdown table (requires tabulate)

## Data Caching
- First run downloads required ranges and saves them under data_cache/.
- Future runs only fetch missing segments (earlier/later) and merge them.
- You can delete files under data_cache/ to force a full refetch (optional).

## Notes
- All calculations and signals are based on Open prices to remain consistent.
- Benchmarks are Open-to-Open (first trading day Open to last trading day Open of each year).
- __pycache__ can be safely deleted and should be git-ignored.
