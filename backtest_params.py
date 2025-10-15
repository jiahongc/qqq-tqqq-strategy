"""
Backtest parameters for TQQQ strategy.

Edit these values to tune your research range and signal thresholds.
"""

# Date range
START_DATE = "2015-01-01"
END_DATE = "2025-12-31"

# Capital
INIT_CAPITAL_PER_YEAR = 10000.0

# Signals
RULE_DROP_THRESHOLD = -0.015  # QQQ open-to-open <= -1.5%
VIX_ABS_THRESHOLD = 25.0      # VIX(Open) >= 25
VIX_REL_MULTIPLIER = 1.25     # or VIX(Open) >= 1.25 * VIX(Open)_20d_avg
RSI_PERIOD = 14
RSI_THRESHOLD = 45.0

# Exit rules
PROFIT_TARGET_PCT = 0.15      # 15% target on TQQQ open-to-open

# Tickers
TICK_QQQ = "QQQ"
TICK_TQQQ = "TQQQ"
TICK_VIX = "^VIX"
TICK_SPY = "SPY"


