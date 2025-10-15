# Code Review Suggestions

## 1. Avoid forward-filling across mismatched trading calendars
- **Issue**: `fetch_data()` builds a union of all ticker calendars and then forward-fills prices for each series. After this reindexing step, the backtest loop sees every date as tradable for QQQ and TQQQ, even on sessions when one of the products was closed. The forward-fill therefore reuses stale opens and can produce trades on synthetic data points (e.g., holiday gaps where VIX traded but QQQ/TQQQ did not).
- **Recommendation**: Align the datasets on the intersection of trading days (or at least drop rows where QQQ/TQQQ are missing) before evaluating signals/executions. Alternatively, keep the raw indexes and join on-demand inside the loop so non-trading days are skipped naturally.
- **References**: `tqqq_qqq_backtest.py` lines 203-212 show the union + `ffill()` logic that introduces the problem.【F:tqqq_qqq_backtest.py†L201-L213】

## 2. Documented profit target disagrees with the implementation
- **Issue**: The README and the module docstring state a +10% profit target (`entry * 1.10`), but `PROFIT_TARGET_PCT` is set to `0.15` (15%). The generated report also reflects a 15% target. This inconsistency makes it hard for readers to trust the results.
- **Recommendation**: Decide on the intended threshold and update either the configuration default or the docs so they agree. If 15% is preferred, update the README and docstring; if 10% is correct, change `PROFIT_TARGET_PCT` and any report text.
- **References**: README profit-target bullet, module docstring, and `backtest_params.py` default.【F:README.md†L13-L14】【F:tqqq_qqq_backtest.py†L12-L15】【F:backtest_params.py†L21-L23】

## 3. Preserve exit metadata for forced final liquidation
- **Issue**: When the script forces an exit at the last available date (end of dataset), it does not populate `exit_reason`. In the Markdown report, the final trade for 2025 therefore shows a blank sell reason, which can confuse readers who expect every exit to have a rationale.
- **Recommendation**: Set a specific `exit_reason` (e.g., `"data_end"` or `"final"`) in the cleanup branch so downstream reports stay consistent.
- **References**: Forced liquidation block at the end of `backtest()` omits the `exit_reason` assignment.【F:tqqq_qqq_backtest.py†L362-L369】

## 4. Derive `YEAR_RANGE` from available data instead of static dates
- **Issue**: `YEAR_RANGE` is computed from the configured `START_DATE`/`END_DATE` without checking whether data exists for the later years. If the cache lacks future observations (e.g., running mid-year), the yearly summary still prints rows with zero trades and placeholder benchmark values, which overstates coverage.
- **Recommendation**: Build the list of years from the actual quote indexes returned by `fetch_data()` (e.g., `range(qqq.index.year.min(), qqq.index.year.max() + 1)`) so the report only includes years that have data.
- **References**: Static `YEAR_RANGE` definition near the top of the script.【F:tqqq_qqq_backtest.py†L60-L61】
