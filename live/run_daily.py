import os
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from live.alpaca_client import AlpacaClient
from live.signals_live import compute_today_signals
from live.notifications import send_email, send_sms_via_email
from backtest_params import PROFIT_TARGET_PCT


def now_et():
    tz = os.getenv("TIMEZONE", "America/New_York")
    return datetime.now(ZoneInfo(tz))


def main():
    load_dotenv()
    et = now_et()
    print(f"[run_daily] Starting at {et.isoformat()}")

    client = AlpacaClient()

    # Fetch daily opens
    syms = ["QQQ", "TQQQ", "^VIX", "VIXY"]
    opens = client.get_daily_opens(syms)

    # Compute today's buy signals using opens
    any_buy, rules, metrics = compute_today_signals(opens)

    # Position status
    tqqq_qty = client.get_position_qty("TQQQ")

    # Get today's open prices
    qqq_open_today = float(opens["QQQ"].iloc[-1]) if not opens["QQQ"].empty else None
    tqqq_open_today = float(opens["TQQQ"].iloc[-1]) if not opens["TQQQ"].empty else None

    if tqqq_open_today is None:
        print("No TQQQ open available for today; exiting.")
        return 0

    actions = []
    if tqqq_qty <= 0:
        # Consider buy
        if any_buy:
            # Use full cash as notional for simplicity
            cash = client.get_account_cash()
            if cash > 5:  # avoid dust
                order = client.market_buy("TQQQ", notional=cash)
                actions.append(f"BUY TQQQ notional=${cash:.2f} @ market: {order}")
            else:
                actions.append("BUY signal but insufficient cash")
        else:
            actions.append("No BUY today")
    else:
        # Consider sell at open >= target
        # Note: We do not track entry price here; a robust version would persist entry
        # For a simple implementation, sell when today's open is >= yesterday's open * (1+target)
        # This approximates but is not exact without stored entry price.
        tqqq_prev_open = float(opens["TQQQ"].iloc[-2]) if len(opens["TQQQ"]) >= 2 else None
        if tqqq_prev_open is not None and tqqq_open_today >= tqqq_prev_open * (1.0 + PROFIT_TARGET_PCT):
            order = client.market_sell("TQQQ", qty=tqqq_qty)
            actions.append(f"SELL TQQQ qty={tqqq_qty} @ market: {order}")
        else:
            actions.append("Holding; target not met")

    # Notifications
    subject = "qqq-tqqq-strategy daily"
    body = (
        f"Date: {et.date()}\n"
        f"Signals: {rules}\n"
        f"Metrics: {metrics}\n"
        f"Actions: {actions}\n"
    )
    send_email(subject, body)
    send_sms_via_email(subject, ". ".join(actions))

    print("\n".join(actions))
    return 0


if __name__ == "__main__":
    sys.exit(main())


