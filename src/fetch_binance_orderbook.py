"""
Binance Order Book Snapshot Collector
-----------------------------------------
Polls Binance's PUBLIC market data API (no account or API key needed -- this
endpoint is open to anyone) at regular intervals and saves each snapshot in
the exact CSV format feature_engineering.py expects:

    timestamp, best_bid, best_ask, best_bid_qty, best_ask_qty,
    bid_depth_total, ask_depth_total, mid_price, spread

Important: this has to run for real wall-clock time to collect real data --
there's no way to "fast forward" and get historical order book snapshots for
free. If you poll every 5 seconds, collecting 1 hour of data takes 1 hour.

Usage:
    python3 fetch_binance_orderbook.py --symbol BTCUSDT --interval 5 --duration_minutes 60

This appends one row to the output CSV every `interval` seconds, so you can
stop it early (Ctrl+C) and still keep whatever was collected so far.
"""

import argparse
import csv
import time
from datetime import datetime, timezone

import requests

BINANCE_DEPTH_URL = "https://api.binance.com/api/v3/depth"
DEPTH_LEVELS = 20   # how many price levels to pull, for computing total depth

CSV_COLUMNS = [
    "timestamp", "best_bid", "best_ask", "best_bid_qty", "best_ask_qty",
    "bid_depth_total", "ask_depth_total", "mid_price", "spread",
]


def fetch_snapshot(symbol: str):
    """
    Calls Binance's public order book endpoint once and returns one row's
    worth of data. Returns None if the request fails (e.g. no internet,
    rate limited, symbol typo) so the caller can skip and retry rather than
    crash the whole collection run over one bad request.
    """
    try:
        response = requests.get(
            BINANCE_DEPTH_URL,
            params={"symbol": symbol.upper(), "limit": DEPTH_LEVELS},
            timeout=5,
        )
        response.raise_for_status()
        data = response.json()

        bids = data["bids"]   # list of [price_str, quantity_str], best first
        asks = data["asks"]   # list of [price_str, quantity_str], best first

        if not bids or not asks:
            print("  (empty order book snapshot, skipping)")
            return None

        best_bid = float(bids[0][0])
        best_bid_qty = float(bids[0][1])
        best_ask = float(asks[0][0])
        best_ask_qty = float(asks[0][1])

        bid_depth_total = sum(float(level[1]) for level in bids)
        ask_depth_total = sum(float(level[1]) for level in asks)

        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "best_bid": best_bid,
            "best_ask": best_ask,
            "best_bid_qty": best_bid_qty,
            "best_ask_qty": best_ask_qty,
            "bid_depth_total": bid_depth_total,
            "ask_depth_total": ask_depth_total,
            "mid_price": mid_price,
            "spread": spread,
        }

    except requests.exceptions.RequestException as e:
        print(f"  Request failed ({e}), skipping this snapshot")
        return None
    except (KeyError, ValueError, IndexError) as e:
        print(f"  Unexpected response format ({e}), skipping this snapshot")
        return None


def collect(symbol: str, interval_seconds: int, duration_minutes: float, output_path: str):
    n_snapshots_target = int((duration_minutes * 60) / interval_seconds)
    print(f"Collecting {symbol} order book snapshots every {interval_seconds}s "
          f"for {duration_minutes} minutes (~{n_snapshots_target} rows).")
    print(f"Saving to: {output_path}")
    print("Press Ctrl+C to stop early -- rows collected so far are kept.\n")

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        collected = 0
        start_time = time.time()
        try:
            while (time.time() - start_time) < duration_minutes * 60:
                row = fetch_snapshot(symbol)
                if row is not None:
                    writer.writerow(row)
                    f.flush()   # write immediately, so Ctrl+C never loses data
                    collected += 1
                    if collected % 10 == 0:
                        print(f"  {collected} snapshots collected...")
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            print("\nStopped early by user.")

    print(f"\nDone. {collected} snapshots saved to {output_path}")
    if collected < 30:
        print("Note: fewer than 30 rows collected -- the feature engineering step "
              "needs a warmup period (rolling windows up to 30 rows), so very "
              "short runs may produce few or zero usable feature rows. Consider "
              "a longer duration or shorter interval.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect real Binance order book snapshots.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair, e.g. BTCUSDT")
    parser.add_argument("--interval", type=int, default=5, help="Seconds between snapshots")
    parser.add_argument("--duration_minutes", type=float, default=60,
                         help="How long to collect for, in minutes")
    parser.add_argument("--output", default="../data/BTCUSDT_orderbook.csv",
                         help="Output CSV path")
    args = parser.parse_args()

    collect(args.symbol, args.interval, args.duration_minutes, args.output)
