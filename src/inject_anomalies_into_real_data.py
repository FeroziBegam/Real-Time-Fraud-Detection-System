"""
Inject Known Anomalies into Real Collected Data
----------------------------------------------------
Real Binance data has no ground truth -- we can't know for certain which
moments were "really" manipulation. So we can't compute precision/recall
on it directly.

This script bridges that gap: take your REAL collected order book data
(real price movements, real natural noise) and deliberately inject the
same three known anomaly types we used in the synthetic generator
(spoofing, liquidity vacuum, one-sided imbalance) at random points,
directly into the real values. This gives us:
  - Real market dynamics and noise (not a fake random walk)
  - Known ground truth (we know exactly which rows we modified)

...so we CAN compute a real precision/recall score, on real data.

Usage:
    python3 inject_anomalies_into_real_data.py
"""

import numpy as np
import pandas as pd

INPUT_PATH = "../data/BTCUSDT_orderbook.csv"
OUTPUT_PATH = "../data/BTCUSDT_orderbook_real_with_injected.csv"


def inject_anomalies(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    n_rows = len(df)
    rng = np.random.default_rng(seed)

    if n_rows < 50:
        raise ValueError(
            f"Only {n_rows} rows collected -- need at least 50 to safely inject "
            "anomaly clusters without them overlapping or running off the edges. "
            "Collect more data first."
        )

    # scale how many anomaly events to inject based on how much data we have,
    # same logic as the large-scale synthetic generator
    n_clusters = max(1, n_rows // 100)
    safe_range = range(10, n_rows - 10)

    anomaly_indices = []

    spoof_idx = rng.choice(safe_range, size=min(n_clusters, len(safe_range)), replace=False)
    for idx in spoof_idx:
        df.loc[idx:idx+2, "bid_depth_total"] *= rng.uniform(8, 15)
        anomaly_indices.extend(range(idx, idx+3))

    vacuum_idx = rng.choice(safe_range, size=min(n_clusters, len(safe_range)), replace=False)
    for idx in vacuum_idx:
        df.loc[idx:idx+3, "spread"] *= rng.uniform(6, 12)
        df.loc[idx:idx+3, "bid_depth_total"] *= 0.15
        df.loc[idx:idx+3, "ask_depth_total"] *= 0.15
        anomaly_indices.extend(range(idx, idx+4))

    imbalance_idx = rng.choice(safe_range, size=min(n_clusters, len(safe_range)), replace=False)
    for idx in imbalance_idx:
        df.loc[idx:idx+1, "best_bid_qty"] *= rng.uniform(5, 9)
        anomaly_indices.extend(range(idx, idx+2))

    # Recompute fields that depend on the ones we just modified, so the
    # injected anomaly is internally consistent (e.g. mid_price should still
    # reflect best_bid/best_ask after a spread change)
    df["best_bid"] = df["mid_price"] - df["spread"] / 2
    df["best_ask"] = df["mid_price"] + df["spread"] / 2

    df["is_injected_anomaly"] = 0
    valid_indices = sorted(set(i for i in anomaly_indices if i < n_rows))
    df.loc[valid_indices, "is_injected_anomaly"] = 1

    return df


if __name__ == "__main__":
    raw = pd.read_csv(INPUT_PATH, parse_dates=["timestamp"])
    print(f"Loaded {len(raw)} real rows from {INPUT_PATH}")

    labeled = inject_anomalies(raw)
    n_injected = labeled["is_injected_anomaly"].sum()
    print(f"Injected anomalies into {n_injected} rows ({n_injected/len(labeled):.2%} of data)")

    labeled.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved -> {OUTPUT_PATH}")
    print("\nNext: run evaluate_real_with_injected.py to get precision/recall on this file.")
