"""
Large-Scale Synthetic Order Book Generator
----------------------------------------------
Same generation logic as generate_synthetic_data.py, but scaled up to
hundreds of thousands of rows -- large enough that pandas' single-threaded,
in-memory processing genuinely starts to strain, which is the actual
justification for bringing in a distributed engine (Spark) for the
feature engineering step.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_synthetic_orderbook(n_rows: int = 300_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    start_time = datetime(2026, 6, 16, 0, 0, 0)
    # 1-second resolution instead of 1-minute, since we now have enough rows
    # to represent a realistic high-frequency stream over several days
    timestamps = [start_time + timedelta(seconds=i) for i in range(n_rows)]

    price_steps = rng.normal(loc=0, scale=2, size=n_rows)
    mid_price = 65000 + np.cumsum(price_steps)

    base_spread = rng.normal(loc=1.5, scale=0.3, size=n_rows).clip(min=0.1)
    bid_depth = rng.normal(loc=15, scale=3, size=n_rows).clip(min=0.5)
    ask_depth = rng.normal(loc=15, scale=3, size=n_rows).clip(min=0.5)
    best_bid_qty = rng.normal(loc=1.2, scale=0.4, size=n_rows).clip(min=0.01)
    best_ask_qty = rng.normal(loc=1.2, scale=0.4, size=n_rows).clip(min=0.01)

    anomaly_indices = []
    n_anomaly_clusters = max(4, n_rows // 5000)   # scale anomaly count with data size

    spoof_idx = rng.choice(range(200, n_rows - 200), size=n_anomaly_clusters, replace=False)
    for idx in spoof_idx:
        bid_depth[idx:idx+3] *= rng.uniform(8, 15)
        anomaly_indices.extend(range(idx, idx+3))

    vacuum_idx = rng.choice(range(200, n_rows - 200), size=n_anomaly_clusters, replace=False)
    for idx in vacuum_idx:
        base_spread[idx:idx+4] *= rng.uniform(6, 12)
        bid_depth[idx:idx+4] *= 0.15
        ask_depth[idx:idx+4] *= 0.15
        anomaly_indices.extend(range(idx, idx+4))

    imbalance_idx = rng.choice(range(200, n_rows - 200), size=n_anomaly_clusters, replace=False)
    for idx in imbalance_idx:
        best_bid_qty[idx:idx+2] *= rng.uniform(5, 9)
        anomaly_indices.extend(range(idx, idx+2))

    best_bid = mid_price - base_spread / 2
    best_ask = mid_price + base_spread / 2

    df = pd.DataFrame({
        "timestamp": timestamps,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "best_bid_qty": best_bid_qty,
        "best_ask_qty": best_ask_qty,
        "bid_depth_total": bid_depth,
        "ask_depth_total": ask_depth,
        "mid_price": mid_price,
        "spread": base_spread,
    })

    df["is_injected_anomaly"] = 0
    df.loc[sorted(set(i for i in anomaly_indices if i < n_rows)), "is_injected_anomaly"] = 1

    return df


if __name__ == "__main__":
    import time
    start = time.time()
    df = generate_synthetic_orderbook(n_rows=300_000)
    out_path = "../data/BTCUSDT_orderbook_large.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df):,} rows -> {out_path} in {time.time()-start:.1f}s")
    print(f"Injected anomalies: {df['is_injected_anomaly'].sum():,} rows")
    print(f"File size: approx {df.memory_usage(deep=True).sum() / 1e6:.1f} MB in memory")
