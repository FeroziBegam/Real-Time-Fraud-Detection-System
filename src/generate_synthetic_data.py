import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_synthetic_orderbook(n_rows: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    start_time = datetime(2026, 6, 16, 0, 0, 0)
    timestamps = [start_time + timedelta(minutes=i) for i in range(n_rows)]

    price_steps = rng.normal(loc=0, scale=8, size=n_rows)
    mid_price = 65000 + np.cumsum(price_steps)

    base_spread = rng.normal(loc=1.5, scale=0.3, size=n_rows).clip(min=0.1)

    bid_depth = rng.normal(loc=15, scale=3, size=n_rows).clip(min=0.5)
    ask_depth = rng.normal(loc=15, scale=3, size=n_rows).clip(min=0.5)

    best_bid_qty = rng.normal(loc=1.2, scale=0.4, size=n_rows).clip(min=0.01)
    best_ask_qty = rng.normal(loc=1.2, scale=0.4, size=n_rows).clip(min=0.01)

    anomaly_indices = []

    spoof_idx = rng.choice(range(200, n_rows - 200), size=4, replace=False)
    for idx in spoof_idx:
        bid_depth[idx:idx+3] *= rng.uniform(8, 15)
        anomaly_indices.extend(range(idx, idx+3))

    vacuum_idx = rng.choice(range(200, n_rows - 200), size=3, replace=False)
    for idx in vacuum_idx:
        base_spread[idx:idx+4] *= rng.uniform(6, 12)
        bid_depth[idx:idx+4] *= 0.15
        ask_depth[idx:idx+4] *= 0.15
        anomaly_indices.extend(range(idx, idx+4))

    imbalance_idx = rng.choice(range(200, n_rows - 200), size=4, replace=False)
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
    df.loc[sorted(set(anomaly_indices)), "is_injected_anomaly"] = 1

    return df


if __name__ == "__main__":
    df = generate_synthetic_orderbook(n_rows=2000)
    out_path = "../data/BTCUSDT_orderbook_synthetic.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    print(f"Injected anomalies: {df['is_injected_anomaly'].sum()} rows")
    print(df.head())
