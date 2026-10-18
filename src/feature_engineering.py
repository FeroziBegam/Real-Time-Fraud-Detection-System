import pandas as pd
import numpy as np


def compute_order_flow_imbalance(df: pd.DataFrame) -> pd.Series:
    num = df["best_bid_qty"] - df["best_ask_qty"]
    denom = df["best_bid_qty"] + df["best_ask_qty"]
    return num / denom.replace(0, np.nan)


def compute_depth_imbalance(df: pd.DataFrame) -> pd.Series:
    num = df["bid_depth_total"] - df["ask_depth_total"]
    denom = df["bid_depth_total"] + df["ask_depth_total"]
    return num / denom.replace(0, np.nan)


def compute_relative_spread(df: pd.DataFrame) -> pd.Series:
    return df["spread"] / df["mid_price"]


def compute_rolling_volatility(df: pd.DataFrame, window: int = 15) -> pd.Series:
    log_returns = np.log(df["mid_price"] / df["mid_price"].shift(1))
    return log_returns.rolling(window=window, min_periods=5).std()


def compute_price_zscore(df: pd.DataFrame, window: int = 30) -> pd.Series:
    rolling_mean = df["mid_price"].rolling(window=window, min_periods=10).mean()
    rolling_std = df["mid_price"].rolling(window=window, min_periods=10).std()
    return (df["mid_price"] - rolling_mean) / rolling_std.replace(0, np.nan)


def compute_quote_intensity(df: pd.DataFrame, window: int = 5) -> pd.Series:
    bid_changes = df["best_bid"].diff().abs()
    ask_changes = df["best_ask"].diff().abs()
    intensity = (bid_changes + ask_changes).rolling(window=window, min_periods=2).sum()
    return intensity


def compute_depth_ratio_change(df: pd.DataFrame) -> pd.Series:
    total_depth = df["bid_depth_total"] + df["ask_depth_total"]
    rolling_avg_depth = total_depth.rolling(window=20, min_periods=5).mean()
    return total_depth / rolling_avg_depth.replace(0, np.nan)


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["ofi"] = compute_order_flow_imbalance(out)
    out["depth_imbalance"] = compute_depth_imbalance(out)
    out["relative_spread"] = compute_relative_spread(out)
    out["rolling_volatility"] = compute_rolling_volatility(out)
    out["price_zscore"] = compute_price_zscore(out)
    out["quote_intensity"] = compute_quote_intensity(out)
    out["depth_ratio_change"] = compute_depth_ratio_change(out)

    feature_cols = [
        "ofi", "depth_imbalance", "relative_spread", "rolling_volatility",
        "price_zscore", "quote_intensity", "depth_ratio_change"
    ]
    out = out.dropna(subset=feature_cols).reset_index(drop=True)

    return out


FEATURE_COLUMNS = [
    "ofi", "depth_imbalance", "relative_spread", "rolling_volatility",
    "price_zscore", "quote_intensity", "depth_ratio_change"
]

# Per-side "ratio to own recent baseline" features (same idea as the Spark version).
# OFI / depth_imbalance are ratios capped at +/-1, which throws away MAGNITUDE
# (a bid quantity 7x normal looks like a modest 0.75). Dividing each side's
# quantity/depth/spread by its own rolling average keeps that magnitude.
RATIO_FEATURE_MAP = [
    ("best_bid_qty", "bid_qty_ratio"),
    ("best_ask_qty", "ask_qty_ratio"),
    ("bid_depth_total", "bid_depth_ratio"),
    ("ask_depth_total", "ask_depth_ratio"),
    ("spread", "spread_ratio"),
]
EXTENDED_FEATURE_COLUMNS = FEATURE_COLUMNS + [name for _, name in RATIO_FEATURE_MAP]


def add_ratio_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Call AFTER build_feature_matrix(). Adds the 5 ratio features."""
    out = df.copy()
    for src, name in RATIO_FEATURE_MAP:
        rolling_avg = out[src].rolling(window=window, min_periods=5).mean()
        out[name] = out[src] / rolling_avg.replace(0, np.nan)
    return out.dropna(subset=[name for _, name in RATIO_FEATURE_MAP]).reset_index(drop=True)


if __name__ == "__main__":
    raw = pd.read_csv("../data/BTCUSDT_orderbook_synthetic.csv", parse_dates=["timestamp"])
    features = build_feature_matrix(raw)

    print(f"Raw rows: {len(raw)} -> Feature rows after warmup drop: {len(features)}")
    print("\nFeature summary stats:")
    print(features[FEATURE_COLUMNS].describe().T[["mean", "std", "min", "max"]])

    out_path = "../data/BTCUSDT_features_synthetic.csv"
    features.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")
