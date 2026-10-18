import pandas as pd
from feature_engineering import build_feature_matrix, add_ratio_features, EXTENDED_FEATURE_COLUMNS
from anomaly_detector import fit_anomaly_detector

RAW_PATH = "../data/BTCUSDT_orderbook.csv"
OUTPUT_PATH = "../output/real_data_anomaly_results.csv"

# Same max_samples fix validated on the large synthetic dataset: sklearn's
# default (256) is too small a sample for each tree once you have more than
# a few thousand rows, giving noisy scores. We use the smaller of this value
# or the actual row count, since a real collection run may be shorter.
MAX_SAMPLES = 4096


def main():
    raw = pd.read_csv(RAW_PATH, parse_dates=["timestamp"])
    print(f"Loaded {len(raw)} raw rows from {RAW_PATH}")
    print(f"Time range: {raw['timestamp'].iloc[0]} -> {raw['timestamp'].iloc[-1]}")

    features = build_feature_matrix(raw)
    features = add_ratio_features(features)   # the 5 extra features that fixed "imbalance" detection
    print(f"\nAfter feature engineering + warmup drop: {len(features)} rows")

    if len(features) < 30:
        print("\nWARNING: Very few rows survived the rolling-window warmup.")
        print("Rolling windows need ~15-30 data points before they produce values.")
        print("With only a few hundred total rows, this can eat a big chunk of your data.")
        print("This is fine for a sanity check, but for the final report you'll want")
        print("a longer collection run, or smaller window sizes.")
        if len(features) == 0:
            print("\nZero usable rows -- stopping here. Collect more data and try again.")
            return

    print("\nFeature summary stats (real data):")
    print(features[EXTENDED_FEATURE_COLUMNS].describe().T[["mean", "std", "min", "max"]])

    max_samples = min(MAX_SAMPLES, len(features))
    model, scaler, anomaly_score, predictions = fit_anomaly_detector(
        features, contamination=0.01,
        feature_columns=EXTENDED_FEATURE_COLUMNS, max_samples=max_samples,
    )
    features["anomaly_score"] = anomaly_score
    features["is_anomaly_pred"] = (predictions == -1).astype(int)

    n_flagged = features["is_anomaly_pred"].sum()
    print(f"\nFlagged {n_flagged} anomalies out of {len(features)} real snapshots")

    print("\n=== Top flagged anomalies (real data) ===")
    top = features.nlargest(min(10, n_flagged if n_flagged > 0 else 5), "anomaly_score")[
        ["timestamp", "mid_price", "ofi", "depth_imbalance",
         "bid_depth_ratio", "anomaly_score"]
    ]
    print(top.to_string(index=False))

    features.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
