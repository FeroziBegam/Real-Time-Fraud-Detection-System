"""
Evaluate Detector on Real Data + Injected Anomalies
---------------------------------------------------------
Runs the full pipeline (feature engineering + ratio features + Isolation
Forest) on the labeled file produced by inject_anomalies_into_real_data.py,
and reports real precision/recall/F1 -- a genuine, verifiable test on real
market dynamics.
"""

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

from feature_engineering import build_feature_matrix, add_ratio_features, EXTENDED_FEATURE_COLUMNS
from anomaly_detector import fit_anomaly_detector

INPUT_PATH = "../data/BTCUSDT_orderbook_real_with_injected.csv"
OUTPUT_PATH = "../output/real_with_injected_results.csv"
MAX_SAMPLES = 4096


def main():
    raw = pd.read_csv(INPUT_PATH, parse_dates=["timestamp"])
    print(f"Loaded {len(raw)} rows ({raw['is_injected_anomaly'].sum()} with injected anomalies)")

    features = build_feature_matrix(raw)
    features = add_ratio_features(features)
    print(f"After feature engineering + warmup drop: {len(features)} rows")

    if len(features) < 20:
        print("Too few rows survived the warmup -- collect more data and try again.")
        return

    true_rate = features["is_injected_anomaly"].mean()
    print(f"True anomaly rate in usable data: {true_rate:.4%}")

    max_samples = min(MAX_SAMPLES, len(features))
    model, scaler, anomaly_score, predictions = fit_anomaly_detector(
        features, contamination=max(true_rate, 0.001),
        feature_columns=EXTENDED_FEATURE_COLUMNS, max_samples=max_samples,
    )
    features["anomaly_score"] = anomaly_score
    features["is_anomaly_pred"] = (predictions == -1).astype(int)

    y_true = features["is_injected_anomaly"].values
    y_pred = features["is_anomaly_pred"].values

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    ap = average_precision_score(y_true, anomaly_score)

    print(f"\n=== Results: real data + injected anomalies ===")
    print(f"Known anomalies : {y_true.sum()}")
    print(f"Flagged by model: {y_pred.sum()}")
    print(f"Precision       : {precision:.3f}")
    print(f"Recall          : {recall:.3f}")
    print(f"F1              : {f1:.3f}")
    print(f"Average Precision (threshold-free): {ap:.3f}")

    print("\n=== Top 10 highest anomaly scores ===")
    top = features.nlargest(10, "anomaly_score")[
        ["timestamp", "mid_price", "ofi", "depth_imbalance",
         "bid_depth_ratio", "anomaly_score", "is_injected_anomaly"]
    ]
    print(top.to_string(index=False))

    features.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
