"""
Anomaly Detection on Spark-Computed Features (Large Scale)
----------------------------------------------------------------
Reads the CSV output of feature_engineering_spark.py (computed by
distributed Spark, collected to pandas for saving) and runs Isolation
Forest on it.

Prints a step-by-step comparison so you can see what each improvement
contributed:
    A) original 7 features, sklearn default max_samples (256)
    B) + 5 per-side ratio features
    C) + larger max_samples (final configuration)

Metrics:
  Precision / Recall / F1 -- computed at a cutoff that flags exactly as many
      rows as there are true anomalies. NOTE: that cutoff uses the true
      anomaly rate, which we only know because the data is synthetic. In a
      real deployment the rate is unknown and the cutoff has to be chosen
      from domain knowledge or a validation set.
  Average precision -- threshold-free measure of how well the SCORES rank
      true anomalies above normal rows. This is the fairer number when the
      cutoff is uncertain.
"""

import time

import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score)

from anomaly_detector import fit_anomaly_detector
from feature_engineering_spark import EXTENDED_FEATURE_COLUMNS, FEATURE_COLUMNS

INPUT_PATH = "../data/BTCUSDT_features_large.csv"
MAX_SAMPLES = 4096   # validated on 3 different random datasets; 16384 scored slightly higher on one


def run_config(features, label, cols, max_samples, contamination):
    y_true = features["is_injected_anomaly"].values
    start = time.time()
    model, scaler, anomaly_score, predictions = fit_anomaly_detector(
        features, contamination=contamination,
        feature_columns=cols, max_samples=max_samples,
    )
    y_pred = (predictions == -1).astype(int)
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    ap = average_precision_score(y_true, anomaly_score)
    print(f"{label:<44} P={p:.3f}  R={r:.3f}  F1={f1:.3f}  AvgPrec={ap:.3f}  ({time.time()-start:.1f}s)")
    return anomaly_score, y_pred


def main():
    start = time.time()
    features = pd.read_csv(INPUT_PATH, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    print(f"Loaded {len(features):,} rows from Spark-computed CSV in {time.time()-start:.2f}s")

    y_true = features["is_injected_anomaly"].values
    contamination = y_true.mean()
    print(f"Known anomalous rows: {y_true.sum()}  (rate {contamination:.4%})\n")

    print("=== Step-by-step improvement (large-scale synthetic data) ===")
    run_config(features, "A) original 7 features, default max_samples",
               FEATURE_COLUMNS, "auto", contamination)
    run_config(features, "B) + 5 per-side ratio features",
               EXTENDED_FEATURE_COLUMNS, "auto", contamination)
    anomaly_score, y_pred = run_config(features, f"C) + max_samples={MAX_SAMPLES}  (final)",
               EXTENDED_FEATURE_COLUMNS, MAX_SAMPLES, contamination)

    features["anomaly_score"] = anomaly_score
    features["is_anomaly_pred"] = y_pred

    print("\n=== Top 10 highest anomaly scores (final configuration) ===")
    top = features.nlargest(10, "anomaly_score")[
        ["timestamp", "mid_price", "ofi", "depth_imbalance",
         "bid_depth_ratio", "anomaly_score", "is_injected_anomaly"]
    ]
    print(top.to_string(index=False))

    out_path = "../output/large_scale_anomaly_results.csv"
    features.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
