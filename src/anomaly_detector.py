import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from feature_engineering import build_feature_matrix, FEATURE_COLUMNS


def fit_anomaly_detector(features_df: pd.DataFrame, contamination: float = 0.01,
                         feature_columns=None, max_samples="auto"):
    # Defaults reproduce the original behavior exactly. The optional arguments
    # are used by the large-scale pipeline:
    #   feature_columns -> use an extended feature set
    #   max_samples     -> rows each tree sees. sklearn's default ("auto") is
    #                      only 256, which is far too small once you have
    #                      hundreds of thousands of rows: each tree then sees
    #                      a tiny random slice and the scores get noisy.
    cols = feature_columns if feature_columns is not None else FEATURE_COLUMNS
    scaler = StandardScaler()
    X = scaler.fit_transform(features_df[cols])

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples=max_samples,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)

    raw_scores = model.decision_function(X)
    anomaly_score = -raw_scores
    predictions = model.predict(X)

    return model, scaler, anomaly_score, predictions


def evaluate_against_known_anomalies(df: pd.DataFrame, predictions: np.ndarray):
    from sklearn.metrics import precision_score, recall_score, f1_score

    y_true = df["is_injected_anomaly"].values
    y_pred = (predictions == -1).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"Known anomalies in data : {y_true.sum()}")
    print(f"Flagged by model        : {y_pred.sum()}")
    print(f"Precision                : {precision:.3f}")
    print(f"Recall                   : {recall:.3f}")
    print(f"F1                       : {f1:.3f}")

    return {"precision": precision, "recall": recall, "f1": f1}


if __name__ == "__main__":
    raw = pd.read_csv("../data/BTCUSDT_orderbook_synthetic.csv", parse_dates=["timestamp"])
    features = build_feature_matrix(raw)

    model, scaler, anomaly_score, predictions = fit_anomaly_detector(
        features, contamination=0.01
    )
    features["anomaly_score"] = anomaly_score
    features["is_anomaly_pred"] = (predictions == -1).astype(int)

    print("=== Validation on synthetic data (known anomalies) ===")
    metrics = evaluate_against_known_anomalies(features, predictions)

    print("\n=== Top 10 highest anomaly scores ===")
    top_anomalies = features.nlargest(10, "anomaly_score")[
        ["timestamp", "mid_price", "ofi", "depth_imbalance",
         "relative_spread", "anomaly_score", "is_injected_anomaly"]
    ]
    print(top_anomalies.to_string(index=False))

    out_path = "../output/synthetic_anomaly_results.csv"
    features.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")
