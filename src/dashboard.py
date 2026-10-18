import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_dashboard(features_df: pd.DataFrame, title: str = "Order Book Anomaly Detection"):
    df = features_df.copy()
    anomalies = df[df["is_anomaly_pred"] == 1]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.08,
        subplot_titles=("Mid Price with Flagged Anomalies", "Order Flow Imbalance (OFI)")
    )

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"], y=df["mid_price"],
            mode="lines", name="Mid Price",
            line=dict(color="#4C78A8", width=1.5),
        ),
        row=1, col=1
    )

    if len(anomalies) > 0:
        hover_text = [
            f"Score: {row.anomaly_score:.4f}<br>"
            f"OFI: {row.ofi:.3f}<br>"
            f"Depth Imb: {row.depth_imbalance:.3f}<br>"
            f"Rel Spread: {row.relative_spread:.6f}"
            for row in anomalies.itertuples()
        ]
        fig.add_trace(
            go.Scatter(
                x=anomalies["timestamp"], y=anomalies["mid_price"],
                mode="markers", name="Flagged Anomaly",
                marker=dict(color="#E45756", size=10, symbol="circle-open", line=dict(width=2)),
                text=hover_text,
                hoverinfo="text+x",
            ),
            row=1, col=1
        )

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"], y=df["ofi"],
            mode="lines", name="OFI",
            line=dict(color="#72B7B2", width=1),
            showlegend=False,
        ),
        row=2, col=1
    )
    if len(anomalies) > 0:
        fig.add_trace(
            go.Scatter(
                x=anomalies["timestamp"], y=anomalies["ofi"],
                mode="markers", name="Anomaly (OFI)",
                marker=dict(color="#E45756", size=7),
                showlegend=False,
            ),
            row=2, col=1
        )
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)

    fig.update_layout(
        title=title,
        height=700,
        template="plotly_white",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
    fig.update_yaxes(title_text="OFI", row=2, col=1, range=[-1.1, 1.1])
    fig.update_xaxes(title_text="Time", row=2, col=1)

    return fig


def main(results_path: str, output_prefix: str, title: str):
    df = pd.read_csv(results_path, parse_dates=["timestamp"])

    fig = build_dashboard(df, title=title)

    html_path = f"../output/{output_prefix}.html"
    fig.write_html(html_path)
    print(f"Saved interactive dashboard -> {html_path}")

    try:
        png_path = f"../output/{output_prefix}.png"
        fig.write_image(png_path, width=1400, height=800, scale=2)
        print(f"Saved static image -> {png_path}")
    except Exception as e:
        print(f"Could not save PNG (likely missing 'kaleido' package): {e}")
        print("Run: pip install -U kaleido")


if __name__ == "__main__":
    main(
        results_path="../output/synthetic_anomaly_results.csv",
        output_prefix="synthetic_dashboard",
        title="Synthetic Order Book — Anomaly Detection (Validation Run)"
    )
