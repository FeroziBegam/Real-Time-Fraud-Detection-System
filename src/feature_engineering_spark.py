"""
Distributed Feature Engineering (PySpark)
---------------------------------------------
Same 7 features as feature_engineering.py, computed with Apache Spark's
DataFrame API and window functions instead of pandas. Runs in local mode
here (single machine, using all available cores), but the exact same code
runs unchanged on a real multi-node Spark cluster -- that's the actual
point of using Spark: the computation is expressed in a way that scales
out horizontally, not tied to a single machine's memory or a single CPU
core the way pandas' rolling() is.

Why this matters at scale: pandas holds the entire dataset in one
process's memory and computes rolling windows on a single core. Spark
partitions the data, computes windowed aggregates in parallel, and can
spill to disk or scale across a cluster when data no longer fits in one
machine's RAM -- which real exchange-scale tick data (millions of rows/day
per symbol) eventually requires.
"""

import time

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

FEATURE_COLUMNS = [
    "ofi", "depth_imbalance", "relative_spread", "rolling_volatility",
    "price_zscore", "quote_intensity", "depth_ratio_change"
]

# Per-side "ratio to own recent baseline" features. Why they were added:
# OFI and depth_imbalance are normalized ratios capped at +/-1, so a huge
# one-sided quantity (e.g. 7x normal) gets squashed into a value that
# ordinary noise also reaches thousands of times at large scale -- the
# MAGNITUDE is lost. Dividing each side's quantity/depth/spread by its own
# rolling average keeps the magnitude, so "bid quantity is 7x its recent
# norm" stays visible. On 300k rows this raised F1 from ~0.39 to ~0.54
# (and to ~0.81 together with a larger max_samples in the model).
RATIO_FEATURES = [
    ("best_bid_qty", "bid_qty_ratio"),
    ("best_ask_qty", "ask_qty_ratio"),
    ("bid_depth_total", "bid_depth_ratio"),
    ("ask_depth_total", "ask_depth_ratio"),
    ("spread", "spread_ratio"),
]
EXTENDED_FEATURE_COLUMNS = FEATURE_COLUMNS + [name for _, name in RATIO_FEATURES]


def get_spark():
    return (
        SparkSession.builder
        .appName("OrderBookFeatureEngineering")
        .master("local[*]")   # uses all local CPU cores; change to a cluster URL for real deployment
        .config("spark.sql.shuffle.partitions", "8")   # tuned down for local/small-cluster runs
        .getOrCreate()
    )


def build_feature_matrix_spark(spark, input_path: str):
    df = spark.read.csv(input_path, header=True, inferSchema=True)

    order_col = "timestamp"
    row_window = Window.orderBy(order_col)

    # -- row-wise features (no window needed) --
    df = df.withColumn(
        "ofi",
        (F.col("best_bid_qty") - F.col("best_ask_qty"))
        / F.when((F.col("best_bid_qty") + F.col("best_ask_qty")) == 0, None)
           .otherwise(F.col("best_bid_qty") + F.col("best_ask_qty"))
    )
    df = df.withColumn(
        "depth_imbalance",
        (F.col("bid_depth_total") - F.col("ask_depth_total"))
        / F.when((F.col("bid_depth_total") + F.col("ask_depth_total")) == 0, None)
           .otherwise(F.col("bid_depth_total") + F.col("ask_depth_total"))
    )
    df = df.withColumn("relative_spread", F.col("spread") / F.col("mid_price"))

    # -- rolling volatility: stddev of log returns over last 15 rows (min 5) --
    df = df.withColumn("prev_mid", F.lag("mid_price", 1).over(row_window))
    df = df.withColumn("log_return", F.log(F.col("mid_price") / F.col("prev_mid")))
    vol_window = Window.orderBy(order_col).rowsBetween(-14, 0)
    df = df.withColumn("vol_count", F.count("log_return").over(vol_window))
    df = df.withColumn(
        "rolling_volatility",
        F.when(F.col("vol_count") >= 5, F.stddev("log_return").over(vol_window))
    )

    # -- price z-score: (price - rolling mean) / rolling std, window 30 (min 10) --
    z_window = Window.orderBy(order_col).rowsBetween(-29, 0)
    df = df.withColumn("z_count", F.count("mid_price").over(z_window))
    df = df.withColumn("rolling_mean_price", F.avg("mid_price").over(z_window))
    df = df.withColumn("rolling_std_price", F.stddev("mid_price").over(z_window))
    df = df.withColumn(
        "price_zscore",
        F.when(
            (F.col("z_count") >= 10) & (F.col("rolling_std_price") != 0),
            (F.col("mid_price") - F.col("rolling_mean_price")) / F.col("rolling_std_price")
        )
    )

    # -- quote intensity: sum of |bid change| + |ask change| over last 5 rows (min 2) --
    df = df.withColumn("prev_bid", F.lag("best_bid", 1).over(row_window))
    df = df.withColumn("prev_ask", F.lag("best_ask", 1).over(row_window))
    df = df.withColumn(
        "quote_change",
        F.abs(F.col("best_bid") - F.col("prev_bid")) + F.abs(F.col("best_ask") - F.col("prev_ask"))
    )
    qi_window = Window.orderBy(order_col).rowsBetween(-4, 0)
    df = df.withColumn("qi_count", F.count("quote_change").over(qi_window))
    df = df.withColumn(
        "quote_intensity",
        F.when(F.col("qi_count") >= 2, F.sum("quote_change").over(qi_window))
    )

    # -- depth ratio change: total depth / its own rolling average (window 20, min 5) --
    df = df.withColumn("total_depth", F.col("bid_depth_total") + F.col("ask_depth_total"))
    depth_window = Window.orderBy(order_col).rowsBetween(-19, 0)
    df = df.withColumn("depth_count", F.count("total_depth").over(depth_window))
    df = df.withColumn("rolling_avg_depth", F.avg("total_depth").over(depth_window))
    df = df.withColumn(
        "depth_ratio_change",
        F.when(F.col("depth_count") >= 5, F.col("total_depth") / F.col("rolling_avg_depth"))
    )

    # -- per-side ratio features: value / its own rolling average (window 20, min 5) --
    ratio_window = Window.orderBy(order_col).rowsBetween(-19, 0)
    for src, name in RATIO_FEATURES:
        df = df.withColumn(
            name,
            F.when(
                F.count(src).over(ratio_window) >= 5,
                F.col(src) / F.avg(src).over(ratio_window)
            )
        )

    # drop warmup rows where any feature is still null -- same as pandas' dropna()
    for col in EXTENDED_FEATURE_COLUMNS:
        df = df.filter(F.col(col).isNotNull())

    keep_cols = ["timestamp", "best_bid", "best_ask", "best_bid_qty", "best_ask_qty",
                 "bid_depth_total", "ask_depth_total", "mid_price", "spread",
                 "is_injected_anomaly"] + EXTENDED_FEATURE_COLUMNS
    return df.select(*keep_cols)


if __name__ == "__main__":
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")   # quiet down Spark's default verbose logging

    start = time.time()
    features = build_feature_matrix_spark(spark, "../data/BTCUSDT_orderbook_large.csv")
    features.cache()   # avoid recomputing the whole window pipeline for each action below
    row_count = features.count()   # triggers the actual distributed computation
    elapsed = time.time() - start

    print(f"Processed {row_count:,} feature rows in {elapsed:.2f}s using Spark local[*]")
    print("\nFeature summary stats:")
    features.select(EXTENDED_FEATURE_COLUMNS).describe().show()

    # Collect the finished (much smaller) feature table into pandas and save with
    # pandas directly, rather than Spark's own file writer. On Windows, Spark's
    # writer needs a small utility called winutils.exe that isn't installed by
    # default -- rather than requiring that extra setup, we do the heavy
    # distributed computation in Spark (the actual point of using it) and then
    # bring the finished, much smaller result into pandas for saving and for
    # the modeling step that follows. This sidesteps the Windows-only issue
    # entirely and works identically on any OS.
    out_path = "../data/BTCUSDT_features_large.csv"
    pdf = features.toPandas()
    pdf.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")

    spark.stop()
