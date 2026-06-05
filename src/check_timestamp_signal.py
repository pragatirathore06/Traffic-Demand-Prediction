# src/check_timestamp_signal.py

import pandas as pd

train = pd.read_csv("data/train.csv")

mean_by_ts = (
    train.groupby("timestamp")["demand"]
    .mean()
)

train["ts_mean"] = (
    train["timestamp"]
    .map(mean_by_ts)
)

corr = train["demand"].corr(
    train["ts_mean"]
)

print("Correlation:", corr)