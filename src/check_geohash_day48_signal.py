# src/check_geohash_day48_signal.py

import pandas as pd

train = pd.read_csv("data/train.csv")

day48 = train[train["day"] == 48]
day49 = train[train["day"] == 49]

day48_mean = (
    day48.groupby("geohash")["demand"]
    .mean()
)

day49 = day49.copy()

day49["day48_mean"] = (
    day49["geohash"]
    .map(day48_mean)
)

print(day49[["demand", "day48_mean"]].head())

corr = day49["demand"].corr(
    day49["day48_mean"]
)

print("\nCorrelation:", corr)