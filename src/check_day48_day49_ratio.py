# src/check_day48_day49_ratio.py

import pandas as pd

train = pd.read_csv("data/train.csv")

day48 = train[train["day"] == 48]
day49 = train[train["day"] == 49]

merged = day49.merge(
    day48,
    on=["geohash", "timestamp"],
    suffixes=("_49", "_48")
)

print("Merged rows:", len(merged))

ratio = (
    merged["demand_49"] /
    (merged["demand_48"] + 1e-9)
)

print("\nRatio stats:")
print(ratio.describe())