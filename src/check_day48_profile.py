# src/check_day48_profile.py

import pandas as pd

train = pd.read_csv("data/train.csv")

day48 = train[train["day"] == 48]

ts_mean = (
    day48.groupby("timestamp")["demand"]
    .mean()
    .sort_index()
)

print(ts_mean)