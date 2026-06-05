# src/check_day49_timestamps.py

import pandas as pd

train = pd.read_csv("data/train.csv")

day49 = train[train["day"] == 49]

print(sorted(day49["timestamp"].unique()))