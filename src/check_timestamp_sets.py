# src/check_timestamp_sets.py

import pandas as pd

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

day49 = train[train["day"] == 49]

train_ts = set(day49["timestamp"].unique())
test_ts = set(test["timestamp"].unique())

print("Day49 timestamps:")
print(sorted(train_ts))

print("\nTest timestamps:")
print(sorted(test_ts))

print("\nIntersection:", len(train_ts & test_ts))