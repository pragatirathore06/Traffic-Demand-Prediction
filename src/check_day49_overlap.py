# src/check_day49_overlap.py

import pandas as pd

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

day49 = train[train["day"] == 49]

print("Train day49 rows:", len(day49))
print("Test rows:", len(test))

keys_train = set(
    zip(day49["geohash"], day49["timestamp"])
)

keys_test = set(
    zip(test["geohash"], test["timestamp"])
)

overlap = len(keys_train & keys_test)

print("Overlap:", overlap)
print("Test coverage:", overlap / len(test))