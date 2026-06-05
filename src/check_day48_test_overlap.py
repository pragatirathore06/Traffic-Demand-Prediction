# src/check_day48_test_overlap.py

import pandas as pd

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

day48 = train[train["day"] == 48]

train_pairs = set(
    zip(day48["geohash"], day48["timestamp"])
)

test_pairs = set(
    zip(test["geohash"], test["timestamp"])
)

overlap = len(train_pairs & test_pairs)

print("Day48 pairs:", len(train_pairs))
print("Test pairs :", len(test_pairs))
print("Overlap    :", overlap)
print("Coverage   :", overlap / len(test_pairs))