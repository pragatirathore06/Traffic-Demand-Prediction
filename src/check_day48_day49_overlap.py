# src/check_day48_day49_overlap.py

import pandas as pd

train = pd.read_csv("data/train.csv")

day48 = train[train["day"] == 48]
day49 = train[train["day"] == 49]

print("Day48 geohashes:", day48["geohash"].nunique())
print("Day49 geohashes:", day49["geohash"].nunique())

common = set(day48["geohash"]) & set(day49["geohash"])

print("Common geohashes:", len(common))

print(
    "Coverage:",
    len(common) / day49["geohash"].nunique()
)