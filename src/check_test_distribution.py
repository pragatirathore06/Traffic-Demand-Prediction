# src/check_day_distribution.py

import pandas as pd

train = pd.read_csv("data/train.csv")

for day in sorted(train["day"].unique()):

    temp = train[train["day"] == day]

    print(f"\nDay {day}")
    print("Rows:", len(temp))
    print("Unique geohashes:", temp["geohash"].nunique())
    print("Unique timestamps:", temp["timestamp"].nunique())