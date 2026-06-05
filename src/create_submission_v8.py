"""
v8 — robust blend submission.

Key insight from validation on held-out day-49 data:
  * The strongest, most day-transferable signal is each geohash's average demand.
  * Day-48 same-time-of-day demand ("pair") helps only a little; weighting it
    heavily overfits to day-48 noise that does not carry to day 49.
  * A heavy LightGBM on raw features scored *worse* (R2 0.59) than this blend
    (R2 ~0.66) on the day-49 holdout.

We therefore predict: demand = w * pair + (1 - w) * geo_mean
Aggregates use ALL training rows (day 48 + the day-49 sliver) so the per-geohash
level estimate is as good as possible for the day-49 test.
"""
import pandas as pd
import numpy as np

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Blend weight on the day-48 same-time signal. 0.15 is a touch above the
# night-time optimum (0.1) because the test is daytime, where the time-of-day
# pattern is a bit more stable and informative.
W_PAIR = 0.15

global_mean = train["demand"].mean()

# Per-geohash average demand (use both days for the best level estimate).
geo_mean = train.groupby("geohash")["demand"].mean()

# Same-time-of-day demand per geohash, from the full day (day 48).
day48 = train[train["day"] == 48]
pair_mean = day48.groupby(["geohash", "timestamp"])["demand"].mean()

# Map onto test.
test["geo"] = test["geohash"].map(geo_mean).fillna(global_mean)
test["pair"] = test.set_index(["geohash", "timestamp"]).index.map(pair_mean)
# Fall back to the geohash mean where we have no same-time observation.
test["pair"] = test["pair"].fillna(test["geo"])

predictions = (W_PAIR * test["pair"] + (1 - W_PAIR) * test["geo"]).clip(0, 1)

submission = pd.DataFrame({"Index": test["Index"], "demand": predictions})
submission.to_csv("submissions/submission_v8.csv", index=False)

print("Saved submissions/submission_v8.csv", submission.shape)
print(submission.head())
