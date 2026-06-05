"""
V15 = V5 feature set, but CatBoost instead of LightGBM.
CatBoost sometimes handles high-cardinality categoricals (geohash) better.
Same features as V5; categoricals passed natively via cat_features.
"""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")


def create_features(df):
    df["hour"] = df["timestamp"].apply(lambda x: int(str(x).split(":")[0]))
    df["minute"] = df["timestamp"].apply(lambda x: int(str(x).split(":")[1]))
    df["total_minutes"] = df["hour"] * 60 + df["minute"]
    df["rush_hour"] = (
        ((df["hour"] >= 7) & (df["hour"] <= 10))
        | ((df["hour"] >= 17) & (df["hour"] <= 20))
    ).astype(int)
    df["geohash_prefix_4"] = df["geohash"].str[:4]
    df["geohash_prefix_5"] = df["geohash"].str[:5]
    return df


train = create_features(train)
test = create_features(test)
global_mean = train["demand"].mean()

geo_mean = train.groupby("geohash")["demand"].mean()
train["geohash_mean_demand"] = train["geohash"].map(geo_mean)
test["geohash_mean_demand"] = test["geohash"].map(geo_mean).fillna(global_mean)

prefix4_mean = train.groupby("geohash_prefix_4")["demand"].mean()
prefix5_mean = train.groupby("geohash_prefix_5")["demand"].mean()
train["prefix4_mean_demand"] = train["geohash_prefix_4"].map(prefix4_mean)
test["prefix4_mean_demand"] = test["geohash_prefix_4"].map(prefix4_mean).fillna(global_mean)
train["prefix5_mean_demand"] = train["geohash_prefix_5"].map(prefix5_mean)
test["prefix5_mean_demand"] = test["geohash_prefix_5"].map(prefix5_mean).fillna(global_mean)

# CatBoost wants categoricals as strings, no NaN
cat_cols = ["geohash", "geohash_prefix_4", "geohash_prefix_5", "RoadType",
            "LargeVehicles", "Landmarks", "Weather"]
for df in [train, test]:
    df["RoadType"] = df["RoadType"].fillna("Unknown")
    df["Weather"] = df["Weather"].fillna("Unknown")
    df["Temperature"] = df["Temperature"].fillna(train["Temperature"].median())
    for c in cat_cols:
        df[c] = df[c].astype(str)

features = ["geohash", "geohash_mean_demand", "prefix4_mean_demand",
            "prefix5_mean_demand", "geohash_prefix_4", "geohash_prefix_5", "day",
            "RoadType", "NumberofLanes", "LargeVehicles", "Landmarks",
            "Temperature", "Weather", "hour", "minute", "total_minutes", "rush_hour"]
cat_idx = [features.index(c) for c in cat_cols]

model = CatBoostRegressor(iterations=3000, learning_rate=0.03, depth=8,
                          loss_function="RMSE", random_seed=42, verbose=500)
model.fit(train[features], train["demand"], cat_features=cat_idx)
pred = np.clip(model.predict(test[features]), 0, 1)

pd.DataFrame({"Index": test["Index"], "demand": pred}).to_csv(
    "submissions/submission_v15_catboost.csv", index=False)
print("Saved submissions/submission_v15_catboost.csv")
