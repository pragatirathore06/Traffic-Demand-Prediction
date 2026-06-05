"""
V17 = V5 EXACTLY, but geohash mean encoding replaced with Bayesian-smoothed mean:

    smooth = (count * geo_mean + m * global_mean) / (count + m),   m = 50

This shrinks sparse / noisy geohashes toward the global mean. Only the geohash
encoding changes; every other V5 feature and the LightGBM params are identical.
"""
import pandas as pd
from lightgbm import LGBMRegressor

M = 50  # smoothing strength

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

# --- SMOOTHED geohash encoding (the only change from V5) ---
gs = train.groupby("geohash")["demand"].agg(["mean", "count"])
gs["smooth"] = (gs["count"] * gs["mean"] + M * global_mean) / (gs["count"] + M)
geo_smooth = gs["smooth"]

train["geohash_mean_demand"] = train["geohash"].map(geo_smooth)
test["geohash_mean_demand"] = test["geohash"].map(geo_smooth).fillna(global_mean)

# --- prefix encodings (V5, unchanged) ---
prefix4_mean = train.groupby("geohash_prefix_4")["demand"].mean()
prefix5_mean = train.groupby("geohash_prefix_5")["demand"].mean()
train["prefix4_mean_demand"] = train["geohash_prefix_4"].map(prefix4_mean)
test["prefix4_mean_demand"] = test["geohash_prefix_4"].map(prefix4_mean).fillna(global_mean)
train["prefix5_mean_demand"] = train["geohash_prefix_5"].map(prefix5_mean)
test["prefix5_mean_demand"] = test["geohash_prefix_5"].map(prefix5_mean).fillna(global_mean)

for df in [train, test]:
    df["RoadType"] = df["RoadType"].fillna("Unknown")
    df["Weather"] = df["Weather"].fillna("Unknown")
    df["Temperature"] = df["Temperature"].fillna(train["Temperature"].median())

cat_cols = ["geohash", "geohash_prefix_4", "geohash_prefix_5", "RoadType",
            "LargeVehicles", "Landmarks", "Weather"]
for col in cat_cols:
    train[col] = train[col].astype("category")
    test[col] = test[col].astype("category")

features = ["geohash", "geohash_mean_demand", "prefix4_mean_demand",
            "prefix5_mean_demand", "geohash_prefix_4", "geohash_prefix_5", "day",
            "RoadType", "NumberofLanes", "LargeVehicles", "Landmarks",
            "Temperature", "Weather", "hour", "minute", "total_minutes", "rush_hour"]

model = LGBMRegressor(n_estimators=1000, learning_rate=0.03, random_state=42)
model.fit(train[features], train["demand"])
pred = model.predict(test[features]).clip(0, 1)

pd.DataFrame({"Index": test["Index"], "demand": pred}).to_csv(
    "submissions/submission_v17_smooth.csv", index=False)
print("Saved submissions/submission_v17_smooth.csv")
