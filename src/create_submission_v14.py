"""
V14 = V5 EXACTLY + three geohash-reliability features.

Added (enrich the existing strong geohash signal, do not replace it):
  * geohash_std_demand   - within-geohash demand spread (how noisy the mean is)
  * geohash_count        - how many obs back the geohash mean (reliability)
  * prefix5_std_demand   - prefix-5 demand spread

Everything else is identical to V5 (same features, same LightGBM params), since
V5 remains the best LB model and the new categoricals in V13 slightly hurt the
real (daytime) test despite looking good on the night-only validation.
"""
import pandas as pd
from lightgbm import LGBMRegressor

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

# --- geohash target encodings (V5) + NEW std/count ---
geo_grp = train.groupby("geohash")["demand"]
geo_mean = geo_grp.mean()
geo_std = geo_grp.std()
geo_count = geo_grp.count()

train["geohash_mean_demand"] = train["geohash"].map(geo_mean)
test["geohash_mean_demand"] = test["geohash"].map(geo_mean).fillna(global_mean)

# NEW features
median_std = geo_std.median()
train["geohash_std_demand"] = train["geohash"].map(geo_std)
test["geohash_std_demand"] = test["geohash"].map(geo_std).fillna(median_std)
train["geohash_count"] = train["geohash"].map(geo_count)
test["geohash_count"] = test["geohash"].map(geo_count).fillna(0)

# --- prefix encodings (V5) + NEW prefix5 std ---
prefix4_mean = train.groupby("geohash_prefix_4")["demand"].mean()
p5_grp = train.groupby("geohash_prefix_5")["demand"]
prefix5_mean = p5_grp.mean()
prefix5_std = p5_grp.std()

train["prefix4_mean_demand"] = train["geohash_prefix_4"].map(prefix4_mean)
test["prefix4_mean_demand"] = test["geohash_prefix_4"].map(prefix4_mean).fillna(global_mean)
train["prefix5_mean_demand"] = train["geohash_prefix_5"].map(prefix5_mean)
test["prefix5_mean_demand"] = test["geohash_prefix_5"].map(prefix5_mean).fillna(global_mean)

train["prefix5_std_demand"] = train["geohash_prefix_5"].map(prefix5_std)
test["prefix5_std_demand"] = test["geohash_prefix_5"].map(prefix5_std).fillna(prefix5_std.median())

# --- missing values (V5) ---
for df in [train, test]:
    df["RoadType"] = df["RoadType"].fillna("Unknown")
    df["Weather"] = df["Weather"].fillna("Unknown")
    df["Temperature"] = df["Temperature"].fillna(train["Temperature"].median())

cat_cols = ["geohash", "geohash_prefix_4", "geohash_prefix_5", "RoadType",
            "LargeVehicles", "Landmarks", "Weather"]
for col in cat_cols:
    train[col] = train[col].astype("category")
    test[col] = test[col].astype("category")

features = [
    "geohash",
    "geohash_mean_demand",
    "geohash_std_demand",     # NEW
    "geohash_count",          # NEW
    "prefix4_mean_demand",
    "prefix5_mean_demand",
    "prefix5_std_demand",     # NEW
    "geohash_prefix_4",
    "geohash_prefix_5",
    "day",
    "RoadType",
    "NumberofLanes",
    "LargeVehicles",
    "Landmarks",
    "Temperature",
    "Weather",
    "hour",
    "minute",
    "total_minutes",
    "rush_hour",
]

model = LGBMRegressor(n_estimators=1000, learning_rate=0.03, random_state=42)
model.fit(train[features], train["demand"])
pred = model.predict(test[features]).clip(0, 1)

sub = pd.DataFrame({"Index": test["Index"], "demand": pred})
sub.to_csv("submissions/submission_v14.csv", index=False)
print("Saved submissions/submission_v14.csv", sub.shape)
print(sub.head())
