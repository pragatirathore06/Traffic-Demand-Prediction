import pandas as pd
from lightgbm import LGBMRegressor

# =====================
# Load Data
# =====================

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# =====================
# Feature Engineering
# =====================

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

    # RoadType + Lanes interaction
    df["road_lane"] = (
        df["RoadType"].fillna("Unknown").astype(str)
        + "_"
        + df["NumberofLanes"].astype(str)
    )

    # Weather + RoadType interaction
    df["weather_road"] = (
        df["Weather"].fillna("Unknown").astype(str)
        + "_"
        + df["RoadType"].fillna("Unknown").astype(str)
    )

    # Temperature bins
    df["temp_bin"] = pd.cut(
        df["Temperature"],
        bins=[-100, 15, 25, 35, 100],
        labels=["cold", "mild", "warm", "hot"],
    )

    return df


train = create_features(train)
test = create_features(test)

# =====================
# Geohash Target Encoding
# =====================

geo_mean = train.groupby("geohash")["demand"].mean()
global_mean = train["demand"].mean()

train["geohash_mean_demand"] = train["geohash"].map(geo_mean)
test["geohash_mean_demand"] = test["geohash"].map(geo_mean).fillna(global_mean)

# =====================
# Geohash Prefix Target Encoding
# =====================

prefix4_mean = train.groupby("geohash_prefix_4")["demand"].mean()
prefix5_mean = train.groupby("geohash_prefix_5")["demand"].mean()

train["prefix4_mean_demand"] = train["geohash_prefix_4"].map(prefix4_mean)
test["prefix4_mean_demand"] = test["geohash_prefix_4"].map(prefix4_mean).fillna(global_mean)

train["prefix5_mean_demand"] = train["geohash_prefix_5"].map(prefix5_mean)
test["prefix5_mean_demand"] = test["geohash_prefix_5"].map(prefix5_mean).fillna(global_mean)

# =====================
# Missing Values
# =====================

for df in [train, test]:
    df["RoadType"] = df["RoadType"].fillna("Unknown")
    df["Weather"] = df["Weather"].fillna("Unknown")
    df["Temperature"] = df["Temperature"].fillna(train["Temperature"].median())

# =====================
# Categories
# =====================

cat_cols = [
    "geohash",
    "geohash_prefix_4",
    "geohash_prefix_5",
    "RoadType",
    "LargeVehicles",
    "Landmarks",
    "Weather",
    "road_lane",
    "weather_road",
    "temp_bin",
]

for col in cat_cols:
    train[col] = train[col].astype("category")
    test[col] = test[col].astype("category")

# =====================
# Features
# =====================

features = [
    "geohash",
    "geohash_mean_demand",
    "prefix4_mean_demand",
    "prefix5_mean_demand",
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
    "road_lane",
    "weather_road",
    "temp_bin",
]

X_train = train[features]
y_train = train["demand"]
X_test = test[features]

# =====================
# Train Model
# =====================

# NOTE: validation (Day48->Day49) showed the tuned hyperparameters
# (n_estimators=2000, num_leaves=63, ...) OVERFIT day-48 noise and dropped the
# score below V5 (0.7263 vs 0.7350). V5's original params won (0.7471 with the
# new interaction features). So we keep V5's params and only add the features.
model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    random_state=42,
)

model.fit(X_train, y_train)

# =====================
# Predict
# =====================

predictions = model.predict(X_test).clip(0, 1)

submission = pd.DataFrame({"Index": test["Index"], "demand": predictions})
submission.to_csv("submissions/submission_v13.csv", index=False)

print("Saved submissions/submission_v13.csv", submission.shape)
print(submission.head())
