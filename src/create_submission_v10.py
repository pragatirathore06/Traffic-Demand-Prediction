"""
V10 = exact V5 pipeline + ONE additional feature: spatial neighbour demand.

Nothing from V5 is removed (that is what scores 91.69). We only ADD a
`geo_knn_demand` column = mean demand of the K nearest geohashes by decoded
lat/lon. This injects real new information (local spatial context) that the
geohash/prefix encodings do not capture, especially for sparse + the 10 unseen
geohashes. If even this does not beat V5, ~91.7 is the hard data ceiling.
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.neighbors import NearestNeighbors
from geo_utils import geo_coords

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")


def create_features(df):
    df["hour"] = df["timestamp"].apply(lambda x: int(str(x).split(":")[0]))
    df["minute"] = df["timestamp"].apply(lambda x: int(str(x).split(":")[1]))
    df["total_minutes"] = df["hour"] * 60 + df["minute"]
    df["rush_hour"] = (((df["hour"] >= 7) & (df["hour"] <= 10)) |
                       ((df["hour"] >= 17) & (df["hour"] <= 20))).astype(int)
    df["geohash_prefix_4"] = df["geohash"].str[:4]
    df["geohash_prefix_5"] = df["geohash"].str[:5]
    return df


train = create_features(train)
test = create_features(test)
global_mean = train["demand"].mean()

# --- V5 target encodings (unchanged) ---
geo_mean = train.groupby("geohash")["demand"].mean()
train["geohash_mean_demand"] = train["geohash"].map(geo_mean)
test["geohash_mean_demand"] = test["geohash"].map(geo_mean).fillna(global_mean)

prefix4_mean = train.groupby("geohash_prefix_4")["demand"].mean()
prefix5_mean = train.groupby("geohash_prefix_5")["demand"].mean()
train["prefix4_mean_demand"] = train["geohash_prefix_4"].map(prefix4_mean)
test["prefix4_mean_demand"] = test["geohash_prefix_4"].map(prefix4_mean).fillna(global_mean)
train["prefix5_mean_demand"] = train["geohash_prefix_5"].map(prefix5_mean)
test["prefix5_mean_demand"] = test["geohash_prefix_5"].map(prefix5_mean).fillna(global_mean)

# --- NEW: spatial kNN neighbour demand ---
K = 8
all_gh = pd.Index(pd.unique(pd.concat([train["geohash"], test["geohash"]])))
coords = geo_coords(all_gh)
known = list(geo_mean.index)
known_xy = np.array([coords[x] for x in known])
known_val = geo_mean.values
nn = NearestNeighbors(n_neighbors=min(K + 1, len(known))).fit(known_xy)
known_set = set(known)


def knn_demand(gh_series):
    ghs = gh_series.tolist()
    xy = np.array([coords[x] for x in ghs])
    dist, idx = nn.kneighbors(xy)
    out = np.empty(len(ghs))
    for i, gh in enumerate(ghs):
        js = [j for j, d in zip(idx[i], dist[i]) if not (gh in known_set and d == 0.0)][:K]
        out[i] = known_val[js].mean() if js else global_mean
    return out


train["geo_knn_demand"] = knn_demand(train["geohash"])
test["geo_knn_demand"] = knn_demand(test["geohash"])

# --- missing values (V5) ---
for df in (train, test):
    df["RoadType"] = df["RoadType"].fillna("Unknown")
    df["Weather"] = df["Weather"].fillna("Unknown")
    df["Temperature"] = df["Temperature"].fillna(train["Temperature"].median())

cat_cols = ["geohash", "geohash_prefix_4", "geohash_prefix_5", "RoadType",
            "LargeVehicles", "Landmarks", "Weather"]
for col in cat_cols:
    train[col] = train[col].astype("category")
    test[col] = test[col].astype("category")

features = [
    "geohash", "geohash_mean_demand", "prefix4_mean_demand", "prefix5_mean_demand",
    "geo_knn_demand",                              # <-- the only addition
    "geohash_prefix_4", "geohash_prefix_5", "day", "RoadType", "NumberofLanes",
    "LargeVehicles", "Landmarks", "Temperature", "Weather", "hour", "minute",
    "total_minutes", "rush_hour",
]

model = LGBMRegressor(n_estimators=1000, learning_rate=0.03, random_state=42)
model.fit(train[features], train["demand"])
pred = model.predict(test[features]).clip(0, 1)

pd.DataFrame({"Index": test["Index"], "demand": pred}).to_csv(
    "submissions/submission_v10.csv", index=False)
print("Saved submissions/submission_v10.csv", (len(pred), 2))
imp = sorted(zip(features, model.feature_importances_), key=lambda x: -x[1])
print("top features:", imp[:6])
