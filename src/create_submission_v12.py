"""
V12 — final submission from the Day48->Day49 validation winner.

Validated recipe (3-seed, train=Day48 / valid=Day49):
    BASE (V5 features)                       R2 0.7352
    BASE + road_lane + knn_demand            R2 0.7436   <-- winner (+0.0084)
Adding lat/lon, missing-flags or weather_temp on top did NOT help, so they are
excluded to avoid overfitting noise.

Changes vs V5:
  1. + road_lane  = RoadType_NumberofLanes  (top transferable interaction)
  2. + knn_demand = mean demand of 8 nearest geohashes (decoded lat/lon)
  3. better unseen-geohash fallback: prefix5 -> prefix4 -> global (was global only;
     affects the 25 test rows on the 10 unseen geohashes)
Trained on FULL train (Day48 + Day49).
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.neighbors import NearestNeighbors
from geo_utils import decode

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
gmean = train["demand"].mean()


def base_feats(df):
    df = df.copy()
    df["hour"] = df["timestamp"].str.split(":").str[0].astype(int)
    df["minute"] = df["timestamp"].str.split(":").str[1].astype(int)
    df["total_minutes"] = df["hour"] * 60 + df["minute"]
    df["rush_hour"] = (((df.hour >= 7) & (df.hour <= 10)) |
                       ((df.hour >= 17) & (df.hour <= 20))).astype(int)
    df["p4"] = df["geohash"].str[:4]
    df["p5"] = df["geohash"].str[:5]
    df["road_lane"] = df["RoadType"].fillna("NA").astype(str) + "_" + df["NumberofLanes"].astype(str)
    return df


train = base_feats(train)
test = base_feats(test)

# target encodings (full train)
geo = train.groupby("geohash")["demand"].mean()
p4m = train.groupby("p4")["demand"].mean()
p5m = train.groupby("p5")["demand"].mean()
for df in (train, test):
    df["geo_mean"] = df["geohash"].map(geo)
    df["p4_mean"] = df["p4"].map(p4m).fillna(gmean)
    df["p5_mean"] = df["p5"].map(p5m).fillna(gmean)
    # unseen fallback: p5 -> p4 -> global (not global only)
    fb = df["p5_mean"].where(df["p5"].isin(p5m.index), df["p4_mean"])
    df["geo_mean"] = df["geo_mean"].fillna(fb).fillna(gmean)

# spatial kNN neighbour demand
all_gh = pd.unique(pd.concat([train["geohash"], test["geohash"]]))
coords = {g: decode(g) for g in all_gh}
known = list(geo.index)
nn = NearestNeighbors(n_neighbors=9).fit(np.array([coords[g] for g in known]))
kval = geo.values
kset = set(known)


def knn_demand(df):
    xy = np.array([coords[g] for g in df["geohash"]])
    dist, idx = nn.kneighbors(xy)
    out = np.empty(len(df))
    for i, g in enumerate(df["geohash"].values):
        js = [j for j, d in zip(idx[i], dist[i]) if not (g in kset and d == 0.0)][:8]
        out[i] = kval[js].mean() if js else gmean
    return out


train["knn_demand"] = knn_demand(train)
test["knn_demand"] = knn_demand(test)

# fills + categories
for df in (train, test):
    df["RoadType"] = df["RoadType"].fillna("Unknown")
    df["Weather"] = df["Weather"].fillna("Unknown")
    df["Temperature"] = df["Temperature"].fillna(train["Temperature"].median())
    for c in ["geohash", "p4", "p5", "RoadType", "LargeVehicles", "Landmarks",
              "Weather", "road_lane"]:
        df[c] = df[c].astype("category")

FEATURES = ["geohash", "geo_mean", "p4_mean", "p5_mean", "p4", "p5",
            "road_lane", "knn_demand",                       # <-- the two additions
            "RoadType", "NumberofLanes", "LargeVehicles", "Landmarks",
            "Temperature", "Weather", "hour", "minute", "total_minutes", "rush_hour"]

model = LGBMRegressor(n_estimators=1000, learning_rate=0.03, random_state=42)
model.fit(train[FEATURES], train["demand"])
pred = model.predict(test[FEATURES]).clip(0, 1)

sub = pd.DataFrame({"Index": test["Index"], "demand": pred})
sub.to_csv("submissions/submission_v12.csv", index=False)
print("Saved submissions/submission_v12.csv", sub.shape, "| NaNs", int(np.isnan(pred).sum()))
print(sub.head())
