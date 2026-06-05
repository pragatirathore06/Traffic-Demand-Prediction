"""
FINAL single submission — strongest honest model.

Every lever was tested (see investigation):
  * location (geohash) is the only real signal (~0.91 of variance);
  * Temperature/Weather correlate 0.002/0.001 with demand -> decoys, dropped;
  * day-48 time-of-day shape is flat on average and its per-cell deviations do
    NOT transfer to day 49, so pair/timestamp features only add noise.
So the best estimator is a clean, well-shrunk per-geohash LEVEL:

  level(g) = hierarchical shrink: geohash mean  -> prefix5 -> prefix4 -> global,
             plus a spatial kNN fallback (decoded lat/lon) for the ~10 unseen /
             sparse geohashes.

Expected score ~92 (a touch above your V5 91.69). It will NOT reach 95/100:
that requires the actual day-49 demand values (a lookup, not a model).
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from geo_utils import geo_coords

K = 8          # spatial neighbours
A5 = 4.0       # shrink geohash -> prefix5
A4 = 8.0       # shrink prefix5 -> prefix4

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
g_global = train["demand"].mean()

train["p5"] = train["geohash"].str[:5]
train["p4"] = train["geohash"].str[:4]
test["p5"] = test["geohash"].str[:5]
test["p4"] = test["geohash"].str[:4]

# group stats (both days -> best level estimate)
g = train.groupby("geohash")["demand"].agg(g_mean="mean", g_n="count")
p5 = train.groupby("p5")["demand"].agg(p5_mean="mean", p5_n="count")
p4 = train.groupby("p4")["demand"].agg(p4_mean="mean", p4_n="count")

# hierarchical shrinkage: prefix4 -> prefix5 -> geohash
p4m = p4["p4_mean"]
p5_sh = (p5["p5_n"] * p5["p5_mean"] + A4 * test["p4"].map(p4m).groupby(test["p5"]).first().reindex(p5.index).fillna(g_global)) / (p5["p5_n"] + A4)

t = test.copy()
t = t.join(g, on="geohash").join(p5, on="p5").join(p4, on="p4")
t["p4_mean"] = t["p4_mean"].fillna(g_global)
t["p4_n"] = t["p4_n"].fillna(0)
# prefix5 shrunk toward prefix4
t["p5_level"] = (t["p5_n"].fillna(0) * t["p5_mean"].fillna(t["p4_mean"]) + A4 * t["p4_mean"]) / (t["p5_n"].fillna(0) + A4)
# geohash shrunk toward prefix5 level
t["level"] = (t["g_n"].fillna(0) * t["g_mean"].fillna(t["p5_level"]) + A5 * t["p5_level"]) / (t["g_n"].fillna(0) + A5)

# spatial kNN fallback for geohashes with no own history
all_gh = pd.Index(pd.unique(pd.concat([train["geohash"], test["geohash"]])))
coords = geo_coords(all_gh)
known = list(g.index)
nn = NearestNeighbors(n_neighbors=min(K + 1, len(known))).fit(
    np.array([coords[x] for x in known]))
kmean = g["g_mean"].values
known_set = set(known)

need = t["g_n"].fillna(0) == 0
if need.any():
    sub_gh = t.loc[need, "geohash"].tolist()
    xy = np.array([coords[x] for x in sub_gh])
    dist, idx = nn.kneighbors(xy)
    knn_vals = []
    for i, gh in enumerate(sub_gh):
        js = [j for j, d in zip(idx[i], dist[i]) if not (gh in known_set and d == 0.0)][:K]
        knn_vals.append(kmean[js].mean() if js else g_global)
    t.loc[need, "level"] = knn_vals

pred = t["level"].clip(0, 1).values
sub = pd.DataFrame({"Index": test["Index"], "demand": pred})
sub.to_csv("submissions/submission_final.csv", index=False)
print("Saved submissions/submission_final.csv", sub.shape)
print("NaNs:", int(np.isnan(pred).sum()), "| range", round(pred.min(), 4), round(pred.max(), 4))
print(sub.head())
