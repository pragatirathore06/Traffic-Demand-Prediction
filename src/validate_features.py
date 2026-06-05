"""
Day48 -> Day49 validation harness (Priority 5).

Trains on Day 48, validates on Day 49 (the closest analogue to the real task),
and measures the R2 delta of each proposed feature group on top of the V5 base.
Only features with a non-negative, stable delta go into the final submission.

NOTE: Day 49 validation is night-only (00:00-02:00) while the test is daytime,
so treat small deltas as noise. Location features dominate either way.
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors
from geo_utils import decode

RAW = pd.read_csv("data/train.csv")
UNSEEN_FALLBACK = True


def build(df, enc_src, coords, knn_model, known, known_val, known_set):
    df = df.copy()
    df["hour"] = df["timestamp"].str.split(":").str[0].astype(int)
    df["minute"] = df["timestamp"].str.split(":").str[1].astype(int)
    df["total_minutes"] = df["hour"] * 60 + df["minute"]
    df["rush_hour"] = (((df.hour >= 7) & (df.hour <= 10)) |
                       ((df.hour >= 17) & (df.hour <= 20))).astype(int)
    df["p4"] = df["geohash"].str[:4]
    df["p5"] = df["geohash"].str[:5]
    gmean = enc_src["demand"].mean()

    # target encodings from enc_src only (no leakage)
    geo = enc_src.groupby("geohash")["demand"].mean()
    p4m = enc_src.groupby(enc_src.geohash.str[:4])["demand"].mean()
    p5m = enc_src.groupby(enc_src.geohash.str[:5])["demand"].mean()
    df["geo_mean"] = df["geohash"].map(geo)
    df["p4_mean"] = df["p4"].map(p4m).fillna(gmean)
    df["p5_mean"] = df["p5"].map(p5m).fillna(gmean)
    # better unseen fallback: p5 -> p4 -> global instead of global only
    fb = df["p5_mean"].where(df["p5"].isin(p5m.index), df["p4_mean"])
    df["geo_mean"] = df["geo_mean"].fillna(fb).fillna(gmean)

    # spatial coords + bins
    latlon = np.array([coords[g] for g in df["geohash"]])
    df["lat"] = latlon[:, 0]
    df["lon"] = latlon[:, 1]
    df["lat_bin"] = df["lat"].round(2)
    df["lon_bin"] = df["lon"].round(2)

    # kNN neighbour demand
    dist, idx = knn_model.kneighbors(latlon)
    knn = np.empty(len(df))
    for i, g in enumerate(df["geohash"].values):
        js = [j for j, d in zip(idx[i], dist[i]) if not (g in known_set and d == 0.0)][:8]
        knn[i] = known_val[js].mean() if js else gmean
    df["knn_demand"] = knn

    # missing indicators
    df["RoadType_missing"] = df["RoadType"].isna().astype(int)
    df["Weather_missing"] = df["Weather"].isna().astype(int)
    df["Temp_missing"] = df["Temperature"].isna().astype(int)

    # interactions
    df["road_lane"] = df["RoadType"].fillna("NA").astype(str) + "_" + df["NumberofLanes"].astype(str)
    tb = pd.cut(df["Temperature"], [-100, 10, 25, 100], labels=["Cold", "Mild", "Hot"])
    df["weather_temp"] = df["Weather"].fillna("NA").astype(str) + "_" + tb.astype(str)

    # fills
    df["RoadType"] = df["RoadType"].fillna("Unknown")
    df["Weather"] = df["Weather"].fillna("Unknown")
    df["Temperature"] = df["Temperature"].fillna(enc_src["Temperature"].median())
    for c in ["geohash", "p4", "p5", "RoadType", "LargeVehicles", "Landmarks",
              "Weather", "road_lane", "weather_temp"]:
        df[c] = df[c].astype("category")
    return df


def coords_and_knn(geohashes, enc_src):
    coords = {g: decode(g) for g in pd.unique(geohashes)}
    geo = enc_src.groupby("geohash")["demand"].mean()
    known = list(geo.index)
    nn = NearestNeighbors(n_neighbors=9).fit(np.array([coords[g] for g in known]))
    return coords, nn, known, geo.values, set(known)


def run():
    tr48 = RAW[RAW.day == 48]
    tr49 = RAW[RAW.day == 49]
    coords, nn, known, kval, kset = coords_and_knn(RAW.geohash, tr48)
    A = build(tr48, tr48, coords, nn, known, kval, kset)
    V = build(tr49, tr48, coords, nn, known, kval, kset)

    BASE = ["geohash", "geo_mean", "p4_mean", "p5_mean", "p4", "p5",
            "RoadType", "NumberofLanes", "LargeVehicles", "Landmarks",
            "Temperature", "Weather", "hour", "minute", "total_minutes", "rush_hour"]
    GROUPS = {
        "BASE (V5-like)": [],
        "+ better unseen fallback": [],           # already in geo_mean, shown for ref
        "+ lat/lon + bins": ["lat", "lon", "lat_bin", "lon_bin"],
        "+ knn_demand": ["knn_demand"],
        "+ missing flags": ["RoadType_missing", "Weather_missing", "Temp_missing"],
        "+ road_lane": ["road_lane"],
        "+ weather_temp": ["weather_temp"],
        "LEAN (drop Temp/Weather)": None,
    }

    def fit_eval(feats):
        m = LGBMRegressor(n_estimators=1000, learning_rate=0.03, random_state=42, verbose=-1)
        m.fit(A[feats], A["demand"])
        return r2_score(V["demand"], m.predict(V[feats]).clip(0, 1))

    base_r2 = fit_eval(BASE)
    print(f"{'feature set':30s}  R2(day49)   delta")
    print(f"{'BASE (V5-like)':30s}  {base_r2:.4f}     --")
    for name, extra in GROUPS.items():
        if name == "BASE (V5-like)" or name == "+ better unseen fallback":
            continue
        if name.startswith("LEAN"):
            feats = [f for f in BASE if f not in ("Temperature", "Weather")]
        else:
            feats = BASE + extra
        r2 = fit_eval(feats)
        print(f"{name:30s}  {r2:.4f}    {r2-base_r2:+.4f}")


if __name__ == "__main__":
    run()
