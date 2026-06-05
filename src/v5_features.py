"""Shared V5 feature builder. Returns train, test, feature list, categorical cols.

build_v5(encode="category") -> pandas categoricals (LightGBM/CatBoost-friendly)
build_v5(encode="label")    -> integer label-encoded cats (ExtraTrees/XGBoost)
"""
import pandas as pd


def _features(df):
    df = df.copy()
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


CAT_COLS = ["geohash", "geohash_prefix_4", "geohash_prefix_5", "RoadType",
            "LargeVehicles", "Landmarks", "Weather"]

FEATURES = ["geohash", "geohash_mean_demand", "prefix4_mean_demand",
            "prefix5_mean_demand", "geohash_prefix_4", "geohash_prefix_5", "day",
            "RoadType", "NumberofLanes", "LargeVehicles", "Landmarks",
            "Temperature", "Weather", "hour", "minute", "total_minutes", "rush_hour"]


def build_v5(encode="category"):
    train = _features(pd.read_csv("data/train.csv"))
    test = _features(pd.read_csv("data/test.csv"))
    gmean = train["demand"].mean()

    geo_mean = train.groupby("geohash")["demand"].mean()
    train["geohash_mean_demand"] = train["geohash"].map(geo_mean)
    test["geohash_mean_demand"] = test["geohash"].map(geo_mean).fillna(gmean)

    p4 = train.groupby("geohash_prefix_4")["demand"].mean()
    p5 = train.groupby("geohash_prefix_5")["demand"].mean()
    train["prefix4_mean_demand"] = train["geohash_prefix_4"].map(p4)
    test["prefix4_mean_demand"] = test["geohash_prefix_4"].map(p4).fillna(gmean)
    train["prefix5_mean_demand"] = train["geohash_prefix_5"].map(p5)
    test["prefix5_mean_demand"] = test["geohash_prefix_5"].map(p5).fillna(gmean)

    for df in (train, test):
        df["RoadType"] = df["RoadType"].fillna("Unknown")
        df["Weather"] = df["Weather"].fillna("Unknown")
        df["Temperature"] = df["Temperature"].fillna(train["Temperature"].median())

    if encode == "category":
        for c in CAT_COLS:
            train[c] = train[c].astype("category")
            test[c] = test[c].astype("category")
    elif encode == "label":
        # consistent integer codes across train+test
        for c in CAT_COLS:
            cats = pd.Categorical(pd.concat([train[c], test[c]]).astype(str)).categories
            train[c] = pd.Categorical(train[c].astype(str), categories=cats).codes
            test[c] = pd.Categorical(test[c].astype(str), categories=cats).codes
    elif encode == "str":
        for c in CAT_COLS:
            train[c] = train[c].astype(str)
            test[c] = test[c].astype(str)

    return train, test
