import pandas as pd

train = pd.read_csv("data/train.csv")

# Time Features
train["hour"] = train["timestamp"].apply(
    lambda x: int(str(x).split(":")[0])
)

train["minute"] = train["timestamp"].apply(
    lambda x: int(str(x).split(":")[1])
)

train["total_minutes"] = (
    train["hour"] * 60 + train["minute"]
)

# Rush Hour
train["rush_hour"] = (
    ((train["hour"] >= 7) & (train["hour"] <= 10))
    |
    ((train["hour"] >= 17) & (train["hour"] <= 20))
).astype(int)

# Geohash Prefixes
train["geohash_prefix_4"] = (
    train["geohash"].str[:4]
)

train["geohash_prefix_5"] = (
    train["geohash"].str[:5]
)

print(
    train[
        [
            "geohash",
            "geohash_prefix_4",
            "geohash_prefix_5",
            "hour",
            "rush_hour"
        ]
    ].head()
)