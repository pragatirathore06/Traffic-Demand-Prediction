import pandas as pd

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")


def create_time_features(df):

    df["hour"] = df["timestamp"].apply(
        lambda x: int(str(x).split(":")[0])
    )

    df["minute"] = df["timestamp"].apply(
        lambda x: int(str(x).split(":")[1])
    )

    df["total_minutes"] = (
        df["hour"] * 60 + df["minute"]
    )

    return df


def create_geohash_features(df):

    df["geohash_prefix_4"] = df["geohash"].str[:4]
    df["geohash_prefix_5"] = df["geohash"].str[:5]

    return df


train = create_time_features(train)
test = create_time_features(test)

train = create_geohash_features(train)
test = create_geohash_features(test)

print(
    train[
        [
            "geohash",
            "geohash_prefix_4",
            "geohash_prefix_5"
        ]
    ].head()
)