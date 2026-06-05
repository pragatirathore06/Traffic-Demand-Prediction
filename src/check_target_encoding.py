import pandas as pd

train = pd.read_csv("data/train.csv")

geo_mean = (
    train.groupby("geohash")["demand"]
    .mean()
    .reset_index()
)

print(geo_mean.head())

print("\nTotal geohashes:")
print(len(geo_mean))