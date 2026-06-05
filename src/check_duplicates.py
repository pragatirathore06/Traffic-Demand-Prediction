import pandas as pd

train = pd.read_csv("data/train.csv")

dup = train.groupby(
    ["geohash", "day", "timestamp"]
)["demand"].count()

print("Max count:", dup.max())
print("Min count:", dup.min())

print("\nValue counts of occurrences:")
print(dup.value_counts().head(20))