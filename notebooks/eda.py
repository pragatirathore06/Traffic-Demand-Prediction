import pandas as pd

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

print("="*50)
print("UNIQUE GEOHASHES")
print("="*50)

print("Train:", train["geohash"].nunique())
print("Test :", test["geohash"].nunique())

print("\n")

print("="*50)
print("UNIQUE DAYS")
print("="*50)

print("Train:", train["day"].nunique())
print("Test :", test["day"].nunique())

print("\n")

print("="*50)
print("UNIQUE TIMESTAMPS")
print("="*50)

print("Train:", train["timestamp"].nunique())
print("Test :", test["timestamp"].nunique())

print("\n")

print("="*50)
print("SAMPLE TIMESTAMPS")
print("="*50)

print(sorted(train["timestamp"].unique())[:20])

print("\n")

print("="*50)
print("UNSEEN GEOHASHES IN TEST")
print("="*50)

unseen = set(test["geohash"]) - set(train["geohash"])

print("Count:", len(unseen))