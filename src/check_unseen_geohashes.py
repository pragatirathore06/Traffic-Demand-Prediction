import pandas as pd

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

train_geohashes = set(train["geohash"])
test_geohashes = set(test["geohash"])

unseen = test_geohashes - train_geohashes

print("Train geohashes:", len(train_geohashes))
print("Test geohashes :", len(test_geohashes))
print("Unseen geohashes:", len(unseen))

print("\nUnseen geohashes:")
print(sorted(list(unseen)))