import pandas as pd

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

print("Train days:")
print(sorted(train["day"].unique()))

print("\nTest days:")
print(sorted(test["day"].unique()))

print("\nTrain timestamps:", train["timestamp"].nunique())
print("Test timestamps :", test["timestamp"].nunique())

print("\nFirst 20 test timestamps:")
print(sorted(test["timestamp"].unique())[:20])

print("\nLast 20 test timestamps:")
print(sorted(test["timestamp"].unique())[-20:])