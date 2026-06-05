import pandas as pd

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

train_ts = set(train["timestamp"].unique())
test_ts = set(test["timestamp"].unique())

missing = sorted(list(train_ts - test_ts))

print("Timestamps missing from test:")
print(missing)

print("\nCount:", len(missing))