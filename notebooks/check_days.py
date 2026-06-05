import pandas as pd

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

print("Train Days")
print(sorted(train["day"].unique()))

print("\nTest Days")
print(sorted(test["day"].unique()))