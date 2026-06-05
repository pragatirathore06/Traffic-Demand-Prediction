import pandas as pd

v3 = pd.read_csv("submissions/submission_v3.csv")
v5 = pd.read_csv("submissions/submission_v5.csv")

ensemble = pd.DataFrame()

ensemble["Index"] = v3["Index"]

ensemble["demand"] = (
    0.4 * v3["demand"] +
    0.6 * v5["demand"]
)

ensemble.to_csv(
    "submissions/submission_ensemble_v1.csv",
    index=False
)

print("Saved:")
print("submissions/submission_ensemble_v1.csv")
print(ensemble.head())