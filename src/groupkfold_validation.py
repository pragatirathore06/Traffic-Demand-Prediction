import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

from lightgbm import LGBMRegressor

# -------------------
# Load Data
# -------------------

train = pd.read_csv("data/train.csv")

# -------------------
# Time Features
# -------------------

train["hour"] = train["timestamp"].apply(
    lambda x: int(str(x).split(":")[0])
)

train["minute"] = train["timestamp"].apply(
    lambda x: int(str(x).split(":")[1])
)

train["total_minutes"] = (
    train["hour"] * 60 + train["minute"]
)

# -------------------
# Missing Values
# -------------------

train["RoadType"] = train["RoadType"].fillna("Unknown")
train["Weather"] = train["Weather"].fillna("Unknown")

train["Temperature"] = train["Temperature"].fillna(
    train["Temperature"].median()
)

# -------------------
# Categories
# -------------------

cat_cols = [
    "geohash",
    "RoadType",
    "LargeVehicles",
    "Landmarks",
    "Weather"
]

for col in cat_cols:
    train[col] = train[col].astype("category")

# -------------------
# Features
# -------------------

features = [
    "geohash",
    "day",
    "RoadType",
    "NumberofLanes",
    "LargeVehicles",
    "Landmarks",
    "Temperature",
    "Weather",
    "hour",
    "minute",
    "total_minutes"
]

X = train[features]
y = train["demand"]

groups = train["geohash"]

gkf = GroupKFold(n_splits=5)

scores = []

for fold, (train_idx, valid_idx) in enumerate(
    gkf.split(X, y, groups)
):

    X_train = X.iloc[train_idx]
    X_valid = X.iloc[valid_idx]

    y_train = y.iloc[train_idx]
    y_valid = y.iloc[valid_idx]

    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    score = r2_score(y_valid, preds)

    scores.append(score)

    print(f"Fold {fold+1}: {score:.5f}")

print("\nMean R²:", sum(scores)/len(scores))