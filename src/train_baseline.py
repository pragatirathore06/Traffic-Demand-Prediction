import pandas as pd
from sklearn.model_selection import train_test_split
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
# Simple Missing Value Handling
# -------------------

train["RoadType"] = train["RoadType"].fillna("Unknown")
train["Weather"] = train["Weather"].fillna("Unknown")

train["Temperature"] = train["Temperature"].fillna(
    train["Temperature"].median()
)

# -------------------
# Encode Categoricals
# -------------------

categorical_cols = [
    "geohash",
    "RoadType",
    "LargeVehicles",
    "Landmarks",
    "Weather"
]

for col in categorical_cols:
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

# -------------------
# Train Validation Split
# -------------------

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------
# Model
# -------------------

model = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_valid)

score = r2_score(y_valid, preds)

print("\nValidation R²:", score)

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance")
print(importance_df)