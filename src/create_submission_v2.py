import pandas as pd
from lightgbm import LGBMRegressor

# =====================
# Load Data
# =====================

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# =====================
# Feature Engineering
# =====================

def create_features(df):

    # Time Features
    df["hour"] = df["timestamp"].apply(
        lambda x: int(str(x).split(":")[0])
    )

    df["minute"] = df["timestamp"].apply(
        lambda x: int(str(x).split(":")[1])
    )

    df["total_minutes"] = (
        df["hour"] * 60 + df["minute"]
    )

    # NEW FEATURE
    df["geohash_hour"] = (
        df["geohash"].astype(str)
        + "_"
        + df["hour"].astype(str)
    )

    # Rush Hour
    df["rush_hour"] = (
        ((df["hour"] >= 7) & (df["hour"] <= 10))
        |
        ((df["hour"] >= 17) & (df["hour"] <= 20))
    ).astype(int)

    # Geohash Prefixes
    df["geohash_prefix_4"] = df["geohash"].str[:4]
    df["geohash_prefix_5"] = df["geohash"].str[:5]

    return df


train = create_features(train)
test = create_features(test)

# =====================
# Missing Values
# =====================

for df in [train, test]:

    df["RoadType"] = df["RoadType"].fillna("Unknown")
    df["Weather"] = df["Weather"].fillna("Unknown")

    df["Temperature"] = df["Temperature"].fillna(
        train["Temperature"].median()
    )

# =====================
# Categorical Columns
# =====================

cat_cols = [
    "geohash",
    "geohash_hour",
    "geohash_prefix_4",
    "geohash_prefix_5",
    "RoadType",
    "LargeVehicles",
    "Landmarks",
    "Weather"
]

for col in cat_cols:

    train[col] = train[col].astype("category")
    test[col] = test[col].astype("category")

# =====================
# Features
# =====================

features = [
    "geohash",
    "geohash_hour",
    "geohash_prefix_4",
    "geohash_prefix_5",
    "day",
    "RoadType",
    "NumberofLanes",
    "LargeVehicles",
    "Landmarks",
    "Temperature",
    "Weather",
    "hour",
    "minute",
    "total_minutes",
    "rush_hour"
]

X_train = train[features]
y_train = train["demand"]

X_test = test[features]

# =====================
# Model
# =====================

model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    random_state=42
)

model.fit(X_train, y_train)

# =====================
# Prediction
# =====================

predictions = model.predict(X_test)

predictions = predictions.clip(0, 1)

# =====================
# Submission
# =====================

submission = pd.DataFrame({
    "Index": test["Index"],
    "demand": predictions
})

submission.to_csv(
    "submissions/submission_v2.csv",
    index=False
)

print("\nSubmission saved:")
print("submissions/submission_v2.csv")

print("\nShape:", submission.shape)

print(submission.head())