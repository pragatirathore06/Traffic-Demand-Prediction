"""Fast subset: V21 ExtraTrees, V22 XGBoost, V24 rank-blend (no CatBoost dep)."""
import sys
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from v5_features import build_v5, FEATURES

test_index = pd.read_csv("data/test.csv")["Index"]
v5 = pd.read_csv("submissions/submission_v5.csv").sort_values("Index").reset_index(drop=True)
v15 = pd.read_csv("submissions/submission_v15_catboost.csv").sort_values("Index").reset_index(drop=True)


def save(name, pred):
    pd.DataFrame({"Index": test_index, "demand": np.clip(pred, 0, 1)}).to_csv(
        f"submissions/{name}.csv", index=False)
    print("Saved", name)


trl, tel = build_v5(encode="label")

# V21 ExtraTrees
from sklearn.ensemble import ExtraTreesRegressor
m21 = ExtraTreesRegressor(n_estimators=1000, max_depth=None, min_samples_leaf=2,
                          random_state=42, n_jobs=-1)
m21.fit(trl[FEATURES], trl["demand"])
save("submission_v21_extratrees", m21.predict(tel[FEATURES]))

# V22 XGBoost
from xgboost import XGBRegressor
m22 = XGBRegressor(n_estimators=2000, learning_rate=0.03, max_depth=8,
                   subsample=0.8, colsample_bytree=0.8, random_state=42,
                   tree_method="hist", n_jobs=-1)
m22.fit(trl[FEATURES], trl["demand"])
save("submission_v22_xgb", m22.predict(tel[FEATURES]))

# V24 rank-blend of V5 + V15, mapped back onto V5's demand scale (R2-safe)
rb = 0.5 * rankdata(v5["demand"].values) + 0.5 * rankdata(v15["demand"].values)
order = np.argsort(np.argsort(rb))
save("submission_v24_rankblend", np.sort(v5["demand"].values)[order])

print("\nDeviation from V5:")
for name in ["submission_v21_extratrees", "submission_v22_xgb", "submission_v24_rankblend"]:
    p = pd.read_csv(f"submissions/{name}.csv").sort_values("Index")["demand"].values
    print(f"  {name:30s} corr={np.corrcoef(p, v5['demand'])[0,1]:.4f}  MAE={np.abs(p-v5['demand'].values).mean():.4f}")
