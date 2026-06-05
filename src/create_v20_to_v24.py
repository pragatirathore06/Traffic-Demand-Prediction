"""
Build five genuinely-different models on the V5 feature set:
  V20 CatBoost deep | V21 ExtraTrees | V22 XGBoost
  V23 = 0.7*LGB(V5) + 0.3*CatBoost(V20) blend
  V24 = rank blend of V5 + V15
Run:  python src/create_v20_to_v24.py
"""
import sys
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from v5_features import build_v5, FEATURES, CAT_COLS


def save(name, pred):
    test_index = pd.read_csv("data/test.csv")["Index"]
    pd.DataFrame({"Index": test_index, "demand": np.clip(pred, 0, 1)}).to_csv(
        f"submissions/{name}.csv", index=False)
    print(f"Saved submissions/{name}.csv")


# ---------- V20: CatBoost deep ----------
from catboost import CatBoostRegressor
tr, te = build_v5(encode="str")
cat_idx = [FEATURES.index(c) for c in CAT_COLS]
m20 = CatBoostRegressor(iterations=5000, learning_rate=0.02, depth=10,
                        l2_leaf_reg=10, random_seed=42, loss_function="RMSE",
                        verbose=1000)
m20.fit(tr[FEATURES], tr["demand"], cat_features=cat_idx)
p20 = m20.predict(te[FEATURES])
save("submission_v20_catboost_deep", p20)

# ---------- V21: ExtraTrees ----------
from sklearn.ensemble import ExtraTreesRegressor
trl, tel = build_v5(encode="label")
m21 = ExtraTreesRegressor(n_estimators=1000, max_depth=None, min_samples_leaf=2,
                          random_state=42, n_jobs=-1)
m21.fit(trl[FEATURES], trl["demand"])
p21 = m21.predict(tel[FEATURES])
save("submission_v21_extratrees", p21)

# ---------- V22: XGBoost ----------
from xgboost import XGBRegressor
m22 = XGBRegressor(n_estimators=2000, learning_rate=0.03, max_depth=8,
                   subsample=0.8, colsample_bytree=0.8, random_state=42,
                   tree_method="hist", n_jobs=-1)
m22.fit(trl[FEATURES], trl["demand"])
p22 = m22.predict(tel[FEATURES])
save("submission_v22_xgb", p22)

# ---------- V23: LightGBM(V5) + CatBoost(V20) blend ----------
v5 = pd.read_csv("submissions/submission_v5.csv").sort_values("Index").reset_index(drop=True)
p23 = 0.7 * v5["demand"].values + 0.3 * np.clip(p20, 0, 1)
save("submission_v23_blend_70_30", p23)

# ---------- V24: rank blend of V5 + V15 ----------
v15 = pd.read_csv("submissions/submission_v15_catboost.csv").sort_values("Index").reset_index(drop=True)
r1 = rankdata(v5["demand"].values)
r2 = rankdata(v15["demand"].values)
rb = 0.5 * r1 + 0.5 * r2
# map ranks back to a real demand scale (V5's distribution) so R2 is meaningful
order = np.argsort(np.argsort(rb))
p24 = np.sort(v5["demand"].values)[order]
save("submission_v24_rankblend", p24)

print("\nDone. Deviations from V5:")
for name in ["submission_v20_catboost_deep", "submission_v21_extratrees",
             "submission_v22_xgb", "submission_v23_blend_70_30",
             "submission_v24_rankblend"]:
    p = pd.read_csv(f"submissions/{name}.csv").sort_values("Index")["demand"].values
    print(f"  {name:32s} corr={np.corrcoef(p, v5['demand'])[0,1]:.4f}  MAE={np.abs(p-v5['demand'].values).mean():.4f}")
