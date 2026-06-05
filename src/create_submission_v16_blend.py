"""
V16 - blend V5 (best, 91.69) with V13 (91.57). Produces two blend files.
Blending barely-different models rarely helps when both are ~99.9% correlated,
but it is cheap and occasionally nets a tiny LB gain. Try the 0.9/0.1 first.
"""
import pandas as pd

v5 = pd.read_csv("submissions/submission_v5.csv").sort_values("Index").reset_index(drop=True)
v13 = pd.read_csv("submissions/submission_v13.csv").sort_values("Index").reset_index(drop=True)
assert (v5["Index"].values == v13["Index"].values).all()

for w5, tag in [(0.9, "91"), (0.8, "82")]:
    blend = (w5 * v5["demand"] + (1 - w5) * v13["demand"]).clip(0, 1)
    out = pd.DataFrame({"Index": v5["Index"], "demand": blend})
    path = f"submissions/submission_v16_blend_{tag}.csv"
    out.to_csv(path, index=False)
    print(f"Saved {path}  ({w5:.1f}*V5 + {1-w5:.1f}*V13)")
