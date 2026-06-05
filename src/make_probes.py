"""
Leaderboard diagnostic probes.

GOAL: find out whether the public leaderboard scores ALL 41,778 test rows or
only a small hidden subset. If it scores a small subset, the teams at "100" are
overfitting that subset and will collapse on the private leaderboard — your
91.7 is actually strong. If it scores everything, the 100s have a real label
leak (see notes at bottom).

METHOD: split the test by Index into 4 quarters. For each quarter q we submit a
file that uses the strong v9 prediction on q and the GLOBAL MEAN everywhere else.
Because the global mean is the best *constant*, the non-q rows contribute their
full variance to the error and ~0 to R². So:

    score_q  ~=  100 * (fraction of total scored variance located in quarter q)
                     * (model quality on quarter q)

Read the 4 returned scores:
  * All four clearly positive and roughly summing to your full v9 score
        -> the LB scores the whole test set. 100 requires a real label leak.
  * Some quarters ~0, others large
        -> the LB scores only a SUBSET (the non-zero quarters). The 100-club is
           subset-overfitting. Re-run this script with finer splits (set N_SPLITS
           higher) to localise the scored rows, then focus all effort there.

Submit these 5 files one at a time and write the returned score next to each.
"""
import numpy as np
import pandas as pd

N_SPLITS = 4

test = pd.read_csv("data/test.csv")
v9 = pd.read_csv("submissions/submission_v9.csv")["demand"].values
gmean = pd.read_csv("data/train.csv")["demand"].mean()
n = len(test)
idx = test["Index"].values

# Baseline: full v9 everywhere (this is also your real submission).
pd.DataFrame({"Index": idx, "demand": v9}).to_csv(
    "submissions/probe_full.csv", index=False)

# Constant global-mean everywhere -> should score ~0 if LB grades all rows.
pd.DataFrame({"Index": idx, "demand": np.full(n, gmean)}).to_csv(
    "submissions/probe_constant.csv", index=False)

# Quarter probes: v9 on the quarter, global mean elsewhere.
bounds = np.linspace(0, n, N_SPLITS + 1).astype(int)
for q in range(N_SPLITS):
    lo, hi = bounds[q], bounds[q + 1]
    pred = np.full(n, gmean)
    pred[lo:hi] = v9[lo:hi]
    pd.DataFrame({"Index": idx, "demand": pred}).to_csv(
        f"submissions/probe_q{q+1}.csv", index=False)
    print(f"probe_q{q+1}.csv  -> v9 on Index[{idx[lo]}..{idx[hi-1]}], mean elsewhere")

print("\nSubmit in this order and record each score:")
print("  probe_constant.csv  (expect ~0 if all rows graded)")
print("  probe_full.csv      (your real v9 score, call it S)")
print("  probe_q1..q4.csv    (should roughly sum to S if all rows graded)")
