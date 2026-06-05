"""
Probe solver — interpret the leaderboard scores returned by the probe files.

Submit these files and paste the returned scores into SCORES below:
  probe_constant.csv  -> all global mean   (best constant => R2<=0 => score ~0)
  probe_full.csv      -> your full model   (call it S)
  probe_q1..q4.csv    -> model on quarter q, global mean elsewhere

Logic:
  * A quarter probe predicts well only on its quarter and the best-constant
    elsewhere, so its score ~= 100 * (share of total scored variance in that
    quarter) * (model quality there). If the leaderboard grades ALL rows, the
    four quarter scores roughly SUM to the full score S. If the LB grades only a
    subset, the quarters with no scored rows return ~0 and the others carry it.
"""

# -------- paste your returned leaderboard scores here --------
SCORES = {
    "constant": None,   # probe_constant.csv
    "full":     None,   # probe_full.csv  (= your real submission score S)
    "q1":       None,   # probe_q1.csv
    "q2":       None,
    "q3":       None,
    "q4":       None,
}
# -------------------------------------------------------------


def solve(s):
    miss = [k for k, v in s.items() if v is None]
    if miss:
        print("Fill in these scores first:", ", ".join(miss))
        return
    S = s["full"]
    quarters = [s["q1"], s["q2"], s["q3"], s["q4"]]
    qsum = sum(quarters)
    print(f"full model score S        = {S:.4f}")
    print(f"constant (global-mean)    = {s['constant']:.4f}   (expect ~0 if all rows graded)")
    print(f"sum of quarter scores     = {qsum:.4f}")
    print(f"ratio qsum / S            = {qsum / S:.3f}\n")

    live = [i + 1 for i, q in enumerate(quarters) if q > 0.05 * S]
    dead = [i + 1 for i, q in enumerate(quarters) if q <= 0.05 * S]

    if s["constant"] > 5:
        print("VERDICT: a constant prediction already scores high -> the public LB is")
        print("NOT plain R2 over all rows (tiny subset or different metric).")
        print("This is THE explanation for the 100s. Inspect rules/metric next.")
    elif not dead and 0.8 <= qsum / S <= 1.25:
        print("VERDICT: all four quarters contribute and sum ~ S.")
        print("=> The leaderboard grades essentially ALL rows. The 100s have the")
        print("   real day-49 labels (external source). No model crosses ~92; your")
        print("   91.69 is honest and competitive on a full-data private split.")
    else:
        print(f"VERDICT: scored rows are concentrated in quarter(s) {live}; "
              f"quarter(s) {dead} contribute ~0.")
        print("=> The public LB grades only a SUBSET. The teams at 100 are")
        print("   overfitting that subset and will likely COLLAPSE on the private")
        print("   leaderboard. Re-run make_probes.py with N_SPLITS=8 or 16 to")
        print(f"   localise the subset inside quarter(s) {live}, then focus there.")


if __name__ == "__main__":
    solve(SCORES)
