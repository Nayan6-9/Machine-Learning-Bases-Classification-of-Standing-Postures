#!/usr/bin/env python3
"""
wbb_qc.py - did this participant actually change posture?

The single biggest limit on this study is not the classifier: it is that some
participants' "slouched" trials look almost identical to their "neutral" ones.
When that happens the recording is unusable and no amount of modelling fixes it,
but it is invisible during collection - so it is only discovered weeks later.

This module answers the question while the person is still standing there:

    within-subject separability = can a model tell THIS person's two postures
    apart, training and testing only on their own data?

If that is near chance, the two conditions were effectively the same posture.
Re-coach and re-record. If it is high, the recording is good.

    python wbb_qc.py --db wbb_db --subject S07
    python wbb_qc.py --db wbb_db            # every subject, as a table
"""

import argparse
import sys

from wbb_dataset import Dataset, FEATURE_NAMES

# Features that moved most consistently between postures in the pilot data
# (within-subject Cohen's dz, n=18). Shown to the operator as a plain-language
# readout of what changed.
REPORT_FEATURES = [
    ("mean_velocity_cm_s", "sway speed", "cm/s"),
    ("path_length_cm", "total sway path", "cm"),
    ("rms_mag_cm", "sway size", "cm"),
    ("cop_shift_cm", "shift from neutral", "cm"),
    ("f50_ml", "left-right sway rate", "Hz"),
]

GOOD, WEAK, BAD = "GOOD", "WEAK", "UNUSABLE"


def subject_contrast(db, subject, win_s=10.0, hop_s=5.0, baseline_source="auto"):
    """How separable are this subject's own neutral and slouched recordings?

    Trains and tests only on this subject, holding out one trial at a time, so
    the score says whether the two conditions differ at all - not whether the
    model generalizes. Returns a dict with a 'report' string.
    """
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
    from sklearn.metrics import accuracy_score

    ds = Dataset(db)
    X, y, trials = ds.load_windowed_normalized(win_s, hop_s, "trial",
                                               baseline_source)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    trials = np.asarray(trials)

    # rows belonging to this subject
    keep = np.array([("_" + subject) in t or t.endswith(subject) for t in trials])
    if keep.sum() == 0:
        raise ValueError(f"No trials found for subject {subject}.")
    X, y, trials = X[keep], y[keep], trials[keep]

    labels = sorted(set(y.tolist()))
    lines = [f"POSTURE CONTRAST CHECK - {subject}"]
    if len(labels) < 2:
        lines.append(f"  Only '{labels[0]}' recorded so far - record the other "
                     "posture, then check again.")
        return {"report": "\n".join(lines), "verdict": None, "accuracy": None}

    n_trials = len(set(trials.tolist()))
    lines.append(f"  {n_trials} trials, {len(X)} windows")
    lines.append("")

    # what actually changed, in plain units
    lines.append("  What changed between the two postures:")
    for feat, plain, unit in REPORT_FEATURES:
        i = FEATURE_NAMES.index(feat)
        a = X[y == "neutral", i].mean()
        b = X[y == "slouched", i].mean()
        lines.append(f"    {plain:22} {a:8.2f} -> {b:8.2f} {unit:5}"
                     f"  ({b - a:+.2f})")
    lines.append("")

    # separability: hold out one trial at a time, this subject only
    if n_trials < 3:
        lines.append("  Need at least 3 trials to score separability.")
        return {"report": "\n".join(lines), "verdict": None, "accuracy": None}
    per_class = {c: len(set(trials[y == c].tolist())) for c in labels}
    if min(per_class.values()) < 1:
        lines.append("  Both postures must be recorded before scoring.")
        return {"report": "\n".join(lines), "verdict": None, "accuracy": None}

    model = RandomForestClassifier(n_estimators=200, random_state=0)
    pred = cross_val_predict(model, X, y, cv=LeaveOneGroupOut(), groups=trials)
    acc = float(accuracy_score(y, pred))

    if acc >= 0.80:
        verdict, advice = GOOD, "Clear difference. Recording is good."
    elif acc >= 0.65:
        verdict, advice = WEAK, ("Weak difference. If there is time, coach the "
                                 "slouch more clearly and add one more pair.")
    else:
        verdict, advice = BAD, ("The two postures look the same to the board. "
                                "Re-coach and re-record this subject.")

    lines.append(f"  Separability (own data only): {acc:.2f}")
    lines.append(f"  --> {verdict}: {advice}")
    if verdict != GOOD:
        lines.append("")
        lines.append("  Coaching that usually helps:")
        lines.append("    - exaggerate: let the upper back round clearly")
        lines.append("    - hold it; do not drift back to neutral mid-trial")
        lines.append("    - keep the feet on the tape (a wider stance is not")
        lines.append("      the posture we are trying to measure)")

    return {"report": "\n".join(lines), "verdict": verdict, "accuracy": acc,
            "n_trials": n_trials, "n_windows": int(len(X))}


def all_subjects(db, win_s=10.0, hop_s=5.0, baseline_source="auto"):
    """One line per subject - use at the end of a session."""
    ds = Dataset(db)
    subs = set()
    import csv
    with open(ds.csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            subs.add(r["subject"])
    lines = ["CONTRAST CHECK - every subject",
             "",
             f"  {'subject':10}{'separability':>14}  verdict"]
    out = {}
    for s in sorted(subs):
        try:
            r = subject_contrast(db, s, win_s, hop_s, baseline_source)
        except Exception as e:
            lines.append(f"  {s:10}{'-':>14}  ({e})")
            continue
        if r["accuracy"] is None:
            lines.append(f"  {s:10}{'-':>14}  incomplete")
            continue
        out[s] = r
        lines.append(f"  {s:10}{r['accuracy']:>14.2f}  {r['verdict']}")
    bad = [s for s, r in out.items() if r["verdict"] == BAD]
    if bad:
        lines.append("")
        lines.append("  Re-record if possible: " + ", ".join(bad))
    return {"report": "\n".join(lines), "subjects": out}


def main():
    ap = argparse.ArgumentParser(description="Did the posture actually change?")
    ap.add_argument("--db", default="wbb_db")
    ap.add_argument("--subject", default=None)
    ap.add_argument("--window", type=float, default=10.0)
    ap.add_argument("--hop", type=float, default=5.0)
    args = ap.parse_args()
    try:
        if args.subject:
            print(subject_contrast(args.db, args.subject, args.window, args.hop)["report"])
        else:
            print(all_subjects(args.db, args.window, args.hop)["report"])
    except ValueError as e:
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
