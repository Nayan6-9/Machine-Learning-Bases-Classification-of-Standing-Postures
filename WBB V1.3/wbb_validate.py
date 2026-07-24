#!/usr/bin/env python3
"""
wbb_validate.py - Phase 2: test a FROZEN model on newly collected subjects.

Nothing is fitted here. The saved model is loaded exactly as it was trained and
scored on a dataset it has never seen, which is the honest estimate of how the
system behaves on a new person.

    python wbb_validate.py --db wbb_db_val --model posture_model.joblib

The validation dataset must be a SEPARATE folder from the training dataset, and
must be collected with the same protocol (same window, same postures). If the
model is baseline-normalized, every validation subject needs their own dedicated
BASELINE recording - that is the 10 s calibration step, and it is part of the
method being tested.
"""

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict

from wbb_dataset import Dataset, FEATURE_NAMES, BASELINE_LABEL


def _trial_subject_map(ds):
    m = {}
    with open(ds.csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            m[r["trial_id"]] = r["subject"]
    return m


def evaluate_saved_model(db, model_path):
    """Score a frozen model on `db`. Returns a dict with a text 'report'.
    Raises ValueError if the data and the model do not match."""
    import numpy as np
    import joblib
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, confusion_matrix)

    if not os.path.exists(model_path):
        raise ValueError(f"No model file: {model_path}\nTrain and freeze one first.")
    b = joblib.load(model_path)
    feature_names = b["feature_names"]
    positive = b["positive"]
    win = b.get("window_s")
    hop = b.get("hop_s") or (win / 2.0 if win else None)
    normalized = bool(b.get("normalized", False))
    model = b["model"]

    if list(feature_names) != list(FEATURE_NAMES):
        raise ValueError(
            "The model was trained on a different feature set than this app "
            f"produces ({len(feature_names)} vs {len(FEATURE_NAMES)}).\n"
            "Re-train the model with the current version.")

    ds = Dataset(db)
    counts = ds.counts()
    if not any(k not in (BASELINE_LABEL,) for k in counts):
        raise ValueError(f"No posture trials in {db}. Collect validation data first.")

    warnings = []
    trained_on = set(b.get("train_subjects") or [])
    if normalized:
        have = ds.baseline_subjects()
        subjects_all = set(_trial_subject_map(ds).values())
        posture_subjects = set()
        with open(ds.csv_path, newline="") as fh:
            for r in csv.DictReader(fh):
                if r["label"] != BASELINE_LABEL:
                    posture_subjects.add(r["subject"])
        missing = sorted(posture_subjects - have)
        if missing:
            warnings.append(
                "These subjects have NO baseline recording, so their calibration "
                "falls back to the group average: " + ", ".join(missing))
        short = {s: d for s, d in ds.baseline_durations().items()
                 if win and d < 0.9 * win}
        if short:
            warnings.append(
                "Baseline shorter than the model's window for: "
                + ", ".join(f"{s} ({d:.0f}s)" for s, d in short.items()))

    if win is None:
        X, y, meta = ds.load()
        trials = [m["trial_id"] for m in meta]
    elif normalized:
        X, y, trials = ds.load_windowed_normalized(win, hop, group_by="trial",
                                                   baseline_source="dedicated")
    else:
        X, y, trials = ds.load_windowed(win, hop, group_by="trial")

    if not X:
        raise ValueError(
            f"No usable windows. The model needs {win}s windows - are the "
            f"validation trials at least that long?")

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    t2s = _trial_subject_map(ds)
    subj = np.asarray([t2s[t] for t in trials])
    labels = sorted(set(y.tolist()))

    pred = model.predict(X)

    lines = []
    lines.append(f"PHASE 2 - frozen model on unseen data")
    lines.append(f"  model   : {os.path.basename(model_path)}")
    lines.append(f"  data    : {db}")
    lines.append(f"  setup   : window {win}s, "
                 + ("baseline-normalized (10s calibration per subject)"
                    if normalized else "no calibration"))
    lines.append(f"  subjects: {len(set(subj.tolist()))}   "
                 f"trials: {len(set(trials))}   windows: {len(X)}")

    # Phase 2 only measures generalization if these are genuinely new people.
    seen = sorted(set(subj.tolist()) & trained_on)
    if seen:
        lines.append("")
        lines.append("  *** NOT A HELD-OUT TEST ***")
        lines.append("  These subjects were in the training set: "
                     + ", ".join(seen))
        lines.append("  The model already learned them, so their scores measure")
        lines.append("  memory, not generalization. Remove them, or report them")
        lines.append("  separately from the genuinely new subjects.")
        warnings.append("Phase 2 contains training subjects: " + ", ".join(seen))
    elif trained_on:
        lines.append(f"  held out : yes - none of these {len(set(subj.tolist()))} "
                     f"people were in the {len(trained_on)}-subject training set")
    lines.append("")

    acc = accuracy_score(y, pred)
    prec = precision_score(y, pred, pos_label=positive, zero_division=0)
    rec = recall_score(y, pred, pos_label=positive, zero_division=0)
    f1 = f1_score(y, pred, pos_label=positive, zero_division=0)
    lines.append(f"{'':8s}{'acc':>7s}{'prec':>7s}{'rec':>7s}{'F1':>7s}")
    lines.append(f"{'window':8s}{acc:>7.3f}{prec:>7.3f}{rec:>7.3f}{f1:>7.3f}")

    # per-trial majority vote = one decision per recording (how it is used)
    gp = defaultdict(list)
    gt = {}
    for i in range(len(y)):
        gp[trials[i]].append(pred[i])
        gt[trials[i]] = y[i]
    tr_pred = {g: Counter(v).most_common(1)[0][0] for g, v in gp.items()}
    tr_true = [gt[g] for g in gp]
    tr_hat = [tr_pred[g] for g in gp]
    tacc = accuracy_score(tr_true, tr_hat)
    tprec = precision_score(tr_true, tr_hat, pos_label=positive, zero_division=0)
    trec = recall_score(tr_true, tr_hat, pos_label=positive, zero_division=0)
    tf1 = f1_score(tr_true, tr_hat, pos_label=positive, zero_division=0)
    lines.append(f"{'trial':8s}{tacc:>7.3f}{tprec:>7.3f}{trec:>7.3f}{tf1:>7.3f}")
    lines.append(f"  (trial = majority vote of that recording's windows)")

    cm = confusion_matrix(tr_true, tr_hat, labels=labels)
    lines.append("")
    lines.append(f"Confusion, per trial (rows=true {labels}):")
    for i, lab in enumerate(labels):
        lines.append(f"  {lab:9s} {cm[i].tolist()}")

    # per-subject accuracy: the unit of generalization is a PERSON
    lines.append("")
    lines.append("Per-subject accuracy (trial-level):")
    per_subj = {}
    for s in sorted(set(subj.tolist())):
        idx = [i for i, g in enumerate(gp) if t2s[g] == s]
        gl = [list(gp.keys())[i] for i in idx]
        if not gl:
            continue
        a = accuracy_score([gt[g] for g in gl], [tr_pred[g] for g in gl])
        per_subj[s] = float(a)
        lines.append(f"  {s:8s} {a:5.3f}   ({len(gl)} trials)")
    if per_subj:
        vals = np.asarray(list(per_subj.values()), dtype=float)
        n = len(vals)
        mean = float(vals.mean())
        sd = float(vals.std(ddof=1)) if n > 1 else 0.0
        half = 1.96 * sd / (n ** 0.5) if n > 1 else 0.0
        lines.append(f"  mean {mean:.3f} +- {sd:.3f} sd   "
                     f"95% CI [{max(0.0, mean-half):.3f}, {min(1.0, mean+half):.3f}]"
                     f"  (n={n} subjects)")
        if n < 12:
            lines.append(f"  NOTE: with {n} subjects the interval is wide - report "
                         "it, do not report the mean alone.")

    if warnings:
        lines.append("")
        for w in warnings:
            lines.append("WARNING: " + w)

    cv_acc = b.get("cv_accuracy")
    if cv_acc is not None:
        gb = b.get("cv_group_by") or "?"
        lines.append("")
        lines.append("Did the Phase 1 estimate hold up?")
        lines.append(f"  Phase 1 CV ({gb}-grouped) : {cv_acc:.3f}")
        lines.append(f"  Phase 2 held-out (trial)  : {tacc:.3f}"
                     f"   difference {tacc - cv_acc:+.3f}")
        if gb == "trial":
            lines.append("  CAUTION: the Phase 1 number was trial-grouped, which is")
            lines.append("  optimistic. Subject-grouped is the fair comparison.")

    return {"report": "\n".join(lines), "window_acc": float(acc),
            "trial_acc": float(tacc), "trial_f1": float(tf1),
            "trial_prec": float(tprec), "trial_rec": float(trec),
            "confusion": cm.tolist(),
            "per_subject": per_subj, "n_subjects": len(set(subj.tolist())),
            "n_trials": len(set(trials)), "n_windows": int(len(X)),
            "normalized": normalized, "labels": labels, "warnings": warnings,
            "window_s": win, "model_name": os.path.basename(model_path),
            # what this model scored in Phase 1, before it was frozen
            "cv_accuracy": b.get("cv_accuracy"), "cv_group_by": b.get("cv_group_by"),
            "cv_folds": b.get("cv_folds"), "cv_model": b.get("cv_model"),
            "train_subjects": sorted(trained_on),
            "contaminated": sorted(set(subj.tolist()) & trained_on)}


def main():
    ap = argparse.ArgumentParser(description="Score a frozen model on new data")
    ap.add_argument("--db", default="wbb_db_val",
                    help="validation dataset folder (must NOT be the training one)")
    ap.add_argument("--model", default="posture_model.joblib")
    args = ap.parse_args()
    try:
        res = evaluate_saved_model(args.db, args.model)
    except ValueError as e:
        print(str(e))
        sys.exit(1)
    print(res["report"])


if __name__ == "__main__":
    main()
