#!/usr/bin/env python3
"""
wbb_train.py - train & compare classifiers on the posture dataset (item 5),
with optional windowed augmentation + leakage-safe grouped CV (item 1).

Compares Logistic Regression, Decision Tree, and Random Forest, reports
accuracy / precision / recall / F1 for the 'slouched' (non-ergonomic) class,
prints confusion matrices, picks the best by F1, refits on all data, and saves
it for live monitoring.

Two modes:
  per-trial (default): one feature row per 30 s trial; StratifiedKFold.
  windowed (--window): each trial is split into overlapping sub-windows for
      augmentation, and StratifiedGroupKFold keeps all windows of one trial in
      the same fold (group = trial_id) so there is NO leakage. The window length
      is stored in the model so the live monitor uses the same window.

    python wbb_train.py --db wbb_db --out posture_model.joblib --cv 5
    python wbb_train.py --db wbb_db --window 10 --hop 5 --out posture_model.joblib

Needs scikit-learn + joblib (pip install scikit-learn joblib).
"""

import argparse
import sys

import numpy as np

from wbb_dataset import Dataset, FEATURE_NAMES, DEFAULT_WINDOW_S, DEFAULT_HOP_S

POSITIVE = "slouched"   # the class we care about detecting


def build_models():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    return {
        "logreg": Pipeline([("scaler", StandardScaler()),
                            ("clf", LogisticRegression(max_iter=1000))]),
        "tree": DecisionTreeClassifier(random_state=0),
        "rf": RandomForestClassifier(n_estimators=300, random_state=0),
    }


def compare_designs(db, window=None, hop=None, normalize=True,
                    baseline_source="auto", trial_cv=5):
    """Run the three analyses a paper should report side by side.

    trials-only k-fold : windows grouped by trial. The same people are in train
                         and test, so it answers "does the signal exist, and can
                         the model read it within a person?" - optimistic.
    subject k-fold     : people grouped, SAME number of folds as trials-only, so
                         both fit on the same fraction of the data. The only
                         thing that changes is whether the test people are known.
                         This is the controlled contrast for the gap.
    LOSO               : one fold per person. The conventional way to report
                         between-subject generalization, and the closest match to
                         the deployed model (it trains on n-1 of n subjects).

    Report LOSO as the headline number and use the matched pair to talk about the
    gap, which measures how person-specific the posture signal is.
    """
    out = {}
    out["trial"] = train_and_compare(db, window=window, hop=hop, cv=trial_cv,
                                     normalize=normalize, group_by="trial",
                                     baseline_source=baseline_source)
    out["subject_k"] = train_and_compare(db, window=window, hop=hop, cv=trial_cv,
                                         normalize=normalize, group_by="subject",
                                         baseline_source=baseline_source)
    out["loso"] = train_and_compare(db, window=window, hop=hop, cv=None,
                                    normalize=normalize, group_by="subject",
                                    baseline_source=baseline_source)

    rows = [("trials-only k-fold", "trial"),
            ("subject k-fold", "subject_k"),
            ("LOSO", "loso")]
    lines = []
    lines.append("HOW THE MODEL IS SCORED - three designs")
    lines.append("")
    lines.append(f"{'':20s}{'folds':>6s}{'best':>8s}{'acc':>8s}{'F1':>8s}"
                 f"{'fold sd':>9s}")
    for name, key in rows:
        r = out[key]
        b = r["best"]
        lines.append(f"{name:20s}{len(r['folds']):>6d}{b:>8s}"
                     f"{r['results'][b][0]:>8.3f}{r['results'][b][3]:>8.3f}"
                     f"{r['fold_sd'][b]:>9.3f}")

    ta = out["trial"]["results"][out["trial"]["best"]][0]
    sa = out["subject_k"]["results"][out["subject_k"]["best"]][0]
    la = out["loso"]["results"][out["loso"]["best"]][0]
    n_subj = len(out["loso"]["folds"])
    matched = (len(out["subject_k"]["folds"]) == len(out["trial"]["folds"]))

    lines.append("")
    lines.append("MATCHED CONTRAST (same folds -> same training size;")
    lines.append("the only difference is whether the test people are known):")
    if matched:
        lines.append(f"  trials-only {ta:.3f}  -  subject k-fold {sa:.3f}"
                     f"  =  gap {ta - sa:+.3f}")
        if ta - sa > 0.15:
            lines.append("  Large gap: the signal is person-specific. Knowing the")
            lines.append("  person is doing much of the work.")
        elif ta - sa < 0.05:
            lines.append("  Small gap: the signal looks consistent across people.")
    else:
        lines.append(f"  Not matched: only {n_subj} subjects, so subject k-fold "
                     f"collapsed to {len(out['subject_k']['folds'])} folds.")
        lines.append("  Collect more people for a clean matched contrast.")

    lines.append("")
    lines.append(f"HEADLINE FOR THE PAPER: LOSO = {la:.3f}  (n={n_subj} subjects)")
    if len(out["subject_k"]["folds"]) == n_subj:
        lines.append(f"  NOTE: with {n_subj} subjects, subject k-fold IS LOSO -")
        lines.append("  the two rows above are the same analysis.")
    lines.append("  LOSO trains on n-1 subjects, closest to the model you ship,")
    lines.append("  and is what this field conventionally reports.")

    lines.append("")
    lines.append("Per-subject accuracy under LOSO (each held out once):")
    lr = out["loso"]
    b = lr["best"]
    for f in lr["folds"]:
        lines.append(f"  {f['held_out']:10s} {f['acc'][b]:5.3f}   "
                     f"({f['n_test']} windows)")
    accs = [f["acc"][b] for f in lr["folds"]]
    if len(accs) > 1:
        import numpy as _np
        a = _np.asarray(accs, dtype=float)
        n = len(a)
        mean, sd = float(a.mean()), float(a.std(ddof=1))
        half = 1.96 * sd / (n ** 0.5)
        lines.append(f"  mean {mean:.3f} +- {sd:.3f} sd   "
                     f"95% CI [{max(0.0, mean-half):.3f}, "
                     f"{min(1.0, mean+half):.3f}]  (n={n})")
        worst = min(lr["folds"], key=lambda f: f["acc"][b])
        lines.append(f"  worst subject: {worst['held_out']} "
                     f"({worst['acc'][b]:.3f})")
        lines.append("  CAUTION: LOSO training sets overlap heavily, so these")
        lines.append("  folds are not independent and the CI is optimistic.")
        lines.append("  Phase 2 on new people is the unbiased check.")

    out["report"] = "\n".join(lines)
    out["gap"] = float(ta - sa) if matched else float(ta - la)
    out["gap_matched"] = float(ta - sa) if matched else None
    out["matched"] = matched
    return out


def _all_subjects(ds):
    """Every subject id present in a dataset (postures and baselines alike)."""
    import csv
    out = set()
    try:
        with open(ds.csv_path, newline="") as fh:
            for r in csv.DictReader(fh):
                out.add(r["subject"])
    except Exception:
        pass
    return out


def train_and_compare(db, out=None, window=None, hop=None, cv=5, normalize=False,
                      group_by="trial", baseline_source="auto"):
    """Train & compare the three models. Returns a dict with a text 'report',
    the 'best' model name, and 'saved' path (or None). Raises ValueError on
    insufficient data. Used by both the CLI and the GUI.

    normalize=True subtracts each subject's neutral baseline from every feature.
    group_by='subject' evaluates generalization to NEW people (all of a subject's
    windows stay in one fold); 'trial' only prevents window leakage.
    """
    import numpy as np
    from sklearn.model_selection import (StratifiedKFold, StratifiedGroupKFold,
                                         cross_val_predict)
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, confusion_matrix)
    import joblib

    ds = Dataset(db)
    windowed = window is not None
    win_s = window
    hop_s = hop if hop is not None else (win_s / 2.0 if win_s else None)

    if windowed:
        if normalize:
            Xl, yl, groups = ds.load_windowed_normalized(win_s, hop_s, group_by,
                                                         baseline_source)
        else:
            Xl, yl, groups = ds.load_windowed(win_s, hop_s, group_by)
        n_groups = len(set(groups))
    else:
        Xl, yl, meta = ds.load()
        if normalize:
            baselines, global_neutral = ds.neutral_baselines(
                source=baseline_source)
            Xl = [[Xl[r][i] - baselines.get(meta[r]["subject"], global_neutral)[i]
                   for i in range(len(FEATURE_NAMES))] for r in range(len(Xl))]
        groups = None
        n_groups = len(Xl)

    X = np.asarray(Xl, dtype=float)
    y = np.asarray(yl)
    labels = sorted(set(y.tolist()))
    if len(X) < 6 or len(labels) < 2:
        raise ValueError(f"Need >=2 classes and several trials each. "
                         f"Have {len(X)} rows, labels={labels}.")
    if POSITIVE not in labels:
        raise ValueError(f"Positive class '{POSITIVE}' not in labels {labels}.")

    if windowed:
        g = np.asarray(groups)
        per_class_groups = {c: len(set(g[y == c].tolist())) for c in labels}
        min_class_groups = min(per_class_groups.values())
        if min_class_groups < 2:
            raise ValueError(f"Need >=2 trials per class for grouped CV. "
                             f"Have {per_class_groups}.")
        # cv=None -> one fold per group. Grouped by subject that is
        # leave-one-subject-out (LOSO): every person is held out exactly once.
        folds = min_class_groups if cv is None else max(2, min(cv, min_class_groups))
        folds = max(2, folds)
        splitter = StratifiedGroupKFold(n_splits=folds)
        cv_kw = {"groups": g}
        if group_by == "subject":
            gtag = ("subject, leave-one-subject-out" if cv is None
                    else "subject (generalizes to new people)")
        else:
            gtag = "trial (no window leakage)"
        header = (f"{len(X)} windows from {n_groups} {group_by}s "
                  f"(win={win_s}s hop={hop_s}s), "
                  f"{', '.join(f'{c}={(y==c).sum()}' for c in labels)} windows; "
                  f"{folds}-fold StratifiedGroupKFold grouped by {gtag}")
    else:
        min_class = min((y == c).sum() for c in labels)
        folds = int(min_class) if cv is None else max(2, min(cv, int(min_class)))
        folds = max(2, folds)
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
        cv_kw = {}
        header = (f"{len(X)} trials, "
                  f"{', '.join(f'{c}={(y==c).sum()}' for c in labels)}; "
                  f"{folds}-fold StratifiedKFold")

    lines = [header, "",
             f"{'model':8s}  {'acc':>5s}  {'prec':>5s}  {'rec':>5s}  {'F1':>5s}"]
    results = {}
    group_acc = {}
    preds = {}
    for name, model in build_models().items():
        pred = cross_val_predict(model, X, y, cv=splitter, **cv_kw)
        preds[name] = pred
        acc = accuracy_score(y, pred)
        prec = precision_score(y, pred, pos_label=POSITIVE, zero_division=0)
        rec = recall_score(y, pred, pos_label=POSITIVE, zero_division=0)
        f1 = f1_score(y, pred, pos_label=POSITIVE, zero_division=0)
        cm = confusion_matrix(y, pred, labels=labels)
        results[name] = (acc, prec, rec, f1, cm)
        lines.append(f"{name:8s}  {acc:5.3f}  {prec:5.3f}  {rec:5.3f}  {f1:5.3f}")
        if windowed and group_by == "trial":
            from collections import defaultdict, Counter
            gp = defaultdict(list)
            gt = {}
            for i in range(len(y)):
                gp[groups[i]].append(pred[i])
                gt[groups[i]] = y[i]
            correct = sum(1 for g in gp
                          if Counter(gp[g]).most_common(1)[0][0] == gt[g])
            group_acc[name] = correct / len(gp)

    if group_acc:
        lines.append("")
        lines.append("Per-trial accuracy (majority vote of a trial's windows):")
        lines.append("  " + "  ".join(f"{k} {v:.3f}" for k, v in group_acc.items()))

    # ---- per-fold detail -------------------------------------------------
    # cross_val_predict already produced out-of-fold predictions with this exact
    # (deterministic) splitter, so the folds can be re-derived and scored without
    # refitting anything. The spread across folds matters as much as the mean:
    # 0.75 from {0.75, 0.75} is a very different result from 0.75 from {0.5, 1.0}.
    model_names = list(results.keys())
    fold_idx = list(splitter.split(X, y, **cv_kw))
    fold_accs = {n: [] for n in model_names}
    fold_rows = []
    for i, (_, te) in enumerate(fold_idx, 1):
        accs = {}
        for n in model_names:
            a = accuracy_score(y[te], preds[n][te])
            accs[n] = a
            fold_accs[n].append(a)
        if windowed:
            held = sorted(set(np.asarray(groups)[te].tolist()))
            desc = ", ".join(held) if group_by == "subject" else f"{len(held)} trials"
        else:
            desc = f"{len(te)} trials"
        fold_rows.append({"fold": i, "n_test": int(len(te)),
                          "held_out": desc, "acc": accs})

    lines.append("")
    lines.append(f"Per-fold accuracy ({len(fold_idx)} folds, held-out data only):")
    lines.append(" fold" + "".join(f"{n:>8s}" for n in model_names)
                 + "     n  held-out")
    for r in fold_rows:
        lines.append(f" {r['fold']:>4d}"
                     + "".join(f"{r['acc'][n]:>8.3f}" for n in model_names)
                     + f"  {r['n_test']:>4d}  {r['held_out']}")
    lines.append(" mean" + "".join(f"{float(np.mean(fold_accs[n])):>8.3f}"
                                   for n in model_names))
    lines.append(" sd  " + "".join(f"{float(np.std(fold_accs[n])):>8.3f}"
                                   for n in model_names))

    best = max(results, key=lambda k: (results[k][3], results[k][0]))

    spread = float(np.std(fold_accs[best]))
    if spread > 0.15:
        lines.append("")
        lines.append(f"NOTE: folds disagree a lot (sd={spread:.2f}) - the mean is "
                     "an unstable")
        lines.append("      estimate. Usually means too few "
                     f"{group_by}s; collect more.")

    # refit best on all data (needed for importance ranking and saving)
    model = build_models()[best]
    model.fit(X, y)

    # An always-available linear explainer (even if the best model is a tree/forest),
    # so the live "why?" panel can show feature contributions for any best model.
    from sklearn.pipeline import Pipeline as _Pipe
    from sklearn.preprocessing import StandardScaler as _Scaler
    from sklearn.linear_model import LogisticRegression as _LR
    explainer = _Pipe([("scaler", _Scaler()), ("clf", _LR(max_iter=1000))]).fit(X, y)

    # Feature ranking. Plain accuracy-drop permutation is all-zero when accuracy
    # saturates at 100% (redundant features), so use a probability-sensitive score
    # (neg_log_loss); if that is still degenerate, fall back to the model's own
    # importances (|coef| for linear, feature_importances_ for trees).
    from sklearn.inspection import permutation_importance

    def _native_importance(m, nf):
        est = m
        if hasattr(m, "named_steps"):
            est = m.named_steps.get("clf", list(m.named_steps.values())[-1])
        if hasattr(est, "feature_importances_"):
            return np.abs(np.asarray(est.feature_importances_, dtype=float))
        if hasattr(est, "coef_"):
            c = np.abs(np.asarray(est.coef_, dtype=float))
            return c.mean(axis=0) if c.ndim > 1 else c
        return np.zeros(nf)

    imp_vals = None
    try:
        pim = permutation_importance(model, X, y, n_repeats=15, random_state=0,
                                     scoring="neg_log_loss")
        imp_vals = pim.importances_mean
    except Exception:
        imp_vals = None
    if imp_vals is None or float(np.max(np.abs(imp_vals))) < 1e-9:
        imp_vals = _native_importance(model, len(FEATURE_NAMES))
    importance = sorted(
        [(FEATURE_NAMES[i], float(imp_vals[i])) for i in range(len(FEATURE_NAMES))],
        key=lambda kv: kv[1], reverse=True)

    lines.append("")
    lines.append(f"(precision/recall/F1 shown for '{POSITIVE}')")
    lines.append(f"confusion (rows=true {labels}):")
    for name, (_, _, _, _, cm) in results.items():
        lines.append(f"  {name}: {cm.tolist()}")
    lines.append("")
    lines.append("Top features (permutation importance, best model):")
    for rank, (nm, imp) in enumerate(importance[:5], 1):
        lines.append(f"  {rank}. {nm}  ({imp:+.3f})")

    saved = None
    if out:
        joblib.dump({"model": model, "feature_names": FEATURE_NAMES,
                     "labels": labels, "positive": POSITIVE,
                     "window_s": win_s if windowed else None,
                     "hop_s": hop_s if windowed else None,
                     "normalized": bool(normalize),
                     "explainer": explainer,
                     # Phase 1 provenance: what this model scored before it was
                     # frozen, so Phase 2 can be compared against its own estimate
                     "cv_accuracy": float(results[best][0]),
                     "cv_f1": float(results[best][3]),
                     "cv_group_by": group_by if windowed else None,
                     "cv_folds": len(fold_idx),
                     "cv_model": best,
                     "train_subjects": sorted(_all_subjects(ds)),
                     "baseline_source": baseline_source if normalize else None},
                    out)
        saved = out
    tag = f"window {win_s}s" if windowed else "per-trial"
    if normalize:
        ded = ds.baseline_subjects()
        if baseline_source == "neutral" or not ded:
            tag += ", per-subject baseline-normalized (baseline = mean of " \
                   "neutral trials)"
        else:
            tag += (f", per-subject baseline-normalized (baseline = dedicated "
                    f"recording, {len(ded)} subject(s))")
            # a dedicated baseline shorter than the window cannot fill one window,
            # so duration-dependent features would be on the wrong scale. Use the
            # same tolerance as _window_vectors: a "5 s" recording spans ~4.99 s.
            if windowed:
                short = {s: d for s, d in ds.baseline_durations().items()
                         if d < 0.9 * win_s}
                if short:
                    lines.append("")
                    lines.append(f"WARNING: baseline shorter than the {win_s}s "
                                 f"window for: "
                                 + ", ".join(f"{s} ({d:.0f}s)"
                                             for s, d in short.items()))
                    lines.append("  -> set Window <= the baseline length, or "
                                 "record a longer baseline.")
    lines.append("")
    lines.append(f"BEST: {best} (F1={results[best][3]:.3f}, {tag})"
                 + (f" -> saved {out}" if saved else ""))
    return {"report": "\n".join(lines), "best": best, "saved": saved,
            "results": results, "labels": labels, "n": len(X),
            "importance": importance,
            "folds": fold_rows,
            "fold_acc": {n: list(v) for n, v in fold_accs.items()},
            "fold_mean": {n: float(np.mean(v)) for n, v in fold_accs.items()},
            "fold_sd": {n: float(np.std(v)) for n, v in fold_accs.items()}}


def main():
    ap = argparse.ArgumentParser(description="Train posture classifiers")
    ap.add_argument("--db", default="wbb_db")
    ap.add_argument("--out", default="posture_model.joblib")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--window", type=float, default=None,
                    help=f"window seconds for augmentation (e.g. {DEFAULT_WINDOW_S})")
    ap.add_argument("--hop", type=float, default=None,
                    help=f"window hop seconds (default = window/2 or {DEFAULT_HOP_S})")
    ap.add_argument("--normalize", action="store_true",
                    help="subtract each subject's neutral baseline (recommended)")
    ap.add_argument("--group", choices=["trial", "subject"], default="trial",
                    help="CV grouping: 'subject' estimates new-person generalization")
    ap.add_argument("--baseline-source", choices=["auto", "dedicated", "neutral"],
                    default="auto",
                    help="where the per-subject neutral origin comes from")
    args = ap.parse_args()
    try:
        res = train_and_compare(args.db, out=args.out, window=args.window,
                                hop=args.hop, cv=args.cv, normalize=args.normalize,
                                group_by=args.group,
                                baseline_source=args.baseline_source)
    except ValueError as e:
        print(str(e), "\nCollect more with collect_posture.py.")
        sys.exit(1)
    print(res["report"])


if __name__ == "__main__":
    main()
