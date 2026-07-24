#!/usr/bin/env python3
"""
make_figures.py - generate poster-ready PNG figures from your dataset.

Runs the training/evaluation and saves four figures you can drop into a poster:
  figures/accuracy_by_model.png   - accuracy of the 3 models
  figures/confusion_matrix.png    - best model's confusion matrix
  figures/feature_importance.png  - top features that drive the decision
  figures/generalization.png      - trial vs subject accuracy (the key story)

Usage:
    python make_figures.py --db wbb_db --window 10
    python make_figures.py --db wbb_db --window 5 --no-normalize --out figures

Needs scikit-learn, joblib, and matplotlib (pip install scikit-learn joblib matplotlib).
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import wbb_train

BLUE, ORANGE, GREEN = "#2d6cdf", "#e0864e", "#4ea172"


def _bar_labels(ax, bars):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=10)


def fig_accuracy(res, path, group):
    names = list(res["results"].keys())
    accs = [res["results"][n][0] for n in names]
    recs = [res["results"][n][2] for n in names]
    x = range(len(names))
    fig, ax = plt.subplots(figsize=(6, 4))
    w = 0.38
    b1 = ax.bar([i - w / 2 for i in x], accs, w, label="Accuracy", color=BLUE)
    b2 = ax.bar([i + w / 2 for i in x], recs, w, label="Recall (slouched)",
                color=ORANGE)
    _bar_labels(ax, b1); _bar_labels(ax, b2)
    ax.set_xticks(list(x)); ax.set_xticklabels([n.upper() for n in names])
    ax.set_ylim(0, 1.1); ax.set_ylabel("Score")
    ax.set_title(f"Model comparison (grouped by {group})")
    ax.legend(loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def fig_confusion(res, path, group):
    best = res["best"]
    cm = res["results"][best][4]
    labels = res["labels"]
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix ({best}, by {group})")
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(cm[i][j]), ha="center", va="center",
                    color="white" if cm[i][j] > thresh else "#14223a",
                    fontsize=14, fontweight="bold")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def fig_importance(res, path, group, top=8):
    imp = res["importance"][:top][::-1]
    names = [n for n, _ in imp]
    vals = [v for _, v in imp]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(names, vals, color=GREEN)
    ax.set_xlabel("Importance (higher = more influential)")
    ax.set_title(f"Top {top} features (by {group})")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def fig_decision_boundary(res, db, window, hop, normalize, group_by, path):
    """2D scatter of all windows on the two most important features, with a
    decision boundary shaded. The most intuitive 'how it separates' plot."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from wbb_dataset import Dataset, FEATURE_NAMES

    ds = Dataset(db)
    if normalize:
        X, y, _ = ds.load_windowed_normalized(window, hop, group_by)
    else:
        X, y, _ = ds.load_windowed(window, hop, group_by)
    X = np.asarray(X, dtype=float); y = np.asarray(y)
    order = [FEATURE_NAMES.index(n) for n, _ in res["importance"]]
    ia, ib = order[0], order[1]
    if ia == ib:
        ib = order[2] if len(order) > 2 else (ia + 1) % len(FEATURE_NAMES)
    X2 = X[:, [ia, ib]]
    clf = LogisticRegression(max_iter=1000).fit(X2, y)
    classes = list(clf.classes_)
    cmap = {classes[0]: GREEN, classes[1]: ORANGE}

    px = 0.1 * (np.ptp(X2[:, 0]) or 1); py = 0.1 * (np.ptp(X2[:, 1]) or 1)
    xx, yy = np.meshgrid(
        np.linspace(X2[:, 0].min() - px, X2[:, 0].max() + px, 250),
        np.linspace(X2[:, 1].min() - py, X2[:, 1].max() + py, 250))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Zi = np.array([classes.index(z) for z in Z]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.contourf(xx, yy, Zi, levels=[-0.5, 0.5, 1.5],
                colors=[GREEN, ORANGE], alpha=0.15)
    for lab in classes:
        m = y == lab
        ax.scatter(X2[m, 0], X2[m, 1], s=20, c=cmap[lab], edgecolor="white",
                   linewidth=0.4, label=lab)
    unit = " (rel. to neutral)" if normalize else ""
    ax.set_xlabel(FEATURE_NAMES[ia] + unit)
    ax.set_ylabel(FEATURE_NAMES[ib] + unit)
    ax.set_title(f"How the classes separate ({group_by})")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def fig_cop_distribution(db, group_by, path, window=10.0, hop=5.0):
    """Scatter of per-window mean CoP positions (cm) colored by posture, over a
    simple board outline. Uses RAW CoP so axes are real centimeters."""
    import numpy as np
    from wbb_dataset import Dataset, FEATURE_NAMES
    from wbb_core import BOARD_X_CM, BOARD_Y_CM
    ix = FEATURE_NAMES.index("mean_cop_x")
    iy = FEATURE_NAMES.index("mean_cop_y")
    ds = Dataset(db)
    X, y, _ = ds.load_windowed(window, hop, group_by)   # raw features (real cm)
    X = np.asarray(X, dtype=float); y = np.asarray(y)
    cmap = {"neutral": GREEN, "slouched": ORANGE}

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    hx, hy = BOARD_X_CM / 2, BOARD_Y_CM / 2
    ax.add_patch(plt.Rectangle((-hx, -hy), BOARD_X_CM, BOARD_Y_CM, fill=False,
                               edgecolor="#3a4250", linewidth=2))
    ax.axhline(0, color="#c8cfd8", lw=0.8); ax.axvline(0, color="#c8cfd8", lw=0.8)
    if X.ndim == 2 and X.shape[0] > 0:
        for lab in sorted(set(y)):
            m = y == lab
            ax.scatter(X[m, ix], X[m, iy], s=22, c=cmap.get(lab, "#888"),
                       edgecolor="white", linewidth=0.4, label=lab, alpha=0.8)
        ax.legend(frameon=False, loc="upper right")
    else:
        ax.text(0, 0, "no windows\n(window > trial length?)", ha="center",
                color="#888")
    ax.set_xlim(-hx - 3, hx + 3); ax.set_ylim(-hy - 3, hy + 3)
    ax.set_xlabel("CoP left-right (cm)   -left | right+")
    ax.set_ylabel("CoP back-front (cm)   -back | front+")
    ax.set_title(f"Where weight centers per posture ({group_by})")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def fig_generalization(acc_trial, acc_subject, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    xs = ["Group by\ntrial", "Group by\nsubject\n(new people)"]
    vals = [acc_trial, acc_subject if acc_subject is not None else 0]
    colors = [BLUE, ORANGE]
    bars = ax.bar(xs, vals, color=colors, width=0.6)
    _bar_labels(ax, bars)
    if acc_subject is None:
        ax.text(1, 0.05, "need >=2 subjects", ha="center", color="#888")
    ax.set_ylim(0, 1.1); ax.set_ylabel("Best-model accuracy")
    ax.set_title("Does it generalize to new people?")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def generate_figures(db, window=10.0, hop=None, normalize=True, out="figures",
                     group_by="trial"):
    """Train and write poster PNGs for the chosen grouping. The accuracy,
    confusion, and importance figures reflect `group_by` and are labeled and
    suffixed with it (so trial and subject runs don't overwrite each other).
    generalization.png always compares trial vs subject. Returns a summary dict."""
    os.makedirs(out, exist_ok=True)
    if hop is None:
        hop = window / 2.0

    res = wbb_train.train_and_compare(db, window=window, hop=hop,
                                      normalize=normalize, group_by=group_by)
    g = group_by
    made = []
    for name, fn in [
        (f"accuracy_by_model_{g}.png", lambda p: fig_accuracy(res, p, g)),
        (f"confusion_matrix_{g}.png", lambda p: fig_confusion(res, p, g)),
        (f"feature_importance_{g}.png", lambda p: fig_importance(res, p, g)),
        (f"decision_boundary_{g}.png",
         lambda p: fig_decision_boundary(res, db, window, hop, normalize, g, p)),
        (f"cop_distribution_{g}.png",
         lambda p: fig_cop_distribution(db, g, p, window, hop)),
    ]:
        try:
            fn(os.path.join(out, name)); made.append(name)
        except Exception as e:
            print(f"({name} skipped:", e, ")")
    sel_acc = res["results"][res["best"]][0]

    # both groupings for the generalization comparison
    acc = {"trial": None, "subject": None}
    acc[g] = sel_acc
    other = "subject" if g == "trial" else "trial"
    try:
        res_o = wbb_train.train_and_compare(db, window=window, hop=hop,
                                            normalize=normalize, group_by=other)
        acc[other] = res_o["results"][res_o["best"]][0]
    except Exception as e:
        print(f"({other}-grouped run skipped:", e, ")")
    fig_generalization(acc["trial"], acc["subject"],
                       os.path.join(out, "generalization.png"))

    return {"out": os.path.abspath(out), "group": g, "best": res["best"],
            "acc": sel_acc, "acc_trial": acc["trial"], "acc_subject": acc["subject"]}


def _per_subject_bars(names, vals, title, xlabel, path):
    """Shared: one bar per person, with the mean and its 95% CI."""
    import numpy as np
    n = len(vals)
    mean = float(np.mean(vals)) if n else 0.0
    sd = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    half = 1.96 * sd / (n ** 0.5) if n > 1 else 0.0
    lo, hi = max(0.0, mean - half), min(1.0, mean + half)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    bars = ax.bar(names, vals, color=BLUE, width=0.6)
    _bar_labels(ax, bars)
    if n > 1:
        ax.axhspan(lo, hi, color=ORANGE, alpha=0.18,
                   label=f"95% CI [{lo:.2f}, {hi:.2f}]")
    ax.axhline(mean, color=ORANGE, lw=2, label=f"mean {mean:.2f}")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    if n > 8:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def fig_design_comparison(cmp_res, path):
    """The three ways of scoring, side by side. The matched pair (same number of
    folds) isolates the effect of testing on a stranger; LOSO is the headline."""
    t, sk, l = cmp_res["trial"], cmp_res["subject_k"], cmp_res["loso"]
    tb, sb, lb = t["best"], sk["best"], l["best"]
    n_subj = len(l["folds"])
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    xs = [f"Trials-only\n{len(t['folds'])}-fold\n(same people)",
          f"Subject\n{len(sk['folds'])}-fold\n(new people)",
          f"LOSO\n{n_subj} folds\n(new people)"]
    vals = [t["results"][tb][0], sk["results"][sb][0], l["results"][lb][0]]
    errs = [t["fold_sd"][tb], sk["fold_sd"][sb], l["fold_sd"][lb]]
    bars = ax.bar(xs, vals, color=[BLUE, ORANGE, GREEN], width=0.6,
                  yerr=errs, capsize=6, ecolor="#3a4250")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.2); ax.set_ylabel("Accuracy")
    ax.set_title("Does the model still work on someone new?")
    ax.spines[["top", "right"]].set_visible(False)
    if cmp_res.get("matched") and cmp_res.get("gap_matched") is not None:
        # bracket over the two bars that share a training-set size
        yb = max(vals[0], vals[1]) + 0.105
        ax.plot([0, 0, 1, 1], [yb - 0.02, yb, yb, yb - 0.02], lw=1.2,
                color="#3a4250")
        ax.text(0.5, yb + 0.01, f"gap {cmp_res['gap_matched']:+.2f}",
                ha="center", fontsize=9.5, color="#3a4250")
    fig.text(0.5, 0.015,
             "matched pair = same training size; error bars = SD across folds",
             ha="center", fontsize=8.5, color="#6b7684")
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    fig.savefig(path, dpi=300); plt.close(fig)


def fig_loso_per_subject(cmp_res, path):
    l = cmp_res["loso"]
    b = l["best"]
    names = [f["held_out"] for f in l["folds"]]
    vals = [f["acc"][b] for f in l["folds"]]
    _per_subject_bars(names, vals,
                      "Leave-one-subject-out: accuracy per held-out person",
                      f"Held-out subject (n={len(names)})", path)


def fig_val_per_subject(res, path):
    """Accuracy for each held-out person, with the mean and its 95% CI.
    Generalization is about people, so this is the Phase 2 headline figure."""
    per = res["per_subject"]
    _per_subject_bars(list(per.keys()), list(per.values()),
                      "Phase 2: accuracy on people the model never saw",
                      f"Held-out subject (n={len(per)})", path)


def fig_val_confusion(res, path):
    import numpy as np
    cm = np.asarray(res["confusion"])
    labels = res["labels"]
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Phase 2 confusion ({res['n_subjects']} new subjects)")
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(cm[i][j]), ha="center", va="center",
                    color="white" if cm[i][j] > thresh else "#14223a",
                    fontsize=14, fontweight="bold")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def fig_phase_compare(res, path):
    """Did the Phase 1 cross-validated estimate survive contact with new people?"""
    cv = res.get("cv_accuracy")
    if cv is None:
        raise ValueError("this model has no stored Phase 1 score (re-train to add it)")
    gb = res.get("cv_group_by") or "?"
    fig, ax = plt.subplots(figsize=(5, 4))
    xs = [f"Phase 1\nCV ({gb}-grouped)", f"Phase 2\nheld-out (n={res['n_subjects']})"]
    vals = [cv, res["trial_acc"]]
    bars = ax.bar(xs, vals, color=[BLUE, ORANGE], width=0.6)
    _bar_labels(ax, bars)
    ax.set_ylim(0, 1.12); ax.set_ylabel("Accuracy")
    ax.set_title("Estimated vs actual performance on new people")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def generate_validation_figures(db, model_path, out="figures"):
    """Phase 2 figures. Scores the frozen model on `db` and plots the result."""
    import wbb_validate
    os.makedirs(out, exist_ok=True)
    res = wbb_validate.evaluate_saved_model(db, model_path)
    made = []
    for name, fn in [
        ("phase2_per_subject.png", lambda p: fig_val_per_subject(res, p)),
        ("phase2_confusion.png", lambda p: fig_val_confusion(res, p)),
        ("phase2_vs_phase1.png", lambda p: fig_phase_compare(res, p)),
    ]:
        try:
            fn(os.path.join(out, name)); made.append(name)
        except Exception as e:
            print(f"({name} skipped:", e, ")")
    return {"out": os.path.abspath(out), "made": made, "result": res}


def main():
    ap = argparse.ArgumentParser(description="Generate poster figures")
    ap.add_argument("--db", default="wbb_db")
    ap.add_argument("--window", type=float, default=10.0)
    ap.add_argument("--hop", type=float, default=None)
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--group", choices=["trial", "subject"], default="trial")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--validate", metavar="HELDOUT_DB", default=None,
                    help="make Phase 2 figures from a held-out dataset instead")
    ap.add_argument("--model", default="posture_model.joblib",
                    help="frozen model to score for --validate")
    args = ap.parse_args()
    if args.validate:
        r = generate_validation_figures(args.validate, args.model, out=args.out)
        print("Saved Phase 2 figures to", r["out"])
        for m in r["made"]:
            print("  " + m)
        return
    r = generate_figures(args.db, window=args.window, hop=args.hop,
                         normalize=not args.no_normalize, out=args.out,
                         group_by=args.group)
    print("Saved figures to", r["out"])
    print(f"  grouping: {r['group']}  best model: {r['best']}  accuracy {r['acc']:.3f}")
    if r["acc_trial"] is not None and r["acc_subject"] is not None:
        print(f"  trial acc {r['acc_trial']:.3f} vs subject acc {r['acc_subject']:.3f}")


if __name__ == "__main__":
    main()
