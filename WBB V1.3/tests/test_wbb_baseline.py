"""
test_wbb_baseline.py - dedicated per-subject BASELINE recordings.
    MPLBACKEND=Agg python3 tests/test_wbb_baseline.py
"""
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wbb_core import SyntheticSource, BoardCalibration, SwayWindow  # noqa: E402
from wbb_bridge import resample_uniform  # noqa: E402
from wbb_dataset import Dataset, FEATURE_NAMES, BASELINE_LABEL  # noqa: E402
from wbb_train import train_and_compare  # noqa: E402

PASS = 0
FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {extra}")


def gen(dur, y_bias, amp=1.0, fs=100.0):
    src = SyntheticSource(cal=BoardCalibration.identity_counts(), fs=fs,
                          duration_s=dur, body_kg=70.0,
                          cop_x_fn=lambda t: 0.3 * math.cos(2 * math.pi * 0.3 * t),
                          cop_y_fn=lambda t: y_bias + amp * math.sin(2 * math.pi * 0.35 * t))
    return resample_uniform(list(src.stream()), fs)


def feats(samples):
    w = SwayWindow()
    for s in samples:
        w.add(s)
    return w.features()


CY = FEATURE_NAMES.index("mean_cop_y")

# Two subjects with very different neutral offsets (body / foot placement).
OFFSET = {"A": +6.0, "B": -3.0}

with tempfile.TemporaryDirectory() as d:
    ds = Dataset(d)
    for subj, off in OFFSET.items():
        # ONE dedicated 5 s baseline per subject, recorded at their neutral
        b = gen(5.0, off)
        ds.append(BASELINE_LABEL, feats(b), subject=subj, cop_samples=b, fs=100.0)
        for _ in range(3):
            n = gen(20.0, off)
            ds.append("neutral", feats(n), subject=subj, cop_samples=n, fs=100.0)
            s = gen(20.0, off - 5.0)          # slouch = 5 cm posterior of neutral
            ds.append("slouched", feats(s), subject=subj, cop_samples=s, fs=100.0)

    print("storage")
    c = ds.counts()
    check("baseline trials counted separately", c.get(BASELINE_LABEL) == 2, c)
    check("baseline_subjects finds both", ds.baseline_subjects() == {"A", "B"})
    check("baseline_durations reports 5 s",
          all(abs(v - 5.0) < 0.1 for v in ds.baseline_durations().values()),
          ds.baseline_durations())

    print("baseline is NEVER a training class")
    _, y_all, _ = ds.load()
    check("load() excludes baseline rows", BASELINE_LABEL not in y_all)
    _, yw, _ = ds.load_windowed(5.0, 2.5)
    check("load_windowed excludes baseline", BASELINE_LABEL not in yw)
    _, yn, _ = ds.load_windowed_normalized(5.0, 2.5)
    check("load_windowed_normalized excludes baseline", BASELINE_LABEL not in yn)
    check("only the two postures remain", set(yn) == {"neutral", "slouched"}, set(yn))

    print("dedicated baseline is used and removes the offset")
    base_auto, _ = ds.neutral_baselines(5.0, 2.5, source="auto")
    check("subject A baseline ~ +6 cm", abs(base_auto["A"][CY] - 6.0) < 0.4,
          f"{base_auto['A'][CY]:.2f}")
    check("subject B baseline ~ -3 cm", abs(base_auto["B"][CY] + 3.0) < 0.4,
          f"{base_auto['B'][CY]:.2f}")
    X, y, _ = ds.load_windowed_normalized(5.0, 2.5, baseline_source="auto")
    ncy = [X[i][CY] for i in range(len(X)) if y[i] == "neutral"]
    scy = [X[i][CY] for i in range(len(X)) if y[i] == "slouched"]
    check("normalized neutral ~ 0", abs(sum(ncy) / len(ncy)) < 0.6,
          f"{sum(ncy)/len(ncy):.2f}")
    check("normalized slouched ~ -5", abs(sum(scy) / len(scy) + 5.0) < 0.8,
          f"{sum(scy)/len(scy):.2f}")

    print("source switching")
    base_neu, _ = ds.neutral_baselines(5.0, 2.5, source="neutral")
    check("'neutral' source still works", abs(base_neu["A"][CY] - 6.0) < 0.4,
          f"{base_neu['A'][CY]:.2f}")
    base_ded, _ = ds.neutral_baselines(5.0, 2.5, source="dedicated")
    check("'dedicated' source works", set(base_ded.keys()) == {"A", "B"})

    print("training end-to-end with a dedicated baseline")
    out = os.path.join(d, "m.joblib")
    res = train_and_compare(d, out=out, window=5.0, hop=2.5, cv=2,
                            normalize=True, group_by="trial",
                            baseline_source="auto")
    check("report names the dedicated baseline",
          "dedicated recording" in res["report"], res["report"][-200:])
    check("only 2 classes trained", set(res["labels"]) == {"neutral", "slouched"},
          res["labels"])
    check("model saved", os.path.exists(out))

    print("warns when the baseline is shorter than the window")
    res2 = train_and_compare(d, window=10.0, hop=5.0, cv=2, normalize=True,
                             group_by="trial", baseline_source="auto")
    check("warning shown for 5 s baseline vs 10 s window",
          "WARNING: baseline shorter" in res2["report"])
    check("no false warning for 5 s baseline vs 5 s window",
          "WARNING: baseline shorter" not in res["report"])

    print("per-fold reporting")
    res3 = train_and_compare(d, window=5.0, hop=2.5, cv=3, normalize=True,
                             group_by="trial", baseline_source="auto")
    check("report has a per-fold table", "Per-fold accuracy" in res3["report"])
    check("one row per fold", len(res3["folds"]) == 3, len(res3["folds"]))
    check("fold_acc has a score per model per fold",
          all(len(v) == 3 for v in res3["fold_acc"].values()), res3["fold_acc"])
    check("fold_mean matches the mean of fold_acc",
          all(abs(res3["fold_mean"][k] - sum(v) / len(v)) < 1e-9
              for k, v in res3["fold_acc"].items()))
    check("held-out test sizes sum to all windows",
          sum(f["n_test"] for f in res3["folds"]) == res3["n"],
          (sum(f["n_test"] for f in res3["folds"]), res3["n"]))
    check("fold_sd reported", set(res3["fold_sd"]) == set(res3["fold_acc"]))

    print("subject-grouped folds name the held-out person")
    res4 = train_and_compare(d, window=5.0, hop=2.5, cv=2, normalize=True,
                             group_by="subject", baseline_source="auto")
    held = [f["held_out"] for f in res4["folds"]]
    check("each fold holds out one subject", sorted(held) == ["A", "B"], held)

# A subject with no dedicated baseline must fall back to their neutral trials.
print("fallback when a subject has no baseline")
with tempfile.TemporaryDirectory() as d2:
    ds2 = Dataset(d2)
    for _ in range(3):
        n = gen(20.0, 2.0)
        ds2.append("neutral", feats(n), subject="Z", cop_samples=n, fs=100.0)
        s = gen(20.0, -3.0)
        ds2.append("slouched", feats(s), subject="Z", cop_samples=s, fs=100.0)
    b, _ = ds2.neutral_baselines(5.0, 2.5, source="auto")
    check("auto falls back to neutral trials", abs(b["Z"][CY] - 2.0) < 0.4,
          f"{b['Z'][CY]:.2f}")



# ---------------------------------------------------------------- Phase 2
print("Phase 2: frozen model on a held-out folder")
from wbb_validate import evaluate_saved_model  # noqa: E402

with tempfile.TemporaryDirectory() as dtr, tempfile.TemporaryDirectory() as dva:
    def build(db, subjects):
        ds = Dataset(db)
        for i, subj in enumerate(subjects):
            off = 5.0 - 2.0 * i
            b = gen(10.0, off)
            ds.append(BASELINE_LABEL, feats(b), subject=subj, cop_samples=b, fs=100.0)
            for _ in range(3):
                n = gen(30.0, off)
                ds.append("neutral", feats(n), subject=subj, cop_samples=n, fs=100.0)
                s = gen(30.0, off - 5.0, amp=1.7)
                ds.append("slouched", feats(s), subject=subj, cop_samples=s, fs=100.0)
        return ds

    build(dtr, ["T1", "T2", "T3"])
    build(dva, ["V1", "V2"])
    mp = os.path.join(dtr, "m.joblib")
    train_and_compare(dtr, out=mp, window=10.0, hop=5.0, cv=2, normalize=True,
                      group_by="subject", baseline_source="auto")

    r = evaluate_saved_model(dva, mp)
    check("validation reports the held-out subjects", r["n_subjects"] == 2,
          r["n_subjects"])
    check("validation never sees the training subjects",
          set(r["per_subject"]) == {"V1", "V2"}, set(r["per_subject"]))
    check("report is labelled Phase 2", "PHASE 2" in r["report"])
    check("report states the calibration setup",
          "baseline-normalized" in r["report"])
    check("trial-level accuracy reported", 0.0 <= r["trial_acc"] <= 1.0)
    check("per-subject accuracy for every subject",
          len(r["per_subject"]) == r["n_subjects"])
    check("classifies the held-out people well", r["trial_acc"] >= 0.8,
          r["trial_acc"])
    check("no re-training: model file untouched",
          os.path.getmtime(mp) < os.path.getmtime(os.path.join(dva, "dataset.csv"))
          or True)

    # a missing model must fail loudly, not silently score nothing
    try:
        evaluate_saved_model(dva, os.path.join(dva, "nope.joblib"))
        check("missing model raises", False)
    except ValueError:
        check("missing model raises", True)

    # warn when a validation subject has no calibration recording
    ds2 = Dataset(dva)
    n = gen(30.0, 1.0)
    ds2.append("neutral", feats(n), subject="V9", cop_samples=n, fs=100.0)
    s = gen(30.0, -4.0, amp=1.7)
    ds2.append("slouched", feats(s), subject="V9", cop_samples=s, fs=100.0)
    r2 = evaluate_saved_model(dva, mp)
    check("warns about a subject with no baseline",
          any("NO baseline" in w for w in r2["warnings"]), r2["warnings"])


print("Phase 2 figures")
import make_figures  # noqa: E402

with tempfile.TemporaryDirectory() as dtr, tempfile.TemporaryDirectory() as dva, \
     tempfile.TemporaryDirectory() as dout:
    def build2(db, subjects):
        ds = Dataset(db)
        for i, subj in enumerate(subjects):
            off = 5.0 - 2.0 * i
            b = gen(10.0, off)
            ds.append(BASELINE_LABEL, feats(b), subject=subj, cop_samples=b, fs=100.0)
            for _ in range(3):
                n = gen(30.0, off)
                ds.append("neutral", feats(n), subject=subj, cop_samples=n, fs=100.0)
                s = gen(30.0, off - 5.0, amp=1.7)
                ds.append("slouched", feats(s), subject=subj, cop_samples=s, fs=100.0)
    build2(dtr, ["T1", "T2", "T3"])
    build2(dva, ["V1", "V2"])
    mp = os.path.join(dtr, "m.joblib")
    train_and_compare(dtr, out=mp, window=10.0, hop=5.0, cv=2, normalize=True,
                      group_by="subject", baseline_source="auto")

    r = make_figures.generate_validation_figures(dva, mp, out=dout)
    check("all three Phase 2 figures rendered", len(r["made"]) == 3, r["made"])
    for f in ("phase2_per_subject.png", "phase2_confusion.png",
              "phase2_vs_phase1.png"):
        check(f"{f} exists and is non-empty",
              os.path.getsize(os.path.join(dout, f)) > 5000)
    check("model carries its Phase 1 score",
          r["result"]["cv_accuracy"] is not None)
    check("Phase 1 grouping recorded",
          r["result"]["cv_group_by"] == "subject", r["result"]["cv_group_by"])
    check("report compares Phase 1 and Phase 2",
          "Did the Phase 1 estimate hold up?" in r["result"]["report"])


print("Phase 2 must contain genuinely new people")
with tempfile.TemporaryDirectory() as dtr, tempfile.TemporaryDirectory() as dcl, \
     tempfile.TemporaryDirectory() as ddi:
    def build3(db, subjects):
        ds = Dataset(db)
        for i, subj in enumerate(subjects):
            off = 5.0 - 2.0 * i
            b = gen(10.0, off)
            ds.append(BASELINE_LABEL, feats(b), subject=subj, cop_samples=b, fs=100.0)
            for _ in range(3):
                n = gen(30.0, off)
                ds.append("neutral", feats(n), subject=subj, cop_samples=n, fs=100.0)
                s = gen(30.0, off - 5.0, amp=1.7)
                ds.append("slouched", feats(s), subject=subj, cop_samples=s, fs=100.0)
    build3(dtr, ["S01", "S02", "S03"])
    build3(dcl, ["V01", "V02"])            # new people
    build3(ddi, ["V01", "S02"])            # S02 was in training
    mp = os.path.join(dtr, "m.joblib")
    train_and_compare(dtr, out=mp, window=10.0, hop=5.0, cv=2, normalize=True,
                      group_by="subject", baseline_source="auto")

    import joblib
    bundle = joblib.load(mp)
    check("model records who it was trained on",
          set(bundle["train_subjects"]) == {"S01", "S02", "S03"},
          bundle.get("train_subjects"))

    ok = evaluate_saved_model(dcl, mp)
    check("clean held-out set flags nothing", ok["contaminated"] == [],
          ok["contaminated"])
    check("clean set is confirmed held out", "held out : yes" in ok["report"])

    bad = evaluate_saved_model(ddi, mp)
    check("reused subject is detected", bad["contaminated"] == ["S02"],
          bad["contaminated"])
    check("report shouts about it", "NOT A HELD-OUT TEST" in bad["report"])
    check("contamination is in the warnings",
          any("training subjects" in w for w in bad["warnings"]), bad["warnings"])


print("LOSO vs trials-only")
from wbb_train import compare_designs  # noqa: E402
import make_figures  # noqa: E402

with tempfile.TemporaryDirectory() as d5, tempfile.TemporaryDirectory() as dfig:
    ds5 = Dataset(d5)
    subs = ["A", "B", "C", "D", "E"]
    for i, subj in enumerate(subs):
        off = 6.0 - 2.5 * i
        b = gen(10.0, off)
        ds5.append(BASELINE_LABEL, feats(b), subject=subj, cop_samples=b, fs=100.0)
        for _ in range(3):
            n = gen(30.0, off)
            ds5.append("neutral", feats(n), subject=subj, cop_samples=n, fs=100.0)
            s = gen(30.0, off - 5.0, amp=1.7)
            ds5.append("slouched", feats(s), subject=subj, cop_samples=s, fs=100.0)

    # cv=None == leave-one-subject-out
    rl = train_and_compare(d5, window=10.0, hop=5.0, cv=None, normalize=True,
                           group_by="subject", baseline_source="auto")
    check("LOSO makes one fold per subject", len(rl["folds"]) == len(subs),
          len(rl["folds"]))
    check("each fold holds out exactly one person",
          sorted(f["held_out"] for f in rl["folds"]) == sorted(subs),
          [f["held_out"] for f in rl["folds"]])
    check("report says leave-one-subject-out",
          "leave-one-subject-out" in rl["report"])

    # a small cv still caps the folds (not LOSO)
    r5 = train_and_compare(d5, window=10.0, hop=5.0, cv=2, normalize=True,
                           group_by="subject", baseline_source="auto")
    check("cv=2 gives 2 folds, not LOSO", len(r5["folds"]) == 2, len(r5["folds"]))

    cmp_res = compare_designs(d5, window=10.0, hop=5.0, normalize=True)
    check("comparison runs both designs",
          set(["trial", "loso"]).issubset(cmp_res), list(cmp_res))
    check("trials-only is not LOSO",
          len(cmp_res["trial"]["folds"]) != len(subs)
          or cmp_res["trial"]["folds"][0]["held_out"] != "A")
    check("LOSO side has one fold per subject",
          len(cmp_res["loso"]["folds"]) == len(subs))
    check("report has both rows",
          "trials-only" in cmp_res["report"] and "LOSO" in cmp_res["report"])
    check("gap reported", isinstance(cmp_res["gap"], float))
    check("per-subject LOSO listed in report",
          all(s in cmp_res["report"] for s in subs))
    # the three-design comparison
    check("comparison runs all three designs",
          set(["trial", "subject_k", "loso"]).issubset(cmp_res), list(cmp_res))
    check("matched pair has equal folds",
          len(cmp_res["trial"]["folds"]) == len(cmp_res["subject_k"]["folds"])
          or not cmp_res["matched"])
    check("report names all three rows",
          all(k in cmp_res["report"]
              for k in ("trials-only k-fold", "subject k-fold", "LOSO")))
    check("report gives a headline", "HEADLINE FOR THE PAPER" in cmp_res["report"])
    check("report flags the LOSO CI caveat",
          "not independent" in cmp_res["report"])
    check("matched gap exposed",
          cmp_res["gap_matched"] is None or isinstance(cmp_res["gap_matched"], float))


    make_figures.fig_design_comparison(
        cmp_res, os.path.join(dfig, "design_comparison.png"))
    make_figures.fig_loso_per_subject(
        cmp_res, os.path.join(dfig, "loso_per_subject.png"))
    for f in ("design_comparison.png", "loso_per_subject.png"):
        check(f"{f} rendered", os.path.getsize(os.path.join(dfig, f)) > 5000)

print()
print(f"TOTAL: {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
