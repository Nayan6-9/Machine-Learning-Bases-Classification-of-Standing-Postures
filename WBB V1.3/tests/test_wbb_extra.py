"""
test_wbb_extra.py — frequency features, clear(), subject-grouped CV.
    MPLBACKEND=Agg python3 tests/test_wbb_extra.py
"""
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wbb_core import SyntheticSource, BoardCalibration, SwayWindow  # noqa: E402
from wbb_bridge import resample_uniform  # noqa: E402
from wbb_dataset import Dataset, FEATURE_NAMES  # noqa: E402
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


def gen(dur, y_bias, amp, freq, fs=100.0):
    src = SyntheticSource(cal=BoardCalibration.identity_counts(), fs=fs,
                          duration_s=dur, body_kg=70.0,
                          cop_x_fn=lambda t: 0.3 * math.cos(2 * math.pi * 0.3 * t),
                          cop_y_fn=lambda t: y_bias + amp * math.sin(2 * math.pi * freq * t))
    return resample_uniform(list(src.stream()), fs)


def feats(samples):
    w = SwayWindow()
    for s in samples:
        w.add(s)
    return w.features()


# --- frequency features ---------------------------------------------------
print("frequency features")
check("18 features total", len(FEATURE_NAMES) == 18, len(FEATURE_NAMES))
check("magnitude features present",
      all(m in FEATURE_NAMES for m in ("cop_shift_cm", "rms_mag_cm", "p95_disp_cm")))
# a 0.8 Hz AP sway should show f50_ap near 0.8 Hz
f = feats(gen(20.0, 0.0, 2.0, 0.8))
check("SwayFeatures has mpf_ap", hasattr(f, "mpf_ap"))
check("f50_ap detects ~0.8 Hz", abs(f.f50_ap - 0.8) < 0.15, f"{f.f50_ap:.3f}")
# a faster 1.6 Hz sway -> higher median frequency
f2 = feats(gen(20.0, 0.0, 2.0, 1.6))
check("higher sway freq -> higher f50_ap", f2.f50_ap > f.f50_ap + 0.3,
      f"{f.f50_ap:.2f} vs {f2.f50_ap:.2f}")

# --- dataset clear (archives raw, keeps it) -------------------------------
print("dataset clear")
with tempfile.TemporaryDirectory() as d:
    ds = Dataset(d)
    for i in range(2):
        s = gen(12.0, 1.0, 1.0, 0.5)
        ds.append("neutral", feats(s), subject=f"S{i}", cop_samples=s, fs=100.0)
    check("has 2 trials before clear", ds.counts().get("neutral") == 2)
    raw_before = os.listdir(os.path.join(d, "samples"))
    check("raw files exist before", len(raw_before) == 2)
    adir, n = ds.clear()
    check("clear reports 2 archived", n == 2)
    check("dataset empty after clear", ds.counts() == {})
    check("raw preserved in archive (not deleted)",
          os.path.isdir(os.path.join(adir, "samples")) and
          len(os.listdir(os.path.join(adir, "samples"))) == 2)
    check("live samples dir is now empty",
          os.listdir(os.path.join(d, "samples")) == [])
    # can collect again cleanly
    s = gen(12.0, 1.0, 1.0, 0.5)
    ds.append("neutral", feats(s), subject="S9", cop_samples=s, fs=100.0)
    check("can append after clear", ds.counts().get("neutral") == 1)

# --- subject-grouped CV ---------------------------------------------------
print("subject-grouped CV")
with tempfile.TemporaryDirectory() as d:
    ds = Dataset(d)
    for subj in ("A", "B", "C"):
        for _ in range(2):
            sn = gen(30.0, 1.0, 1.0, 0.5)
            ds.append("neutral", feats(sn), subject=subj, cop_samples=sn, fs=100.0)
            sl = gen(30.0, -4.0, 1.0, 0.5)
            ds.append("slouched", feats(sl), subject=subj, cop_samples=sl, fs=100.0)
    X, y, groups = ds.load_windowed(10.0, 5.0, group_by="subject")
    check("groups are subjects", set(groups) == {"A", "B", "C"}, set(groups))
    res = train_and_compare(d, window=10.0, hop=5.0, cv=3, group_by="subject")
    check("report says grouped by subject", "grouped by subject" in res["report"])
    check("subject CV runs and picks best", res["best"] in ("logreg", "tree", "rf"))

print("window == trial length must not drop trials")
with tempfile.TemporaryDirectory() as d:
    ds = Dataset(d)
    for i in range(2):
        s = gen(5.0, 1.0, 1.0, 0.5)          # a "5 s" trial actually spans ~4.99 s
        ds.append("neutral", feats(s), subject=f"S{i}", cop_samples=s, fs=100.0)
        s2 = gen(5.0, -4.0, 1.0, 0.5)
        ds.append("slouched", feats(s2), subject=f"S{i}", cop_samples=s2, fs=100.0)
    X, y, _ = ds.load_windowed(5.0, 2.5)
    check("5 s window on 5 s trials still yields data", len(X) == 4, len(X))
    check("both classes survive", set(y) == {"neutral", "slouched"}, set(y))
    X2, _, _ = ds.load_windowed(10.0, 5.0)   # window clearly longer than trials
    check("window much longer than trial yields nothing", len(X2) == 0, len(X2))

print("schema migration")
with tempfile.TemporaryDirectory() as d:
    ds = Dataset(d)
    for i in range(2):
        s = gen(12.0, 1.0, 1.0, 0.5)
        ds.append("neutral", feats(s), subject=f"S{i}", cop_samples=s, fs=100.0)
    # simulate an OLD-schema dataset.csv (drop the frequency feature columns)
    import csv as _csv
    with open(ds.csv_path, newline="") as fh:
        rows = list(_csv.reader(fh))
    freq = ["mpf_ml", "f50_ml", "mpf_ap", "f50_ap"]
    keep = [i for i, c in enumerate(rows[0]) if c not in freq]
    with open(ds.csv_path, "w", newline="") as fh:
        w = _csv.writer(fh)
        for r in rows:
            w.writerow([r[i] for i in keep])
    # re-open: should auto-migrate from raw and restore full schema
    ds2 = Dataset(d)
    with open(ds2.csv_path, newline="") as fh:
        hdr = next(_csv.reader(fh))
    check("migrated header matches current schema", hdr == list(__import__(
        "wbb_dataset").DATASET_COLUMNS))
    check("counts work after migration", ds2.counts().get("neutral") == 2,
          ds2.counts())
    X, y, _ = ds2.load()
    check("migrated rows load with all features",
          len(X) == 2 and len(X[0]) == len(FEATURE_NAMES))


print("importing a model trained outside this app")
import numpy as np  # noqa: E402
import joblib as _jl  # noqa: E402
from sklearn.ensemble import RandomForestClassifier as _RF  # noqa: E402
from sklearn.pipeline import Pipeline as _Pipe  # noqa: E402
from sklearn.preprocessing import StandardScaler as _Sc  # noqa: E402
from sklearn.linear_model import LogisticRegression as _LR  # noqa: E402
from wbb_monitor import PostureClassifier as _PC  # noqa: E402

with tempfile.TemporaryDirectory() as _d:
    _rng = np.random.default_rng(0)
    _X = _rng.normal(size=(60, len(FEATURE_NAMES)))
    _y = np.array(["neutral"] * 30 + ["slouched"] * 30)

    def _save(name, obj):
        p = os.path.join(_d, name)
        _jl.dump(obj, p)
        return p

    # a bare estimator, which is what a separate training script usually saves
    _p = _save("bare.joblib", _RF(n_estimators=20, random_state=0).fit(_X, _y))
    _c = _PC.load(_p)
    check("bare estimator loads", _c.model is not None)
    check("features default to the app's set",
          _c.feature_names == list(FEATURE_NAMES))
    check("classes read from the fitted model",
          _c.labels == ["neutral", "slouched"], _c.labels)
    check("positive class inferred", _c.positive == "slouched", _c.positive)
    check("describe() is human readable", "18 features" in _c.describe())

    # a bare pipeline
    _p = _save("pipe.joblib", _Pipe([("scaler", _Sc()), ("clf", _LR(max_iter=500))]).fit(_X, _y))
    check("bare pipeline loads", _PC.load(_p).model is not None)

    # other people name the key differently
    _p = _save("alt.joblib", {"estimator": _RF(n_estimators=20, random_state=0).fit(_X, _y)})
    check("alternate dict key accepted", _PC.load(_p).model is not None)

    # a model trained on a different feature count must be refused, not used
    _p = _save("old.joblib", _RF(n_estimators=20, random_state=0)
               .fit(_rng.normal(size=(60, 15)), _y))
    try:
        _PC.load(_p)
        check("feature-count mismatch refused", False)
    except ValueError as e:
        check("feature-count mismatch refused", "15" in str(e) and "18" in str(e))

    # a file with no model in it must fail clearly
    _p = _save("junk.joblib", {"notes": "hi"})
    try:
        _PC.load(_p)
        check("file with no model refused", False)
    except ValueError:
        check("file with no model refused", True)

    # numeric labels still work; positive falls back to the last class
    _p = _save("num.joblib", _RF(n_estimators=20, random_state=0)
               .fit(_X, np.array([0] * 30 + [1] * 30)))
    _c = _PC.load(_p)
    check("numeric class labels supported", _c.labels == ["0", "1"], _c.labels)

print()
print(f"TOTAL: {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
