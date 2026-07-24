"""
test_wbb_ml.py — dataset + training + alarm tests.
    MPLBACKEND=Agg python3 tests/test_wbb_ml.py
"""
import math
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.join(os.path.dirname(__file__), "..")

from wbb_core import SyntheticSource, BoardCalibration, SwayWindow  # noqa: E402
from wbb_bridge import resample_uniform  # noqa: E402
from wbb_dataset import Dataset, FEATURE_NAMES, feature_vector  # noqa: E402
from wbb_monitor import AlarmController  # noqa: E402

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


def synth_features(y_bias, amp):
    src = SyntheticSource(cal=BoardCalibration.identity_counts(), fs=100.0,
                          duration_s=6.0, body_kg=70.0,
                          cop_x_fn=lambda t: 0.5 * math.cos(2 * math.pi * 0.3 * t),
                          cop_y_fn=lambda t: y_bias + amp * math.sin(2 * math.pi * 0.35 * t))
    w = SwayWindow()
    for s in resample_uniform(list(src.stream()), 100.0):
        w.add(s)
    return w.features(), w.samples


# --- dataset --------------------------------------------------------------
print("dataset")
with tempfile.TemporaryDirectory() as d:
    ds = Dataset(d)
    check("feature_vector length matches names",
          len(feature_vector(synth_features(1.0, 1.2)[0])) == len(FEATURE_NAMES))
    f, samp = synth_features(1.0, 1.2)
    tid = ds.append("neutral", f, subject="S01", cop_samples=samp, fs=100.0)
    check("append returns trial_id", isinstance(tid, str) and "neutral" in tid)
    check("per-trial CoP CSV written",
          os.path.exists(os.path.join(d, "samples", tid + "_cop.csv")))
    f2, _ = synth_features(-4.0, 2.5)
    ds.append("slouched", f2, subject="S01")
    X, y, meta = ds.load()
    check("load returns 2 rows", len(X) == 2 and len(y) == 2)
    check("labels present", set(y) == {"neutral", "slouched"})
    check("counts correct", ds.counts() == {"neutral": 1, "slouched": 1})
    check("feature width correct", len(X[0]) == len(FEATURE_NAMES))

# --- training pipeline end-to-end (subprocess, like real usage) ----------
print("training pipeline")
with tempfile.TemporaryDirectory() as d:
    ds = Dataset(d)
    # build a separable dataset: neutral (y~+1) vs slouched (y~-4)
    for i in range(12):
        fn, sn = synth_features(1.0 + 0.3 * math.sin(i), 1.2)
        ds.append("neutral", fn, subject=f"S{i:02d}")
        fs_, ss = synth_features(-4.0 + 0.3 * math.cos(i), 2.5)
        ds.append("slouched", fs_, subject=f"S{i:02d}")
    out = os.path.join(d, "model.joblib")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "wbb_train.py"),
                        "--db", d, "--out", out, "--cv", "4"],
                       capture_output=True, text=True)
    check("train.py ran ok", r.returncode == 0, r.stderr[-300:])
    check("model file saved", os.path.exists(out))
    check("reports all three models",
          all(m in r.stdout for m in ("logreg", "tree", "rf")), r.stdout[-300:])
    check("picks a best model", "BEST:" in r.stdout)

    # load model and predict on a fresh slouched sample
    from wbb_monitor import PostureClassifier
    clf = PostureClassifier.load(out)
    fpred, _ = synth_features(-4.0, 2.5)
    label, proba = clf.predict(fpred)
    check("predicts slouched on slouched-like input", label == "slouched",
          f"got {label}")
    check("proba in [0,1] or None", proba is None or (0.0 <= proba <= 1.0))

# --- alarm state machine --------------------------------------------------
print("alarm controller")
a = AlarmController(positive="slouched", sustain_s=5.0, clear_s=3.0, cooldown_s=15.0)
events = []
# neutral 0-4s: nothing
for t in [0, 1, 2, 3, 4]:
    events.append((t, a.update(float(t), "neutral")))
check("no alarm while neutral", all(e is None for _, e in events))
# slouch from t=5; should alarm once >= sustain (at t=10)
fire = [a.update(float(t), "slouched") for t in range(5, 12)]
check("alarm fires after sustain", "ALARM" in fire, f"{fire}")
check("alarm fires once", fire.count("ALARM") == 1, f"{fire}")
# brief neutral (< clear_s) shouldn't clear
check("short good posture doesn't clear", a.update(12.0, "neutral") is None)
check("still alarming", a.alarming)
# sustained neutral clears
clr = [a.update(float(t), "neutral") for t in range(13, 17)]
check("clears after good posture", "CLEAR" in clr, f"{clr}")
check("not alarming after clear", not a.alarming)

print()
print(f"TOTAL: {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
