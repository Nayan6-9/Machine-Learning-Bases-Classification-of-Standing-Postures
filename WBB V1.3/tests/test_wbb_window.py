"""
test_wbb_window.py — windowed augmentation + grouped CV (item 1).
    MPLBACKEND=Agg python3 tests/test_wbb_window.py
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
from wbb_record import iter_windows  # noqa: E402
from wbb_dataset import (Dataset, FEATURE_NAMES, DEFAULT_WINDOW_S,  # noqa: E402
                         DEFAULT_HOP_S)

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


def gen_samples(dur, y_bias, amp, fs=100.0):
    src = SyntheticSource(cal=BoardCalibration.identity_counts(), fs=fs,
                          duration_s=dur, body_kg=70.0,
                          cop_x_fn=lambda t: 0.4 * math.cos(2 * math.pi * 0.3 * t),
                          cop_y_fn=lambda t: y_bias + amp * math.sin(2 * math.pi * 0.35 * t))
    return resample_uniform(list(src.stream()), fs)


def expected_windows(samples, win, hop):
    """Number of FULL-length windows iter_windows should yield."""
    if len(samples) < 2:
        return 0
    span = samples[-1].t - samples[0].t
    if span < win:
        return 0
    return int((span - win) / hop + 1e-9) + 1


def feats(samples):
    w = SwayWindow()
    for s in samples:
        w.add(s)
    return w.features()


# --- iter_windows counting ------------------------------------------------
print("iter_windows")
s30 = gen_samples(30.0, 1.0, 1.2)
wins = list(iter_windows(s30, 10.0, 5.0))
ew = expected_windows(s30, 10.0, 5.0)
check(f"30s @ win10 hop5 -> {ew} full windows", len(wins) == ew, f"got {len(wins)}")
check("each window ~10s long",
      all(9.0 <= (w[-1].t - w[0].t) <= 10.0 for w in wins))
short = gen_samples(6.0, 1.0, 1.2)
check("trial shorter than window -> 0 windows",
      len(list(iter_windows(short, 10.0, 5.0))) == 0)
check("defaults exported", DEFAULT_WINDOW_S == 10.0 and DEFAULT_HOP_S == 5.0)

# --- load_windowed --------------------------------------------------------
print("load_windowed")
with tempfile.TemporaryDirectory() as d:
    ds = Dataset(d)
    for i in range(2):
        sn = gen_samples(30.0, 1.0, 1.2)
        ds.append("neutral", feats(sn), subject=f"S{i}", cop_samples=sn, fs=100.0)
        sl = gen_samples(30.0, -4.0, 2.5)
        ds.append("slouched", feats(sl), subject=f"S{i}", cop_samples=sl, fs=100.0)
    X, y, groups = ds.load_windowed(10.0, 5.0)
    per = expected_windows(gen_samples(30.0, 1.0, 1.2), 10.0, 5.0)
    total = 4 * per
    check(f"4 trials x {per} windows = {total} rows", len(X) == total, f"got {len(X)}")
    check("labels aligned", len(y) == total and set(y) == {"neutral", "slouched"})
    check("groups = 4 distinct trials", len(set(groups)) == 4, f"{set(groups)}")
    check("feature width correct", len(X[0]) == len(FEATURE_NAMES))
    # each group has the same window count (no trial dropped/duplicated)
    from collections import Counter
    check(f"{per} windows per trial", set(Counter(groups).values()) == {per},
          f"{Counter(groups)}")

# --- grouped training end-to-end -----------------------------------------
print("grouped training")
with tempfile.TemporaryDirectory() as d:
    ds = Dataset(d)
    for i in range(3):
        sn = gen_samples(30.0, 1.0 + 0.2 * i, 1.2)
        ds.append("neutral", feats(sn), subject=f"S{i}", cop_samples=sn, fs=100.0)
        sl = gen_samples(30.0, -4.0 + 0.2 * i, 2.5)
        ds.append("slouched", feats(sl), subject=f"S{i}", cop_samples=sl, fs=100.0)
    out = os.path.join(d, "wmodel.joblib")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "wbb_train.py"),
                        "--db", d, "--window", "10", "--hop", "5",
                        "--out", out, "--cv", "3"],
                       capture_output=True, text=True)
    check("windowed train ran", r.returncode == 0, r.stderr[-300:])
    check("uses grouped CV (no leakage)",
          "StratifiedGroupKFold" in r.stdout and "leakage" in r.stdout,
          r.stdout[-200:])
    check("reports windows from trials", "windows from" in r.stdout)
    check("model saved", os.path.exists(out))
    # window_s persisted, and classifier loads it
    from wbb_monitor import PostureClassifier
    clf = PostureClassifier.load(out)
    check("model carries window_s=10", clf.window_s == 10.0, f"{clf.window_s}")
    label, _ = clf.predict(feats(gen_samples(12.0, -4.0, 2.5)))
    check("predicts slouched on slouched window", label == "slouched", label)
    ex = clf.explain(feats(gen_samples(12.0, -4.0, 2.5)))
    if ex is not None:
        check("explain returns 1-3 contributions", 1 <= len(ex) <= 3, len(ex))
        check("contribs are (name, float)",
              all(isinstance(n, str) and isinstance(v, float) for n, v in ex))
        check("contribs sorted by magnitude",
              all(abs(ex[i][1]) >= abs(ex[i + 1][1]) for i in range(len(ex) - 1)))



def _api_test():
    import tempfile as _tf
    from wbb_train import train_and_compare
    with _tf.TemporaryDirectory() as d:
        ds = Dataset(d)
        for i in range(3):
            sn = gen_samples(30.0, 1.0 + 0.2 * i, 1.2)
            ds.append("neutral", feats(sn), subject=f"S{i}", cop_samples=sn, fs=100.0)
            sl = gen_samples(30.0, -4.0 + 0.2 * i, 2.5)
            ds.append("slouched", feats(sl), subject=f"S{i}", cop_samples=sl, fs=100.0)
        out = os.path.join(d, "m.joblib")
        res = train_and_compare(d, out=out, window=10.0, hop=5.0, cv=3)
        check("train_and_compare returns report", "BEST" in res["report"])
        check("best is a known model", res["best"] in ("logreg", "tree", "rf"))
        check("model saved by API", res["saved"] == out and os.path.exists(out))
        check("n windows > trials", res["n"] > 6, res["n"])
        check("returns feature importance ranking",
              "importance" in res and len(res["importance"]) == len(FEATURE_NAMES))
        check("importance sorted descending",
              all(res["importance"][i][1] >= res["importance"][i + 1][1]
                  for i in range(len(res["importance"]) - 1)))
        check("report shows Top features", "Top features" in res["report"])


print("train_and_compare API")
_api_test()
print()
print(f"TOTAL: {PASS} passed, {FAIL} failed")

sys.exit(1 if FAIL else 0)
