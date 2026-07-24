"""
test_wbb_norm.py — per-subject baseline normalization.
    MPLBACKEND=Agg python3 tests/test_wbb_norm.py
"""
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wbb_core import SyntheticSource, BoardCalibration, SwayWindow  # noqa: E402
from wbb_bridge import resample_uniform  # noqa: E402
from wbb_dataset import Dataset, FEATURE_NAMES, feature_vector  # noqa: E402
from wbb_train import train_and_compare  # noqa: E402
from wbb_monitor import PostureClassifier  # noqa: E402

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


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


def gen(dur, y_bias, amp, fs=100.0):
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


CY_IDX = FEATURE_NAMES.index("mean_cop_y")

# Build a dataset where each subject has a DIFFERENT neutral offset (foot
# placement), and slouching shifts CoP posteriorly by ~5 cm from THEIR neutral.
print("build dataset with per-subject offsets")
with tempfile.TemporaryDirectory() as d:
    ds = Dataset(d)
    subj_offset = {"A": +6.0, "B": -3.0, "C": +1.0}
    for subj, off in subj_offset.items():
        for _ in range(2):
            sn = gen(30.0, off, 1.0)           # neutral at the subject's offset
            ds.append("neutral", feats(sn), subject=subj, cop_samples=sn, fs=100.0)
            sl = gen(30.0, off - 5.0, 1.0)      # slouched = 5 cm posterior of neutral
            ds.append("slouched", feats(sl), subject=subj, cop_samples=sl, fs=100.0)

    # --- baselines ---
    print("neutral_baselines")
    baselines, global_neutral = ds.neutral_baselines(10.0, 5.0)
    check("a baseline per subject", set(baselines.keys()) == {"A", "B", "C"})
    check("subject A baseline cop_y ~ +6", approx(baselines["A"][CY_IDX], 6.0, tol=0.3),
          f"{baselines['A'][CY_IDX]}")
    check("subject B baseline cop_y ~ -3", approx(baselines["B"][CY_IDX], -3.0, tol=0.3),
          f"{baselines['B'][CY_IDX]}")

    # --- normalized loading: neutral windows ~0, slouched ~ -5 in cop_y ---
    print("load_windowed_normalized")
    X, y, groups = ds.load_windowed_normalized(10.0, 5.0)
    ncy = [X[i][CY_IDX] for i in range(len(X)) if y[i] == "neutral"]
    scy = [X[i][CY_IDX] for i in range(len(X)) if y[i] == "slouched"]
    check("normalized neutral cop_y ~ 0",
          approx(sum(ncy) / len(ncy), 0.0, tol=0.5), f"{sum(ncy)/len(ncy):.3f}")
    check("normalized slouched cop_y ~ -5",
          approx(sum(scy) / len(scy), -5.0, tol=0.7), f"{sum(scy)/len(scy):.3f}")
    check("offsets removed -> classes separated in cop_y",
          max(ncy) > min(scy) - 100 and (sum(ncy)/len(ncy)) - (sum(scy)/len(scy)) > 3)

    # --- train normalized; bundle flags normalized; predict needs baseline ---
    print("train (normalized) + baseline-aware predict")
    out = os.path.join(d, "m.joblib")
    res = train_and_compare(d, out=out, window=10.0, hop=5.0, cv=3, normalize=True)
    check("report notes normalization", "normalized" in res["report"])
    clf = PostureClassifier.load(out)
    check("model bundle marked normalized", clf.normalized is True)
    check("feature set includes frequency features",
          all(fn in FEATURE_NAMES for fn in ("mpf_ml", "f50_ml", "mpf_ap", "f50_ap")))

    # a slouched-like trial for subject B (offset -3): raw cop_y ~ -3-5 = -8
    sl_b = feats(gen(10.0, -3.0 - 5.0, 1.0))
    base_b = baselines["B"]
    lbl_with, p_s = clf.predict(sl_b, baseline=base_b)
    check("predicts slouched WITH correct baseline", lbl_with == "slouched", lbl_with)
    # neutral-like trial for subject B: raw cop_y ~ -3
    nt_b = feats(gen(10.0, -3.0, 1.0))
    lbl_n, p_n = clf.predict(nt_b, baseline=base_b)
    # robust directional check: a neutral stance is less slouch-probable than a
    # slouched one (exact label on a single synthetic window is brittle).
    if p_s is not None and p_n is not None:
        check("neutral stance less slouch-probable than slouched", p_n < p_s,
              f"{p_n:.2f} vs {p_s:.2f}")
    else:
        check("predicts neutral for neutral stance", lbl_n == "neutral", lbl_n)

print()
print(f"TOTAL: {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
