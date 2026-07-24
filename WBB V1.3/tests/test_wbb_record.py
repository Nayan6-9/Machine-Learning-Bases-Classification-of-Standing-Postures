"""
test_wbb_record.py — recording/export tests.
    MPLBACKEND=Agg python3 tests/test_wbb_record.py
"""
import csv
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wbb_core import BoardCalibration, SyntheticSource, HALF_X  # noqa: E402
from wbb_bridge import BridgeSource, Tare, parse_sample_line  # noqa: E402
from wbb_record import (  # noqa: E402
    record_trial, write_cop_csv, write_features_summary,
    WELAB_COLUMNS, BRAINBLOX_COLUMNS, cop_rows,
)

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


def approx(a, b, tol=1e-4):
    return abs(a - b) <= tol


# --- timed trial truncation ----------------------------------------------
print("timed trial")
# 10 s of synthetic data at 50 Hz; ask for a 2 s timed trial
src = SyntheticSource(cal=BoardCalibration.identity_counts(), fs=50.0,
                      duration_s=10.0, body_kg=70.0, cop_x_fn=lambda t: 3.0)
trial = record_trial(src, duration_s=2.0, fs=100.0)
check("timed trial ~2 s", approx(trial.duration_s, 2.0, tol=0.05), f"{trial.duration_s}")
check("resampled to 100 Hz (~201 samples)", abs(len(trial.samples) - 201) <= 2,
      f"{len(trial.samples)}")
f = trial.features()
check("features computed on trial", f is not None and approx(f.mean_cop_x, 3.0, tol=1e-2),
      f"{f.mean_cop_x if f else None}")

# untimed trial consumes whole source
src2 = SyntheticSource(fs=50.0, duration_s=3.0, body_kg=70.0)
trial2 = record_trial(src2, duration_s=None, fs=None)
check("untimed keeps full native stream (~150)", abs(len(trial2.samples) - 150) <= 1,
      f"{len(trial2.samples)}")

# --- CSV export: WELAB layout --------------------------------------------
print("CSV export (WELAB)")
lines = [f"{i/50.0},20,0,20,0" for i in range(100)]  # load to right corners
trial3 = record_trial(BridgeSource(lines=lines), duration_s=None, fs=50.0)
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "trial.csv")
    n = write_cop_csv(p, trial3, brainblox=False)
    with open(p) as f:
        rdr = list(csv.reader(f))
    check("WELAB header matches", rdr[0] == WELAB_COLUMNS, f"{rdr[0]}")
    check("row count matches", len(rdr) - 1 == n)
    # first data row: cop_x ~ +HALF_X, mass ~ 40
    row = rdr[1]
    check("cop_x_cm ~ +HALF_X", approx(float(row[1]), HALF_X, tol=1e-2), f"{row[1]}")
    check("mass_kg ~ 40", approx(float(row[3]), 40.0, tol=1e-2), f"{row[3]}")
    check("per-corner columns present (8 cols)", len(row) == 8)

# --- CSV export: BrainBLoX-compatible layout -----------------------------
print("CSV export (BrainBLoX-compatible)")
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "trial_bblox.csv")
    write_cop_csv(p, trial3, brainblox=True)
    with open(p) as f:
        rdr = list(csv.reader(f))
    check("BrainBLoX header = time,x,y,mass", rdr[0] == BRAINBLOX_COLUMNS, f"{rdr[0]}")
    check("BrainBLoX rows have 4 cols", len(rdr[1]) == 4)

# --- features summary -----------------------------------------------------
print("features summary")
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "summary.csv")
    write_features_summary(p, trial3.features())
    with open(p) as f:
        rdr = list(csv.reader(f))
    keys = {r[0] for r in rdr[1:]}
    check("summary has path_length_cm", "path_length_cm" in keys)
    check("summary has ellipse_area_cm2", "ellipse_area_cm2" in keys)
    check("summary has lr_asymmetry", "lr_asymmetry" in keys)

# --- tare path through recorder ------------------------------------------
print("recorder + tare")
tare = Tare.from_samples([parse_sample_line("0,1,1,1,1"),
                          parse_sample_line("0.02,1,1,1,1")])
lines2 = [f"{i/50.0},11,11,11,11" for i in range(60)]
trial4 = record_trial(BridgeSource(lines=lines2), fs=50.0, tare=tare)
f4 = trial4.features()
check("tared mass ~ 40 kg", approx(f4.mean_total_kg, 40.0, tol=1e-2), f"{f4.mean_total_kg}")

print()
print(f"TOTAL: {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
