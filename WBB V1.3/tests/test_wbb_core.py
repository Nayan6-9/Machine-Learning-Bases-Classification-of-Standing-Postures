"""
test_wbb_core.py — pure-logic tests, WELAB style. Run with:
    MPLBACKEND=Agg python3 tests/test_wbb_core.py
Exits non-zero on first failure.
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wbb_core import (  # noqa: E402
    SensorCalibration, BoardCalibration, Sample, make_sample,
    SwayWindow, SyntheticSource, collect, HALF_X, HALF_Y,
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


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


# --- calibration ----------------------------------------------------------
print("calibration")
c = SensorCalibration(raw_0kg=1000, raw_17kg=2000, raw_34kg=3000)
check("0 kg anchor", approx(c.raw_to_kg(1000), 0.0))
check("17 kg anchor", approx(c.raw_to_kg(2000), 17.0))
check("34 kg anchor", approx(c.raw_to_kg(3000), 34.0))
check("midpoint lower seg = 8.5 kg", approx(c.raw_to_kg(1500), 8.5))
check("midpoint upper seg = 25.5 kg", approx(c.raw_to_kg(2500), 25.5))
check("below 0 clamps to 0", approx(c.raw_to_kg(500), 0.0))

# --- CoP geometry ---------------------------------------------------------
print("CoP geometry")
# perfectly centered, equal load -> CoP at origin
s = Sample(t=0, tr=10, tl=10, br=10, bl=10)
check("centered total", approx(s.total, 40.0))
check("centered cop_x", approx(s.cop_x, 0.0))
check("centered cop_y", approx(s.cop_y, 0.0))
check("centered 50/50 LR", approx(s.left_right_pct[0], 50.0))

# all load on right side -> cop_x at +HALF_X
s = Sample(t=0, tr=10, tl=0, br=10, bl=0)
check("full-right cop_x = +HALF_X", approx(s.cop_x, HALF_X))
check("full-right 0/100 LR", approx(s.left_right_pct[1], 100.0))

# all load on top -> cop_y at +HALF_Y
s = Sample(t=0, tr=10, tl=10, br=0, bl=0)
check("full-top cop_y = +HALF_Y", approx(s.cop_y, HALF_Y))

# single corner (top-right) -> both axes max
s = Sample(t=0, tr=40, tl=0, br=0, bl=0)
check("TR-only cop_x", approx(s.cop_x, HALF_X))
check("TR-only cop_y", approx(s.cop_y, HALF_Y))

# unloaded board -> safe defaults, no div-by-zero
s = Sample(t=0, tr=0, tl=0, br=0, bl=0)
check("empty cop_x = 0", approx(s.cop_x, 0.0))
check("empty LR = 50/50", approx(s.left_right_pct[0], 50.0))

# --- synthetic round-trip: drive a known CoP, recover it ------------------
print("synthetic round-trip")
cal = BoardCalibration.identity_counts(per_kg=100.0)
src = SyntheticSource(cal=cal, fs=100.0, duration_s=1.0, body_kg=80.0,
                      cop_x_fn=lambda t: 5.0, cop_y_fn=lambda t: -3.0)
win = collect(src)
f = win.features()
check("recovers mean total ~80 kg", approx(f.mean_total_kg, 80.0, tol=1e-3),
      f"got {f.mean_total_kg}")
check("recovers cop_x ~5.0", approx(f.mean_cop_x, 5.0, tol=1e-3), f"got {f.mean_cop_x}")
check("recovers cop_y ~-3.0", approx(f.mean_cop_y, -3.0, tol=1e-3), f"got {f.mean_cop_y}")
check("static trajectory -> ~0 path", f.path_length_cm < 1e-3, f"got {f.path_length_cm}")
check("static -> ~0 ellipse", f.ellipse_area_cm2 < 1e-3, f"got {f.ellipse_area_cm2}")

# --- sway features on a circular CoP path ---------------------------------
print("sway features (circular sway)")
R = 2.0  # cm radius
src = SyntheticSource(cal=cal, fs=200.0, duration_s=4.0, body_kg=75.0,
                      cop_x_fn=lambda t: R * math.cos(2 * math.pi * 0.5 * t),
                      cop_y_fn=lambda t: R * math.sin(2 * math.pi * 0.5 * t))
f = collect(src).features()
# 0.5 Hz for 4 s = 2 revolutions; path ~ 2 * (2*pi*R)
expected_path = 2 * (2 * math.pi * R)
check("circular path length ~ 2 laps", approx(f.path_length_cm, expected_path, tol=0.2),
      f"got {f.path_length_cm:.3f} vs {expected_path:.3f}")
# 95% ellipse for a filled circle radius R: cov is isotropic var=R^2/2 -> area = pi*chi2*R^2/2
expected_area = math.pi * 5.991 * (R * R / 2.0)
check("circular ellipse area", approx(f.ellipse_area_cm2, expected_area, tol=0.5),
      f"got {f.ellipse_area_cm2:.3f} vs {expected_area:.3f}")
check("balanced LR on symmetric sway", approx(f.lr_asymmetry, 0.0, tol=0.2),
      f"got {f.lr_asymmetry}")

# --- weight-shift asymmetry detection -------------------------------------
print("weight-shift asymmetry")
src = SyntheticSource(cal=cal, fs=100.0, duration_s=2.0, body_kg=70.0,
                      cop_x_fn=lambda t: HALF_X * 0.6)  # held to the right
f = collect(src).features()
check("right shift -> right% > left%", f.right_pct > f.left_pct, f"L{f.left_pct} R{f.right_pct}")
check("asymmetry ~60%", approx(f.lr_asymmetry, 60.0, tol=1.0), f"got {f.lr_asymmetry}")

print()
print(f"TOTAL: {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
