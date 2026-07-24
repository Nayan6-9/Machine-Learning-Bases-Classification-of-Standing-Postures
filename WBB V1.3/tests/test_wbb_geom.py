"""
test_wbb_geom.py — geometry helper tests (GUI math, no Tkinter needed).
    MPLBACKEND=Agg python3 tests/test_wbb_geom.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wbb_core import (  # noqa: E402
    eig_2x2_sym, confidence_ellipse_points, board_cm_to_canvas,
    HALF_X, HALF_Y,
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


# --- eigen ----------------------------------------------------------------
print("eig_2x2_sym")
l1, l2, v1, v2 = eig_2x2_sym(4.0, 1.0, 0.0)  # diagonal, x-dominant
check("l1 = 4", approx(l1, 4.0))
check("l2 = 1", approx(l2, 1.0))
check("v1 along x", approx(abs(v1[0]), 1.0) and approx(v1[1], 0.0))
check("v2 orthonormal", approx(v1[0]*v2[0] + v1[1]*v2[1], 0.0))
# rotated case: equal variance, positive covariance -> principal axis at 45°
l1, l2, v1, v2 = eig_2x2_sym(2.0, 2.0, 1.0)
check("rotated l1 = 3", approx(l1, 3.0))
check("rotated l2 = 1", approx(l2, 1.0))
check("rotated v1 at 45°", approx(abs(v1[0]), abs(v1[1]), tol=1e-9))

# --- confidence ellipse points -------------------------------------------
print("confidence_ellipse_points")
check("too few points -> []", confidence_ellipse_points([0, 1], [0, 1]) == [])
# x-elongated cloud
xs = [-3, -1, 0, 1, 3, 0, 0, 0]
ys = [0, 0, 0, 0, 0, 1, -1, 0]
pts = confidence_ellipse_points(xs, ys, n=40)
check("returns n points", len(pts) == 40)
mx = sum(p[0] for p in pts) / len(pts)
my = sum(p[1] for p in pts) / len(pts)
check("ellipse centered at cloud mean x", approx(mx, sum(xs)/len(xs), tol=1e-6))
check("ellipse centered at cloud mean y", approx(my, sum(ys)/len(ys), tol=1e-6))
x_ext = max(p[0] for p in pts) - min(p[0] for p in pts)
y_ext = max(p[1] for p in pts) - min(p[1] for p in pts)
check("x-elongated cloud -> wider ellipse in x", x_ext > y_ext, f"{x_ext} vs {y_ext}")

# --- cm -> canvas ---------------------------------------------------------
print("board_cm_to_canvas")
cx, cy, bw, bh = 260.0, 160.0, 400.0, 200.0
px, py = board_cm_to_canvas(0, 0, bw, bh, cx, cy)
check("center -> canvas center", approx(px, cx) and approx(py, cy))
px, py = board_cm_to_canvas(HALF_X, 0, bw, bh, cx, cy)
check("full right -> +x edge", approx(px, cx + bw/2) and approx(py, cy))
px, py = board_cm_to_canvas(0, HALF_Y, bw, bh, cx, cy)
check("anterior -> UP (smaller py)", approx(px, cx) and approx(py, cy - bh/2))
px, py = board_cm_to_canvas(0, -HALF_Y, bw, bh, cx, cy)
check("posterior -> DOWN (larger py)", approx(py, cy + bh/2))

print()
print(f"TOTAL: {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
