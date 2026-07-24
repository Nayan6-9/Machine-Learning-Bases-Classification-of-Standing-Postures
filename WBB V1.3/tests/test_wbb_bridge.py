"""
test_wbb_bridge.py — pure-logic tests for the ingest adapter.
    MPLBACKEND=Agg python3 tests/test_wbb_bridge.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wbb_core import Sample, HALF_X  # noqa: E402
from wbb_bridge import (  # noqa: E402
    parse_sample_line, Tare, resample_uniform, BridgeSource, record,
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


# --- parsing --------------------------------------------------------------
print("parsing")
s = parse_sample_line("1.5,10,20,30,40")
check("parses 5 fields", s is not None)
check("t", approx(s.t, 1.5))
check("tr/tl/br/bl", s.tr == 10 and s.tl == 20 and s.br == 30 and s.bl == 40)
check("blank -> None", parse_sample_line("   ") is None)
check("garbage -> None", parse_sample_line("hello,world") is None)
check("short -> None", parse_sample_line("1,2,3") is None)
check("trailing junk tolerated", parse_sample_line("0,1,2,3,4,extra") is not None)

# --- tare -----------------------------------------------------------------
print("tare")
baseline = [Sample(t=0, tr=2, tl=2, br=2, bl=2),
            Sample(t=0.1, tr=2, tl=2, br=2, bl=2)]
tare = Tare.from_samples(baseline)
check("tare captures 2 kg offset", approx(tare.tr, 2.0))
loaded = Sample(t=1.0, tr=12, tl=11, br=10, bl=9)
z = tare.apply(loaded)
check("tare subtracts offset", z.tr == 10 and z.tl == 9 and z.br == 8 and z.bl == 7)
neg = tare.apply(Sample(t=2.0, tr=1, tl=1, br=1, bl=1))
check("tare clamps negatives to 0", neg.tr == 0.0)

# --- resampling -----------------------------------------------------------
print("resampling")
# ramp tr from 0 at t=0 to 10 at t=1; resample at 10 Hz -> midpoint t=0.5 -> 5
ramp = [Sample(t=0.0, tr=0, tl=0, br=0, bl=0),
        Sample(t=1.0, tr=10, tl=0, br=0, bl=0)]
rs = resample_uniform(ramp, fs=10.0)
check("resample count ~ 11 (0..1 @10Hz)", len(rs) == 11, f"got {len(rs)}")
mid = [p for p in rs if approx(p.t, 0.5, tol=1e-6)]
check("linear interp midpoint = 5", mid and approx(mid[0].tr, 5.0), f"{mid}")
check("resample endpoints preserved", approx(rs[0].tr, 0.0) and approx(rs[-1].tr, 10.0))

# jittery timestamps still resample monotonically
jitter = [Sample(t=0.00, tr=0, tl=0, br=0, bl=0),
          Sample(t=0.013, tr=1, tl=0, br=0, bl=0),
          Sample(t=0.031, tr=2, tl=0, br=0, bl=0),
          Sample(t=0.052, tr=3, tl=0, br=0, bl=0)]
rsj = resample_uniform(jitter, fs=100.0)
check("jitter resample is time-ordered",
      all(rsj[i].t <= rsj[i + 1].t for i in range(len(rsj) - 1)))

# --- end-to-end ingest ----------------------------------------------------
print("end-to-end ingest")
# simulate a bridge holding load to the right edge; check features pipeline
lines = []
fs = 50.0
for i in range(100):
    t = i / fs
    # all load on right corners -> CoP at +HALF_X
    lines.append(f"{t},20,0,20,0")
win = record(BridgeSource(lines=lines), fs=fs)
f = win.features()
check("ingest produced samples", len(win) > 0)
check("ingest total ~40 kg", approx(f.mean_total_kg, 40.0, tol=1e-3), f"{f.mean_total_kg}")
check("ingest cop_x ~ +HALF_X", approx(f.mean_cop_x, HALF_X, tol=1e-3), f"{f.mean_cop_x}")
check("ingest right% = 100", approx(f.right_pct, 100.0, tol=1e-3), f"{f.right_pct}")

# end-to-end with tare from an unloaded prefix
print("end-to-end with tare")
tare = Tare.from_samples([parse_sample_line("0,1,1,1,1"),
                          parse_sample_line("0.02,1,1,1,1")])
lines2 = [f"{i/fs},11,11,11,11" for i in range(50)]  # 11 kg/corner, 1 is baseline
win2 = record(BridgeSource(lines=lines2, tare=tare), fs=fs)
f2 = win2.features()
check("tared total ~ 40 kg (4 x 10)", approx(f2.mean_total_kg, 40.0, tol=1e-3),
      f"{f2.mean_total_kg}")

print()
print(f"TOTAL: {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
