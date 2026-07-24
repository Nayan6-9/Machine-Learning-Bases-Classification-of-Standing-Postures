#!/usr/bin/env python3
"""
run_wbb.py — run the WELAB Wii Balance Board pipeline.

Two modes:
  DEMO  (no hardware): synthetic board, to verify the software path.
      python3 run_wbb.py --demo --duration 5 --no-tare
  LIVE  (real board): listens for the C# bridge (WiiBoardBridge.cs) on UDP.
      python3 run_wbb.py --duration 30 --tare-seconds 3 --out trial1

Output: prints derived posturography features and writes
  <out>_cop.csv      (CoP+mass time series, WELAB schema)
  <out>_bblox.csv    (BrainBLoX-compatible: time, x_cm, y_cm, mass_kg)
  <out>_features.csv (path length, velocity, ellipse, asymmetry, ...)
"""

import argparse
import math
import sys
import time
from typing import List, Optional

from wbb_core import Sample, SwayWindow, SyntheticSource, BoardCalibration
from wbb_bridge import BridgeSource, Tare, resample_uniform
from wbb_record import Trial, write_cop_csv, write_features_summary


def make_source(args):
    """Return a fresh source each call (demo regenerates; live rebinds UDP)."""
    if args.demo:
        # A gentle 0.4 Hz sway + slight rightward bias, ~70 kg.
        return SyntheticSource(
            cal=BoardCalibration.identity_counts(),
            fs=args.fs, duration_s=args.tare_seconds + args.duration + 1.0,
            body_kg=70.0,
            cop_x_fn=lambda t: 1.5 + 2.0 * math.cos(2 * math.pi * 0.4 * t),
            cop_y_fn=lambda t: 1.0 * math.sin(2 * math.pi * 0.4 * t),
        )
    return BridgeSource(host=args.host, port=args.port)


def collect_for(source, seconds: float, tare: Optional[Tare],
                live: bool, fs: float) -> List[Sample]:
    out: List[Sample] = []
    t0 = None
    print_every = max(1, int(fs / 2))   # ~2 console updates/sec
    i = 0
    for s in source.stream():
        if tare is not None:
            s = tare.apply(s)
        if t0 is None:
            t0 = s.t
        out.append(s)
        if live and i % print_every == 0:
            l, r = s.left_right_pct
            sys.stdout.write(
                f"\r  CoP x={s.cop_x:+6.2f}cm  y={s.cop_y:+6.2f}cm  "
                f"mass={s.total:5.1f}kg  L/R={l:4.1f}/{r:4.1f}%   ")
            sys.stdout.flush()
        i += 1
        if (s.t - t0) >= seconds:
            break
    if live:
        sys.stdout.write("\n")
    return out


def main():
    ap = argparse.ArgumentParser(description="WELAB Wii Balance Board runner")
    ap.add_argument("--demo", action="store_true", help="synthetic board, no hardware")
    ap.add_argument("--host", default="127.0.0.1", help="UDP host the bridge sends to")
    ap.add_argument("--port", type=int, default=8674, help="UDP port (matches the bridge)")
    ap.add_argument("--duration", type=float, default=30.0, help="trial length, seconds")
    ap.add_argument("--fs", type=float, default=100.0, help="resample rate, Hz")
    ap.add_argument("--tare-seconds", type=float, default=3.0,
                    help="seconds to capture the unloaded baseline (0 to skip)")
    ap.add_argument("--no-tare", action="store_true", help="skip taring entirely")
    ap.add_argument("--out", default="trial", help="output filename prefix")
    args = ap.parse_args()
    if args.no_tare:
        args.tare_seconds = 0.0

    # 1) Tare (optional): capture the unloaded baseline.
    tare = None
    if args.tare_seconds > 0:
        print(f"Taring: keep OFF the board for {args.tare_seconds:.0f}s ...")
        base = collect_for(make_source(args), args.tare_seconds, None, False, args.fs)
        if base:
            tare = Tare.from_samples(base)
            print(f"  baseline captured ({len(base)} samples).")
        else:
            print("  no data received during tare (is the bridge running?).")

    # 2) Record the trial.
    print(f"Recording {args.duration:.0f}s — stand on the board now.")
    samples = collect_for(make_source(args), args.duration, tare, True, args.fs)
    if not samples:
        print("No samples received. In live mode, confirm the C# bridge is "
              "streaming to udp://{}:{}.".format(args.host, args.port))
        sys.exit(2)

    # 3) Resample + features.
    samples = resample_uniform(samples, args.fs)
    trial = Trial(samples=samples, fs=args.fs)
    feats = trial.features()

    print("\n--- Posturography summary ---")
    print(f"  duration         : {feats.duration_s:.2f} s  ({feats.n} samples)")
    print(f"  mean load        : {feats.mean_total_kg:.1f} kg")
    print(f"  mean CoP (x,y)   : ({feats.mean_cop_x:+.2f}, {feats.mean_cop_y:+.2f}) cm")
    print(f"  path length      : {feats.path_length_cm:.2f} cm")
    print(f"  mean velocity    : {feats.mean_velocity_cm_s:.2f} cm/s")
    print(f"  95% ellipse area : {feats.ellipse_area_cm2:.2f} cm^2")
    print(f"  RMS ML / AP      : {feats.rms_ml:.2f} / {feats.rms_ap:.2f} cm")
    print(f"  load L / R       : {feats.left_pct:.1f} / {feats.right_pct:.1f} %"
          f"   (asymmetry {feats.lr_asymmetry:.1f})")
    print(f"  load ant / post  : {feats.anterior_pct:.1f} / {feats.posterior_pct:.1f} %")

    # 4) Export.
    n1 = write_cop_csv(f"{args.out}_cop.csv", trial, brainblox=False)
    write_cop_csv(f"{args.out}_bblox.csv", trial, brainblox=True)
    write_features_summary(f"{args.out}_features.csv", feats)
    print(f"\nSaved: {args.out}_cop.csv ({n1} rows), "
          f"{args.out}_bblox.csv, {args.out}_features.csv")


if __name__ == "__main__":
    main()
