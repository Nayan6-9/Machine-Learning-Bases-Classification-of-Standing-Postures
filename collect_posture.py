#!/usr/bin/env python3
"""
collect_posture.py — collect ONE labeled 30 s trial and append to the database.
(Items 2, 3, 4 of the plan: neutral / slouched collection, accumulated.)

Examples:
    # neutral, live board, default 30 s, subject S01, into ./wbb_db
    python collect_posture.py --label neutral  --subject S01 --db wbb_db
    python collect_posture.py --label slouched --subject S01 --db wbb_db

    # no hardware, try the workflow with synthetic data:
    python collect_posture.py --label neutral  --demo --db wbb_db
    python collect_posture.py --label slouched --demo --db wbb_db

Run it many times per posture to grow the dataset for training.
"""

import argparse
import math
import sys

from wbb_core import SyntheticSource, BoardCalibration
from wbb_bridge import BridgeSource, Tare, resample_uniform
from wbb_record import Trial
from wbb_dataset import Dataset

VALID_LABELS = ("neutral", "slouched", "baseline")


def make_source(args, slouch_bias: bool):
    if args.demo:
        # crude synthetic difference so the demo dataset is separable:
        # slouched -> CoP shifted posteriorly (-y) with a bit more sway.
        bias = -4.0 if slouch_bias else 1.0
        amp = 2.5 if slouch_bias else 1.2
        return SyntheticSource(
            cal=BoardCalibration.identity_counts(), fs=args.fs,
            duration_s=args.tare_seconds + args.duration + 1.0, body_kg=70.0,
            cop_x_fn=lambda t: 0.5 * math.cos(2 * math.pi * 0.3 * t),
            cop_y_fn=lambda t: bias + amp * math.sin(2 * math.pi * 0.35 * t))
    return BridgeSource(host=args.host, port=args.port)


def collect_for(source, seconds, tare, fs):
    out, t0 = [], None
    for s in source.stream():
        if tare is not None:
            s = tare.apply(s)
        if t0 is None:
            t0 = s.t
        out.append(s)
        if (s.t - t0) >= seconds:
            break
    return out


def main():
    ap = argparse.ArgumentParser(description="Collect a labeled posture trial")
    ap.add_argument("--label", required=True, choices=VALID_LABELS)
    ap.add_argument("--subject", default="S00")
    ap.add_argument("--db", default="wbb_db", help="dataset directory")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--tare-seconds", type=float, default=3.0)
    ap.add_argument("--no-tare", action="store_true")
    ap.add_argument("--fs", type=float, default=100.0)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8674)
    args = ap.parse_args()
    if args.no_tare:
        args.tare_seconds = 0.0

    slouch = (args.label == "slouched")
    ds = Dataset(args.db)

    tare = None
    if args.tare_seconds > 0:
        print(f"Taring: keep OFF the board for {args.tare_seconds:.0f}s ...")
        base = collect_for(make_source(args, slouch), args.tare_seconds, None, args.fs)
        if base:
            tare = Tare.from_samples(base)

    print(f"Recording {args.duration:.0f}s of '{args.label}' — assume the posture now.")
    samples = collect_for(make_source(args, slouch), args.duration, tare, args.fs)
    if len(samples) < 2:
        print("No data received. Is the bridge running (or use --demo)?")
        sys.exit(2)

    samples = resample_uniform(samples, args.fs)
    trial = Trial(samples=samples, fs=args.fs)
    feats = trial.features()

    trial_id = ds.append(args.label, feats, subject=args.subject,
                         cop_samples=samples, fs=args.fs)
    counts = ds.counts()
    print(f"\nSaved trial {trial_id}")
    print(f"  CoP(x,y) {feats.mean_cop_x:+.2f},{feats.mean_cop_y:+.2f} cm · "
          f"ant {feats.anterior_pct:.0f}% · path {feats.path_length_cm:.1f} cm · "
          f"ellipse {feats.ellipse_area_cm2:.1f} cm²")
    print(f"  dataset now: " +
          ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
