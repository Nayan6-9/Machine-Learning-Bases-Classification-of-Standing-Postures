"""
wbb_record.py — recording + export, benchmarked against CU BrainBLoX.

BrainBLoX records a CoP+mass time series at 100 Hz (variable WBB rate resampled),
supports tare and timed trials, and saves a time series that downstream tools
analyze for path length / velocity / etc. This module reaches that parity and
adds the derived posturography features in-tool (via wbb_core).

Two export layouts:
  - WELAB schema (default): documented header, includes per-corner kg.
  - BrainBLoX-compatible: time, CoP x (cm), CoP y (cm), mass (kg) — the same
    four series BrainBLoX plots/saves, so existing analysis scripts work.

NOTE on exact byte-level compatibility: BrainBLoX saves an extension-less file;
its precise column order/units should be confirmed against a real sample file
(in the lab's distribution) before assuming drop-in equivalence. The column set
here is therefore configurable so matching is a one-line change once verified.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import List, Optional

from wbb_core import Sample, SwayWindow, SwayFeatures, BalanceBoardSource
from wbb_bridge import resample_uniform, Tare


# Canonical WELAB columns for the CoP time series.
WELAB_COLUMNS = ["time_s", "cop_x_cm", "cop_y_cm", "mass_kg",
                 "tr_kg", "tl_kg", "br_kg", "bl_kg"]
# The four series BrainBLoX records (origin = board center, cm / kg).
BRAINBLOX_COLUMNS = ["time_s", "cop_x_cm", "cop_y_cm", "mass_kg"]


@dataclass
class Trial:
    """A recorded trial: the (resampled) samples plus convenience accessors."""
    samples: List[Sample]
    fs: Optional[float] = None

    @property
    def duration_s(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        return self.samples[-1].t - self.samples[0].t

    def window(self) -> SwayWindow:
        w = SwayWindow()
        for s in self.samples:
            w.add(s)
        return w

    def features(self) -> Optional[SwayFeatures]:
        return self.window().features()


def record_trial(source: BalanceBoardSource,
                 duration_s: Optional[float] = None,
                 fs: Optional[float] = 100.0,
                 tare: Optional[Tare] = None) -> Trial:
    """Collect a trial from a source.

    duration_s: stop once this much wall-time of samples has been gathered
                (timed trial, like BrainBLoX Record Time). None = until the
                source ends.
    fs:         resample to this rate (Hz). None = keep native timestamps.
    tare:       per-corner zero offset to subtract.
    """
    collected: List[Sample] = []
    t0: Optional[float] = None
    for s in source.stream():
        if tare is not None:
            s = tare.apply(s)
        if t0 is None:
            t0 = s.t
        collected.append(s)
        if duration_s is not None and (s.t - t0) >= duration_s:
            break
    if fs is not None:
        collected = resample_uniform(collected, fs)
    return Trial(samples=collected, fs=fs)


def cop_rows(samples: List[Sample], brainblox: bool = False) -> List[list]:
    rows = []
    for s in samples:
        if brainblox:
            rows.append([s.t, s.cop_x, s.cop_y, s.total])
        else:
            rows.append([s.t, s.cop_x, s.cop_y, s.total,
                         s.tr, s.tl, s.br, s.bl])
    return rows


def write_cop_csv(path: str, trial: Trial, brainblox: bool = False) -> int:
    """Write the CoP+mass time series. Returns the number of data rows written."""
    cols = BRAINBLOX_COLUMNS if brainblox else WELAB_COLUMNS
    rows = cop_rows(trial.samples, brainblox=brainblox)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([f"{v:.5f}" for v in r])
    return len(rows)


def write_features_summary(path: str, features: SwayFeatures) -> None:
    """Write the derived posturography metrics as a key,value CSV."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in features.as_dict().items():
            w.writerow([k, f"{v:.5f}" if isinstance(v, float) else v])


def iter_windows(samples: List[Sample], win_s: float, hop_s: float):
    """Yield overlapping sub-windows (lists of Samples) of length win_s, stepped
    by hop_s, across a trial. Used for data augmentation at train time and to
    match the rolling window used at inference time."""
    if not samples or win_s <= 0 or hop_s <= 0:
        return
    s = sorted(samples, key=lambda p: p.t)
    t0, tend = s[0].t, s[-1].t
    start = t0
    while start + win_s <= tend + 1e-9:
        win = [p for p in s if start <= p.t < start + win_s]
        if len(win) >= 2:
            yield win
        start += hop_s
