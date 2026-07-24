"""
wbb_dataset.py — labeled posture dataset for ML (items 2-4 of the plan).

Each collected trial -> one row in a growing dataset.csv (the "database"),
plus its raw CoP time series saved under samples/ for provenance and future
re-analysis (e.g. windowed augmentation).

Feature vector (excludes body mass, which is not posture):
    mean_cop_x, mean_cop_y, rms_ml, rms_ap, range_ml, range_ap,
    path_length_cm, mean_velocity_cm_s, ellipse_area_cm2,
    anterior_pct, lr_asymmetry

Note: CoP-position features (mean_cop_y / anterior_pct) are the strongest
slouch discriminators but depend on foot placement on the board. For best
results, keep foot position consistent across trials, or train per subject.
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from wbb_core import Sample, SwayFeatures, SwayWindow
from wbb_record import Trial, write_cop_csv, iter_windows

# Window length / hop for augmentation + train-inference alignment.
# Training windows the raw CoP into these segments; the live monitor uses the
# same window length so train and inference see comparable features.
DEFAULT_WINDOW_S = 10.0
DEFAULT_HOP_S = 5.0

FEATURE_NAMES: List[str] = [
    "mean_cop_x", "mean_cop_y", "rms_ml", "rms_ap", "range_ml", "range_ap",
    "path_length_cm", "mean_velocity_cm_s", "ellipse_area_cm2",
    "cop_shift_cm", "rms_mag_cm", "p95_disp_cm",
    "anterior_pct", "lr_asymmetry",
    "mpf_ml", "f50_ml", "mpf_ap", "f50_ap",
]

META_COLUMNS = ["trial_id", "label", "subject", "timestamp", "duration_s", "n_samples"]
DATASET_COLUMNS = META_COLUMNS + FEATURE_NAMES

# A dedicated per-subject baseline recording (quiet neutral stance). It is stored
# like any other trial (raw CoP included) but is NEVER used as a training class -
# it only defines that subject's "neutral origin" for baseline normalization.
BASELINE_LABEL = "baseline"


def _window_vectors(samples: List[Sample], win_s: float, hop_s: float
                    ) -> List[List[float]]:
    """Feature vectors for every full window of a trial.

    If no full window fits but the trial is *almost* as long as the window, the
    whole trial is used as a single window: sampling means a "5 s" recording
    actually spans ~4.99 s, which would otherwise yield zero windows and make the
    trial silently disappear. Trials clearly shorter than the window still yield
    nothing, because their duration-dependent features (path length, range, ...)
    would be on the wrong scale.
    """
    out: List[List[float]] = []
    for win in iter_windows(samples, win_s, hop_s):
        w = SwayWindow()
        for s in win:
            w.add(s)
        f = w.features()
        if f is not None:
            out.append(feature_vector(f))
    if not out and len(samples) >= 2:
        span = samples[-1].t - samples[0].t
        if span >= 0.9 * win_s:
            w = SwayWindow()
            for s in samples:
                w.add(s)
            f = w.features()
            if f is not None:
                out.append(feature_vector(f))
    return out


def feature_vector(f: SwayFeatures) -> List[float]:
    return [float(getattr(f, name)) for name in FEATURE_NAMES]


@dataclass
class Dataset:
    """Directory-backed accumulating store of labeled trials."""
    root: str

    def __post_init__(self):
        os.makedirs(self.root, exist_ok=True)
        os.makedirs(os.path.join(self.root, "samples"), exist_ok=True)
        self.csv_path = os.path.join(self.root, "dataset.csv")
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as fh:
                csv.writer(fh).writerow(DATASET_COLUMNS)
            return
        # If the file's columns differ from the current schema (e.g. an older
        # feature set), rebuild it from the raw samples so no data is lost.
        with open(self.csv_path, newline="") as fh:
            header = next(csv.reader(fh), [])
        if header != DATASET_COLUMNS:
            self._migrate_schema()

    def _migrate_schema(self):
        """Rebuild dataset.csv under the current schema by recomputing features
        from each trial's raw CoP in samples/. Meta columns (trial_id/label/
        subject) are position-stable, so old rows are read reliably."""
        import shutil
        old_rows = []
        with open(self.csv_path, newline="") as fh:
            for r in csv.DictReader(fh):
                old_rows.append(r)
        adir = os.path.join(self.root, "archive",
                            "schema-" + time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(adir, exist_ok=True)
        shutil.copy(self.csv_path, os.path.join(adir, "dataset_old.csv"))
        migrated = 0
        with open(self.csv_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(DATASET_COLUMNS)
            for r in old_rows:
                tid = r.get("trial_id")
                label = r.get("label")
                subject = r.get("subject", "S00")
                if not tid or not label:
                    continue
                samples = self.load_trial_cop(tid)
                if len(samples) < 2:
                    continue
                win = SwayWindow()
                for s in samples:
                    win.add(s)
                f = win.features()
                if f is None:
                    continue
                ts = r.get("timestamp") or time.strftime("%Y%m%d-%H%M%S")
                w.writerow([tid, label, subject, ts,
                            f"{f.duration_s:.3f}", f.n]
                           + [f"{v:.6f}" for v in feature_vector(f)])
                migrated += 1
        print(f"[wbb] migrated dataset to new schema: {migrated} trials "
              f"rebuilt from raw (old file archived in {adir}).")

    def _next_index(self) -> int:
        with open(self.csv_path, newline="") as fh:
            return max(0, sum(1 for _ in fh) - 1)  # minus header

    def append(self, label: str, features: SwayFeatures,
               subject: str = "S00",
               cop_samples: Optional[List[Sample]] = None,
               fs: Optional[float] = None) -> str:
        idx = self._next_index()
        ts = time.strftime("%Y%m%d-%H%M%S")
        trial_id = f"{idx:04d}_{label}_{subject}"

        if cop_samples:
            trial = Trial(samples=cop_samples, fs=fs)
            write_cop_csv(os.path.join(self.root, "samples", trial_id + "_cop.csv"),
                          trial, brainblox=False)

        row = [trial_id, label, subject, ts,
               f"{features.duration_s:.3f}", features.n] + \
              [f"{v:.6f}" for v in feature_vector(features)]
        with open(self.csv_path, "a", newline="") as fh:
            csv.writer(fh).writerow(row)
        return trial_id

    def load(self) -> Tuple[List[List[float]], List[str], List[dict]]:
        """Return (X, y, meta) for TRAINING rows. X = feature matrix, y = labels.
        Dedicated baseline recordings are excluded (they are not a posture class)."""
        X: List[List[float]] = []
        y: List[str] = []
        meta: List[dict] = []
        with open(self.csv_path, newline="") as fh:
            rdr = csv.DictReader(fh)
            for r in rdr:
                if r["label"] == BASELINE_LABEL:
                    continue
                X.append([float(r[name]) for name in FEATURE_NAMES])
                y.append(r["label"])
                meta.append({k: r[k] for k in META_COLUMNS})
        return X, y, meta

    def counts(self) -> dict:
        """Trial counts per label, including dedicated baselines."""
        out: dict = {}
        with open(self.csv_path, newline="") as fh:
            for r in csv.DictReader(fh):
                out[r["label"]] = out.get(r["label"], 0) + 1
        return out

    def baseline_subjects(self) -> set:
        """Subjects that have at least one dedicated BASELINE recording."""
        out = set()
        with open(self.csv_path, newline="") as fh:
            for r in csv.DictReader(fh):
                if r["label"] == BASELINE_LABEL:
                    out.add(r["subject"])
        return out

    def baseline_durations(self) -> dict:
        """subject -> shortest dedicated baseline duration (s). Empty if none."""
        out: dict = {}
        with open(self.csv_path, newline="") as fh:
            for r in csv.DictReader(fh):
                if r["label"] != BASELINE_LABEL:
                    continue
                d = float(r["duration_s"])
                s = r["subject"]
                out[s] = min(out[s], d) if s in out else d
        return out

    def load_trial_cop(self, trial_id: str) -> List[Sample]:
        """Reload a trial's raw CoP time series from samples/ as Samples."""
        path = os.path.join(self.root, "samples", trial_id + "_cop.csv")
        out: List[Sample] = []
        if not os.path.exists(path):
            return out
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                out.append(Sample(t=float(r["time_s"]), tr=float(r["tr_kg"]),
                                  tl=float(r["tl_kg"]), br=float(r["br_kg"]),
                                  bl=float(r["bl_kg"])))
        return out

    def clear(self) -> Tuple[str, int]:
        """Reset the dataset for a fresh start WITHOUT deleting raw data.

        Moves the current dataset.csv and the samples/ folder into an archive
        subfolder (archive/<timestamp>/), then recreates an empty dataset.csv and
        an empty samples/ folder. Returns (archive_path, n_trials_archived).
        """
        import shutil
        _, y, _ = self.load()
        n = len(y)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        adir = os.path.join(self.root, "archive", stamp)
        os.makedirs(adir, exist_ok=True)
        if os.path.exists(self.csv_path):
            shutil.move(self.csv_path, os.path.join(adir, "dataset.csv"))
        samples_dir = os.path.join(self.root, "samples")
        if os.path.isdir(samples_dir) and os.listdir(samples_dir):
            shutil.move(samples_dir, os.path.join(adir, "samples"))
        # recreate empty structure
        os.makedirs(os.path.join(self.root, "samples"), exist_ok=True)
        with open(self.csv_path, "w", newline="") as fh:
            csv.writer(fh).writerow(DATASET_COLUMNS)
        return adir, n

    def load_windowed(self, win_s: float, hop_s: float, group_by: str = "trial"
                      ) -> Tuple[List[List[float]], List[str], List[str]]:
        """Expand each trial into overlapping windows for augmentation.

        Returns (X, y, groups). group_by='trial' groups windows by trial_id (no
        window leakage); group_by='subject' groups by subject, so cross-validation
        estimates generalization to NEW people. Trials with no saved CoP series
        fall back to their stored aggregate feature row.
        """
        X: List[List[float]] = []
        y: List[str] = []
        groups: List[str] = []
        with open(self.csv_path, newline="") as fh:
            for r in csv.DictReader(fh):
                tid, label = r["trial_id"], r["label"]
                if label == BASELINE_LABEL:
                    continue
                g = r["subject"] if group_by == "subject" else tid
                samples = self.load_trial_cop(tid)
                if len(samples) < 2:
                    X.append([float(r[nm]) for nm in FEATURE_NAMES])
                    y.append(label)
                    groups.append(g)
                    continue
                for v in _window_vectors(samples, win_s, hop_s):
                    X.append(v)
                    y.append(label)
                    groups.append(g)
        return X, y, groups

    # ---- per-subject baseline normalization -----------------------------
    def neutral_baselines(self, win_s: Optional[float] = None,
                          hop_s: Optional[float] = None,
                          source: str = "auto") -> Tuple[dict, List[float]]:
        """Per-subject "neutral origin" feature vector, plus a global fallback.
        Subtracting it removes that person's foot-placement / body offset.

        source:
          'auto'      - use a subject's dedicated BASELINE recording if they have
                        one, otherwise fall back to their neutral trials.
          'dedicated' - dedicated BASELINE recordings only.
          'neutral'   - mean of the subject's neutral trials (legacy behaviour).

        A dedicated baseline matches how live monitoring works (one short quiet
        capture) and keeps the baseline independent of the training labels; the
        'neutral' source uses far more data but is computed from the same rows
        that are being classified.

        If win_s is given, the baseline is averaged over windows of the same
        length used for training/inference, so duration-dependent features (path
        length, range, ...) are on a comparable scale.
        """
        from collections import defaultdict
        sums = defaultdict(lambda: [0.0] * len(FEATURE_NAMES))
        counts = defaultdict(int)
        gsum = [0.0] * len(FEATURE_NAMES)
        gcount = 0
        have_dedicated = self.baseline_subjects() if source == "auto" else set()

        def add(subject, vec):
            nonlocal gcount
            for i in range(len(vec)):
                sums[subject][i] += vec[i]
                gsum[i] += vec[i]
            counts[subject] += 1
            gcount += 1

        def _wanted(label, subject):
            if source == "dedicated":
                return label == BASELINE_LABEL
            if source == "neutral":
                return label == "neutral"
            # auto: prefer the subject's own dedicated baseline when it exists
            if subject in have_dedicated:
                return label == BASELINE_LABEL
            return label == "neutral"

        with open(self.csv_path, newline="") as fh:
            for r in csv.DictReader(fh):
                subject = r["subject"]
                if not _wanted(r["label"], subject):
                    continue
                if win_s is not None:
                    samples = self.load_trial_cop(r["trial_id"])
                    if len(samples) >= 2:
                        vecs = _window_vectors(samples, win_s, hop_s)
                        for v in vecs:
                            add(subject, v)
                        if vecs:
                            continue
                        # baseline too short for this window: fall through to the
                        # whole-trial summary (train reports a scale warning)
                add(subject, [float(r[n]) for n in FEATURE_NAMES])

        baselines = {s: [sums[s][i] / counts[s] for i in range(len(FEATURE_NAMES))]
                     for s in counts}
        global_neutral = ([gsum[i] / gcount for i in range(len(FEATURE_NAMES))]
                          if gcount else [0.0] * len(FEATURE_NAMES))
        return baselines, global_neutral

    def load_windowed_normalized(self, win_s: float, hop_s: float,
                                 group_by: str = "trial",
                                 baseline_source: str = "auto"
                                 ) -> Tuple[List[List[float]], List[str], List[str]]:
        """Like load_windowed, but subtract each subject's neutral baseline
        (computed over the same window length). group_by='trial'|'subject';
        baseline_source='auto'|'dedicated'|'neutral'."""
        baselines, global_neutral = self.neutral_baselines(win_s, hop_s,
                                                           baseline_source)
        X: List[List[float]] = []
        y: List[str] = []
        groups: List[str] = []
        with open(self.csv_path, newline="") as fh:
            for r in csv.DictReader(fh):
                tid, label, subject = r["trial_id"], r["label"], r["subject"]
                if label == BASELINE_LABEL:
                    continue
                g = subject if group_by == "subject" else tid
                base = baselines.get(subject, global_neutral)
                samples = self.load_trial_cop(tid)
                if len(samples) < 2:
                    v = [float(r[nm]) for nm in FEATURE_NAMES]
                    X.append([v[i] - base[i] for i in range(len(v))])
                    y.append(label)
                    groups.append(g)
                    continue
                for v in _window_vectors(samples, win_s, hop_s):
                    X.append([v[i] - base[i] for i in range(len(v))])
                    y.append(label)
                    groups.append(g)
        return X, y, groups
