"""
wbb_core.py — Wii Balance Board (WBB) as a low-cost force plate.

Pure-logic core for WELAB. No third-party dependencies (stdlib math only),
so it runs under the same pure-logic test harness as the scoring modules.

Pipeline:  raw 16-bit sensor counts
              -> raw_to_kg() per quadrant (piecewise-linear, like RSI multipliers)
              -> Sample (TR, TL, BR, BL kg, total, CoP x/y, L/R + A/P load %)
              -> SwayWindow.features() (path length, velocity, ellipse, RMS, asymmetry)

Transport (Bluetooth/HID) is deliberately NOT in this file. See the
`BalanceBoardSource` protocol at the bottom; a synthetic source is provided
for testing, and the real Windows HID backend plugs in behind the same API.

Sensor naming follows WiiBrew: TR top-right, TL top-left, BR bottom-right,
BL bottom-left.  Default transducer spacing from Pagnacco/Carey datasheets and
the STS literature: 43.3 cm (x, medio-lateral) x 22.8 cm (y, antero-posterior).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Iterator, List, Optional, Protocol, Tuple

# ----------------------------------------------------------------------------
# Geometry (board half-extents, in cm). CoP returned in cm, origin = board center.
# ----------------------------------------------------------------------------
BOARD_X_CM = 43.3   # distance between left and right transducers
BOARD_Y_CM = 22.8   # distance between top and bottom transducers
HALF_X = BOARD_X_CM / 2.0
HALF_Y = BOARD_Y_CM / 2.0

CHI2_95_2DOF = 5.991  # 95% confidence ellipse, chi-square, 2 DOF


# ----------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------
@dataclass
class SensorCalibration:
    """Raw uint16 anchors stored on the board for 0 / 17 / 34 kg loads.

    The WBB ships 3 calibration anchors per sensor (register 0x24). We convert a
    raw reading to kg by piecewise-linear interpolation between the bracketing
    anchors -- identical in spirit to the RSI worksheet interpolation in WELAB.
    """
    raw_0kg: float
    raw_17kg: float
    raw_34kg: float

    def raw_to_kg(self, raw: float) -> float:
        lo, mid, hi = self.raw_0kg, self.raw_17kg, self.raw_34kg
        if raw <= lo:
            # extrapolate below 0 kg using the lower segment slope (clamped >= 0)
            if mid == lo:
                return 0.0
            return max(0.0, 17.0 * (raw - lo) / (mid - lo))
        if raw <= mid:
            return 17.0 * (raw - lo) / (mid - lo) if mid != lo else 0.0
        if raw <= hi:
            return 17.0 + 17.0 * (raw - mid) / (hi - mid) if hi != mid else 17.0
        # above 34 kg: extrapolate on the upper segment slope
        return 17.0 + 17.0 * (raw - mid) / (hi - mid) if hi != mid else 34.0


@dataclass
class BoardCalibration:
    tr: SensorCalibration
    tl: SensorCalibration
    br: SensorCalibration
    bl: SensorCalibration

    @staticmethod
    def identity_counts(per_kg: float = 100.0) -> "BoardCalibration":
        """A synthetic linear calibration: raw = kg * per_kg. Useful for tests
        and for boards whose stored anchors you want to bypass."""
        def s() -> SensorCalibration:
            return SensorCalibration(0.0, 17.0 * per_kg, 34.0 * per_kg)
        return BoardCalibration(s(), s(), s(), s())


# ----------------------------------------------------------------------------
# A single calibrated sample
# ----------------------------------------------------------------------------
@dataclass
class Sample:
    t: float          # timestamp, seconds
    tr: float         # kg
    tl: float
    br: float
    bl: float

    @property
    def total(self) -> float:
        return self.tr + self.tl + self.br + self.bl

    @property
    def cop_x(self) -> float:
        """Medio-lateral CoP in cm. + = subject's load shifted toward RIGHT side."""
        tot = self.total
        if tot <= 1e-6:
            return 0.0
        return HALF_X * ((self.tr + self.br) - (self.tl + self.bl)) / tot

    @property
    def cop_y(self) -> float:
        """Antero-posterior CoP in cm. + = load shifted toward TOP (anterior)."""
        tot = self.total
        if tot <= 1e-6:
            return 0.0
        return HALF_Y * ((self.tl + self.tr) - (self.bl + self.br)) / tot

    @property
    def left_right_pct(self) -> Tuple[float, float]:
        """(% load on LEFT, % load on RIGHT). Sums to 100 (when loaded)."""
        tot = self.total
        if tot <= 1e-6:
            return (50.0, 50.0)
        right = (self.tr + self.br) / tot * 100.0
        return (100.0 - right, right)

    @property
    def ant_post_pct(self) -> Tuple[float, float]:
        """(% load anterior/top, % load posterior/bottom)."""
        tot = self.total
        if tot <= 1e-6:
            return (50.0, 50.0)
        ant = (self.tl + self.tr) / tot * 100.0
        return (ant, 100.0 - ant)


def make_sample(t: float,
                raw_tr: float, raw_tl: float, raw_br: float, raw_bl: float,
                cal: BoardCalibration) -> Sample:
    return Sample(
        t=t,
        tr=cal.tr.raw_to_kg(raw_tr),
        tl=cal.tl.raw_to_kg(raw_tl),
        br=cal.br.raw_to_kg(raw_br),
        bl=cal.bl.raw_to_kg(raw_bl),
    )


# ----------------------------------------------------------------------------
# Sway / posture features over a window of samples
# ----------------------------------------------------------------------------
@dataclass
class SwayFeatures:
    n: int
    duration_s: float
    mean_total_kg: float
    # CoP location
    mean_cop_x: float
    mean_cop_y: float
    # excursion / variability (cm)
    rms_ml: float            # RMS of medio-lateral CoP about its mean
    rms_ap: float            # RMS of antero-posterior CoP about its mean
    range_ml: float
    range_ap: float
    path_length_cm: float    # total CoP path length
    mean_velocity_cm_s: float
    ellipse_area_cm2: float  # 95% confidence ellipse
    # magnitude features: direction-agnostic size of the CoP shift, added so a
    # model can read "how far did weight move" directly, and so a subject whose
    # slouch is subtle still shows up. cop_shift/rms_mag combine the ML+AP axes;
    # p95_disp is a robust peak (95th pct of instantaneous displacement).
    cop_shift_cm: float      # |mean CoP| (from origin; from neutral if normalized)
    rms_mag_cm: float        # sqrt(rms_ml^2 + rms_ap^2)
    p95_disp_cm: float       # 95th percentile of per-sample |CoP - mean|
    # load distribution (%)
    left_pct: float
    right_pct: float
    anterior_pct: float
    posterior_pct: float
    lr_asymmetry: float      # |left% - right%|
    # frequency-domain sway (Hz); 0 if not computable / numpy unavailable
    mpf_ml: float = 0.0      # mean power frequency, medio-lateral
    f50_ml: float = 0.0      # median frequency, medio-lateral
    mpf_ap: float = 0.0      # mean power frequency, antero-posterior
    f50_ap: float = 0.0      # median frequency, antero-posterior

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _spectrum_features(vals, fs):
    """(mean power frequency, median frequency) in Hz of a detrended signal.
    Slouched vs. neutral stance often differ in sway frequency content, not just
    amplitude. Returns (0, 0) if not computable or numpy is unavailable (keeps
    wbb_core importable without numpy)."""
    n = len(vals)
    if n < 8 or fs <= 0:
        return 0.0, 0.0
    try:
        import numpy as np
    except Exception:
        return 0.0, 0.0
    x = np.asarray(vals, dtype=float)
    x = x - x.mean()
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    psd = spec.real ** 2 + spec.imag ** 2
    if psd.size:
        psd[0] = 0.0  # drop DC
    tot = float(psd.sum())
    if tot <= 0:
        return 0.0, 0.0
    mpf = float((freqs * psd).sum() / tot)
    cum = np.cumsum(psd)
    idx = min(int(np.searchsorted(cum, tot / 2.0)), len(freqs) - 1)
    return mpf, float(freqs[idx])


class SwayWindow:
    """Accumulates samples and computes posturography features.

    Usage:
        w = SwayWindow()
        for s in source: w.add(s)
        feats = w.features()
    """

    def __init__(self) -> None:
        self._samples: List[Sample] = []

    def add(self, s: Sample) -> None:
        self._samples.append(s)

    def clear(self) -> None:
        self._samples.clear()

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def samples(self) -> List[Sample]:
        return list(self._samples)

    def features(self) -> Optional[SwayFeatures]:
        s = self._samples
        n = len(s)
        if n < 2:
            return None

        xs = [p.cop_x for p in s]
        ys = [p.cop_y for p in s]
        totals = [p.total for p in s]
        duration = max(1e-6, s[-1].t - s[0].t)

        mx = sum(xs) / n
        my = sum(ys) / n

        rms_ml = math.sqrt(sum((x - mx) ** 2 for x in xs) / n)
        rms_ap = math.sqrt(sum((y - my) ** 2 for y in ys) / n)
        range_ml = max(xs) - min(xs)
        range_ap = max(ys) - min(ys)

        # direction-agnostic magnitudes
        cop_shift = math.hypot(mx, my)          # |mean CoP| (rel. to neutral if norm.)
        rms_mag = math.hypot(rms_ml, rms_ap)    # overall sway amplitude
        disp = sorted(math.hypot(xs[i] - mx, ys[i] - my) for i in range(n))
        # 95th percentile via linear interpolation (robust "peak" excursion)
        idx = 0.95 * (n - 1)
        lo = int(idx)
        p95_disp = disp[lo] if lo + 1 >= n else \
            disp[lo] + (idx - lo) * (disp[lo + 1] - disp[lo])

        path = 0.0
        for i in range(1, n):
            path += math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1])
        velocity = path / duration

        # 95% confidence ellipse area = pi * chi2 * sqrt(det(cov))
        sxx = sum((x - mx) ** 2 for x in xs) / n
        syy = sum((y - my) ** 2 for y in ys) / n
        sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / n
        det = max(0.0, sxx * syy - sxy * sxy)
        ellipse = math.pi * CHI2_95_2DOF * math.sqrt(det)

        # mean load distribution
        left = sum(p.left_right_pct[0] for p in s) / n
        right = 100.0 - left
        ant = sum(p.ant_post_pct[0] for p in s) / n
        post = 100.0 - ant

        # frequency-domain sway (estimate fs from timestamps)
        fs_est = (n - 1) / duration if duration > 0 else 0.0
        mpf_ml, f50_ml = _spectrum_features(xs, fs_est)
        mpf_ap, f50_ap = _spectrum_features(ys, fs_est)

        return SwayFeatures(
            n=n,
            duration_s=duration,
            mean_total_kg=sum(totals) / n,
            mean_cop_x=mx,
            mean_cop_y=my,
            rms_ml=rms_ml,
            rms_ap=rms_ap,
            range_ml=range_ml,
            range_ap=range_ap,
            path_length_cm=path,
            mean_velocity_cm_s=velocity,
            ellipse_area_cm2=ellipse,
            cop_shift_cm=cop_shift,
            rms_mag_cm=rms_mag,
            p95_disp_cm=p95_disp,
            left_pct=left,
            right_pct=right,
            anterior_pct=ant,
            posterior_pct=post,
            lr_asymmetry=abs(left - right),
            mpf_ml=mpf_ml,
            f50_ml=f50_ml,
            mpf_ap=mpf_ap,
            f50_ap=f50_ap,
        )


# ----------------------------------------------------------------------------
# Transport abstraction.  The real Bluetooth/HID backend implements this same
# protocol; nothing above depends on how bytes arrive.
# ----------------------------------------------------------------------------
class BalanceBoardSource(Protocol):
    def stream(self) -> Iterator[Sample]:
        """Yield calibrated Samples until the board disconnects / is stopped."""
        ...


@dataclass
class SyntheticSource:
    """Deterministic synthetic board for tests and offline development.

    Generates a body load with a controllable CoP trajectory so the whole
    feature pipeline can be validated without hardware.
    """
    cal: BoardCalibration = field(default_factory=BoardCalibration.identity_counts)
    fs: float = 60.0
    duration_s: float = 5.0
    body_kg: float = 70.0
    # CoP trajectory in cm as functions of time
    cop_x_fn: Optional[callable] = None
    cop_y_fn: Optional[callable] = None

    def _raw_for(self, cop_x: float, cop_y: float) -> Tuple[float, float, float, float]:
        """Invert the CoP equations to get quadrant loads (kg), then to raw counts."""
        w = self.body_kg
        fx = cop_x / HALF_X        # in [-1, 1]
        fy = cop_y / HALF_Y
        # right share / top share in [0,1]
        right = (1 + fx) / 2
        top = (1 + fy) / 2
        tr = w * right * top
        tl = w * (1 - right) * top
        br = w * right * (1 - top)
        bl = w * (1 - right) * (1 - top)
        # forward to raw via the (linear) identity calibration inverse
        def to_raw(c: SensorCalibration, kg: float) -> float:
            return c.raw_0kg + (kg / 34.0) * (c.raw_34kg - c.raw_0kg)
        return (to_raw(self.cal.tr, tr), to_raw(self.cal.tl, tl),
                to_raw(self.cal.br, br), to_raw(self.cal.bl, bl))

    def stream(self) -> Iterator[Sample]:
        n = int(self.duration_s * self.fs)
        for i in range(n):
            t = i / self.fs
            cx = self.cop_x_fn(t) if self.cop_x_fn else 0.0
            cy = self.cop_y_fn(t) if self.cop_y_fn else 0.0
            rtr, rtl, rbr, rbl = self._raw_for(cx, cy)
            yield make_sample(t, rtr, rtl, rbr, rbl, self.cal)


def collect(source: BalanceBoardSource) -> SwayWindow:
    w = SwayWindow()
    for s in source.stream():
        w.add(s)
    return w


# ----------------------------------------------------------------------------
# Geometry helpers for visualization (pure math; no GUI dependency).
# ----------------------------------------------------------------------------
def eig_2x2_sym(sxx: float, syy: float, sxy: float):
    """Eigenvalues/vectors of [[sxx, sxy],[sxy, syy]], sorted l1 >= l2.
    Returns (l1, l2, (v1x, v1y), (v2x, v2y)) with unit eigenvectors."""
    tr = sxx + syy
    diff = math.sqrt(max(0.0, ((sxx - syy) / 2.0) ** 2 + sxy * sxy))
    l1 = tr / 2.0 + diff
    l2 = tr / 2.0 - diff
    if abs(sxy) > 1e-12:
        v1 = (l1 - syy, sxy)
    elif sxx >= syy:
        v1 = (1.0, 0.0)
    else:
        v1 = (0.0, 1.0)
    n1 = math.hypot(*v1) or 1.0
    v1 = (v1[0] / n1, v1[1] / n1)
    v2 = (-v1[1], v1[0])  # orthogonal
    return l1, l2, v1, v2


def confidence_ellipse_points(xs, ys, chi2: float = CHI2_95_2DOF, n: int = 60):
    """Points (cm) tracing the chi2 confidence ellipse of a CoP cloud.
    Returns a list of (x, y); empty if fewer than 3 points."""
    k = len(xs)
    if k < 3:
        return []
    mx = sum(xs) / k
    my = sum(ys) / k
    sxx = sum((x - mx) ** 2 for x in xs) / k
    syy = sum((y - my) ** 2 for y in ys) / k
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(k)) / k
    l1, l2, v1, _ = eig_2x2_sym(sxx, syy, sxy)
    a = math.sqrt(max(0.0, chi2 * l1))
    b = math.sqrt(max(0.0, chi2 * l2))
    ang = math.atan2(v1[1], v1[0])
    ca, sa = math.cos(ang), math.sin(ang)
    pts = []
    for i in range(n):
        t = 2 * math.pi * i / n
        ex, ey = a * math.cos(t), b * math.sin(t)
        pts.append((mx + ex * ca - ey * sa, my + ex * sa + ey * ca))
    return pts


def board_cm_to_canvas(cop_x_cm: float, cop_y_cm: float,
                       board_px_w: float, board_px_h: float,
                       cx: float, cy: float):
    """Map a CoP point in cm (origin = board center, +x right, +y anterior)
    to canvas pixels (origin top-left, +y DOWN). Returns (px, py)."""
    px = cx + (cop_x_cm / HALF_X) * (board_px_w / 2.0)
    py = cy - (cop_y_cm / HALF_Y) * (board_px_h / 2.0)
    return px, py
