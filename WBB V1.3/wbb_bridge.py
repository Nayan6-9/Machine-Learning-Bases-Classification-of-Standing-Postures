"""
wbb_bridge.py — Python ingest for the proven Windows Bluetooth stack.

Rather than re-solving Windows Bluetooth in Python, we reuse the battle-tested
C# stack (32Feet.NET + WiimoteLib, the same one WiiBalanceWalker uses). A tiny
C# bridge program (see WiiBoardBridge.cs) connects to the board and emits one
ASCII line per WiimoteChanged event over localhost UDP:

        <t_seconds>,<tr_kg>,<tl_kg>,<br_kg>,<bl_kg>

WiimoteLib already calibrates each corner to kg using the board's stored
Kg0/Kg17/Kg34 anchors (identical piecewise interpolation to wbb_core), so the
Python side just parses, optionally tares + resamples, and produces Samples.

Everything here is pure-logic-testable: the parsing/tare/resample functions take
plain iterables of strings, and BridgeSource accepts an injected line iterator
in place of a live socket, so the whole path is validated without hardware.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass, field
from typing import Iterable, Iterator, List, Optional, Tuple

from wbb_core import Sample, SwayWindow, BalanceBoardSource


# ----------------------------------------------------------------------------
# Tare: capture an unloaded baseline per corner and subtract it thereafter.
# Matches the WiiBalanceWalker tare button.
# ----------------------------------------------------------------------------
@dataclass
class Tare:
    tr: float = 0.0
    tl: float = 0.0
    br: float = 0.0
    bl: float = 0.0

    @staticmethod
    def from_samples(samples: Iterable[Sample]) -> "Tare":
        s = list(samples)
        if not s:
            return Tare()
        n = len(s)
        return Tare(
            tr=sum(p.tr for p in s) / n,
            tl=sum(p.tl for p in s) / n,
            br=sum(p.br for p in s) / n,
            bl=sum(p.bl for p in s) / n,
        )

    def apply(self, s: Sample) -> Sample:
        return Sample(
            t=s.t,
            tr=max(0.0, s.tr - self.tr),
            tl=max(0.0, s.tl - self.tl),
            br=max(0.0, s.br - self.br),
            bl=max(0.0, s.bl - self.bl),
        )


# ----------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------
def parse_sample_line(line: str) -> Optional[Sample]:
    """Parse 't,tr,tl,br,bl' -> Sample. Returns None on blank/garbage lines
    so a noisy stream never crashes acquisition."""
    line = line.strip()
    if not line:
        return None
    parts = line.split(",")
    if len(parts) < 5:
        return None
    try:
        t, tr, tl, br, bl = (float(parts[i]) for i in range(5))
    except ValueError:
        return None
    return Sample(t=t, tr=tr, tl=tl, br=br, bl=bl)


# ----------------------------------------------------------------------------
# Uniform resampling (WBB rate is ~100 Hz but jittery; resample for consistent
# posturography features across recordings). Linear interpolation per corner.
# ----------------------------------------------------------------------------
def resample_uniform(samples: List[Sample], fs: float) -> List[Sample]:
    if len(samples) < 2:
        return list(samples)
    s = sorted(samples, key=lambda p: p.t)
    t0, t1 = s[0].t, s[-1].t
    if t1 <= t0:
        return list(samples)
    dt = 1.0 / fs
    out: List[Sample] = []
    j = 0
    n = len(s)
    t = t0
    while t <= t1 + 1e-9:
        # advance j so that s[j].t <= t <= s[j+1].t
        while j < n - 2 and s[j + 1].t < t:
            j += 1
        a, b = s[j], s[j + 1]
        span = b.t - a.t
        w = 0.0 if span <= 0 else (t - a.t) / span
        w = min(1.0, max(0.0, w))
        out.append(Sample(
            t=t,
            tr=a.tr + (b.tr - a.tr) * w,
            tl=a.tl + (b.tl - a.tl) * w,
            br=a.br + (b.br - a.br) * w,
            bl=a.bl + (b.bl - a.bl) * w,
        ))
        t += dt
    return out


# ----------------------------------------------------------------------------
# BridgeSource: implements BalanceBoardSource over either a UDP socket (live)
# or an injected iterator of lines (tests / replay).
# ----------------------------------------------------------------------------
@dataclass
class BridgeSource:
    lines: Optional[Iterable[str]] = None      # for tests/replay; if None, use UDP
    host: str = "127.0.0.1"
    port: int = 8674                            # "TORI" -> arbitrary local port
    tare: Optional[Tare] = None
    max_samples: Optional[int] = None           # stop after N (live streams)
    stop_event: object = None                   # threading.Event; stop when set
    recv_timeout: float = 0.5                   # so stop_event is checked promptly

    def _stopped(self) -> bool:
        return self.stop_event is not None and self.stop_event.is_set()

    def _raw_lines(self) -> Iterator[str]:
        if self.lines is not None:
            for ln in self.lines:
                if self._stopped():
                    return
                yield ln
            return
        # Live UDP listener. Each datagram is one ASCII line.
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(self.recv_timeout)
        sock.bind((self.host, self.port))
        try:
            count = 0
            while not self._stopped():
                try:
                    data, _ = sock.recvfrom(256)
                except socket.timeout:
                    continue
                yield data.decode("ascii", errors="ignore")
                count += 1
                if self.max_samples is not None and count >= self.max_samples:
                    break
        finally:
            sock.close()

    def stream(self) -> Iterator[Sample]:
        for line in self._raw_lines():
            s = parse_sample_line(line)
            if s is None:
                continue
            if self.tare is not None:
                s = self.tare.apply(s)
            yield s


def record(source: BalanceBoardSource, fs: Optional[float] = None) -> SwayWindow:
    """Collect a full source into a SwayWindow, optionally resampling to fs Hz."""
    samples = list(source.stream())
    if fs is not None:
        samples = resample_uniform(samples, fs)
    w = SwayWindow()
    for s in samples:
        w.add(s)
    return w
