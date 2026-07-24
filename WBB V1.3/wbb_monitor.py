"""
wbb_monitor.py — live posture inference + alarm logic (item 6).

PostureClassifier wraps a saved model and predicts from a SwayFeatures window.
AlarmController is a debounced state machine: it raises an alarm only when the
non-ergonomic posture is *sustained*, clears it once good posture returns for a
while, and respects a cooldown so it doesn't nag continuously. The decision
logic is pure and unit-tested; the GUI only renders the events it returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from wbb_core import SwayFeatures

POSITIVE = "slouched"   # the class we care about detecting


class PostureClassifier:
    """Loads a joblib bundle from wbb_train.py and predicts a label + p(slouch)."""
    def __init__(self, model, feature_names: List[str], labels: List[str],
                 positive: str, window_s: Optional[float] = None,
                 normalized: bool = False, explainer=None):
        self.model = model
        self.feature_names = feature_names
        self.labels = labels
        self.positive = positive
        self.window_s = window_s
        self.normalized = normalized
        self.explainer = explainer

    @staticmethod
    def load(path: str) -> "PostureClassifier":
        """Load a model saved by this app, or one trained elsewhere.

        A model trained in someone's own script is usually a bare scikit-learn
        estimator with no metadata attached, so anything missing is inferred and
        the important assumptions are checked rather than guessed at silently.
        Raises ValueError with a readable explanation when it cannot be used.
        """
        import joblib
        from wbb_dataset import FEATURE_NAMES

        obj = joblib.load(path)

        if isinstance(obj, dict):
            # this app's bundle, or a dict from another script using other names
            model = (obj.get("model") or obj.get("clf") or obj.get("estimator")
                     or obj.get("pipeline"))
            if model is None:
                raise ValueError(
                    "This .joblib holds a dictionary with no model in it.\n"
                    f"Keys found: {sorted(obj)[:8]}\n"
                    "Save the fitted estimator under the key 'model'.")
            feature_names = list(obj.get("feature_names") or FEATURE_NAMES)
            labels = list(obj.get("labels") or [])
            positive = obj.get("positive")
            window_s = obj.get("window_s")
            normalized = bool(obj.get("normalized", False))
            explainer = obj.get("explainer")
        else:
            model, feature_names = obj, list(FEATURE_NAMES)
            labels, positive, window_s, normalized, explainer = [], None, None, False, None

        if not hasattr(model, "predict"):
            raise ValueError(
                f"{type(model).__name__} has no .predict(), so it is not a "
                "usable classifier.")

        # classes: prefer what the fitted model itself reports
        model_classes = [str(c) for c in getattr(model, "classes_", [])]
        if model_classes:
            labels = model_classes
        if not labels:
            raise ValueError("The model does not report its class labels "
                             "(classes_), so its output cannot be interpreted.")
        if positive is None or positive not in labels:
            positive = POSITIVE if POSITIVE in labels else labels[-1]

        # feature contract: a mismatch here silently produces garbage predictions
        n_expected = getattr(model, "n_features_in_", None)
        if n_expected is not None and int(n_expected) != len(feature_names):
            raise ValueError(
                f"This model expects {int(n_expected)} input features, but the "
                f"app computes {len(feature_names)}.\n"
                "It was trained on a different feature set, so it cannot be "
                "used live. Re-train it on this version's features, or save it "
                "with a 'feature_names' list that matches how it was trained.")
        unknown = [n for n in feature_names if n not in FEATURE_NAMES]
        if unknown:
            raise ValueError("This model expects features the app does not "
                             f"compute: {unknown[:5]}")

        return PostureClassifier(model, feature_names, labels, positive,
                                 window_s, normalized, explainer)

    def describe(self) -> str:
        """Short human summary, shown after a model is loaded."""
        kind = type(self.model).__name__
        if hasattr(self.model, "named_steps"):
            kind = " + ".join(type(v).__name__ for v in self.model.named_steps.values())
        bits = [f"{kind}", f"{len(self.feature_names)} features",
                f"classes: {', '.join(self.labels)}"]
        bits.append(f"window {self.window_s:g}s" if self.window_s
                    else "window: not recorded")
        bits.append("per-subject baseline: required" if self.normalized
                    else "per-subject baseline: not used")
        return "  |  ".join(bits)

    def vector(self, f: SwayFeatures) -> List[float]:
        return [float(getattr(f, n)) for n in self.feature_names]

    def _vector(self, f: SwayFeatures, baseline: Optional[List[float]] = None):
        v = self.vector(f)
        if self.normalized and baseline is not None:
            v = [v[i] - baseline[i] for i in range(len(v))]
        return [v]

    def predict(self, f: SwayFeatures,
                baseline: Optional[List[float]] = None) -> Tuple[str, Optional[float]]:
        X = self._vector(f, baseline)
        label = str(self.model.predict(X)[0])
        proba = None
        if hasattr(self.model, "predict_proba"):
            classes = list(self.model.classes_)
            if self.positive in classes:
                proba = float(self.model.predict_proba(X)[0][classes.index(self.positive)])
        return label, proba

    @staticmethod
    def _linear_parts(est):
        """Return (clf, scaler) if est is a linear model (has coef_), else (None,None)."""
        if est is None:
            return None, None
        clf, scaler = est, None
        if hasattr(est, "named_steps"):
            clf = est.named_steps.get("clf")
            scaler = est.named_steps.get("scaler")
        if clf is not None and hasattr(clf, "coef_"):
            return clf, scaler
        return None, None

    def explain(self, f: SwayFeatures, baseline: Optional[List[float]] = None,
                top: int = 3):
        """Top signed feature contributions to the 'slouched' decision right now:
        list of (feature_name, contribution). Positive pushes toward slouched.
        Uses the main model if it is linear, otherwise the stored linear explainer
        (so this works even when the best model is a tree/forest). None if neither
        is available."""
        import numpy as np
        clf, scaler = self._linear_parts(self.model)
        if clf is None:
            clf, scaler = self._linear_parts(self.explainer)
        if clf is None:
            return None
        v = self.vector(f)
        if self.normalized and baseline is not None:
            v = [v[i] - baseline[i] for i in range(len(v))]
        x = np.asarray([v], dtype=float)
        if scaler is not None:
            x = scaler.transform(x)[0]
            scale = np.asarray(getattr(scaler, "scale_", np.ones_like(x)), dtype=float)
            x = np.clip(x, -4.0, 4.0)
            x = np.where(scale < 1e-6, 0.0, x)
        else:
            x = x[0]
        classes = list(clf.classes_)
        coef = np.asarray(clf.coef_[0], dtype=float)
        if self.positive == classes[0]:
            coef = -coef
        contrib = coef * x
        idx = sorted(range(len(contrib)), key=lambda i: abs(contrib[i]),
                     reverse=True)[:top]
        return [(self.feature_names[i], float(contrib[i])) for i in idx]


@dataclass
class AlarmController:
    """Debounced alarm. Feed (t, predicted_label); get 'ALARM'/'CLEAR'/None."""
    positive: str = "slouched"
    sustain_s: float = 5.0     # slouch must persist this long before alarming
    clear_s: float = 3.0       # good posture must persist this long to clear
    cooldown_s: float = 15.0   # min gap between alarms

    def __post_init__(self):
        self.alarming = False
        self._pos_since: Optional[float] = None
        self._neg_since: Optional[float] = None
        self._last_alarm_t: float = -1e18

    def update(self, t: float, label: str) -> Optional[str]:
        if label == self.positive:
            self._neg_since = None
            if self._pos_since is None:
                self._pos_since = t
            if (not self.alarming
                    and (t - self._pos_since) >= self.sustain_s
                    and (t - self._last_alarm_t) >= self.cooldown_s):
                self.alarming = True
                self._last_alarm_t = t
                return "ALARM"
        else:
            self._pos_since = None
            if self._neg_since is None:
                self._neg_since = t
            if self.alarming and (t - self._neg_since) >= self.clear_s:
                self.alarming = False
                return "CLEAR"
        return None
