"""Aggregate sliding-window predictions into events, counts and a file-level severity rating."""

from __future__ import annotations

import numpy as np

from .config import InferConfig
from .labels import LABELS


def merge_events(starts: np.ndarray, ends: np.ndarray, probs: np.ndarray, thresholds: np.ndarray, labels=LABELS) -> list[dict]:
    """Merge consecutive positive windows of the same type into a single event."""
    events = []
    for k, name in enumerate(labels):
        active = probs[:, k] >= thresholds[k]
        i = 0
        while i < len(active):
            if not active[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(active) and active[j + 1] and starts[j + 1] <= ends[j]:
                j += 1
            events.append({
                "type": name,
                "start_s": round(float(starts[i]), 2),
                "end_s": round(float(ends[j]), 2),
                "confidence": round(float(probs[i : j + 1, k].max()), 3),
            })
            i = j + 1
    return sorted(events, key=lambda e: (e["start_s"], e["type"]))


def severity_band(score: float, cfg: InferConfig) -> str:
    idx = int(np.searchsorted(np.asarray(cfg.band_cutoffs), score, side="right"))
    return cfg.band_names[min(idx, len(cfg.band_names) - 1)]


def summarize(
    starts: np.ndarray,
    ends: np.ndarray,
    probs: np.ndarray,
    severity: np.ndarray,
    thresholds: np.ndarray,
    cfg: InferConfig,
    duration: float,
) -> dict:
    events = merge_events(starts, ends, probs, thresholds)
    positive = (probs >= thresholds[None, :]).any(axis=1)
    stuttered_ratio = float(positive.mean()) if len(positive) else 0.0
    mean_sev = float(severity.mean()) if len(severity) else 0.0
    # Composite 0-100 score: share of disfluent windows blended with mean predicted severity.
    composite = 100.0 * (0.5 * stuttered_ratio + 0.5 * mean_sev)
    return {
        "duration_s": round(duration, 2),
        "num_windows": int(len(probs)),
        "events": events,
        "counts": {name: sum(e["type"] == name for e in events) for name in LABELS},
        "percent_disfluent_windows": round(100.0 * stuttered_ratio, 1),
        "mean_window_severity": round(mean_sev, 3),
        "max_window_severity": round(float(severity.max()) if len(severity) else 0.0, 3),
        "composite_severity_score": round(composite, 1),
        "severity_band": severity_band(composite, cfg),
        "disclaimer": "Research prototype. Window-level estimates (approximate %SS / SSI-4-style band); not a clinical diagnosis.",
    }
