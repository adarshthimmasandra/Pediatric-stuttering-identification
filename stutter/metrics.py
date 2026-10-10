"""Evaluation metrics: per-class / macro / micro F1 for disfluency types, MAE for severity."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from .labels import LABELS


def tune_thresholds(probs: np.ndarray, targets: np.ndarray, grid: np.ndarray | None = None) -> np.ndarray:
    """Per-class decision threshold maximising F1 (fit on validation data only)."""
    grid = np.arange(0.05, 0.96, 0.05) if grid is None else grid
    y = targets >= 0.5
    best = np.full(probs.shape[1], 0.5, dtype=np.float32)
    for k in range(probs.shape[1]):
        if y[:, k].sum() == 0:
            continue
        scores = [f1_score(y[:, k], probs[:, k] >= t, zero_division=0) for t in grid]
        best[k] = grid[int(np.argmax(scores))]
    return best


def compute_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    sev_pred: np.ndarray,
    sev_true: np.ndarray,
    thresholds: np.ndarray | None = None,
    labels: list[str] = LABELS,
) -> dict:
    thresholds = np.full(probs.shape[1], 0.5) if thresholds is None else np.asarray(thresholds)
    y = targets >= 0.5
    pred = probs >= thresholds[None, :]
    out: dict = {"num_samples": int(len(probs))}
    if len(probs) == 0:
        return out
    per_class = {}
    for k, name in enumerate(labels):
        per_class[name] = {
            "f1": float(f1_score(y[:, k], pred[:, k], zero_division=0)),
            "precision": float(precision_score(y[:, k], pred[:, k], zero_division=0)),
            "recall": float(recall_score(y[:, k], pred[:, k], zero_division=0)),
            "support": int(y[:, k].sum()),
            "threshold": float(thresholds[k]),
        }
    out["per_class"] = per_class
    out["macro_f1"] = float(f1_score(y, pred, average="macro", zero_division=0))
    out["micro_f1"] = float(f1_score(y, pred, average="micro", zero_division=0))
    out["any_stutter_f1"] = float(f1_score(y.any(axis=1), pred.any(axis=1), zero_division=0))

    err = sev_pred - sev_true
    out["severity_mae"] = float(np.abs(err).mean())
    out["severity_rmse"] = float(np.sqrt((err**2).mean()))
    if len(sev_true) > 1 and sev_true.std() > 0 and sev_pred.std() > 0:
        out["severity_pearson"] = float(np.corrcoef(sev_pred, sev_true)[0, 1])
    return out


def format_metrics(m: dict) -> str:
    if "macro_f1" not in m:
        return "no samples"
    cls = " ".join(f"{k}={v['f1']:.3f}" for k, v in m["per_class"].items())
    return f"macroF1={m['macro_f1']:.3f} microF1={m['micro_f1']:.3f} anyF1={m['any_stutter_f1']:.3f} sevMAE={m['severity_mae']:.4f} | {cls}"
