"""Label definitions, normalisation and the annotator-derived severity proxy."""

from __future__ import annotations

import numpy as np
import pandas as pd

LABELS: list[str] = ["Block", "Prolongation", "SoundRep", "WordRep", "Interjection"]
NUM_ANNOTATORS = 3
UNUSABLE_COLUMNS = ["NoSpeech", "Music", "PoorAudioQuality"]


def label_matrix(df: pd.DataFrame) -> np.ndarray:
    """Raw label values (annotator counts 0-3, or binary 0/1) as float array of shape (N, len(LABELS))."""
    return df[LABELS].to_numpy(dtype=np.float32)


def is_count_scale(values: np.ndarray) -> bool:
    return bool(np.nanmax(values, initial=0) > 1)


def binary_targets(values: np.ndarray, threshold: int) -> np.ndarray:
    """Binarise. Counts (0-3) use ``>= threshold``; already-binary labels are kept."""
    if is_count_scale(values):
        return (values >= threshold).astype(np.float32)
    return (values >= 0.5).astype(np.float32)


def soft_targets(values: np.ndarray) -> np.ndarray:
    if is_count_scale(values):
        return np.clip(values / NUM_ANNOTATORS, 0.0, 1.0).astype(np.float32)
    return np.clip(values, 0.0, 1.0).astype(np.float32)


def severity_proxy(values: np.ndarray, weights: dict[str, float]) -> np.ndarray:
    """Per-clip severity in [0, 1]: weighted sum of annotator agreement across disfluency types,
    normalised by the largest weight and clipped.

    This is a heuristic proxy (SEP-28k has no clinical severity rating). If a manifest provides a
    ``severity`` column (e.g. clinician SSI-4 normalised to 0-1) it takes precedence.
    """
    w = np.array([weights.get(name, 1.0) for name in LABELS], dtype=np.float32)
    agreement = soft_targets(values)
    # Several simultaneous disfluency types push severity up; normalise by the max weight so a
    # unanimous single block already yields a high (1.0) score.
    score = (agreement * w).sum(axis=1) / max(w.max(), 1e-6)
    return np.clip(score, 0.0, 1.0).astype(np.float32)
