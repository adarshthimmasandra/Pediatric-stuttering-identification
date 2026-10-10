"""Manifest-based datasets.

Every data source (SEP-28k, FluencyBank, UCLASS, your own recordings) is converted into one CSV
manifest with the columns below, so the training code never needs to know where data came from:

    path        audio file (absolute, or relative to the manifest's folder)
    split       train / val / test (optional; derived from ``group`` when missing)
    group       speaker / episode id used for leakage-free splitting (optional)
    start, end  segment boundaries in seconds inside ``path`` (optional)
    Block, Prolongation, SoundRep, WordRep, Interjection
                annotator counts (0-3) or binary labels; leave empty for unlabelled audio
    severity    optional clinician score normalised to [0, 1]; otherwise a proxy is derived
"""

from __future__ import annotations

import random
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import audio as audio_utils
from .config import DataConfig
from .labels import LABELS, UNUSABLE_COLUMNS, binary_targets, label_matrix, severity_proxy, soft_targets

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
SPLITS = ("train", "val", "test")


def assign_split(group: str, val_frac: float = 0.1, test_frac: float = 0.1) -> str:
    """Deterministic group-level split so the same speaker/episode never spans splits."""
    bucket = (zlib.crc32(str(group).encode("utf-8")) % 10_000) / 10_000
    if bucket < test_frac:
        return "test"
    if bucket < test_frac + val_frac:
        return "val"
    return "train"


def read_manifest(path: str | Path, cfg: DataConfig) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    if "path" not in df.columns:
        raise ValueError(f"Manifest {path} needs a 'path' column")
    root = path.parent
    df["path"] = [str(p if Path(p).is_absolute() else (root / p)) for p in df["path"]]
    for name in LABELS:
        if name not in df.columns:
            df[name] = np.nan
    if cfg.drop_unusable:
        for col in UNUSABLE_COLUMNS:
            if col in df.columns:
                df = df[~(df[col].fillna(0) >= cfg.label_threshold)]
    if "group" not in df.columns:
        df["group"] = [Path(p).parent.name for p in df["path"]]
    if "split" not in df.columns or df["split"].isna().any():
        derived = df["group"].map(assign_split)
        df["split"] = df["split"].fillna(derived) if "split" in df.columns else derived
    return df.reset_index(drop=True)


def split_manifest(df: pd.DataFrame, max_samples: int = 0, seed: int = 42) -> dict[str, pd.DataFrame]:
    out = {}
    for split in SPLITS:
        part = df[df["split"] == split]
        cap = max_samples if split == "train" else max(max_samples // 4, 1)
        if max_samples and len(part) > cap:
            part = part.sample(n=cap, random_state=seed)
        out[split] = part.reset_index(drop=True)
    return out


class ClipDataset(Dataset):
    """Fixed-length waveform clips with multi-label targets and a severity target."""

    def __init__(self, df: pd.DataFrame, cfg: DataConfig, augment: bool = False, random_crop: bool = False):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.augment = augment
        self.random_crop = random_crop
        self.num_samples = int(cfg.clip_seconds * cfg.sample_rate)

        raw = label_matrix(self.df)
        self.has_labels = ~np.isnan(raw).any(axis=1)
        raw = np.nan_to_num(raw, nan=0.0)
        self.targets = soft_targets(raw) if cfg.soft_labels else binary_targets(raw, cfg.label_threshold)
        severity = severity_proxy(raw, cfg.severity_weights)
        if "severity" in self.df.columns:
            given = self.df["severity"].to_numpy(dtype=np.float32)
            severity = np.where(np.isnan(given), severity, given).astype(np.float32)
        self.severity = severity

    def __len__(self) -> int:
        return len(self.df)

    def _segment(self, row) -> tuple[float | None, float | None]:
        start = row.get("start")
        end = row.get("end")
        start = None if start is None or pd.isna(start) else float(start)
        end = None if end is None or pd.isna(end) else float(end)
        return start, end

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        start, end = self._segment(row)
        wave = audio_utils.load_audio(row["path"], self.cfg.sample_rate, start, end)
        if self.augment:
            wave = audio_utils.augment(wave, self.cfg.sample_rate)
        length = min(len(wave), self.num_samples)
        wave = audio_utils.fix_length(wave, self.num_samples, random_crop=self.random_crop)
        return {
            "wave": torch.from_numpy(np.ascontiguousarray(wave, dtype=np.float32)),
            "length": torch.tensor(max(length, 1), dtype=torch.long),
            "labels": torch.from_numpy(self.targets[idx]),
            "severity": torch.tensor(self.severity[idx], dtype=torch.float32),
            "has_labels": torch.tensor(bool(self.has_labels[idx])),
        }


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    out = {key: torch.stack([b[key] for b in batch]) for key in batch[0]}
    if "wave" in out:
        positions = torch.arange(out["wave"].shape[1])[None, :]
        out["attention_mask"] = (positions < out["length"][:, None]).long()
    return out


def positive_rates(dataset: ClipDataset) -> np.ndarray:
    labelled = dataset.targets[dataset.has_labels]
    return labelled.mean(axis=0) if len(labelled) else np.full(len(LABELS), 0.5, dtype=np.float32)


# --------------------------------------------------------------------------------------------
# Local corpora (UCLASS, FluencyBank, own recordings)
# --------------------------------------------------------------------------------------------


def find_audio_files(audio_dir: str | Path) -> list[Path]:
    return sorted(p for p in Path(audio_dir).rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS)


def segment_rows(path: Path, group: str, segment_seconds: float, hop_seconds: float) -> list[dict]:
    duration = audio_utils.audio_duration(str(path))
    rows, start = [], 0.0
    while start < max(duration - 0.5, 1e-6):
        rows.append({"path": str(path), "group": group, "start": round(start, 3),
                     "end": round(min(start + segment_seconds, duration), 3)})
        start += hop_seconds
    return rows


def build_local_manifest(
    audio_dir: str | Path,
    annotations: pd.DataFrame | None = None,
    column_map: dict[str, str] | None = None,
    segment_seconds: float = 3.0,
    hop_seconds: float = 3.0,
    group_from: str = "parent",
) -> pd.DataFrame:
    """Build a manifest for a manually downloaded corpus.

    * ``annotations`` (optional) is any table with a file column (``path`` or ``file``) and label
      columns; ``column_map`` renames corpus-specific columns to ours, e.g. ``{"blk": "Block"}``.
      Rows may carry ``start``/``end`` seconds for segment-level labels. Files are matched by
      relative path, file name or stem.
    * Audio files without annotations are split into ``segment_seconds`` windows (unlabelled),
      ready for inference, pseudo-labelling, or for you to annotate.
    """
    audio_dir = Path(audio_dir)
    files = find_audio_files(audio_dir)

    def group_of(p: Path) -> str:
        return p.parent.name if group_from == "parent" else p.stem

    rows: list[dict] = []
    annotated: set[Path] = set()
    if annotations is not None and len(annotations):
        ann = annotations.rename(columns=column_map or {})
        file_col = "path" if "path" in ann.columns else "file"
        if file_col not in ann.columns:
            raise ValueError("Annotations need a 'path' or 'file' column (use column_map to rename)")
        lookup: dict[str, Path] = {}
        for p in files:
            for key in (str(p.relative_to(audio_dir)).replace("\\", "/"), p.name, p.stem):
                lookup.setdefault(key.lower(), p)
        missing = 0
        for rec in ann.to_dict("records"):
            key = str(rec[file_col]).replace("\\", "/").lower()
            match = lookup.get(key) or lookup.get(Path(key).name) or lookup.get(Path(key).stem)
            if match is None:
                missing += 1
                continue
            annotated.add(match)
            row = {"path": str(match), "group": rec.get("group", group_of(match))}
            for col in ("start", "end", "severity", "split", *LABELS, *UNUSABLE_COLUMNS):
                if col in rec and not pd.isna(rec[col]):
                    row[col] = rec[col]
            rows.append(row)
        if missing:
            print(f"[build_local_manifest] {missing} annotation rows had no matching audio file")

    for p in files:
        if p not in annotated:
            rows.extend(segment_rows(p, group_of(p), segment_seconds, hop_seconds))

    df = pd.DataFrame(rows)
    if len(df):
        df["split"] = df["split"].fillna(df["group"].map(assign_split)) if "split" in df else df["group"].map(assign_split)
    return df


def apple_labels_to_annotations(labels_csv: str | Path, episodes_csv: str | Path, sample_rate: int = 16000) -> pd.DataFrame:
    """Convert Apple's ``fluencybank_labels.csv`` (+ ``fluencybank_episodes.csv``) to annotations
    keyed by the original media file name, so they can be matched to a local FluencyBank download."""
    labels = pd.read_csv(labels_csv, skipinitialspace=True)
    episodes = pd.read_csv(episodes_csv, header=None, sep=r",\s*", engine="python")
    url_by_ep = {(str(r.iloc[-2]), int(r.iloc[-1])): str(r.iloc[2]) for _, r in episodes.iterrows()}
    labels["file"] = [
        Path(url_by_ep.get((str(s), int(e)), f"{e}")).stem for s, e in zip(labels["Show"], labels["EpId"])
    ]
    labels["start"] = labels["Start"] / sample_rate
    labels["end"] = labels["Stop"] / sample_rate
    labels["group"] = labels["file"]
    return labels


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
