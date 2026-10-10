"""Cache frozen-backbone statistics so the head can be trained quickly on CPU."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import Config
from .datasets import ClipDataset, collate


def cache_key(cfg: Config, split: str, dataset: ClipDataset) -> str:
    h = hashlib.sha1()
    h.update(f"{cfg.model.backbone}|{cfg.data.sample_rate}|{cfg.data.clip_seconds}|{split}".encode())
    cols = [c for c in ("path", "start", "end") if c in dataset.df.columns]
    h.update(dataset.df[cols].astype(str).to_csv(index=False).encode())
    return f"{Path(cfg.model.backbone).name}_{split}_{h.hexdigest()[:12]}"


@torch.no_grad()
def compute_stats(model, dataset: ClipDataset, batch_size: int, device: torch.device, num_workers: int = 0) -> torch.Tensor:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=num_workers)
    model.eval()
    chunks = []
    for batch in tqdm(loader, desc="Extracting backbone features"):
        stats = model.extract_stats(batch["wave"].to(device), batch["attention_mask"].to(device))
        chunks.append(stats.to(torch.float16).cpu())
    return torch.cat(chunks) if chunks else torch.empty(0)


def load_or_compute_stats(model, dataset: ClipDataset, cfg: Config, split: str, device: torch.device) -> torch.Tensor:
    cache_dir = Path(cfg.train.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{cache_key(cfg, split, dataset)}.pt"
    if path.exists():
        print(f"Loading cached features: {path}")
        return torch.load(path)
    stats = compute_stats(model, dataset, cfg.train.batch_size, device, cfg.train.num_workers)
    torch.save(stats, path)
    print(f"Cached {len(stats)} feature vectors -> {path}")
    return stats


class StatsDataset(Dataset):
    """Pre-computed (L, 2H) statistics paired with the targets of a ``ClipDataset``."""

    def __init__(self, stats: torch.Tensor, source: ClipDataset):
        self.stats = stats
        self.source = source

    def __len__(self) -> int:
        return len(self.stats)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "stats": self.stats[idx],
            "labels": torch.from_numpy(self.source.targets[idx]),
            "severity": torch.tensor(self.source.severity[idx], dtype=torch.float32),
            "has_labels": torch.tensor(bool(self.source.has_labels[idx])),
        }
