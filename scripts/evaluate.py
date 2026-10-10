"""Evaluate a checkpoint on a manifest split (F1 per disfluency type, macro/micro F1, severity MAE).

    python scripts/evaluate.py --checkpoint checkpoints/best.pt --split test
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --manifest data/fluencybank/manifest.csv --split all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from torch.utils.data import DataLoader

from stutter.datasets import ClipDataset, collate, read_manifest
from stutter.embeddings import StatsDataset, load_or_compute_stats
from stutter.engine import evaluate_predictions, load_model, predict, resolve_device
from stutter.metrics import format_metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--manifest", default=None, help="Defaults to the manifest used for training")
    p.add_argument("--split", default="test", help="train / val / test / all")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--no-cache", action="store_true", help="Run the backbone on raw audio instead of cached features")
    p.add_argument("--out", default=None, help="JSON report path")
    args = p.parse_args()

    device = resolve_device(args.device)
    model, cfg, thresholds = load_model(args.checkpoint, device)
    if args.manifest:
        cfg.data.manifest = args.manifest
    df = read_manifest(cfg.data.manifest, cfg.data)
    if args.split != "all":
        df = df[df["split"] == args.split]
    dataset = ClipDataset(df, cfg.data)
    print(f"Evaluating {len(dataset)} clips from {cfg.data.manifest} [{args.split}]")

    use_cache = not args.no_cache and cfg.model.pooling == "mean_std" and not model.backbone_trainable
    if use_cache:
        cfg.train.batch_size = args.batch_size
        dataset = StatsDataset(load_or_compute_stats(model, dataset, cfg, f"eval_{args.split}", device), dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    metrics = evaluate_predictions(predict(model, loader, device), thresholds)
    print(format_metrics(metrics))

    out = Path(args.out) if args.out else Path(args.checkpoint).with_name(f"eval_{args.split}.json")
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Report -> {out}")


if __name__ == "__main__":
    main()
