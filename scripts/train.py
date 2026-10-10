"""Train the multi-task stuttering model.

CPU (default config: frozen backbone, cached features, head-only training):
    python scripts/train.py --config configs/default.yaml --max-samples 2000 --epochs 10
GPU end-to-end fine-tuning:
    python scripts/train.py --config configs/gpu_finetune.yaml
Arbitrary overrides:
    python scripts/train.py --set model.backbone=facebook/hubert-base-ls960 train.batch_size=16
"""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from stutter.config import load_config
from stutter.engine import train


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--manifest", default=None,
                 help="Path to manifest.csv, or hf://owner/repo for a Hugging Face Datasets audio dataset")
    p.add_argument("--backbone", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None, help="Cap training clips (val/test get 1/4)")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--hf-dataset", default=None, help="Hugging Face Datasets repo id (shorthand for --manifest hf://owner/repo)")
    p.add_argument("--set", nargs="*", default=[], metavar="section.key=value")
    args = p.parse_args()

    overrides = list(args.set)
    if args.hf_dataset:
        overrides.append(f"data.manifest=hf://{args.hf_dataset}")
    for flag, key in [("manifest", "data.manifest"), ("backbone", "model.backbone"), ("epochs", "train.epochs"),
                      ("batch_size", "train.batch_size"), ("max_samples", "train.max_samples"),
                      ("output_dir", "train.output_dir"), ("device", "train.device")]:
        value = getattr(args, flag)
        if value is not None:
            overrides.append(f"{key}={value}")
    cfg = load_config(args.config, overrides)
    report = train(cfg)
    print(json.dumps({"best_epoch": report["best_epoch"], "test": {k: v for k, v in report["test"].items() if k != "per_class"}}, indent=2))


if __name__ == "__main__":
    main()
