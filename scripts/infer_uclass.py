"""Batch-infer UCLASS recordings with a trained checkpoint and aggregate the results.

Runs window-level inference over every recording (or children only), writes one JSON
per recording plus a summary CSV:

    python scripts/infer_uclass.py \
        --uclass-dir /root/uclass \
        --checkpoint /workspace/checkpoints/finetune_wav2vec2/best.pt \
        --children-only --out /workspace/results/uclass_inference
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd
import torch

from stutter.engine import load_model, resolve_device

# Reuse infer.py's analyzer rather than duplicating it
sys.path.insert(0, str(Path(__file__).parent))
from infer import analyze  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--uclass-dir", required=True, type=Path)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True, type=Path, help="Output directory for per-file JSON + summary.csv")
    p.add_argument("--children-only", action="store_true", help="Only recordings with is_child in the manifest")
    p.add_argument("--max-files", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    manifest = pd.read_csv(args.uclass_dir / "manifest.csv").drop_duplicates("path")
    if args.children_only:
        manifest = manifest[manifest["is_child"] == True]
    manifest = manifest[manifest["path"].str.endswith(".wav")]
    files = [args.uclass_dir / p for p in manifest["path"].tolist()]
    if args.max_files:
        files = files[: args.max_files]
    meta = manifest.set_index("path")

    args.out.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    model, cfg, thresholds = load_model(args.checkpoint, device)
    print(f"{len(files)} recordings -> {args.out} (device={device})")

    rows, failed = [], 0
    t0 = time.time()
    for i, f in enumerate(files, 1):
        rel = str(f.relative_to(args.uclass_dir))
        try:
            r = analyze(f, model, cfg, thresholds, device)
        except Exception as exc:
            failed += 1
            print(f"  FAILED {rel}: {exc}")
            continue
        out_file = args.out / (Path(rel).stem + ".json")
        out_file.write_text(json.dumps(r, indent=2), encoding="utf-8")
        m = meta.loc[rel] if rel in meta.index else {}
        rows.append({
            "file": rel,
            "speaker": m.get("speaker"), "age_years": m.get("age_years"),
            "sex": m.get("sex"), "task": m.get("task"), "release": m.get("release"),
            "duration_s": r["duration_s"],
            **r["counts"],
            "pct_disfluent": r["percent_disfluent_windows"],
            "mean_severity": r["mean_window_severity"],
            "composite_score": r["composite_severity_score"],
            "severity_band": r["severity_band"],
        })
        if i % 25 == 0 or i == len(files):
            rate = i / (time.time() - t0)
            eta = (len(files) - i) / max(rate, 1e-9)
            print(f"  {i}/{len(files)} ({rate:.1f} files/s, ETA {eta/60:.1f}m)", flush=True)

    summary = pd.DataFrame(rows)
    summary.to_csv(args.out / "summary.csv", index=False)
    print(f"\nDone: {len(rows)} analyzed, {failed} failed -> {args.out/'summary.csv'}")
    if len(summary):
        print(summary.groupby("severity_band").size().to_string())
        print("\nBy task:")
        print(summary.groupby("task")[["pct_disfluent", "composite_score"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
