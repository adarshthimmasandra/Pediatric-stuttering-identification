"""Build a training/inference manifest for a manually downloaded corpus (UCLASS, FluencyBank, own data).

Place audio anywhere, e.g.  data/uclass/<speaker>/*.wav  or  data/fluencybank/*.mp3

1) Unlabelled audio (segmented into 3 s windows, usable for inference or annotation):
     python scripts/prepare_local_corpus.py --audio-dir data/uclass --out data/uclass/manifest.csv

2) Your own annotation CSV (one row per file or per segment with start/end seconds):
     python scripts/prepare_local_corpus.py --audio-dir data/uclass --annotations my_labels.csv \
         --column-map file=path blk=Block prol=Prolongation snd=SoundRep wrd=WordRep intj=Interjection

3) FluencyBank audio + Apple's SEP-28k-style FluencyBank annotations
   (run `python scripts/fetch_sep28k.py --source official --labels-only` first):
     python scripts/prepare_local_corpus.py --audio-dir data/fluencybank \
         --apple-labels data/sep28k/official/fluencybank_labels.csv \
         --apple-episodes data/sep28k/official/fluencybank_episodes.csv

Then train with:  python scripts/train.py --set data.manifest=<manifest.csv>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from stutter.datasets import apple_labels_to_annotations, build_local_manifest
from stutter.labels import LABELS


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio-dir", required=True)
    p.add_argument("--out", default=None, help="Manifest path (default: <audio-dir>/manifest.csv)")
    p.add_argument("--annotations", default=None, help="CSV with a file/path column and label columns")
    p.add_argument("--column-map", nargs="*", default=[], help="Rename columns: src=dst (e.g. blk=Block)")
    p.add_argument("--apple-labels", default=None, help="Apple fluencybank_labels.csv")
    p.add_argument("--apple-episodes", default=None, help="Apple fluencybank_episodes.csv")
    p.add_argument("--segment-seconds", type=float, default=3.0)
    p.add_argument("--hop-seconds", type=float, default=3.0)
    p.add_argument("--group-from", choices=["parent", "file"], default="parent",
                   help="Speaker/group id for splits: parent folder name or file stem")
    args = p.parse_args()

    annotations = None
    if args.apple_labels:
        if not args.apple_episodes:
            p.error("--apple-labels requires --apple-episodes")
        annotations = apple_labels_to_annotations(args.apple_labels, args.apple_episodes)
    elif args.annotations:
        annotations = pd.read_csv(args.annotations)
    column_map = dict(item.split("=", 1) for item in args.column_map)

    df = build_local_manifest(args.audio_dir, annotations, column_map, args.segment_seconds, args.hop_seconds, args.group_from)
    out = Path(args.out) if args.out else Path(args.audio_dir) / "manifest.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    labelled = df[LABELS].notna().all(axis=1).sum() if set(LABELS) <= set(df.columns) else 0
    print(f"Wrote {len(df)} rows ({labelled} labelled) -> {out}")
    if len(df):
        print(df["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
