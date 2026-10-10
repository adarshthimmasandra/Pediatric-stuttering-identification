"""Package the downloaded SEP-28k and UCLASS data into a single, self-describing dataset folder.

Intended for use inside Colab (after Cells 1-6) before uploading to Hugging Face Datasets:

    python scripts/package_dataset.py \
        --src-sep28k "$LOCAL_DATA/sep28k" \
        --src-uclass "$LOCAL_DATA/uclass" \
        --out /content/dataset \
        --copy-audio
    # optional tar archives
    tar -cf "$DATA_TARS/sep28k.tar" -C /content/dataset sep28k
    tar -cf "$DATA_TARS/uclass.tar" -C /content/dataset uclass

The output layout is:

    /content/dataset/
    ├── sep28k/
    │   ├── manifest.csv
    │   ├── clips/
    │   │   └── {dataset}_{show}_{ep}_{clipid}.wav
    │   └── dataset_info.json
    └── uclass/
        ├── manifest.csv
        ├── wav/
        │   └── {speaker}_{...}.wav
        └── dataset_info.json

"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd
from stutter.labels import LABELS


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def package_sep28k(src: Path, out: Path, copy_audio: bool) -> dict:
    manifest_path = src / "manifest.csv"
    df = pd.read_csv(manifest_path)

    audio_dir = out / "clips"
    audio_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for rec in df.to_dict("records"):
        old_path = Path(rec["path"])
        if not old_path.is_absolute():
            old_path = src / old_path

        # uniform naming: <dataset>_<show>_<ep>_<clipid>.wav
        stem = old_path.stem
        new_name = f"{stem}.wav"
        new_path = audio_dir / new_name

        if copy_audio:
            if not new_path.exists():
                shutil.copy2(old_path, new_path)
        elif not new_path.exists():
            new_path.symlink_to(old_path.resolve())

        row = {k: rec[k] for k in ["split", "group"] + LABELS if k in rec}
        row["path"] = f"clips/{new_name}"
        if "dataset" in rec:
            row["dataset"] = rec["dataset"]
        if "severity" in rec and not pd.isna(rec["severity"]):
            row["severity"] = float(rec["severity"])
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out / "manifest.csv", index=False)

    info = {
        "name": "sep28k_fluencybank_clips",
        "source": "SEP-28k + FluencyBank via Hugging Face mirror and Apple official labels",
        "num_clips": len(out_df),
        "num_recordings": out_df["path"].apply(lambda p: Path(p).stem.rsplit("_", 1)[0]).nunique(),
        "sample_rate_hz": 16000,
        "clip_seconds": 3.0,
        "splits": out_df["split"].value_counts().to_dict(),
        "positive_rates": {k: float((out_df[k] >= 2).mean()) for k in LABELS},
        "columns": list(out_df.columns),
        "manifest": "manifest.csv",
        "audio_dir": "clips",
    }
    (out / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def package_uclass(src: Path, out: Path, copy_audio: bool) -> dict:
    manifest_path = src / "manifest.csv"
    df = pd.read_csv(manifest_path)

    audio_dir = out / "wav"
    audio_dir.mkdir(parents=True, exist_ok=True)

    if not df.empty:
        df["path"] = df["path"].astype(str)
        # Build a mapping from absolute audio file to a normalized relative path.
        # The UCLASS manifest paths are relative to src; many rows are 3s windows over the same file.
        file_map: dict[str, str] = {}
        rel_paths = []
        for p in df["path"]:
            src_file = src / p if not Path(p).is_absolute() else Path(p)
            key = str(src_file.resolve())
            if key not in file_map:
                new_name = src_file.name
                file_map[key] = f"wav/{new_name}"
                new_file = audio_dir / new_name
                if copy_audio and not new_file.exists():
                    shutil.copy2(src_file, new_file)
                elif not copy_audio and not new_file.exists():
                    new_file.symlink_to(src_file.resolve())
            rel_paths.append(file_map[key])
        df["path"] = rel_paths

    # Keep the original UCLASS manifest metadata columns plus window columns.
    meta_cols = ["release", "task", "sex", "speaker", "age_years", "is_child"]
    keep = ["path", "group", "split", "start", "end"] + meta_cols
    out_df = df[[c for c in keep if c in df.columns]].copy()
    out_df.to_csv(out / "manifest.csv", index=False)

    info = {
        "name": "uclass_child_speech_windows",
        "source": "UCLASS Release 1 + 2 (UCL Archive of Stuttered Speech)",
        "num_windows": len(out_df),
        "num_recordings": out_df["path"].nunique() if not out_df.empty else 0,
        "sample_rate_hz": 16000,
        "window_seconds": 3.0,
        "hop_seconds": 1.5,
        "child_recordings": int(out_df["is_child"].fillna(False).sum()) if "is_child" in out_df else None,
        "age_years_min": float(out_df["age_years"].min()) if "age_years" in out_df else None,
        "age_years_max": float(out_df["age_years"].max()) if "age_years" in out_df else None,
        "columns": list(out_df.columns),
        "manifest": "manifest.csv",
        "audio_dir": "wav",
    }
    (out / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src-sep28k", default="data/sep28k", help="Folder with SEP-28k manifest + clips")
    p.add_argument("--src-uclass", default="data/uclass", help="Folder with UCLASS manifest + audio")
    p.add_argument("--out", default="dataset", help="Output root folder")
    p.add_argument("--copy-audio", action="store_true", help="Copy audio files instead of symlinking")
    p.add_argument("--skip-uclass", action="store_true", help="Only package SEP-28k")
    args = p.parse_args()

    src_sep28k = Path(args.src_sep28k)
    src_uclass = Path(args.src_uclass)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    summaries = {}
    summaries["sep28k"] = package_sep28k(src_sep28k, out_root / "sep28k", args.copy_audio)
    if not args.skip_uclass and src_uclass.exists() and (src_uclass / "manifest.csv").exists():
        summaries["uclass"] = package_uclass(src_uclass, out_root / "uclass", args.copy_audio)

    (out_root / "dataset_info.json").write_text(json.dumps({
        "datasets": summaries,
        "root": str(out_root),
        "description": "Combined pediatric stuttering research dataset: SEP-28k/FluencyBank labeled clips and UCLASS child speech windows."
    }, indent=2), encoding="utf-8")

    print(json.dumps(summaries, indent=2))
    print(f"\nPackaged dataset -> {out_root}")


if __name__ == "__main__":
    main()
