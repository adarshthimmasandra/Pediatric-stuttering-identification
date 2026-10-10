"""Push the packaged SEP-28k/UCLASS datasets to Hugging Face Datasets for reliable RunPod/Colab access.

Requires:

    pip install datasets
    huggingface-cli login        # or set HF_TOKEN environment variable

Example:

    python scripts/push_to_hub.py \
        --dataset-dir /content/dataset/sep28k \
        --repo-id your-username/pediatric-stutter-sep28k \
        --audio-dir clips \
        --audio-column path \
        --split-column split \
        --private

For UCLASS (unlabelled windows, used for child inference / pseudo-labelling):

    python scripts/push_to_hub.py \
        --dataset-dir /content/dataset/uclass \
        --repo-id your-username/pediatric-stutter-uclass \
        --audio-dir wav \
        --audio-column path \
        --split-column split \
        --private

"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd
from datasets import Audio, Dataset, DatasetDict


def relative_audio_path(example: dict, audio_dir: str) -> dict:
    """Join the audio_dir prefix if the stored path is relative."""
    p = example["path"]
    if p and not Path(p).is_absolute():
        p = f"{audio_dir}/{p}"
    example["path"] = p
    return example


def push_dataset(
    dataset_dir: Path,
    repo_id: str,
    audio_dir: str,
    audio_column: str,
    split_column: str,
    private: bool,
    dedupe_audio: bool,
) -> None:
    manifest = pd.read_csv(dataset_dir / "manifest.csv")
    manifest = manifest.where(manifest.notna(), None)

    if dedupe_audio and audio_column in manifest.columns:
        # Windowed manifests (UCLASS) repeat the same recording hundreds of times.
        # Embedding audio per-row would re-encode every window; keep one row per file.
        manifest = manifest.drop_duplicates(audio_column).reset_index(drop=True)
        print(f"De-duplicated to {len(manifest)} unique audio files")

    # Manifest paths are relative to the dataset root (e.g. "clips/x.wav").
    # Only prepend audio_dir if the path doesn't already start with it.
    audio_dir_path = dataset_dir / audio_dir

    def abs_audio_path(p: str) -> str:
        if Path(p).is_absolute():
            return str(Path(p).resolve())
        if p.startswith(f"{audio_dir}/") or p.startswith(f"{audio_dir}\\"):
            return str((dataset_dir / p).resolve())
        return str((audio_dir_path / p).resolve())

    manifest["audio"] = [abs_audio_path(str(p)) for p in manifest[audio_column]]

    def make_ds(df: pd.DataFrame) -> Dataset:
        ds = Dataset.from_pandas(df, preserve_index=False)
        return ds.cast_column("audio", Audio(sampling_rate=16000))

    if split_column in manifest.columns:
        parts = {s: manifest[manifest[split_column] == s] for s in ("train", "val", "test")}
        ds = DatasetDict({s: make_ds(df) for s, df in parts.items() if len(df)}) or {"train": make_ds(manifest)}
    else:
        ds = {"train": make_ds(manifest)}

    print(f"Pushing {len(manifest)} rows to https://huggingface.co/datasets/{repo_id}")
    if isinstance(ds, dict):
        DatasetDict(ds).push_to_hub(repo_id, private=private)
    else:
        ds.push_to_hub(repo_id, private=private)
    print("Done.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", required=True, type=Path, help="Packaged dataset folder with manifest.csv and audio")
    p.add_argument("--repo-id", required=True, help="Hugging Face Datasets repo, e.g. your-username/pediatric-stutter-sep28k")
    p.add_argument("--audio-dir", default="clips", help="Subfolder containing audio files")
    p.add_argument("--audio-column", default="audio", help="Column that will hold the audio path")
    p.add_argument("--split-column", default="split", help="Column used to split train/val/test")
    p.add_argument("--dedupe-audio", action="store_true",
                   help="Keep one row per unique audio file (windowed manifests like UCLASS)")
    p.add_argument("--private", action="store_true", help="Make the repository private")
    args = p.parse_args()

    push_dataset(
        dataset_dir=args.dataset_dir,
        repo_id=args.repo_id,
        audio_dir=args.audio_dir,
        audio_column=args.audio_column,
        split_column=args.split_column,
        private=args.private,
        dedupe_audio=args.dedupe_audio,
    )


if __name__ == "__main__":
    main()
