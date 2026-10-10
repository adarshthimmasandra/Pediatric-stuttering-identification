"""Fetch SEP-28k (+ FluencyBank annotations) and write 16 kHz wav clips plus a training manifest.

Sources
  --source hf        (default) pre-clipped audio + labels from a Hugging Face mirror (~2.9 GB, 7 shards).
                     Use --max-shards N for a smaller subset (~4.5k clips per shard).
  --source official  Apple's label CSVs from GitHub, original podcast audio from the episode URLs
                     (~32 GB raw, some links are dead), clipped locally with ffmpeg/soundfile.

Examples
  python scripts/fetch_sep28k.py --max-shards 1
  python scripts/fetch_sep28k.py --source official --labels-only
  python scripts/fetch_sep28k.py --source official --datasets sep28k --max-episodes 20
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd
import requests
import soundfile as sf
from tqdm import tqdm

from stutter.audio import decode_audio_bytes
from stutter.datasets import assign_split
from stutter.labels import LABELS

HF_REPO = "saeedzou/sep28k-fluencybank-stutter-dataset"
APPLE_RAW = "https://raw.githubusercontent.com/apple/ml-stuttering-events-dataset/main"
APPLE_FILES = {
    "sep28k": ("SEP-28k_episodes.csv", "SEP-28k_labels.csv"),
    "fluencybank": ("fluencybank_episodes.csv", "fluencybank_labels.csv"),
}
EXTRA_COLUMNS = ["Unsure", "PoorAudioQuality", "DifficultToUnderstand", "NoStutteredWords", "NaturalPause", "Music", "NoSpeech"]
SR = 16000
MIN_CLIP_SECONDS = 0.25


def manifest_row(out_dir: Path, wav_path: Path, rec: dict, split_by: str) -> dict:
    show, ep = str(rec["Show"]).strip(), int(rec["EpId"])
    group = show if split_by == "show" else f"{show}_{ep}"
    row = {
        "path": wav_path.relative_to(out_dir).as_posix(),
        "group": group,
        "split": assign_split(group),
        "dataset": "FluencyBank" if show.lower() == "fluencybank" else "SEP-28k",
        "Show": show,
        "EpId": ep,
        "ClipId": int(rec["ClipId"]),
    }
    for col in LABELS + EXTRA_COLUMNS:
        if col in rec and rec[col] is not None:
            row[col] = int(rec[col])
    return row


def write_manifest(out_dir: Path, rows: list[dict]) -> None:
    """Merge ``rows`` into ``<out_dir>/manifest.csv`` (so HF and official runs can be combined)."""
    path = out_dir / "manifest.csv"
    df = pd.DataFrame(rows)
    if path.exists():
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    if len(df):
        df = df.drop_duplicates("path", keep="last")
        df = df[[(out_dir / p).exists() for p in df["path"]]].reset_index(drop=True)
    df.to_csv(path, index=False)
    print(f"\nAdded {len(rows)} clips; manifest now has {len(df)} clips -> {path}")
    if len(df):
        print(df.groupby(["dataset", "split"]).size().to_string())
        print("Positive rate (>=2/3 annotators): " + ", ".join(f"{c}={(df[c] >= 2).mean():.3f}" for c in LABELS if c in df))


# --------------------------------------------------------------------------------------------
# Hugging Face mirror
# --------------------------------------------------------------------------------------------


def hf_shards(repo: str, parquet_dir: Path, max_shards: int) -> list[Path | str]:
    """Shard list: local copies in ``parquet_dir`` if present (offline re-extraction), else from the Hub."""
    local = sorted(parquet_dir.rglob("*.parquet"))
    if local and (not max_shards or len(local) >= max_shards):
        return local[: max_shards or None]
    from huggingface_hub import HfApi

    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet"))
    return files[: max_shards or None]


def fetch_hf(out_dir: Path, repo: str, max_shards: int, split_by: str, keep_parquet: bool, parquet_dir: Path | None) -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    keep_parquet = keep_parquet or parquet_dir is not None
    parquet_dir = parquet_dir or out_dir / "parquet"
    files = hf_shards(repo, parquet_dir, max_shards)
    print(f"Processing {len(files)} parquet shard(s) from {repo} (shard folder: {parquet_dir})")
    rows, skipped = [], 0
    for fname in files:
        local = fname if isinstance(fname, Path) else hf_hub_download(repo, fname, repo_type="dataset", local_dir=parquet_dir)
        with pq.ParquetFile(local) as pf, tqdm(total=pf.metadata.num_rows, desc=Path(local).name) as bar:
            for batch in pf.iter_batches(batch_size=256):
                for rec in batch.to_pylist():
                    bar.update(1)
                    show = str(rec["Show"]).strip()
                    wav_path = out_dir / "clips" / show / f"{show}_{rec['EpId']}_{rec['ClipId']}.wav"
                    if not wav_path.exists():
                        audio = rec.get("audio") or {}
                        data = audio.get("bytes")
                        if not data:
                            skipped += 1
                            continue
                        wave = decode_audio_bytes(data, SR)
                        if len(wave) < MIN_CLIP_SECONDS * SR:
                            skipped += 1
                            continue
                        wav_path.parent.mkdir(parents=True, exist_ok=True)
                        sf.write(wav_path, wave, SR, subtype="PCM_16")
                    rows.append(manifest_row(out_dir, wav_path, rec, split_by))
        if not keep_parquet:
            Path(local).unlink(missing_ok=True)
    if skipped:
        print(f"Skipped {skipped} empty/too-short clips")
    if not keep_parquet:
        shutil.rmtree(parquet_dir, ignore_errors=True)
    write_manifest(out_dir, rows)


# --------------------------------------------------------------------------------------------
# Official Apple route
# --------------------------------------------------------------------------------------------


def download(url: str, dest: Path, timeout: int = 60) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"}) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.replace(dest)


def read_episodes(path: Path) -> list[tuple[str, str, str]]:
    episodes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        cols = [c.strip() for c in line.split(", ")]
        if len(cols) >= 4:
            episodes.append((cols[-2], cols[-1], cols[2]))  # show, episode id, url
    return episodes


def to_wav16k(src: Path, dst: Path) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", str(SR), str(dst)], check=True)


def fetch_official(out_dir: Path, datasets: list[str], labels_only: bool, max_episodes: int, split_by: str, keep_episodes: bool) -> None:
    meta_dir = out_dir / "official"
    for name in datasets:
        for fname in APPLE_FILES[name]:
            dest = meta_dir / fname
            if not dest.exists():
                download(f"{APPLE_RAW}/{fname}", dest)
            print(f"Labels: {dest}")
    if labels_only:
        return
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required for --source official (decoding mp3/m4a podcast audio)")

    rows, failed = [], []
    for name in datasets:
        episodes_csv, labels_csv = (meta_dir / f for f in APPLE_FILES[name])
        episodes = read_episodes(episodes_csv)[: max_episodes or None]
        labels = pd.read_csv(labels_csv, skipinitialspace=True)
        labels["Show"] = labels["Show"].astype(str).str.strip()
        skipped_complete = 0
        for show, ep, url in tqdm(episodes, desc=f"{name} episodes"):
            wav = out_dir / "episodes" / show / f"{ep}.wav"
            clips = labels[(labels["Show"] == show) & (labels["EpId"] == int(ep))]
            if clips.empty:
                continue
            clip_paths = [out_dir / "clips" / show / f"{show}_{int(ep)}_{c}.wav" for c in clips["ClipId"]]
            if all(p.exists() for p in clip_paths):  # e.g. already provided by the HF mirror
                skipped_complete += 1
                rows.extend(manifest_row(out_dir, p, rec, split_by) for p, rec in zip(clip_paths, clips.to_dict("records")))
                continue
            try:
                if not wav.exists():
                    ext = next((e for e in (".mp3", ".m4a", ".mp4") if e in url), ".mp3")
                    raw = wav.with_suffix(ext)
                    download(url, raw)
                    to_wav16k(raw, wav)
                    raw.unlink(missing_ok=True)
                audio, sr = sf.read(wav, dtype="float32")
            except Exception as exc:  # dead links, HTTP errors, corrupt media
                failed.append(f"{show},{ep},{url},{type(exc).__name__}: {exc}")
                continue
            for rec in clips.to_dict("records"):
                clip_path = out_dir / "clips" / show / f"{show}_{int(ep)}_{rec['ClipId']}.wav"
                if not clip_path.exists():
                    seg = audio[int(rec["Start"]) : int(rec["Stop"])]
                    if len(seg) < MIN_CLIP_SECONDS * SR:
                        continue
                    clip_path.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(clip_path, seg, sr, subtype="PCM_16")
                rows.append(manifest_row(out_dir, clip_path, rec, split_by))
            if not keep_episodes:
                wav.unlink(missing_ok=True)
        print(f"{name}: {skipped_complete} episodes already complete (not re-downloaded)")
    if failed:
        (out_dir / "failed_episodes.txt").write_text("\n".join(failed), encoding="utf-8")
        print(f"{len(failed)} episodes failed to download; see {out_dir / 'failed_episodes.txt'}")
    write_manifest(out_dir, rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["hf", "official"], default="hf")
    p.add_argument("--out", default="data/sep28k")
    p.add_argument("--split-by", choices=["episode", "show"], default="episode",
                   help="Group used for leakage-free train/val/test splits")
    p.add_argument("--hf-repo", default=HF_REPO)
    p.add_argument("--max-shards", type=int, default=0, help="[hf] number of parquet shards (0 = all 7)")
    p.add_argument("--keep-parquet", action="store_true", help="[hf] keep downloaded parquet shards")
    p.add_argument("--parquet-dir", default=None,
                   help="[hf] persistent shard folder (e.g. on Google Drive); reused if shards exist, implies --keep-parquet")
    p.add_argument("--datasets", nargs="+", choices=list(APPLE_FILES), default=["sep28k", "fluencybank"], help="[official]")
    p.add_argument("--labels-only", action="store_true", help="[official] only download the label/episode CSVs")
    p.add_argument("--max-episodes", type=int, default=0, help="[official] limit episodes per dataset")
    p.add_argument("--keep-episodes", action="store_true", help="[official] keep full-length episode wavs")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.source == "hf":
        parquet_dir = Path(args.parquet_dir) if args.parquet_dir else None
        fetch_hf(out_dir, args.hf_repo, args.max_shards, args.split_by, args.keep_parquet, parquet_dir)
    else:
        fetch_official(out_dir, args.datasets, args.labels_only, args.max_episodes, args.split_by, args.keep_episodes)


if __name__ == "__main__":
    main()
