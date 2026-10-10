"""Download UCLASS Release One + Two (UCL Archive of Stuttered Speech) and build a manifest.

UCLASS is openly downloadable: https://www.uclass.psychol.ucl.ac.uk/  (please cite Howell et al., 2009).
Every listing folder offers an ``All*.zip`` bundle; those are downloaded (with retries - the server
frequently drops connections) and extracted. Speaker sex / id / age are parsed from file names such
as ``M_0016_7y10m_1.wav`` so you can evaluate on children only.

    python scripts/fetch_uclass.py                                  # wav audio + all transcripts/annotations
    python scripts/fetch_uclass.py --audio-format both              # also mp3 copies
    python scripts/fetch_uclass.py --zip-dir /content/drive/MyDrive/StutterProject/raw/uclass   # keep zips (e.g. on Drive)
"""

from __future__ import annotations

import argparse
import re
import time
import zipfile
from pathlib import Path
from urllib.parse import urljoin

import _bootstrap  # noqa: F401
import pandas as pd
import requests
from tqdm import tqdm

from stutter.datasets import AUDIO_EXTENSIONS, assign_split, build_local_manifest

BASE = "https://www.uclass.psychol.ucl.ac.uk/"
AUDIO = {
    "release1": "Audio/{fmt}",
    "release2_monologue": "Release2/Monologue/AudioOnly/{fmt}",
    "release2_reading": "Release2/Reading/AudioOnly/{fmt}",
    "release2_conversation": "Release2/Conversation/AudioOnly/{fmt}",
}
TEXT = {  # transcripts / time-aligned annotations
    "release1/transcript_ortho": "Transcript/Flat/Ortho",
    "release1/transcript_phon": "Transcript/Flat/Phon",
    "release1/aligned_chat": "Transcript/TAligned/CHAT",
    "release1/aligned_textgrid": "Transcript/TAligned/TextGrid",
    "release1/aligned_sfs_annotation": "Transcript/TAligned/Annotation",
    "release2_monologue/flat_phon": "Release2/Monologue/FlatPhon/index.htm",
    "release2_monologue/flat_ortho": "Release2/Monologue/FlatOrtho/index.htm",
    "release2_reading/flat_ortho": "Release2/Reading/FlatOrtho/index.htm",
    "release2_monologue/aligned_phon": "Release2/Monologue/AlignedPhon",
    "release2_monologue/aligned_ortho": "Release2/Monologue/AlignedOrtho",
    "release2_reading/aligned_ortho": "Release2/Reading/AlignedOrtho",
    "release2_monologue/chat": "Release2/Monologue/Annotation/chat/index.htm",
    "release2_reading/chat": "Release2/Reading/Annotation/chat/index.htm",
    "release2_monologue/chat_non_aligned": "Release2/Monologue/Annotation/chat/Non-Aligned/index.htm",
    "release2_reading/chat_non_aligned": "Release2/Reading/Annotation/chat/Non-Aligned/index.htm",
    "release2_monologue/praat": "Release2/Monologue/Annotation/praat",
    "release2_reading/praat": "Release2/Reading/Annotation/praat",
    "release2_monologue/sfs_annotation": "Release2/Monologue/Annotation/sfs",
    "release2_reading/sfs_annotation": "Release2/Reading/Annotation/sfs",
}
METADATA = [  # speaker listings (the linked MS Access .mdb files return HTTP 403)
    "information.htm", "Release2/info_monologue.htm", "Release2/info_reading.htm", "Release2/info_conversation.htm",
]
SKIP_LINK = re.compile(r"^/|index_files/|^mailto:|^#|\.html?$")  # absolute UCLASS links are kept (filtered by BASE)
NAME_RE = re.compile(r"^(?P<sex>[MF])_(?P<speaker>\d+)_(?P<years>\d+)y(?P<months>\d+)m", re.I)

session = requests.Session()
session.headers["User-Agent"] = "Mozilla/5.0 (research dataset download)"


def get(url: str, stream: bool = False, attempts: int = 8) -> requests.Response:
    """GET with exponential back-off on connection errors and 5xx responses (4xx fail immediately)."""
    for i in range(attempts):
        try:
            r = session.get(url, stream=stream, timeout=120)
            if r.status_code < 500:
                r.raise_for_status()
                return r
            reason = f"HTTP {r.status_code}"
        except requests.HTTPError:
            raise
        except requests.RequestException as exc:
            reason = type(exc).__name__
        print(f"  retry {i + 1}/{attempts} for {url}: {reason}")
        time.sleep(min(60, 3 * 2**i))
    raise RuntimeError(f"Failed to fetch {url} after {attempts} attempts")


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for i in range(5):
        try:
            with get(url, stream=True) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("content-length", 0)) or None
                with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name, leave=False) as bar:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                        bar.update(len(chunk))
            if total and tmp.stat().st_size < total:
                raise IOError("incomplete download")
            tmp.replace(dest)
            return dest
        except requests.HTTPError:
            raise  # 403/404 will not fix themselves
        except (requests.RequestException, IOError) as exc:
            print(f"  download retry {i + 1}/5 for {dest.name}: {exc}")
            time.sleep(10 * (i + 1))
    raise RuntimeError(f"Could not download {url}")


def listing_files(folder: str) -> tuple[list[str], list[str]]:
    """Return (zip bundle urls, individual file urls) linked from a UCLASS listing page."""
    r = get(BASE + folder)
    base = r.url if r.url.endswith((".htm", ".html", "/")) else r.url + "/"
    links = [urljoin(base, l) for l in re.findall(r'href="([^"]+)"', r.text, re.I) if not SKIP_LINK.search(l)]
    links = [l for l in dict.fromkeys(links) if l.startswith(BASE) and "." in l.rsplit("/", 1)[-1]]
    return [l for l in links if l.lower().endswith(".zip")], [l for l in links if not l.lower().endswith(".zip")]


def fetch_folder(folder: str, target: Path, zip_dir: Path) -> int:
    done = target / ".complete"
    if not done.exists():
        zips, files = listing_files(folder)
        for url in zips:
            archive = download(url, zip_dir / folder.replace("/index.htm", "").replace("/", "_") / url.rsplit("/", 1)[-1])
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(target)
        if not zips:  # no bundle offered: fetch files one by one
            for url in files:
                download(url, target / url.rsplit("/", 1)[-1])
        if not zips and not files:
            raise RuntimeError("no downloadable files found on listing page")
        done.touch()
    return sum(1 for p in target.rglob("*") if p.is_file() and p != done)


def speaker_metadata(path: str) -> dict:
    m = NAME_RE.match(Path(path).stem)
    if not m:
        return {}
    age = int(m["years"]) + int(m["months"]) / 12
    return {"sex": m["sex"].upper(), "speaker": f"{m['sex'].upper()}_{m['speaker']}", "age_years": round(age, 2),
            "is_child": age < 18}


def build_manifest(out_dir: Path, segment_seconds: float, hop_seconds: float) -> pd.DataFrame:
    frames = []
    for release in AUDIO:
        audio_dir = out_dir / "audio" / release
        if not audio_dir.exists():
            continue
        wavs = {p.stem for p in audio_dir.rglob("*.wav")}
        df = build_local_manifest(audio_dir, segment_seconds=segment_seconds, hop_seconds=hop_seconds)
        if df.empty:
            continue
        # prefer wav when both wav and mp3 of the same recording exist
        df = df[[Path(p).suffix.lower() == ".wav" or Path(p).stem not in wavs for p in df["path"]]]
        df["release"], df["task"] = release.split("_")[0], (release.split("_", 1) + ["monologue"])[1]
        meta = pd.DataFrame([speaker_metadata(p) for p in df["path"]], index=df.index)
        df = pd.concat([df, meta], axis=1)
        if "speaker" in df:
            df["group"] = df["speaker"].fillna(df["group"])  # speaker-disjoint splits
        df["split"] = df["group"].map(assign_split)
        df["path"] = [Path(p).relative_to(out_dir).as_posix() for p in df["path"]]
        frames.append(df)
    manifest = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/uclass")
    p.add_argument("--zip-dir", default=None, help="Where to keep downloaded zip bundles (default: <out>/zips)")
    p.add_argument("--audio-format", choices=["wav", "mp3", "both"], default="wav")
    p.add_argument("--skip-text", action="store_true", help="Audio only (no transcripts / annotations)")
    p.add_argument("--segment-seconds", type=float, default=3.0)
    p.add_argument("--hop-seconds", type=float, default=1.5)
    args = p.parse_args()

    out_dir = Path(args.out)
    zip_dir = Path(args.zip_dir) if args.zip_dir else out_dir / "zips"
    formats = ["wav", "mp3"] if args.audio_format == "both" else [args.audio_format]
    jobs = [(f"audio/{name}", folder.format(fmt=fmt)) for name, folder in AUDIO.items() for fmt in formats]
    if not args.skip_text:
        jobs += [(f"text/{name}", folder) for name, folder in TEXT.items()]

    failed = []
    for target, folder in jobs:
        print(f"[{target}] {BASE}{folder}")
        try:
            n = fetch_folder(folder, out_dir / target, zip_dir)
            print(f"  -> {n} files")
        except Exception as exc:
            failed.append(f"{folder}: {exc}")
            print(f"  FAILED: {exc}")
    for item in METADATA:
        try:
            download(BASE + item, out_dir / "metadata" / item.replace("/", "_"))
        except Exception as exc:
            failed.append(f"{item}: {exc}")

    df = build_manifest(out_dir, args.segment_seconds, args.hop_seconds)
    n_audio = sum(1 for f in (out_dir / "audio").rglob("*") if f.suffix.lower() in AUDIO_EXTENSIONS) if (out_dir / "audio").exists() else 0
    print(f"\n{n_audio} audio files; manifest with {len(df)} unlabelled 3 s windows -> {out_dir / 'manifest.csv'}")
    if len(df):
        rec = df.drop_duplicates("path")
        if "speaker" in rec:
            print(f"Recordings: {len(rec)} | speakers: {rec['speaker'].nunique()} | child recordings (<18 y): {int(rec['is_child'].fillna(False).sum())}")
        print(rec.groupby(["release", "task"]).size().to_string())
    if failed:
        (out_dir / "failed_downloads.txt").write_text("\n".join(failed), encoding="utf-8")
        print(f"\n{len(failed)} items failed (re-run to retry; finished files are skipped): {out_dir / 'failed_downloads.txt'}")


if __name__ == "__main__":
    main()
