"""Run the trained model on raw audio and print structured JSON.

    python scripts/infer.py path/to/child.wav
    python scripts/infer.py recordings/ --checkpoint checkpoints/best.pt --out results.json

Each file is cut into overlapping windows (default 3 s, 1.5 s hop). The output lists detected
events with approximate timestamps, per-type counts, % disfluent windows and a heuristic severity band.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch

from stutter.audio import fix_length, load_audio
from stutter.datasets import AUDIO_EXTENSIONS
from stutter.engine import load_model, resolve_device
from stutter.labels import LABELS
from stutter.severity import summarize


@torch.no_grad()
def analyze(path: Path, model, cfg, thresholds: np.ndarray, device: torch.device) -> dict:
    sr = cfg.data.sample_rate
    wave = load_audio(str(path), sr)
    duration = len(wave) / sr
    win, hop = int(cfg.infer.window_seconds * sr), int(cfg.infer.hop_seconds * sr)
    offsets = list(range(0, max(len(wave) - win, 0) + 1, hop))
    if offsets[-1] + win < len(wave):  # cover the tail
        offsets.append(len(wave) - win)

    probs, sev = [], []
    for i in range(0, len(offsets), cfg.infer.batch_size):
        chunk = offsets[i : i + cfg.infer.batch_size]
        segs = [wave[o : o + win] for o in chunk]
        lengths = torch.tensor([max(len(s), 1) for s in segs])
        batch = torch.from_numpy(np.stack([fix_length(s, win) for s in segs])).to(device)
        mask = (torch.arange(win)[None, :] < lengths[:, None]).long().to(device)
        out = model(wave=batch, attention_mask=mask)
        probs.append(torch.sigmoid(out["logits"].float()).cpu().numpy())
        sev.append(out["severity"].float().cpu().numpy())
    probs, sev = np.concatenate(probs), np.concatenate(sev)
    starts = np.array(offsets) / sr
    ends = np.minimum(starts + cfg.infer.window_seconds, duration)

    result = {"file": str(path), **summarize(starts, ends, probs, sev, thresholds, cfg.infer, duration)}
    result["windows"] = [
        {"start_s": round(float(s), 2), "end_s": round(float(e), 2), "severity": round(float(v), 3),
         **{name: round(float(p[k]), 3) for k, name in enumerate(LABELS)}}
        for s, e, p, v in zip(starts, ends, probs, sev)
    ]
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help=".wav files or folders")
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--device", default="auto")
    p.add_argument("--window-seconds", type=float, default=None)
    p.add_argument("--hop-seconds", type=float, default=None)
    p.add_argument("--no-windows", action="store_true", help="Omit per-window probabilities from output")
    p.add_argument("--out", default=None, help="Write JSON to this file instead of stdout")
    args = p.parse_args()

    device = resolve_device(args.device)
    model, cfg, thresholds = load_model(args.checkpoint, device)
    if args.window_seconds:
        cfg.infer.window_seconds = args.window_seconds
    if args.hop_seconds:
        cfg.infer.hop_seconds = args.hop_seconds

    files = []
    for item in map(Path, args.inputs):
        files.extend(sorted(f for f in item.rglob("*") if f.suffix.lower() in AUDIO_EXTENSIONS) if item.is_dir() else [item])
    results = []
    for f in files:
        r = analyze(f, model, cfg, thresholds, device)
        if args.no_windows:
            r.pop("windows")
        results.append(r)

    text = json.dumps(results[0] if len(results) == 1 else results, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Wrote {len(results)} result(s) -> {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
