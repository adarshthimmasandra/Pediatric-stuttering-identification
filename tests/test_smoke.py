"""Offline smoke tests: synthetic audio + a tiny randomly initialised wav2vec2 (no downloads)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from stutter.config import Config  # noqa: E402
from stutter.datasets import ClipDataset, build_local_manifest, collate, read_manifest  # noqa: E402
from stutter.engine import load_model, train  # noqa: E402
from stutter.labels import LABELS, binary_targets, severity_proxy  # noqa: E402
from stutter.metrics import compute_metrics, tune_thresholds  # noqa: E402
from stutter.model import tiny_backbone_config  # noqa: E402

SR = 16000


def make_corpus(root: Path, n: int = 24) -> Path:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        t = np.arange(int(SR * rng.uniform(2.0, 3.5))) / SR
        wave = 0.3 * np.sin(2 * np.pi * rng.uniform(100, 400) * t) + 0.05 * rng.standard_normal(len(t))
        path = root / "clips" / f"spk{i % 6}" / f"clip{i}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, wave.astype(np.float32), SR)
        counts = rng.integers(0, 4, size=len(LABELS))
        rows.append({"path": path.relative_to(root).as_posix(), "group": f"spk{i % 6}",
                     "split": ["train", "train", "train", "train", "val", "test"][i % 6],
                     **dict(zip(LABELS, counts.tolist()))})
    manifest = root / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def tiny_cfg(tmp_path: Path, manifest: Path, **model_overrides) -> Config:
    cfg = Config()
    cfg.data.manifest = str(manifest)
    cfg.train.output_dir = str(tmp_path / "ckpt")
    cfg.train.cache_dir = str(tmp_path / "cache")
    cfg.train.epochs = 2
    cfg.train.batch_size = 4
    cfg.train.device = "cpu"
    cfg.model.proj_dim = 16
    for k, v in model_overrides.items():
        setattr(cfg.model, k, v)
    return cfg


def test_labels_and_severity():
    counts = np.array([[3, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]], dtype=np.float32)
    assert binary_targets(counts, 2).tolist() == [[1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    sev = severity_proxy(counts, Config().data.severity_weights)
    assert sev[0] == pytest.approx(1.0) and 0 < sev[1] < 0.2 and sev[2] == 0


def test_metrics():
    rng = np.random.default_rng(1)
    targets = (rng.random((50, 5)) > 0.7).astype(np.float32)
    probs = np.clip(targets * 0.6 + rng.random((50, 5)) * 0.4, 0, 1)
    th = tune_thresholds(probs, targets)
    m = compute_metrics(probs, targets, rng.random(50), rng.random(50), th)
    assert m["macro_f1"] > 0.8 and 0 <= m["severity_mae"] <= 1 and set(m["per_class"]) == set(LABELS)


def test_dataset_and_collate(tmp_path):
    manifest = make_corpus(tmp_path, n=6)
    ds = ClipDataset(read_manifest(manifest, Config().data), Config().data, augment=True)
    batch = collate([ds[i] for i in range(3)])
    assert batch["wave"].shape == (3, 3 * SR) and batch["labels"].shape == (3, 5)
    assert batch["attention_mask"].sum(dim=1).tolist() == batch["length"].tolist()


@pytest.mark.parametrize("mode", ["cached", "end_to_end"])
def test_train_and_infer(tmp_path, mode):
    import infer

    manifest = make_corpus(tmp_path)
    overrides = {} if mode == "cached" else {"freeze_backbone": False, "pooling": "attentive"}
    cfg = tiny_cfg(tmp_path, manifest, **overrides)
    report = train(cfg, backbone_config=tiny_backbone_config())
    assert "macro_f1" in report["test"] and "severity_mae" in report["test"]

    model, loaded_cfg, thresholds = load_model(Path(cfg.train.output_dir) / "best.pt")
    long_wav = tmp_path / "long.wav"
    sf.write(long_wav, (0.1 * np.random.randn(SR * 7)).astype(np.float32), SR)
    result = infer.analyze(long_wav, model, loaded_cfg, thresholds, model.backbone.device)
    assert result["num_windows"] == 4 and set(result["counts"]) == set(LABELS)
    assert result["severity_band"] in loaded_cfg.infer.band_names


def test_local_manifest(tmp_path):
    for spk in ("child1", "child2"):
        (tmp_path / spk).mkdir()
        sf.write(tmp_path / spk / "rec.wav", np.zeros(SR * 7, dtype=np.float32), SR)
    ann = pd.DataFrame([{"file": "child1/rec.wav", "blk": 3, "Prolongation": 0, "SoundRep": 1, "WordRep": 0,
                         "Interjection": 0, "start": 0.0, "end": 3.0}])
    df = build_local_manifest(tmp_path, ann, {"blk": "Block"})
    assert len(df) == 1 + 3  # one annotated segment + child2 split into 3 unlabelled windows
    assert df.loc[df["Block"].notna(), "Block"].tolist() == [3]
