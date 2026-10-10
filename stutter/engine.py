"""Training / evaluation loops, checkpointing and model loading."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CONFIG_MAPPING, PretrainedConfig

from .config import Config
from .datasets import ClipDataset, collate, positive_rates, read_manifest, seed_everything, split_manifest
from .embeddings import StatsDataset, load_or_compute_stats
from .labels import LABELS
from .metrics import compute_metrics, format_metrics, tune_thresholds
from .model import StutterModel


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def multitask_loss(out: dict, batch: dict, pos_weight: torch.Tensor, sev_weight: float) -> tuple[torch.Tensor, dict]:
    has = batch["has_labels"]
    if has.any():
        cls = F.binary_cross_entropy_with_logits(out["logits"][has].float(), batch["labels"][has], pos_weight=pos_weight)
        sev = F.smooth_l1_loss(out["severity"][has].float(), batch["severity"][has], beta=0.1)
    else:
        cls = sev = out["logits"].sum() * 0.0
    return cls + sev_weight * sev, {"cls": cls.detach().item(), "sev": sev.detach().item()}


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def model_inputs(batch: dict) -> dict:
    if "stats" in batch:
        return {"stats": batch["stats"]}
    return {"wave": batch["wave"], "attention_mask": batch["attention_mask"]}


@torch.no_grad()
def predict(model: StutterModel, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    probs, targets, sev_pred, sev_true, has = [], [], [], [], []
    for batch in loader:
        batch = to_device(batch, device)
        out = model(**model_inputs(batch))
        probs.append(torch.sigmoid(out["logits"].float()).cpu())
        sev_pred.append(out["severity"].float().cpu())
        targets.append(batch["labels"].cpu())
        sev_true.append(batch["severity"].cpu())
        has.append(batch["has_labels"].cpu())
    cat = lambda xs: torch.cat(xs).numpy() if xs else np.zeros((0,))
    return {"probs": cat(probs), "targets": cat(targets), "sev_pred": cat(sev_pred), "sev_true": cat(sev_true), "has": cat(has).astype(bool)}


def evaluate_predictions(pred: dict, thresholds: np.ndarray | None) -> dict:
    h = pred["has"]
    if not h.any():
        return {"num_samples": 0}
    return compute_metrics(pred["probs"][h], pred["targets"][h], pred["sev_pred"][h], pred["sev_true"][h], thresholds)


def build_optimizer(model: StutterModel, cfg: Config, total_steps: int):
    head = [p for p in model.head.parameters() if p.requires_grad]
    backbone = [p for p in model.backbone.parameters() if p.requires_grad]
    groups = [{"params": head, "lr": cfg.train.lr_head}]
    if backbone:
        groups.append({"params": backbone, "lr": cfg.train.lr_backbone})
    opt = torch.optim.AdamW(groups, weight_decay=cfg.train.weight_decay)
    warmup = max(1, int(cfg.train.warmup_ratio * total_steps))

    def schedule(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return opt, torch.optim.lr_scheduler.LambdaLR(opt, schedule)


# --------------------------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------------------------


def save_checkpoint(path: Path, model: StutterModel, cfg: Config, thresholds: np.ndarray, metrics: dict, random_init: bool) -> None:
    state = {f"head.{k}": v for k, v in model.head.state_dict().items()}
    if model.backbone_trainable or random_init:
        state.update({f"backbone.{k}": v for k, v in model.backbone.state_dict().items()})
    torch.save({
        "config": cfg.to_dict(),
        "labels": LABELS,
        "state_dict": state,
        "thresholds": [float(t) for t in thresholds],
        "metrics": metrics,
        "normalize_input": model.normalize_input,
        "backbone_config": model.backbone.config.to_dict() if random_init else None,
    }, path)


def load_model(path: str | Path, device: torch.device | str = "cpu") -> tuple[StutterModel, Config, np.ndarray]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Config.from_dict(ckpt["config"])
    bcfg = None
    if ckpt.get("backbone_config"):
        d = ckpt["backbone_config"]
        bcfg = CONFIG_MAPPING[d["model_type"]].from_dict(d)
    model = StutterModel(cfg.model, num_labels=len(ckpt["labels"]), backbone_config=bcfg)
    model.normalize_input = ckpt.get("normalize_input", model.normalize_input)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    missing = [k for k in missing if k.startswith("head.")]
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch. missing={missing} unexpected={unexpected}")
    model.to(device).eval()
    return model, cfg, np.asarray(ckpt["thresholds"], dtype=np.float32)


# --------------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------------


_AUDIO_MAGIC = [(b"RIFF", ".wav"), (b"fLaC", ".flac"), (b"OggS", ".ogg"), (b"ID3", ".mp3"), (b"\xff\xfb", ".mp3")]


def _audio_suffix(raw: bytes, fallback: str = ".wav") -> str:
    for magic, ext in _AUDIO_MAGIC:
        if raw[:len(magic)] == magic:
            return ext
    return fallback


def _ensure_decodable(path: Path) -> bool:
    """sf.info can pass on a file whose body still fails to decode; verify a real read."""
    import soundfile as sf
    try:
        info = sf.info(path)
        wave, _ = sf.read(path, frames=min(info.frames, 8000), dtype="float32")
        return len(wave) > 0
    except Exception:
        return False


def manifest_from_hf_dataset(repo_id: str, tmp_root: Path = Path(".hf_manifest")) -> Path:
    """Download a Hugging Face audio dataset and write a local manifest CSV the trainer understands.

    Audio is loaded with ``decode=False`` so raw file bytes are written to disk without a
    decoder dependency; the extension is sniffed from magic bytes, and each file is validated
    with soundfile so undecodable rows are excluded from the manifest.
    """
    import soundfile as sf
    from datasets import Audio, DatasetDict, load_dataset

    dset = load_dataset(repo_id)
    if not isinstance(dset, DatasetDict):
        dset = DatasetDict({"train": dset})

    out_dir = tmp_root / repo_id.replace("/", "_")
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    rows, bad = [], 0
    total_rows = sum(len(s) for s in dset.values())
    done = 0
    for split_name, split_ds in dset.items():
        if "audio" in split_ds.column_names:
            split_ds = split_ds.cast_column("audio", Audio(decode=False))
        for i, ex in enumerate(split_ds):
            done += 1
            if done % 2000 == 0 or done == total_rows:
                print(f"  extracting audio {done}/{total_rows} ({100*done/total_rows:.0f}%)", flush=True)
            audio = ex.get("audio") or {}
            raw, src = audio.get("bytes"), audio.get("path")
            if raw:
                raw = bytes(raw)
                src_path = audio_dir / f"{split_name}_{i}{_audio_suffix(raw)}"
                if not src_path.exists():
                    src_path.write_bytes(raw)
            elif src and Path(src).exists():
                src_path = Path(src)
            else:
                bad += 1
                continue
            if not _ensure_decodable(src_path):
                bad += 1
                src_path.unlink(missing_ok=True)
                continue
            row = dict(ex)
            row.pop("audio", None)
            row["path"] = str(src_path.resolve())  # absolute: read_manifest re-roots relative paths
            row["split"] = split_name
            rows.append(row)

    df = pd.DataFrame(rows)
    manifest_path = out_dir / "manifest.csv"
    df.to_csv(manifest_path, index=False)
    print(f"Wrote Hugging Face dataset manifest: {manifest_path} ({len(df)} rows, {bad} unusable audio files skipped)")
    return manifest_path


def build_datasets(cfg: Config, model: StutterModel, device: torch.device, use_cache: bool) -> dict:
    if cfg.data.manifest is None or str(cfg.data.manifest).startswith("hf://"):
        repo_id = str(cfg.data.manifest).replace("hf://", "") if cfg.data.manifest else None
        if not repo_id:
            raise ValueError("Set data.manifest to a local path or hf://owner/repo")
        cfg.data.manifest = str(manifest_from_hf_dataset(repo_id))
    df = read_manifest(cfg.data.manifest, cfg.data)
    splits = split_manifest(df, cfg.train.max_samples, cfg.train.seed)
    if len(splits["val"]) == 0:  # small corpora: carve a validation set out of train
        train = splits["train"].sample(frac=1.0, random_state=cfg.train.seed)
        n_val = max(1, len(train) // 10)
        splits["val"], splits["train"] = train.iloc[:n_val].reset_index(drop=True), train.iloc[n_val:].reset_index(drop=True)
    print("Split sizes: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))

    augment = cfg.train.augment and not use_cache
    clips = {
        name: ClipDataset(part, cfg.data, augment=augment and name == "train", random_crop=augment and name == "train")
        for name, part in splits.items()
    }
    if not use_cache:
        return clips
    return {name: StatsDataset(load_or_compute_stats(model, ds, cfg, name, device), ds) for name, ds in clips.items()}


def train(cfg: Config, backbone_config: PretrainedConfig | None = None) -> dict:
    seed_everything(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    out_dir = Path(cfg.train.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = StutterModel(cfg.model, backbone_config=backbone_config).to(device)
    use_cache = cfg.train.cache_embeddings and not model.backbone_trainable
    if use_cache and cfg.model.pooling != "mean_std":
        print("Embedding cache requires pooling=mean_std; training end-to-end instead.")
        use_cache = False
    print(f"Device={device} | backbone={cfg.model.backbone} | trainable backbone={model.backbone_trainable} | cached features={use_cache}")

    data = build_datasets(cfg, model, device, use_cache)
    if use_cache:
        model.backbone.to("cpu")  # not needed while training the head on cached features
    rates = positive_rates(data["train"].source if use_cache else data["train"])
    pos_weight = torch.tensor(np.clip((1 - rates) / np.clip(rates, 1e-3, None), 1.0, cfg.train.max_pos_weight), dtype=torch.float32, device=device)
    print("Train positive rates: " + ", ".join(f"{n}={r:.3f}" for n, r in zip(LABELS, rates)))

    loader_kw = {"batch_size": cfg.train.batch_size, "collate_fn": collate, "num_workers": cfg.train.num_workers}
    loaders = {
        "train": DataLoader(data["train"], shuffle=True, drop_last=len(data["train"]) > cfg.train.batch_size, **loader_kw),
        "val": DataLoader(data["val"], shuffle=False, **loader_kw),
        "test": DataLoader(data["test"], shuffle=False, **loader_kw),
    }
    steps_per_epoch = max(1, math.ceil(len(loaders["train"]) / cfg.train.grad_accum_steps))
    optimizer, scheduler = build_optimizer(model, cfg, steps_per_epoch * cfg.train.epochs)
    use_amp = cfg.train.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    random_init = backbone_config is not None
    best_path = out_dir / "best.pt"
    best_score, best_epoch, history = -1.0, 0, []

    total_steps = len(loaders["train"])
    run_t0 = time.time()
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loaders["train"], 1):
            batch = to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(**model_inputs(batch))
            loss, parts = multitask_loss(out, batch, pos_weight, cfg.train.severity_loss_weight)
            scaler.scale(loss / cfg.train.grad_accum_steps).backward()
            if step % cfg.train.grad_accum_steps == 0 or step == len(loaders["train"]):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            losses.append(loss.detach().item())
            if step % 25 == 0 or step == total_steps:
                elapsed = time.time() - t0
                sps = step / elapsed
                eta_e = (total_steps - step) / max(sps, 1e-9)
                done_frac = ((epoch - 1) + step / total_steps) / cfg.train.epochs
                eta_total = (time.time() - run_t0) / max(done_frac, 1e-9) - (time.time() - run_t0)
                print(f"  ep {epoch}/{cfg.train.epochs} step {step}/{total_steps} "
                      f"({100*step/total_steps:5.1f}%) loss={np.mean(losses[-25:]):.4f} "
                      f"| {sps:.1f} it/s | epoch ETA {eta_e/60:.1f}m | run ETA {eta_total/60:.0f}m", flush=True)

        val_pred = predict(model, loaders["val"], device)
        h = val_pred["has"]
        thresholds = tune_thresholds(val_pred["probs"][h], val_pred["targets"][h]) if h.any() else np.full(len(LABELS), 0.5)
        val_metrics = evaluate_predictions(val_pred, thresholds)
        score = val_metrics.get("macro_f1", 0.0)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else 0.0, **{k: v for k, v in val_metrics.items() if k != "per_class"}})
        print(f"Epoch {epoch:3d} | loss={history[-1]['train_loss']:.4f} | val {format_metrics(val_metrics)} | {time.time() - t0:.1f}s")

        if score > best_score:
            best_score, best_epoch = score, epoch
            save_checkpoint(best_path, model, cfg, thresholds, val_metrics, random_init)
        elif epoch - best_epoch >= cfg.train.patience:
            print(f"Early stopping (no val macro-F1 improvement for {cfg.train.patience} epochs)")
            break

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    model.head.load_state_dict({k[5:]: v for k, v in best["state_dict"].items() if k.startswith("head.")})
    if any(k.startswith("backbone.") for k in best["state_dict"]):
        model.backbone.load_state_dict({k[9:]: v for k, v in best["state_dict"].items() if k.startswith("backbone.")})
    model.to(device)
    thresholds = np.asarray(best["thresholds"])
    test_metrics = evaluate_predictions(predict(model, loaders["test"], device), thresholds)
    print(f"Best epoch {best_epoch} | test {format_metrics(test_metrics)}")

    report = {"best_epoch": best_epoch, "val": best["metrics"], "test": test_metrics, "history": history}
    with open(out_dir / "train_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report
