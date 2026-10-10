"""Typed configuration loaded from YAML with optional dotted CLI overrides."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    manifest: str | None = "data/sep28k/manifest.csv"  # local CSV or hf://owner/repo
    sample_rate: int = 16000
    clip_seconds: float = 3.0
    label_threshold: int = 2  # annotator count (out of 3) needed for a positive label
    soft_labels: bool = False  # use count/3 as BCE target instead of binary labels
    drop_unusable: bool = True  # drop clips flagged NoSpeech / Music / PoorAudioQuality by majority
    severity_weights: dict[str, float] = field(
        default_factory=lambda: {
            "Block": 1.0,
            "Prolongation": 0.8,
            "SoundRep": 0.8,
            "WordRep": 0.6,
            "Interjection": 0.3,
        }
    )


@dataclass
class ModelConfig:
    backbone: str = "facebook/wav2vec2-base"
    pooling: str = "mean_std"  # "mean_std" (cacheable) or "attentive"
    freeze_backbone: bool = True
    freeze_feature_encoder: bool = True
    unfreeze_top_n_layers: int = 0  # >0 fine-tunes only the top N transformer layers
    proj_dim: int = 256
    dropout: float = 0.2


@dataclass
class TrainConfig:
    output_dir: str = "checkpoints"
    cache_dir: str = "cache"
    cache_embeddings: bool = True  # only used when the backbone is fully frozen
    epochs: int = 20
    batch_size: int = 32
    lr_head: float = 1e-3
    lr_backbone: float = 2e-5
    weight_decay: float = 1e-2
    warmup_ratio: float = 0.1
    severity_loss_weight: float = 1.0
    max_pos_weight: float = 10.0
    grad_accum_steps: int = 1
    patience: int = 5
    amp: bool = True
    augment: bool = True
    num_workers: int = 0
    max_samples: int = 0  # 0 = use everything; otherwise cap train size (val/test capped at 1/4)
    seed: int = 42
    device: str = "auto"


@dataclass
class InferConfig:
    window_seconds: float = 3.0
    hop_seconds: float = 1.5
    batch_size: int = 16
    # Cut-offs on the 0-100 composite score -> heuristic severity band (NOT a clinical SSI-4 score).
    band_cutoffs: list[float] = field(default_factory=lambda: [5.0, 15.0, 30.0, 50.0, 70.0])
    band_names: list[str] = field(
        default_factory=lambda: ["fluent", "very mild", "mild", "moderate", "severe", "very severe"]
    )


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferConfig = field(default_factory=InferConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "Config":
        d = d or {}
        sections = {f.name: f.type for f in dataclasses.fields(cls)}
        kwargs = {}
        for name, sub_cls in (("data", DataConfig), ("model", ModelConfig), ("train", TrainConfig), ("infer", InferConfig)):
            known = {f.name for f in dataclasses.fields(sub_cls)}
            values = d.get(name) or {}
            unknown = set(values) - known
            if unknown:
                raise ValueError(f"Unknown keys in config section '{name}': {sorted(unknown)}")
            kwargs[name] = sub_cls(**values)
        unknown_sections = set(d) - set(sections)
        if unknown_sections:
            raise ValueError(f"Unknown config sections: {sorted(unknown_sections)}")
        return cls(**kwargs)


def _parse_value(raw: str) -> Any:
    return yaml.safe_load(raw)


def apply_overrides(cfg: Config, overrides: list[str] | None) -> Config:
    """Apply overrides such as ``train.epochs=3`` or ``model.backbone=facebook/hubert-base-ls960``."""
    for item in overrides or []:
        key, _, raw = item.partition("=")
        section, _, attr = key.strip().partition(".")
        target = getattr(cfg, section, None)
        if target is None or not hasattr(target, attr):
            raise ValueError(f"Invalid override key: {key}")
        setattr(target, attr, _parse_value(raw))
    return cfg


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    data = {}
    if path:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    return apply_overrides(Config.from_dict(data), overrides)
