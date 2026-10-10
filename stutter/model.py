"""Self-supervised speech backbone (Wav2Vec 2.0 / HuBERT / WavLM) + multi-task head.

Head outputs:
    logits   (B, num_labels)  multi-label disfluency logits (Block, Prolongation, SoundRep, WordRep, Interjection)
    severity (B,)             severity score in [0, 1]

With ``pooling="mean_std"`` the head only needs per-layer mean/std statistics, which can be
pre-computed once with a frozen backbone (see ``embeddings.py``) for fast CPU training. The same
checkpoint then runs end-to-end on raw audio at inference time.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoFeatureExtractor, AutoModel, PretrainedConfig

from .config import ModelConfig
from .labels import LABELS


class AttentiveStatsPool(nn.Module):
    def __init__(self, hidden: int, bottleneck: int = 128):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(hidden, bottleneck), nn.Tanh(), nn.Linear(bottleneck, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.attn(x).squeeze(-1).masked_fill(~mask, float("-inf"))
        w = torch.softmax(scores, dim=-1).unsqueeze(-1)
        mean = (w * x).sum(dim=1)
        std = ((w * (x - mean.unsqueeze(1)) ** 2).sum(dim=1).clamp_min(1e-6)).sqrt()
        return torch.cat([mean, std], dim=-1)


def masked_mean_std(hs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """hs: (L, B, T, H), mask: (B, T) -> (B, L, 2H)."""
    m = mask[None, :, :, None].to(hs.dtype)
    count = m.sum(dim=2).clamp_min(1.0)
    mean = (hs * m).sum(dim=2) / count
    var = (((hs - mean.unsqueeze(2)) ** 2) * m).sum(dim=2) / count
    stats = torch.cat([mean, var.clamp_min(1e-6).sqrt()], dim=-1)
    return stats.transpose(0, 1)


class MultiTaskHead(nn.Module):
    def __init__(self, num_layers: int, hidden: int, num_labels: int, proj_dim: int, dropout: float, pooling: str):
        super().__init__()
        self.pooling = pooling
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))
        if pooling == "attentive":
            self.pool = AttentiveStatsPool(hidden)
        elif pooling != "mean_std":
            raise ValueError(f"Unknown pooling: {pooling}")
        self.shared = nn.Sequential(
            nn.LayerNorm(2 * hidden),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(proj_dim, num_labels)
        self.regressor = nn.Linear(proj_dim, 1)

    def _outputs(self, pooled: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.shared(pooled)
        return {"logits": self.classifier(z), "severity": torch.sigmoid(self.regressor(z)).squeeze(-1)}

    def forward_stats(self, stats: torch.Tensor) -> dict[str, torch.Tensor]:
        """stats: (B, L, 2H) per-layer mean/std statistics."""
        w = torch.softmax(self.layer_weights, dim=0)
        return self._outputs((stats.float() * w[None, :, None]).sum(dim=1))

    def forward_hidden(self, hs: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """hs: (L, B, T, H) hidden states for all layers."""
        if self.pooling == "mean_std":
            return self.forward_stats(masked_mean_std(hs, mask))
        w = torch.softmax(self.layer_weights, dim=0)
        x = (hs * w[:, None, None, None]).sum(dim=0)
        return self._outputs(self.pool(x, mask))


class StutterModel(nn.Module):
    def __init__(self, cfg: ModelConfig, num_labels: int = len(LABELS), backbone_config: PretrainedConfig | None = None):
        super().__init__()
        self.cfg = cfg
        if backbone_config is not None:  # random init (tests / custom architectures)
            self.backbone = AutoModel.from_config(backbone_config)
            self.normalize_input = True
        else:
            self.backbone = AutoModel.from_pretrained(cfg.backbone)
            try:
                self.normalize_input = bool(getattr(AutoFeatureExtractor.from_pretrained(cfg.backbone), "do_normalize", True))
            except Exception:
                self.normalize_input = True
        bcfg = self.backbone.config
        # Models with group-norm feature extractors (e.g. wav2vec2-base) must not receive an attention mask.
        self.pass_attention_mask = getattr(bcfg, "feat_extract_norm", "layer") == "layer"
        self.head = MultiTaskHead(bcfg.num_hidden_layers + 1, bcfg.hidden_size, num_labels, cfg.proj_dim, cfg.dropout, cfg.pooling)
        self.apply_freezing()

    # -- freezing -------------------------------------------------------------------------------
    def apply_freezing(self) -> None:
        cfg = self.cfg
        for p in self.backbone.parameters():
            p.requires_grad = not cfg.freeze_backbone
        if cfg.unfreeze_top_n_layers > 0:
            for p in self.backbone.parameters():
                p.requires_grad = False
            for layer in self.backbone.encoder.layers[-cfg.unfreeze_top_n_layers :]:
                for p in layer.parameters():
                    p.requires_grad = True
        if cfg.freeze_feature_encoder and hasattr(self.backbone, "feature_extractor"):
            for p in self.backbone.feature_extractor.parameters():
                p.requires_grad = False
        if hasattr(self.backbone.config, "layerdrop"):
            self.backbone.config.layerdrop = 0.0  # keep all hidden states available

    @property
    def backbone_trainable(self) -> bool:
        return any(p.requires_grad for p in self.backbone.parameters())

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.backbone_trainable:
            self.backbone.eval()  # disable dropout in a frozen backbone
        return self

    # -- forward --------------------------------------------------------------------------------
    def _normalize(self, wave: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if not self.normalize_input:
            return wave
        m = attention_mask.to(wave.dtype)
        n = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (wave * m).sum(dim=1, keepdim=True) / n
        var = (((wave - mean) ** 2) * m).sum(dim=1, keepdim=True) / n
        return ((wave - mean) / (var + 1e-7).sqrt()) * m

    def hidden_states(self, wave: torch.Tensor, attention_mask: torch.Tensor | None = None):
        if attention_mask is None:
            attention_mask = torch.ones_like(wave, dtype=torch.long)
        wave = self._normalize(wave, attention_mask)
        out = self.backbone(
            wave,
            attention_mask=attention_mask if self.pass_attention_mask else None,
            output_hidden_states=True,
        )
        hs = torch.stack(out.hidden_states, dim=0)  # (L, B, T, H)
        lengths = self.backbone._get_feat_extract_output_lengths(attention_mask.sum(dim=1))
        frames = torch.arange(hs.shape[2], device=hs.device)[None, :]
        mask = frames < lengths[:, None].clamp(min=1, max=hs.shape[2])
        return hs, mask

    @torch.no_grad()
    def extract_stats(self, wave: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        hs, mask = self.hidden_states(wave, attention_mask)
        return masked_mean_std(hs, mask)

    def forward(self, wave=None, attention_mask=None, stats=None) -> dict[str, torch.Tensor]:
        if stats is not None:
            return self.head.forward_stats(stats)
        hs, mask = self.hidden_states(wave, attention_mask)
        return self.head.forward_hidden(hs, mask)


def tiny_backbone_config(model_type: str = "wav2vec2") -> PretrainedConfig:
    """Very small random-init config for offline smoke tests."""
    return AutoConfig.for_model(
        model_type,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        conv_dim=(16, 16, 16),
        conv_stride=(5, 4, 4),
        conv_kernel=(10, 8, 8),
        num_conv_pos_embeddings=16,
        num_conv_pos_embedding_groups=2,
        feat_extract_norm="layer",
        vocab_size=8,
    )
