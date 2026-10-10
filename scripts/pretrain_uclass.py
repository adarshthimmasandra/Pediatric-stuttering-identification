"""Domain-adaptive pretraining of wav2vec2 on UCLASS child speech (self-supervised, no labels).

Continues the masked-span + contrastive pretraining objective on UCLASS audio so the backbone
adapts from adult podcast speech to child speech. The result is saved as an HF-format checkpoint
that can replace ``model.backbone`` for the subsequent SEP-28k fine-tune:

    python scripts/pretrain_uclass.py \
        --uclass-dir /root/uclass \
        --out /root/uclass_wav2vec2_adapted \
        --epochs 3

    python scripts/train.py \
        --config configs/runpod.yaml \
        --hf-dataset adarshcpt/pediatric-stutter-sep28k \
        --output-dir /workspace/checkpoints/finetune_adapted \
        --set model.backbone=/root/uclass_wav2vec2_adapted
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2ForPreTraining, get_linear_schedule_with_warmup

from stutter.audio import load_audio

SAMPLE_RATE = 16000


class UclassAudioDataset(Dataset):
    """Random ``crop_seconds`` crops from full UCLASS recordings (unlabelled pretraining)."""

    def __init__(self, uclass_dir: Path, crop_seconds: float = 10.0, samples_per_epoch: int = 4000):
        df = pd.read_csv(uclass_dir / "manifest.csv").drop_duplicates("path")
        self.paths = [str(uclass_dir / p) for p in df["path"]]
        self.crop = int(crop_seconds * SAMPLE_RATE)
        self.samples_per_epoch = samples_per_epoch

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, _: int) -> torch.Tensor:
        for _ in range(5):
            try:
                wave = load_audio(random.choice(self.paths), SAMPLE_RATE)
            except Exception:
                continue
            if len(wave) >= self.crop:
                off = random.randint(0, len(wave) - self.crop)
                seg = wave[off : off + self.crop]
            else:
                seg = np.pad(wave, (0, self.crop - len(wave)))
            return torch.from_numpy(np.asarray(seg, dtype=np.float32))
        return torch.zeros(self.crop, dtype=torch.float32)


def collate(batch: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(batch)


def span_mask(batch: int, seq_len: int, prob: float, span: int, device) -> torch.BoolTensor:
    """Sample non-overlapping masked spans like the wav2vec2 objective."""
    mask = torch.zeros(batch, seq_len, dtype=torch.bool)
    num_spans = max(1, int(prob * seq_len / span))
    for b in range(batch):
        starts = torch.randperm(seq_len - span)[:num_spans]
        for s in starts:
            mask[b, s : s + span] = True
    return mask.to(device)


def wav2vec2_pretrain_loss(model, out, mask: torch.BoolTensor) -> torch.Tensor:
    """Manual wav2vec2 pretraining loss for transformers versions that don't return `loss`.

    Contrastive: at masked positions, pull the context projection toward the correct quantized
    latent vs negatives sampled from the same batch. Diversity: penalize codebook under-use via
    codevector_perplexity.
    """
    cfg = model.config
    ctx = out.projected_states[mask]            # (N, d_proj) context reps at masked positions
    q = out.projected_quantized_states[mask]    # (N, d_proj) quantized targets
    n = ctx.shape[0]
    neg = getattr(cfg, "num_negatives", 100)
    neg_idx = torch.randint(0, n, (n, neg), device=q.device)
    targets = torch.cat([q.unsqueeze(1), q[neg_idx]], dim=1)          # (N, 1+neg, d)
    logits = F.cosine_similarity(ctx.unsqueeze(1).float(), targets.float(), dim=-1)
    logits = logits / getattr(cfg, "logit_temp", 0.1)
    contrastive = F.cross_entropy(logits, torch.zeros(n, dtype=torch.long, device=q.device))
    # diversity: maximize codevector perplexity toward uniform codebook usage
    ppl = out.codevector_perplexity
    n_codes = getattr(cfg, "num_codevector_groups", 2) * getattr(cfg, "num_codevectors_per_group", 320)
    diversity = (n_codes - ppl) / n_codes
    return contrastive + getattr(cfg, "diversity_loss_weight", 0.1) * diversity


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--uclass-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--backbone", default="facebook/wav2vec2-base")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8, help="10 s crops; 8 fits 24 GB")
    p.add_argument("--crop-seconds", type=float, default=10.0)
    p.add_argument("--samples-per-epoch", type=int, default=4000)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--mask-prob", type=float, default=0.065)
    p.add_argument("--mask-span", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = UclassAudioDataset(args.uclass_dir, args.crop_seconds, args.samples_per_epoch)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate, drop_last=True)
    print(f"{len(ds.paths)} recordings | {args.samples_per_epoch} crops/epoch of {args.crop_seconds}s")

    model = Wav2Vec2ForPreTraining.from_pretrained(args.backbone).to(device)
    for prm in model.wav2vec2.feature_extractor.parameters():  # keep acoustic frontend frozen
        prm.requires_grad = False
    model.train()

    steps = math.ceil(len(loader)) * args.epochs
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    t0, history = time.time(), []
    for epoch in range(1, args.epochs + 1):
        losses = []
        for step, wave in enumerate(loader, 1):
            wave = wave.to(device, non_blocking=True)
            # input normalization the wav2vec2 feature extractor expects
            wave = (wave - wave.mean(dim=1, keepdim=True)) / (wave.std(dim=1, keepdim=True) + 1e-7)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                feat_len = model.wav2vec2._get_feat_extract_output_lengths(torch.tensor([wave.shape[1]])).item()
                mask = span_mask(wave.shape[0], feat_len, args.mask_prob, args.mask_span, device)
                out = model(input_values=wave, mask_time_indices=mask)
                # Transformers versions differ: some return `loss`, some expose the parts, the
                # newest only return the projections - so compute the objective manually.
                loss = getattr(out, "loss", None)
                if not torch.is_tensor(loss):
                    loss = getattr(out, "contrastive_loss", None)
                    if not torch.is_tensor(loss):
                        loss = wav2vec2_pretrain_loss(model, out, mask)
                    else:
                        div = getattr(out, "diversity_loss", None)
                        if torch.is_tensor(div):
                            loss = loss + getattr(model.config, "diversity_loss_weight", 0.1) * div
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            losses.append(loss.detach().item())
            if step % 25 == 0 or step == len(loader):
                sps = step / (time.time() - t0)
                print(f"  ep {epoch}/{args.epochs} step {step}/{len(loader)} ({100*step/len(loader):.0f}%)"
                      f" contrastive={np.mean(losses[-25:]):.4f} | {sps:.1f} it/s", flush=True)
        history.append({"epoch": epoch, "loss": float(np.mean(losses))})
        print(f"Epoch {epoch} | contrastive+diversity loss={history[-1]['loss']:.4f}")

    # Save as an HF-format dir that AutoModel/Wav2Vec2Model.from_pretrained can load as backbone
    args.out.mkdir(parents=True, exist_ok=True)
    model.wav2vec2.save_pretrained(args.out)
    (args.out / "pretrain_info.json").write_text(json.dumps({
        "base": args.backbone, "objective": "wav2vec2 masked-span contrastive + diversity",
        "epochs": args.epochs, "crop_seconds": args.crop_seconds, "history": history,
        "note": "domain-adapted on UCLASS child speech; use as model.backbone for fine-tuning",
    }, indent=2), encoding="utf-8")
    print(f"Adapted backbone saved -> {args.out}  (use --set model.backbone={args.out})")


if __name__ == "__main__":
    main()
