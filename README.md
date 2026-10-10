# Pediatric Stuttering Identification & Severity Scoring

PyTorch + Hugging Face pipeline that fine-tunes a self-supervised speech backbone (Wav2Vec 2.0 /
HuBERT / WavLM) with a **multi-task head**:

* **Multi-label classification** of 5 disfluency types: `Block`, `Prolongation`, `SoundRep`,
  `WordRep`, `Interjection` (SEP-28k label set)
* **Regression** of a 0-1 severity score

Inference on a raw `.wav` returns JSON with event timestamps, per-type counts, % disfluent
windows and a heuristic severity band. See `Child_Stuttering_Speech_Identification___AI_Project_Roadmap.md`.

> Research prototype, not a medical device. SEP-28k is mostly adult podcast speech; validate on
> child data (UCLASS / FluencyBank / your own) before drawing conclusions.

## Layout

```
configs/default.yaml        CPU-friendly: frozen backbone, cached features, head-only training
configs/gpu_finetune.yaml   end-to-end fine-tuning on a CUDA GPU
scripts/fetch_sep28k.py     download SEP-28k (+ FluencyBank annotations) -> clips + manifest.csv
scripts/fetch_uclass.py     download UCLASS Release 1 + 2 (audio, transcripts, annotations) + manifest
colab/stutter_colab.ipynb   Colab workflow: download everything to Google Drive, train on GPU
scripts/make_colab_bundle.py  zip the code for upload to Drive
scripts/prepare_local_corpus.py  manifest for manually downloaded UCLASS / FluencyBank audio
scripts/train.py            training loop (F1 + MAE on val, early stopping, test report)
scripts/evaluate.py         evaluate a checkpoint on any manifest/split
scripts/infer.py            raw .wav -> JSON
stutter/                    package: config, labels, audio, datasets, model, embeddings, metrics, engine, severity
tests/test_smoke.py         offline tests (synthetic audio, tiny random backbone)
```

## 1. Setup (Windows PowerShell)

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_env.ps1            # CPU wheels
powershell -ExecutionPolicy Bypass -File .\setup_env.ps1 -Cuda cu128 # NVIDIA GPU
.\.venv\Scripts\Activate.ps1
python -m pytest tests -q
```

`ffmpeg` on PATH is only needed for the official SEP-28k route and for mp3/m4a inputs.

## Google Colab + Drive (recommended for downloading all data and GPU training)

1. `python scripts/make_colab_bundle.py` creates `stutter_project_code.zip` (re-run after code changes).
2. Upload it to Google Drive as `My Drive/StutterProject/stutter_project_code.zip`.
3. Open `colab/stutter_colab.ipynb` in Colab (drag it into colab.research.google.com -> Upload),
   select a T4 GPU runtime and run the cells in order. Approve the Drive login popup in cell 1.

The notebook downloads SEP-28k/FluencyBank (HF mirror), the official Apple podcast route (only the
clips the mirror lacks) and all of UCLASS Release 1 + 2 **directly to Drive**. It stores them as a few
large files (parquet shards, zip bundles, tar archives) because Drive is slow with thousands of small
files. Later sessions restore everything to the Colab disk in a minute or two (section B) and write
checkpoints and results back to Drive.

## 2. Public data: SEP-28k

```powershell
# Hugging Face mirror (pre-clipped, 7 shards ~2.9 GB, ~4.5k clips per shard)
python scripts/fetch_sep28k.py --max-shards 1     # quick subset
python scripts/fetch_sep28k.py                    # everything (SEP-28k + FluencyBank clips)

# Official Apple route: label CSVs from GitHub + original podcast audio (~32 GB raw, some dead links)
python scripts/fetch_sep28k.py --source official --labels-only
python scripts/fetch_sep28k.py --source official --datasets sep28k --max-episodes 20
```

Both write `data/sep28k/clips/<Show>/<Show>_<EpId>_<ClipId>.wav` and `data/sep28k/manifest.csv`.
Splits are assigned per **episode** (`--split-by show` for stricter speaker separation).

### Manifest format (all data sources)

| column | meaning |
| --- | --- |
| `path` | audio file (absolute or relative to the manifest) |
| `split`, `group` | train/val/test; group = speaker/episode (split derived from group if missing) |
| `start`, `end` | optional segment in seconds |
| `Block` ... `Interjection` | annotator counts 0-3 (positive if >= `data.label_threshold`) or 0/1; empty = unlabelled |
| `severity` | optional clinician score in [0,1] (e.g. SSI-4 / max); otherwise a proxy is used |

**Severity proxy**: SEP-28k has no clinical severity, so the target is
`clip(sum_k w_k * count_k/3 / max(w), 0, 1)` with weights from `data.severity_weights`.

## 3. Child corpora (UCLASS, FluencyBank)

**UCLASS** (Release 1 + 2) is openly downloadable and fetched automatically: 455 recordings from 120
speakers, of which 425 are under 18. It covers monologue, reading and conversation. Speaker sex, id and
age are parsed from the file names, and splits are speaker-disjoint. The recordings are unlabelled
3 s windows (useful for inference and child-speech evaluation). The few time-aligned CHAT/TextGrid
annotations are saved under `data/uclass/text/`. Cite Howell et al. (2009) when you use it.

```powershell
python scripts/fetch_uclass.py                     # wav audio + transcripts/annotations -> data/uclass
```

**FluencyBank** requires a TalkBank account (and approval for restricted sets). Download it manually,
place it under `data/fluencybank/`, then:

```powershell
# unlabelled -> 3 s windows (inference / annotation)
python scripts/prepare_local_corpus.py --audio-dir data/uclass
# your annotations (any column names, remapped)
python scripts/prepare_local_corpus.py --audio-dir data/uclass --annotations labels.csv `
    --column-map file=path blk=Block prol=Prolongation
# FluencyBank audio + Apple's FluencyBank clip annotations
python scripts/prepare_local_corpus.py --audio-dir data/fluencybank `
    --apple-labels data/sep28k/official/fluencybank_labels.csv `
    --apple-episodes data/sep28k/official/fluencybank_episodes.csv
python scripts/train.py --manifest data/fluencybank/manifest.csv
```

## 4. Training

```powershell
# CPU: backbone features are extracted once into cache/, then only the head trains (fast epochs)
python scripts/train.py --max-samples 2000 --epochs 15
python scripts/train.py                                   # full manifest
python scripts/train.py --backbone facebook/hubert-base-ls960

# GPU: end-to-end fine-tuning with attentive pooling, AMP, augmentation
python scripts/train.py --config configs/gpu_finetune.yaml
python scripts/train.py --set model.unfreeze_top_n_layers=4 model.freeze_backbone=false train.cache_embeddings=false
```

Loss = BCE-with-logits (class-balanced `pos_weight`) + `severity_loss_weight` * SmoothL1.
Per-class thresholds are tuned on validation; model selection uses validation macro-F1.
Outputs: `checkpoints/best.pt` and `checkpoints/train_report.json` (per-class F1/P/R, macro/micro
F1, any-stutter F1, severity MAE/RMSE/Pearson).

Feature extraction is the only slow step on CPU: wav2vec2-base runs at ~1 clip/s on a dual-core
laptop (Intel HD 620 class machine), i.e. ~25 min for `--max-samples 1000` (1000 train + 250 val +
250 test) and ~8 h for all of SEP-28k. It runs once and is cached in `cache/`; head-training epochs
then take seconds. For the full dataset or end-to-end fine-tuning, use a GPU (e.g. Colab).

## 5. Evaluation & inference

```powershell
python scripts/evaluate.py --checkpoint checkpoints/best.pt --split test
python scripts/infer.py path\to\child.wav --no-windows
python scripts/infer.py recordings\ --out results.json
```

Example output (abridged):

```json
{
  "file": "child.wav", "duration_s": 10.0, "num_windows": 6,
  "events": [{"type": "Block", "start_s": 1.5, "end_s": 6.0, "confidence": 0.81}],
  "counts": {"Block": 1, "Prolongation": 0, "SoundRep": 0, "WordRep": 0, "Interjection": 0},
  "percent_disfluent_windows": 50.0, "mean_window_severity": 0.31,
  "composite_severity_score": 40.5, "severity_band": "moderate"
}
```

Timestamps have window resolution (3 s window, 1.5 s hop; tune with `--window-seconds/--hop-seconds`).
`percent_disfluent_windows` is a proxy for %SS and the band is a heuristic SSI-4-style mapping
(`infer.band_cutoffs`); neither replaces a clinical assessment.
