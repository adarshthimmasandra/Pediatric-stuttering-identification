"""Generate colab/stutter_colab.ipynb from an explicit, ordered list of cells.

Each entry is (kind, source). Run: python scripts/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "colab" / "stutter_colab.ipynb"

SELF_HEAL = """import os, subprocess
PROJECT = '/content/project'
DRIVE_ROOT = os.environ.get('DRIVE_ROOT', '/content/drive/MyDrive/StutterProject')
LOCAL_DATA = os.environ.get('LOCAL_DATA', '/content/data')
# Push into os.environ so $VAR expansion works inside ! shell commands
os.environ.update(PROJECT=PROJECT, DRIVE_ROOT=DRIVE_ROOT, LOCAL_DATA=LOCAL_DATA,
                  RAW=os.environ.get('RAW', f'{DRIVE_ROOT}/raw'),
                  DATA_TARS=os.environ.get('DATA_TARS', f'{DRIVE_ROOT}/data'))
if not os.path.exists(f'{PROJECT}/{NEED}'):
    os.makedirs(PROJECT, exist_ok=True)
    subprocess.run(['unzip', '-q', '-o', f'{DRIVE_ROOT}/stutter_project_code.zip', '-d', PROJECT], check=True)
    print('Project code extracted from Drive zip')
os.chdir(PROJECT)
"""

cells = [
    ("md", """# Pediatric Stuttering Identification - Colab workflow

Downloads all public datasets **directly to Google Drive** (nothing passes through your laptop), then trains on a GPU.

**Before running**
1. Upload `stutter_project_code.zip` to `My Drive/StutterProject/` (or edit `DRIVE_ROOT` below).
2. `Runtime -> Change runtime type -> T4 GPU` (downloads work on CPU too).
3. Run cells top to bottom. The Drive login popup appears in cell 1 - sign in with your own Google account.

**What ends up on Drive (`My Drive/StutterProject/`)**

| Folder | Contents | Size |
|---|---|---|
| `raw/sep28k_parquet/` | SEP-28k + FluencyBank clips (HF mirror, 7 parquet shards) | ~2.9 GB |
| `raw/sep28k_official/` | Apple label CSVs + `failed_episodes.txt` | small |
| `raw/uclass_zips/` | UCLASS Release 1 + 2 audio (wav) + transcripts/annotations | ~2-4 GB |
| `data/*.tar` | ready-to-use extracted clips + manifests (fast restore) | ~4-6 GB |
| `checkpoints/`, `cache/`, `results/` | models, cached features, reports | varies |

Drive handles many small files badly, so clips are processed on the Colab disk (`/content`) and stored on Drive as a few large files.
Every download step is resumable: re-running a cell skips finished files."""),

    ("code", """# 1) Mount Google Drive and configure paths
from google.colab import drive
drive.mount('/content/drive')

import os, subprocess
DRIVE_ROOT = '/content/drive/MyDrive/StutterProject'
RAW = f'{DRIVE_ROOT}/raw'
DATA_TARS = f'{DRIVE_ROOT}/data'
LOCAL_DATA = '/content/data'
PROJECT = '/content/project'
for d in (RAW, DATA_TARS, f'{DRIVE_ROOT}/checkpoints', f'{DRIVE_ROOT}/cache', f'{DRIVE_ROOT}/results', LOCAL_DATA):
    os.makedirs(d, exist_ok=True)
os.environ.update(DRIVE_ROOT=DRIVE_ROOT, RAW=RAW, DATA_TARS=DATA_TARS, LOCAL_DATA=LOCAL_DATA, PROJECT=PROJECT)
!nvidia-smi --query-gpu=name,memory.total --format=csv || echo "No GPU - fine for downloading, slow for training"
!df -h /content | tail -1"""),

    ("code", """# 2) Get the project code and install dependencies (PyTorch is preinstalled on Colab)
!rm -rf "$PROJECT" && mkdir -p "$PROJECT"
!unzip -q -o "$DRIVE_ROOT/stutter_project_code.zip" -d "$PROJECT"
%cd /content/project
!grep -v '^#' requirements.txt | grep -v -E '^(torch|torchaudio)' > /tmp/req.txt && pip install -q -r /tmp/req.txt
!python -c "import torch, transformers, librosa; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__)\""""),

    ("md", "## A. Download everything (first session only - later sessions jump to section B)"),

    ("code", """# 3) SEP-28k + FluencyBank clips from the Hugging Face mirror (~2.9 GB).
#    Shards are saved on Drive (raw/sep28k_parquet) and clips are extracted to the local Colab disk.
!python scripts/fetch_sep28k.py --out "$LOCAL_DATA/sep28k" --parquet-dir "$RAW/sep28k_parquet\""""),

    ("code", """# 4) Official Apple route: label CSVs + original podcast audio for any clips the mirror lacks.
#    Episodes whose clips already exist are skipped; dead podcast links are listed in failed_episodes.txt.
#    Raw episodes are deleted after clipping, so the ~32 GB of podcast audio is never stored.
!python scripts/fetch_sep28k.py --source official --out "$LOCAL_DATA/sep28k"
!mkdir -p "$RAW/sep28k_official" && cp -r "$LOCAL_DATA/sep28k/official/." "$RAW/sep28k_official/"
!cp "$LOCAL_DATA/sep28k/failed_episodes.txt" "$RAW/sep28k_official/" 2>/dev/null || true"""),

    ("code", """# 5) UCLASS Release 1 + 2 (children/adolescents who stutter): wav audio, transcripts, annotations, metadata.
#    Zip bundles are kept on Drive (raw/uclass_zips); the UCL server is flaky, so re-run this cell if anything failed.
!python scripts/fetch_uclass.py --out "$LOCAL_DATA/uclass" --zip-dir "$RAW/uclass_zips" --audio-format wav
!cp -r "$LOCAL_DATA/uclass/metadata" "$RAW/uclass_metadata" 2>/dev/null || true"""),

    ("code", """# 6) Save ready-to-use extracted data to Drive as single tar files (fast restore in later sessions)
!tar -cf "$DATA_TARS/sep28k.tar" --exclude=sep28k/episodes --exclude=sep28k/parquet -C "$LOCAL_DATA" sep28k
!tar -cf "$DATA_TARS/uclass.tar" --exclude=uclass/zips -C "$LOCAL_DATA" uclass
!ls -lh "$DATA_TARS" && du -sh "$RAW"/*"""),

    ("md", "## B. Later sessions: restore, package, and upload to Hugging Face"),

    ("code", """# 7) Restore extracted data from Drive (safe to run independently in later sessions)
import os, subprocess
DRIVE_ROOT = os.environ.get('DRIVE_ROOT', '/content/drive/MyDrive/StutterProject')
DATA_TARS = os.environ.get('DATA_TARS', f'{DRIVE_ROOT}/data')
LOCAL_DATA = os.environ.get('LOCAL_DATA', '/content/data')
os.makedirs(LOCAL_DATA, exist_ok=True)

for name in ('sep28k', 'uclass'):
    tar = f'{DATA_TARS}/{name}.tar'
    if os.path.exists(f'{LOCAL_DATA}/{name}/manifest.csv'):
        print(name, 'already present locally')
    elif os.path.exists(tar):
        subprocess.run(['tar', '-xf', tar, '-C', LOCAL_DATA], check=True)
        print('restored', name)
    else:
        print(f'{tar} missing - run section A')"""),

    ("code", """# 8) Dataset summary
import pandas as pd
sep = pd.read_csv(f'{LOCAL_DATA}/sep28k/manifest.csv')
print('SEP-28k/FluencyBank clips:', len(sep))
print(sep.groupby(['dataset', 'split']).size().unstack(fill_value=0))
labels = ['Block', 'Prolongation', 'SoundRep', 'WordRep', 'Interjection']
print('positive rate (>=2/3 annotators):', (sep[labels] >= 2).mean().round(3).to_dict())
if os.path.exists(f'{LOCAL_DATA}/uclass/manifest.csv'):
    uc = pd.read_csv(f'{LOCAL_DATA}/uclass/manifest.csv').drop_duplicates('path')
    print('\\nUCLASS recordings:', len(uc), '| speakers:', uc['speaker'].nunique(),
          '| child recordings:', int((uc['is_child'] == True).sum()))
    print(uc.groupby(['release', 'task']).size())
!du -sh "$RAW"/* "$DATA_TARS"/* 2>/dev/null"""),

    ("code", """# 9) Normalize downloaded SEP-28k + UCLASS into one clean /content/dataset folder
#    Audio is copied so the result is self-contained. The packaged dataset is also
#    tarred to Drive so it survives Colab restarts (skip with SKIP_TAR=1).
""" + SELF_HEAL.replace("{NEED}", "scripts/package_dataset.py") + """
if os.path.exists('/content/dataset/dataset_info.json') and os.environ.get('SKIP_TAR') != '1':
    print('/content/dataset already packaged - reusing')
else:
    subprocess.run(['python', 'scripts/package_dataset.py',
                    '--src-sep28k', f'{LOCAL_DATA}/sep28k',
                    '--src-uclass', f'{LOCAL_DATA}/uclass',
                    '--out', '/content/dataset', '--copy-audio'], check=True)

import json
print(json.dumps(json.load(open('/content/dataset/dataset_info.json')), indent=2))

# Persist the packaged dataset to Drive so pushes work even after a runtime restart
os.makedirs(f'{DRIVE_ROOT}/packaged', exist_ok=True)
subprocess.run(['tar', '-cf', f'{DRIVE_ROOT}/packaged/dataset.tar', '-C', '/content', 'dataset'], check=True)
print('Packaged dataset archived to Drive:', f'{DRIVE_ROOT}/packaged/dataset.tar')"""),

    ("code", """# 10) Log in to Hugging Face (required once per Colab session)
#    Either use the notebook login widget, or set HF_TOKEN as a Colab secret.
from huggingface_hub import notebook_login
notebook_login()

!pip install -q datasets"""),

    ("code", """# 11) Push SEP-28k/FluencyBank clips to Hugging Face Datasets
#     Change YOUR_USERNAME to your Hugging Face username/organization.
""" + SELF_HEAL.replace("{NEED}", "scripts/push_to_hub.py") + """
if not os.path.exists('/content/dataset/sep28k/manifest.csv'):
    tar = f'{DRIVE_ROOT}/packaged/dataset.tar'
    if os.path.exists(tar):
        subprocess.run(['tar', '-xf', tar, '-C', '/content'], check=True)
        print('Restored packaged dataset from Drive tar')
    else:
        raise FileNotFoundError('Packaged dataset missing - run Cell 9 first')

YOUR_USERNAME = 'your-username'  # <-- EDIT THIS
SEP_REPO = f'{YOUR_USERNAME}/pediatric-stutter-sep28k'

subprocess.run(['python', 'scripts/push_to_hub.py',
                '--dataset-dir', '/content/dataset/sep28k',
                '--repo-id', SEP_REPO,
                '--audio-dir', 'clips',
                '--audio-column', 'path',
                '--split-column', 'split'], check=True)

print(f"SEP-28k dataset now at https://huggingface.co/datasets/{SEP_REPO}")"""),

    ("code", """# 12) Push UCLASS child-speech windows to Hugging Face Datasets
""" + SELF_HEAL.replace("{NEED}", "scripts/push_to_hub.py") + """
UCLASS_REPO = f'{YOUR_USERNAME}/pediatric-stutter-uclass'

subprocess.run(['python', 'scripts/push_to_hub.py',
                '--dataset-dir', '/content/dataset/uclass',
                '--repo-id', UCLASS_REPO,
                '--audio-dir', 'wav',
                '--audio-column', 'path',
                '--split-column', 'split',
                '--dedupe-audio'], check=True)  # 53k window rows -> ~455 unique recordings

print(f"UCLASS dataset now at https://huggingface.co/datasets/{UCLASS_REPO}")"""),

    ("md", """## C. Train on RunPod (recommended) or continue on Colab

For RunPod, create a GPU pod, upload `stutter_project_code.zip`, install requirements, then run:

```bash
cd /workspace && unzip -q stutter_project_code.zip -d project && cd project
pip install -r requirements.txt datasets
python scripts/train.py --config configs/runpod.yaml \\
    --hf-dataset your-username/pediatric-stutter-sep28k \\
    --output-dir /workspace/checkpoints/finetune_wav2vec2
```

For Colab (slower / session-limited), use the cells below."""),

    ("code", """# 13a) Colab frozen-backbone baseline (safe but weaker)
""" + SELF_HEAL.replace("{NEED}", "scripts/train.py") + """
!python scripts/train.py --config configs/default.yaml --manifest "$LOCAL_DATA/sep28k/manifest.csv" \\
    --output-dir "$DRIVE_ROOT/checkpoints/frozen_wav2vec2" --set train.cache_dir="$DRIVE_ROOT/cache" train.batch_size=64"""),

    ("code", """# 13b) Colab full fine-tuning (only if you have a stable session)
""" + SELF_HEAL.replace("{NEED}", "scripts/train.py") + """
!python scripts/train.py --config configs/gpu_finetune.yaml --manifest "$LOCAL_DATA/sep28k/manifest.csv" \\
    --output-dir "$DRIVE_ROOT/checkpoints/finetune_wav2vec2" \\
    --set train.batch_size=8 train.grad_accum_steps=4 train.num_workers=0"""),

    ("md", """## D. Evaluate and test on children's speech (UCLASS)

If you trained on RunPod, copy `best.pt` to Drive first, or change the checkpoint path below."""),

    ("code", """# 14) Test-set metrics (F1 per disfluency type, macro F1, severity MAE)
""" + SELF_HEAL.replace("{NEED}", "scripts/evaluate.py") + """
CKPT = f'{DRIVE_ROOT}/checkpoints/finetune_wav2vec2/best.pt'
if not os.path.exists(CKPT):
    CKPT = f'{DRIVE_ROOT}/checkpoints/frozen_wav2vec2/best.pt'
os.environ['CKPT'] = CKPT
!python scripts/evaluate.py --checkpoint "$CKPT" --split test --no-cache --out "$DRIVE_ROOT/results/eval_test.json\""""),

    ("code", """# 15) Run inference on UCLASS child recordings (< 18 years) -> JSON on Drive
import json
import pandas as pd
uc = pd.read_csv(f'{LOCAL_DATA}/uclass/manifest.csv').drop_duplicates('path')
uc = uc[(uc['is_child'] == True) & uc['path'].str.endswith('.wav')].sort_values('age_years')
child_wavs = [f'{LOCAL_DATA}/uclass/{p}' for p in uc['path'].head(20)]
out_json = f'{DRIVE_ROOT}/results/uclass_inference.json'
subprocess.run(['python', 'scripts/infer.py', *child_wavs, '--checkpoint', CKPT, '--no-windows', '--out', out_json], check=True)
res = json.load(open(out_json))
res = res if isinstance(res, list) else [res]
print(f"Wrote UCLASS inference results to {out_json}")
pd.DataFrame([{'file': os.path.basename(r['file']), 'duration_s': r['duration_s'], **r['counts'],
               'pct_disfluent': r['percent_disfluent_windows'], 'band': r['severity_band']} for r in res])"""),
]


def cell(kind: str, source: str) -> dict:
    base = {"cell_type": "code" if kind == "code" else "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.split("\n")]}
    if kind == "code":
        base.update(execution_count=None, outputs=[])
    return base


nb = {
    "cells": [cell(k, s) for k, s in cells],
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {OUT} ({len(cells)} cells)")
