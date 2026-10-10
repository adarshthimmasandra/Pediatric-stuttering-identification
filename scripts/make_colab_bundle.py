"""Zip the project code (no .venv / data / cache / checkpoints) for upload to Google Drive.

    python scripts/make_colab_bundle.py      # -> stutter_project_code.zip
Upload it to  My Drive/StutterProject/stutter_project_code.zip  and run colab/stutter_colab.ipynb.
"""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["stutter", "scripts", "configs", "tests", "colab", "requirements.txt", "README.md", "AGENTS.md"]

out = ROOT / "stutter_project_code.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
    for item in INCLUDE:
        path = ROOT / item
        for f in [path] if path.is_file() else sorted(path.rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts:
                zf.write(f, f.relative_to(ROOT).as_posix())  # forward slashes for Linux unzip
print(f"Created {out} ({out.stat().st_size // 1024} KB, {len(zipfile.ZipFile(out).namelist())} files)")
