# Creates a local virtual environment and installs all dependencies.
# Usage:  powershell -ExecutionPolicy Bypass -File .\setup_env.ps1 [-Cuda cu128]
param(
    [string]$Python = "python",
    [string]$Cuda = ""
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv")) {
    & $Python -m venv .venv
}
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $py -m pip install --upgrade pip

$index = if ($Cuda) { "https://download.pytorch.org/whl/$Cuda" } else { "https://download.pytorch.org/whl/cpu" }
& $py -m pip install torch torchaudio --index-url $index
& $py -m pip install -r requirements.txt

& $py -c "import torch, torchaudio, transformers, librosa, soundfile; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| transformers', transformers.__version__, '| librosa', librosa.__version__)"
Write-Host "Done. Activate with: .\.venv\Scripts\Activate.ps1"
