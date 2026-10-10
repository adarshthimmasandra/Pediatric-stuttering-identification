# Agent notes

- Windows / PowerShell, Python 3.14, CPU only (no CUDA). Use `.\.venv\Scripts\python.exe`.
- Setup: `powershell -ExecutionPolicy Bypass -File .\setup_env.ps1`
- Tests (offline, ~3 min): `.\.venv\Scripts\python.exe -m pytest tests -q`
- Audio I/O uses soundfile/librosa, not `torchaudio.load` (torchcodec issues); torchaudio only for resampling.
- CPU feature extraction for wav2vec2-base runs at ~1 clip/s; use `--max-samples` for local runs.
- Colab/Drive is the main path for full downloads + GPU training: after code changes run
  `python scripts/make_colab_bundle.py` and re-upload `stutter_project_code.zip` to `My Drive/StutterProject/`.
- UCLASS server drops connections often (fetch_uclass.py retries); its `.mdb` files always return 403.
- PowerShell reports exit code 1 when tqdm writes to stderr with `2>&1`/`*>`; check the log, not the exit code.
- Colab packaging flow: after downloads run `package_dataset.py`, then `push_to_hub.py` to upload SEP-28k and UCLASS to Hugging Face Datasets for RunPod training.
- RunPod training uses `configs/runpod.yaml` with `--hf-dataset your-username/pediatric-stutter-sep28k`.
- Full fine-tuning on Colab T4 is slow and may stall if data is not on local disk; prefer RunPod for full fine-tuning.
