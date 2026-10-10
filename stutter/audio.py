"""Audio I/O and augmentation.

Loading uses soundfile (wav/flac/ogg) with a librosa/audioread fallback (mp3/m4a via ffmpeg), which
avoids torchaudio's torchcodec dependency on recent PyTorch/Python versions.
"""

from __future__ import annotations

import io
import random

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF


def to_mono_resampled(wave: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim == 2:
        wave = wave.mean(axis=1)
    if sr != target_sr and wave.size:
        wave = AF.resample(torch.from_numpy(wave), sr, target_sr).numpy()
    return wave


def load_audio(
    path: str,
    target_sr: int = 16000,
    start: float | None = None,
    end: float | None = None,
) -> np.ndarray:
    """Load ``path`` as mono float32 at ``target_sr``. ``start``/``end`` are in seconds."""
    try:
        info = sf.info(path)
        frame_start = int(start * info.samplerate) if start else 0
        frame_stop = int(end * info.samplerate) if end else None
        wave, sr = sf.read(path, start=frame_start, stop=frame_stop, dtype="float32", always_2d=False)
    except (sf.LibsndfileError, RuntimeError):
        import librosa

        duration = (end - (start or 0.0)) if end else None
        wave, sr = librosa.load(path, sr=None, mono=True, offset=start or 0.0, duration=duration)
    return to_mono_resampled(wave, sr, target_sr)


def decode_audio_bytes(data: bytes, target_sr: int = 16000) -> np.ndarray:
    wave, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    return to_mono_resampled(wave, sr, target_sr)


def audio_duration(path: str) -> float:
    try:
        return float(sf.info(path).duration)
    except (sf.LibsndfileError, RuntimeError):
        import librosa

        return float(librosa.get_duration(path=path))


def fix_length(wave: np.ndarray, num_samples: int, random_crop: bool = False) -> np.ndarray:
    if len(wave) > num_samples:
        offset = random.randint(0, len(wave) - num_samples) if random_crop else 0
        return wave[offset : offset + num_samples]
    if len(wave) < num_samples:
        return np.pad(wave, (0, num_samples - len(wave)))
    return wave


def _speed_perturb(wave: np.ndarray, factor: float) -> np.ndarray:
    """Linear-interpolation speed perturbation - ~100x cheaper than AF.resample per item."""
    n = len(wave)
    idx = np.clip(np.arange(0, n, factor), 0, n - 1)
    return np.interp(idx, np.arange(n), wave).astype(np.float32)


def augment(wave: np.ndarray, sr: int) -> np.ndarray:
    """Light, label-preserving augmentation: gain, additive noise at random SNR, speed perturbation."""
    wave = wave * random.uniform(0.6, 1.2)
    if random.random() < 0.5:
        snr_db = random.uniform(10, 30)
        power = float(np.mean(wave**2)) + 1e-8
        noise = np.random.randn(len(wave)).astype(np.float32)
        wave = wave + noise * np.sqrt(power / (10 ** (snr_db / 10)))
    if random.random() < 0.3:
        wave = _speed_perturb(wave, random.choice([0.9, 1.1]))
    return np.clip(wave, -1.0, 1.0).astype(np.float32)
