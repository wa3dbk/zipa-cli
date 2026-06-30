"""Shared audio loading and fbank feature extraction.

These helpers mirror the behaviour of the original ZIPA inference scripts
exactly so results match the reference implementation:

* audio is converted to **mono** and resampled to **16 kHz**;
* features are 80-dim kaldi-style log-mel fbank from lhotse, configured with
  ``dither=0.0`` and ``snip_edges=False`` (see ``inference/utils.py``).

Imports of the heavy optional dependencies (lhotse, soundfile, librosa) are kept
lazy so that ``import zipa_cli`` works in a minimal environment, and so the
PyTorch backend (which also pulls a fbank from lhotse) and the ONNX backend can
share one cached extractor.
"""

from __future__ import annotations

import functools
from typing import List

import numpy as np

TARGET_SR = 16000


@functools.lru_cache(maxsize=1)
def get_fbank_extractor():
    """Return a cached lhotse ``Fbank`` extractor (80 filters, no dither)."""
    from lhotse.features.kaldi.extractors import Fbank, FbankConfig

    config = FbankConfig(num_filters=80, dither=0.0, snip_edges=False)
    return Fbank(config)


def load_audio(path: str, target_sr: int = TARGET_SR) -> np.ndarray:
    """Load a single audio file as a mono float32 array at ``target_sr``.

    Supports wav/flac/mp3/etc via soundfile, falling back to librosa for formats
    soundfile cannot decode.
    """
    try:
        import soundfile as sf

        audio, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        import librosa

        audio, sr = librosa.load(path, sr=None, mono=False)

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        # soundfile returns (frames, channels); librosa returns (channels, frames).
        # Reduce to the first channel either way.
        audio = audio[:, 0] if audio.shape[0] >= audio.shape[1] else audio[0, :]
    if sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return np.ascontiguousarray(audio, dtype=np.float32)


def to_mono_16k(array: np.ndarray, sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    """Coerce an already-decoded array to mono float32 at ``target_sr``."""
    audio = np.asarray(array, dtype=np.float32)
    if audio.ndim > 1:
        # Average channels if shaped (frames, channels) or (channels, frames).
        axis = 1 if audio.shape[0] >= audio.shape[1] else 0
        audio = audio.mean(axis=axis).astype(np.float32)
    if sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return np.ascontiguousarray(audio, dtype=np.float32)


def extract_fbank_batch(audio_arrays: List[np.ndarray]):
    """Extract padded fbank features for a batch of mono 16 kHz audio arrays.

    Returns ``(features_padded, feat_lens)`` where ``features_padded`` is a
    ``torch.FloatTensor`` of shape ``(B, T, 80)`` and ``feat_lens`` is an
    ``int64`` numpy array of pre-padding frame counts.
    """
    import torch
    import torch.nn.utils.rnn as rnn_utils

    extractor = get_fbank_extractor()
    tensors = [
        a if isinstance(a, torch.Tensor) else torch.from_numpy(np.asarray(a, dtype=np.float32))
        for a in audio_arrays
    ]
    features = extractor.extract_batch(tensors, sampling_rate=TARGET_SR)

    if isinstance(features, list):
        feat_lens = np.array([f.shape[0] for f in features], dtype=np.int64)
        features_padded = rnn_utils.pad_sequence(features, batch_first=True)
    else:
        # Already a single padded tensor (B, T, 80).
        features_padded = features
        feat_lens = np.array([features_padded.shape[1]] * len(audio_arrays), dtype=np.int64)

    return features_padded, feat_lens
