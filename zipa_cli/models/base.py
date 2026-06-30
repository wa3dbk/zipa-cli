"""Backend-agnostic phone-recognizer interface.

The pipeline owns audio loading and fbank extraction (so precomputed-feature
cuts can skip it and the extractor is shared). A recognizer therefore consumes
padded fbank features and returns one token list per batch item.
"""

from __future__ import annotations

import abc
from typing import List, Optional

import numpy as np


class PhoneRecognizer(abc.ABC):
    """Common interface implemented by the ONNX and PyTorch backends."""

    arch: str  # "ctc" or "transducer"

    @abc.abstractmethod
    def infer_features(self, features_padded, feat_lens: np.ndarray) -> List[List[str]]:
        """Decode a padded fbank batch.

        Args:
            features_padded: ``torch.FloatTensor`` of shape ``(B, T, 80)``.
            feat_lens: ``int64`` numpy array of pre-padding frame counts ``(B,)``.

        Returns:
            A list of length ``B``; each element is a list of phone tokens.
        """

    def infer_features_timed(self, features_padded, feat_lens: np.ndarray):
        """Decode a batch returning per-phone ``(token, start_s, end_s)`` tuples.

        Returns a list of length ``B``; each element is a list of timed phones.
        Backends that cannot produce timings should raise ``NotImplementedError``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support timestamped decoding yet; "
            f"use the ONNX backend for --timestamps / align-json / ctm output."
        )

    def infer_audio(self, audio_arrays: List[np.ndarray]) -> List[List[str]]:
        """Convenience: extract fbank for raw audio then decode."""
        from ..features import extract_fbank_batch

        features_padded, feat_lens = extract_fbank_batch(audio_arrays)
        return self.infer_features(features_padded, feat_lens)

    def close(self) -> None:  # pragma: no cover - optional cleanup hook
        pass
