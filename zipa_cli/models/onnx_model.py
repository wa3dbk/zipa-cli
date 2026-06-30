"""ONNX inference backend (minimal-dependency path).

Wraps onnxruntime sessions for the CTC (single ``model.onnx``) and transducer
(``encoder``/``decoder``/``joiner``) exports. The subsampling-length heuristic
(``//2`` for CTC, ``//4`` for transducer) matches ``inference/batch_inference.py``;
output lengths are clipped to the actual session output to stay safe.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..decode import (
    ctc_greedy_decode,
    ctc_greedy_decode_with_times,
    load_tokens,
    transducer_greedy_decode,
    transducer_greedy_decode_with_times,
)
from .base import PhoneRecognizer


def _make_session(path: str, providers: Optional[List[str]] = None):
    import onnxruntime as ort

    if providers is None:
        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
    return ort.InferenceSession(str(path), providers=providers)


class OnnxCTCRecognizer(PhoneRecognizer):
    arch = "ctc"

    def __init__(self, model_path: str, vocab: Dict[int, str], providers=None):
        self.session = _make_session(model_path, providers)
        self.vocab = vocab

    def infer_features(self, features_padded, feat_lens: np.ndarray) -> List[List[str]]:
        x = features_padded.numpy() if hasattr(features_padded, "numpy") else np.asarray(features_padded)
        feat_lens = np.asarray(feat_lens, dtype=np.int64)
        log_probs = self.session.run(None, {"x": x, "x_lens": feat_lens})[0]  # (B, T_sub, V)
        decoded_lens = np.clip(feat_lens // 2, 0, log_probs.shape[1])
        return ctc_greedy_decode(log_probs, self.vocab, lengths=decoded_lens)

    def infer_features_timed(self, features_padded, feat_lens: np.ndarray):
        x = features_padded.numpy() if hasattr(features_padded, "numpy") else np.asarray(features_padded)
        feat_lens = np.asarray(feat_lens, dtype=np.int64)
        log_probs = self.session.run(None, {"x": x, "x_lens": feat_lens})[0]
        decoded_lens = np.clip(feat_lens // 2, 0, log_probs.shape[1])
        return ctc_greedy_decode_with_times(log_probs, self.vocab, lengths=decoded_lens)


class OnnxTransducerRecognizer(PhoneRecognizer):
    arch = "transducer"

    def __init__(self, encoder: str, decoder: str, joiner: str, vocab: Dict[int, str], providers=None):
        self.sess_enc = _make_session(encoder, providers)
        self.sess_dec = _make_session(decoder, providers)
        self.sess_join = _make_session(joiner, providers)
        self.vocab = vocab

    def infer_features(self, features_padded, feat_lens: np.ndarray) -> List[List[str]]:
        x = features_padded.numpy() if hasattr(features_padded, "numpy") else np.asarray(features_padded)
        feat_lens = np.asarray(feat_lens, dtype=np.int64)
        enc_out = self.sess_enc.run(None, {"x": x, "x_lens": feat_lens})[0]  # (B, T, D)
        decoded_lens = np.clip(feat_lens // 4, 0, enc_out.shape[1])
        return transducer_greedy_decode(
            enc_out, self.sess_dec, self.sess_join, self.vocab, lengths=decoded_lens
        )

    def infer_features_timed(self, features_padded, feat_lens: np.ndarray):
        x = features_padded.numpy() if hasattr(features_padded, "numpy") else np.asarray(features_padded)
        feat_lens = np.asarray(feat_lens, dtype=np.int64)
        enc_out = self.sess_enc.run(None, {"x": x, "x_lens": feat_lens})[0]
        decoded_lens = np.clip(feat_lens // 4, 0, enc_out.shape[1])
        return transducer_greedy_decode_with_times(
            enc_out, self.sess_dec, self.sess_join, self.vocab, lengths=decoded_lens
        )


def build_onnx_recognizer(
    arch: str,
    tokens_path: str,
    *,
    ctc_model: Optional[str] = None,
    transducer_files: Optional[Dict[str, str]] = None,
    providers: Optional[List[str]] = None,
) -> PhoneRecognizer:
    """Construct the right ONNX recognizer for ``arch``."""
    vocab = load_tokens(tokens_path)
    if not vocab:
        raise ValueError(f"No tokens loaded from {tokens_path!r}.")
    if arch == "ctc":
        if not ctc_model:
            raise ValueError("CTC ONNX backend requires a model file path.")
        return OnnxCTCRecognizer(ctc_model, vocab, providers)
    if arch == "transducer":
        if not transducer_files:
            raise ValueError("Transducer ONNX backend requires encoder/decoder/joiner files.")
        return OnnxTransducerRecognizer(
            transducer_files["encoder"],
            transducer_files["decoder"],
            transducer_files["joiner"],
            vocab,
            providers,
        )
    raise ValueError(f"Unknown arch: {arch!r}")
