"""Zipformer architecture parameter blocks for the PyTorch backend.

Lifted verbatim from ``zipa_ctc_inference.py`` and ``zipa_transducer_inference.py``
so the reconstructed models match the released checkpoints exactly. Selection is
by ``(arch, size)`` taken from the registry, replacing the original brittle
``"small" in path`` / ``"large" in path`` string match.
"""

from __future__ import annotations

from typing import Dict

# Shared zipformer encoder geometry --------------------------------------------
_SMALL_ENCODER = {
    "num_encoder_layers": "2,2,3,4,3,2",
    "downsampling_factor": "1,2,4,8,4,2",
    "feedforward_dim": "512,768,1024,1536,1024,768",
    "num_heads": "4,4,4,8,4,4",
    "encoder_dim": "192,256,384,512,384,256",
    "query_head_dim": "32",
    "value_head_dim": "12",
    "pos_head_dim": "4",
    "pos_dim": 48,
    "encoder_unmasked_dim": "192, 192, 256, 256, 256, 192",
    "cnn_module_kernel": "31,31,15,15,15,31",
    "decoder_dim": 512,
    "joiner_dim": 512,
}

_LARGE_ENCODER = {
    "num_encoder_layers": "4,3,4,5,4,4",
    "downsampling_factor": "1,2,4,8,4,2",
    "feedforward_dim": "768,768,1536,2048,1536,768",
    "num_heads": "6,6,6,8,6,6",
    "encoder_dim": "512,512,768,1024,768,512",
    "query_head_dim": "64",
    "value_head_dim": "48",
    "pos_head_dim": "4",
    "pos_dim": 48,
    "encoder_unmasked_dim": "192,192,256,320,256,192",
    "cnn_module_kernel": "31,31,15,15,15,31",
    "decoder_dim": 1024,
    "joiner_dim": 1024,
}

_COMMON = {
    "feature_dim": 80,
    "subsampling_factor": 4,
    "vocab_size": 127,
    # attention decoder (unused at inference, but get_model expects them)
    "attention_decoder_dim": 512,
    "attention_decoder_num_layers": 6,
    "attention_decoder_attention_dim": 512,
    "attention_decoder_num_heads": 8,
    "attention_decoder_feedforward_dim": 2048,
    "causal": False,
    "chunk_size": "16,32,64,-1",
    "left_context_frames": "64,128,256,-1",
    "use_attention_decoder": False,
    "use_unsup_cr_ctc": False,
}

_CTC_FLAGS = {"use_transducer": False, "use_ctc": True, "use_cr_ctc": True}
_TRANSDUCER_FLAGS = {
    "use_transducer": True,
    "use_ctc": False,
    "use_cr_ctc": False,
    "context_size": 2,
}


def get_params(arch: str, size: str):
    """Return an ``icefall.utils.AttributeDict`` of params for ``(arch, size)``."""
    from icefall.utils import AttributeDict

    if size not in ("small", "large"):
        raise ValueError(f"size must be 'small' or 'large', got {size!r}")
    encoder = _SMALL_ENCODER if size == "small" else _LARGE_ENCODER
    flags = _CTC_FLAGS if arch == "ctc" else _TRANSDUCER_FLAGS
    params: Dict = {}
    params.update(_COMMON)
    params.update(encoder)
    params.update(flags)
    return AttributeDict(params)
