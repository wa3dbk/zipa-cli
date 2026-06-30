"""PyTorch inference backend (full GPU path).

Reconstructs the released ZIPA checkpoints with the ZIPA repo's own ``get_model``
and runs greedy decoding, mirroring ``zipa_ctc_inference.py`` /
``zipa_transducer_inference.py`` — but with the architecture/size chosen
explicitly from the registry instead of string-matching the path, and using the
encoder's *exact* output lengths (no subsampling heuristic).

Requires ``torch`` plus ``icefall`` + ``k2`` installed to match your torch/cuda
versions (see the ZIPA README). The relevant arch sub-package of the ZIPA repo
(``zipformer_crctc/`` or ``zipformer_transducer/``) is placed on ``sys.path`` so
its modules' bare imports resolve.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from .base import PhoneRecognizer
from .params import get_params

_ARCH_SUBDIR = {"ctc": "zipformer_crctc", "transducer": "zipformer_transducer"}


def _ensure_on_path(zipa_repo: str, arch: str) -> None:
    if not zipa_repo:
        raise FileNotFoundError(
            "PyTorch backend needs the ZIPA repo. Pass --zipa-repo or set $ZIPA_REPO."
        )
    repo = Path(zipa_repo)
    subdir = repo / _ARCH_SUBDIR[arch]
    if not subdir.is_dir():
        raise FileNotFoundError(f"Expected {subdir} in the ZIPA repo.")
    for p in (str(subdir), str(repo)):
        if p not in sys.path:
            sys.path.insert(0, p)


def _resolve_device(device: str):
    import torch

    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class TorchCTCRecognizer(PhoneRecognizer):
    arch = "ctc"

    def __init__(self, checkpoint: str, bpe_model: str, size: str, device, zipa_repo: str):
        import torch
        import sentencepiece as spm

        _ensure_on_path(zipa_repo, "ctc")
        from train import get_model  # type: ignore

        self.device = _resolve_device(device)
        params = get_params("ctc", size)
        self.model = get_model(params)
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.bpe = spm.SentencePieceProcessor()
        self.bpe.load(bpe_model)

    def infer_features(self, features_padded, feat_lens: np.ndarray) -> List[List[str]]:
        import torch
        from icefall.decode import ctc_greedy_search

        feats = features_padded.to(self.device)
        lens = torch.as_tensor(np.asarray(feat_lens), dtype=torch.int64, device=self.device)
        with torch.no_grad():
            encoder_out, encoder_out_lens = self.model.forward_encoder(feats, lens)
            ctc_output = self.model.ctc_output(encoder_out)
            hyps = ctc_greedy_search(ctc_output, encoder_out_lens)
        return [s.split() for s in self.bpe.decode(hyps)]


class TorchTransducerRecognizer(PhoneRecognizer):
    arch = "transducer"

    def __init__(self, checkpoint: str, bpe_model: str, size: str, device, zipa_repo: str):
        import torch
        import sentencepiece as spm

        _ensure_on_path(zipa_repo, "transducer")
        from train import get_model  # type: ignore
        from beam_search import greedy_search_batch  # type: ignore

        self._greedy_search_batch = greedy_search_batch
        self.device = _resolve_device(device)

        self.bpe = spm.SentencePieceProcessor()
        self.bpe.load(bpe_model)

        params = get_params("transducer", size)
        params.blank_id = self.bpe.piece_to_id("<blk>")
        params.sos_id = params.eos_id = self.bpe.piece_to_id("<sos/eos>")
        params.vocab_size = self.bpe.get_piece_size()

        self.model = get_model(params)
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
        self.model.to(self.device)
        self.model.eval()

    def infer_features(self, features_padded, feat_lens: np.ndarray) -> List[List[str]]:
        import torch

        feats = features_padded.to(self.device)
        lens = torch.as_tensor(np.asarray(feat_lens), dtype=torch.int64, device=self.device)
        with torch.no_grad():
            encoder_out, encoder_out_lens = self.model.forward_encoder(feats, lens)
            hyp_tokens = self._greedy_search_batch(
                model=self.model,
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
            )
        return [self.bpe.decode(h).split() for h in hyp_tokens]


def build_torch_recognizer(
    arch: str,
    checkpoint: str,
    bpe_model: str,
    size: Optional[str],
    device: str = "auto",
    zipa_repo: Optional[str] = None,
) -> PhoneRecognizer:
    if size is None:
        raise ValueError(
            "PyTorch backend needs the model size ('small'/'large'). It is known for "
            "registry tags; for a raw --model path pass it via the registry or use ONNX."
        )
    if arch == "ctc":
        return TorchCTCRecognizer(checkpoint, bpe_model, size, device, zipa_repo)
    if arch == "transducer":
        return TorchTransducerRecognizer(checkpoint, bpe_model, size, device, zipa_repo)
    raise ValueError(f"Unknown arch: {arch!r}")
