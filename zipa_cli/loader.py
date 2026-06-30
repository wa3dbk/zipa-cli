"""Resolve a ``--model`` argument into a ready :class:`PhoneRecognizer`.

Handles three things the user shouldn't have to think about:

* **tag vs path** — a registry tag is auto-downloaded/cached; a local path is
  used as-is (offline);
* **architecture** — taken from the registry, or from ``--model-type``, or
  inferred from the on-disk layout;
* **tokenizer** — ``tokens.txt`` for ONNX, the sentencepiece model for PyTorch,
  defaulting to the bundled ``ipa_simplified`` assets.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

from . import config, registry
from .models.base import PhoneRecognizer


def _infer_arch_from_path(path: Path) -> Optional[str]:
    if path.is_file() and path.suffix == ".onnx":
        return "ctc"
    if path.is_dir():
        if any(path.glob("encoder-*.onnx")):
            return "transducer"
        if any(path.glob("*.onnx")):
            return "ctc"
    return None


def _resolve_tokens_path(explicit: Optional[str], repo_root: Optional[Path]) -> str:
    if explicit:
        return explicit
    p = config.default_tokens_txt(repo_root)
    if p is None:
        raise FileNotFoundError(
            "Could not find tokens.txt. Pass --tokens or set --zipa-repo / $ZIPA_REPO "
            "to point at the ZIPA repo containing ipa_simplified/tokens.txt."
        )
    return str(p)


def _resolve_bpe_path(explicit: Optional[str], repo_root: Optional[Path]) -> str:
    if explicit:
        return explicit
    p = config.default_bpe_model(repo_root)
    if p is None:
        raise FileNotFoundError(
            "Could not find the sentencepiece model. Pass --bpe-model or set "
            "--zipa-repo / $ZIPA_REPO to point at ipa_simplified/unigram_127.model."
        )
    return str(p)


def load_recognizer(args) -> PhoneRecognizer:
    """Build a recognizer from parsed CLI ``args`` (decode / transcribe)."""
    repo_root = config.zipa_repo_root(getattr(args, "zipa_repo", None))
    model_arg = args.model
    backend = args.backend
    precision = getattr(args, "precision", "fp32")

    entry = registry.get_entry(model_arg)
    arch = getattr(args, "model_type", None)

    if entry is not None:
        arch = entry.arch
        local = registry.local_model_dir(entry, backend)
        if not local.exists():
            print(f"[zipa-cli] model '{entry.tag}' not cached; downloading [{backend}, {precision}] ...")
            local = registry.download(entry, backend, precision)
        model_root = local
    else:
        model_root = Path(model_arg).expanduser()
        if not model_root.exists():
            raise FileNotFoundError(
                f"--model {model_arg!r} is neither a known tag nor an existing path. "
                f"Run `zipa-cli models list` to see tags."
            )
        if arch is None:
            arch = _infer_arch_from_path(model_root)
        if arch is None:
            raise ValueError(
                "Could not infer --model-type from the path; pass --model-type ctc|transducer."
            )

    if backend == "onnx":
        return _build_onnx(arch, model_root, precision, args, repo_root)
    if backend == "torch":
        return _build_torch(arch, model_root, entry, args, repo_root)
    raise ValueError(f"Unknown backend: {backend!r}")


def _build_onnx(arch, model_root: Path, precision, args, repo_root):
    from .models.onnx_model import build_onnx_recognizer

    tokens = _resolve_tokens_path(getattr(args, "tokens", None), repo_root)
    if arch == "ctc":
        if model_root.is_file():
            ctc_model = model_root
        else:
            ctc_model = registry.resolve_onnx_ctc(model_root, precision)
        if not ctc_model or not Path(ctc_model).exists():
            raise FileNotFoundError(f"No CTC ONNX model found under {model_root}")
        return build_onnx_recognizer("ctc", tokens, ctc_model=str(ctc_model))
    else:
        files = registry.resolve_onnx_transducer(model_root, precision)
        if not files:
            raise FileNotFoundError(
                f"No transducer encoder/decoder/joiner ONNX files (precision {precision}) "
                f"found under {model_root}"
            )
        return build_onnx_recognizer(
            "transducer", tokens, transducer_files={k: str(v) for k, v in files.items()}
        )


def _build_torch(arch, model_root: Path, entry, args, repo_root):
    from .models.torch_model import build_torch_recognizer

    bpe = _resolve_bpe_path(getattr(args, "bpe_model", None), repo_root)
    size = entry.size if entry is not None else None
    if model_root.is_file():
        ckpt = model_root
    else:
        ckpt = registry.resolve_torch_checkpoint(model_root)
    if not ckpt or not Path(ckpt).exists():
        raise FileNotFoundError(f"No .pth checkpoint found under {model_root}")
    return build_torch_recognizer(
        arch=arch,
        checkpoint=str(ckpt),
        bpe_model=bpe,
        size=size,
        device=getattr(args, "device", "auto"),
        zipa_repo=str(repo_root) if repo_root else None,
    )
