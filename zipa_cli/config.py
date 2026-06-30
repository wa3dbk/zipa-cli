"""Paths, cache locations, and asset resolution for zipa-cli.

The CLI ships alongside the ZIPA training repo (the ``zipa/`` directory that
contains ``zipformer_crctc/``, ``zipformer_transducer/`` and ``ipa_simplified/``).
We need three things at runtime:

* the **cache dir** where downloaded models live;
* the **tokenizer assets** (``tokens.txt`` for ONNX, ``unigram_127.model`` for
  PyTorch), which are bundled in ``ipa_simplified/``;
* the **path to the ZIPA repo**, so the PyTorch backend can import the
  ``zipformer_crctc`` / ``zipformer_transducer`` model definitions.

All of these can be overridden with environment variables or CLI flags so the
tool works regardless of where it is installed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------- #
# Environment variable names (documented in the README).
# --------------------------------------------------------------------------- #
ENV_CACHE_DIR = "ZIPA_CLI_CACHE"
ENV_ZIPA_REPO = "ZIPA_REPO"


def cache_dir() -> Path:
    """Root directory for downloaded models and tokenizers.

    Defaults to ``~/.cache/zipa-cli`` and can be overridden via ``$ZIPA_CLI_CACHE``.
    """
    raw = os.environ.get(ENV_CACHE_DIR)
    if raw:
        path = Path(raw).expanduser()
    else:
        path = Path.home() / ".cache" / "zipa-cli"
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir() -> Path:
    """Sub-directory of the cache where model weights are stored, one dir per tag."""
    path = cache_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _candidate_repo_roots() -> list[Path]:
    """Plausible locations of the ZIPA training repo, most-specific first."""
    candidates: list[Path] = []
    env = os.environ.get(ENV_ZIPA_REPO)
    if env:
        candidates.append(Path(env).expanduser())
    # The repo is conventionally a sibling of this package's parent.
    here = Path(__file__).resolve().parent.parent  # .../waad
    candidates.append(here / "zipa")
    candidates.append(here)  # in case the CLI is vendored inside the zipa repo
    candidates.append(Path.cwd() / "zipa")
    candidates.append(Path.cwd())
    return candidates


def zipa_repo_root(override: Optional[str] = None) -> Optional[Path]:
    """Locate the ZIPA training repo (the dir holding ``ipa_simplified/``).

    Returns ``None`` if it cannot be found; callers that need it (the PyTorch
    backend, the default tokenizer assets) should raise a helpful error.
    """
    candidates = []
    if override:
        candidates.append(Path(override).expanduser())
    candidates.extend(_candidate_repo_roots())
    for c in candidates:
        if (c / "ipa_simplified").is_dir():
            return c
    return None


def default_tokens_txt(repo_root: Optional[Path] = None) -> Optional[Path]:
    """Path to ``ipa_simplified/tokens.txt`` (the ONNX/CTC token table)."""
    root = repo_root or zipa_repo_root()
    if root is None:
        return None
    p = root / "ipa_simplified" / "tokens.txt"
    return p if p.exists() else None


def default_bpe_model(repo_root: Optional[Path] = None) -> Optional[Path]:
    """Path to ``ipa_simplified/unigram_127.model`` (the PyTorch sentencepiece model)."""
    root = repo_root or zipa_repo_root()
    if root is None:
        return None
    p = root / "ipa_simplified" / "unigram_127.model"
    return p if p.exists() else None
