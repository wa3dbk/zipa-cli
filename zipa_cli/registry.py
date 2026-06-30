"""Model registry: short tags -> ZIPA HuggingFace checkpoints, plus download/resolve.

The registry encodes everything the rest of the CLI needs to know about a model
*without* relying on brittle filename string-matching (the original inference
scripts inferred 'small'/'large' from the path). Each entry records the
architecture (ctc / transducer) and the size config name so the PyTorch backend
can pick the right parameter block deterministically.

Tags follow the pattern ``zipa-<arch>-<size>-<steps>`` with ``-ns`` (new-scaling
"Ns" variant) and ``-nd`` (no-diacritics) qualifiers, e.g.::

    zipa-t-s-300k          transducer, small, 300k steps
    zipa-cr-l-500k         crctc, large, 500k steps
    zipa-cr-ns-l-800k      crctc Ns, large, 800k steps
    zipa-cr-ns-nd-l-780k   crctc Ns no-diacritics, large, 780k steps
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from . import config


@dataclass(frozen=True)
class ModelEntry:
    tag: str
    repo: str                  # HuggingFace repo id (averaged checkpoint hub)
    arch: str                  # "ctc" or "transducer"
    size: str                  # "small" or "large" (selects the param block)
    steps: int                 # training steps
    params_m: int              # approx params in millions
    ns: bool = False           # the "Ns" new-scaling variant
    no_diacritics: bool = False
    aliases: List[str] = field(default_factory=list)

    @property
    def description(self) -> str:
        bits = [self.arch.upper(), self.size, f"{self.steps // 1000}k steps", f"~{self.params_m}M"]
        if self.ns:
            bits.append("Ns")
        if self.no_diacritics:
            bits.append("no-diacritics")
        return ", ".join(bits)


# --------------------------------------------------------------------------- #
# The 12 final-averaged checkpoints from the ZIPA README.
# --------------------------------------------------------------------------- #
_ENTRIES: List[ModelEntry] = [
    # Transducer
    ModelEntry("zipa-t-s-300k", "anyspeech/zipa-small-noncausal-300k", "transducer", "small", 300_000, 65),
    ModelEntry("zipa-t-l-300k", "anyspeech/zipa-large-noncausal-300k", "transducer", "large", 300_000, 302),
    ModelEntry("zipa-t-s-500k", "anyspeech/zipa-small-noncausal-500k", "transducer", "small", 500_000, 65),
    ModelEntry("zipa-t-l-500k", "anyspeech/zipa-large-noncausal-500k", "transducer", "large", 500_000, 302),
    # CRCTC
    ModelEntry("zipa-cr-s-300k", "anyspeech/zipa-small-crctc-300k", "ctc", "small", 300_000, 64),
    ModelEntry("zipa-cr-l-300k", "anyspeech/zipa-large-crctc-300k", "ctc", "large", 300_000, 300),
    ModelEntry("zipa-cr-s-500k", "anyspeech/zipa-small-crctc-500k", "ctc", "small", 500_000, 64),
    ModelEntry("zipa-cr-l-500k", "anyspeech/zipa-large-crctc-500k", "ctc", "large", 500_000, 300),
    # CRCTC Ns
    ModelEntry("zipa-cr-ns-s-700k", "anyspeech/zipa-small-crctc-ns-700k", "ctc", "small", 700_000, 64, ns=True),
    ModelEntry("zipa-cr-ns-l-800k", "anyspeech/zipa-large-crctc-ns-800k", "ctc", "large", 800_000, 300, ns=True),
    # CRCTC Ns, no diacritics
    ModelEntry(
        "zipa-cr-ns-nd-s-700k",
        "anyspeech/zipa-small-crctc-ns-no-diacritics-700k",
        "ctc", "small", 700_000, 64, ns=True, no_diacritics=True,
    ),
    ModelEntry(
        "zipa-cr-ns-nd-l-780k",
        "anyspeech/zipa-large-crctc-ns-no-diacritics-780k",
        "ctc", "large", 780_000, 300, ns=True, no_diacritics=True,
    ),
]

MODELS: Dict[str, ModelEntry] = {}
for _e in _ENTRIES:
    MODELS[_e.tag] = _e
    for _a in _e.aliases:
        MODELS[_a] = _e


def get_entry(tag: str) -> Optional[ModelEntry]:
    return MODELS.get(tag)


def is_tag(model_arg: str) -> bool:
    return model_arg in MODELS


# --------------------------------------------------------------------------- #
# Download + file resolution
# --------------------------------------------------------------------------- #
def _allow_patterns(backend: str, precision: str) -> List[str]:
    """Restrict the snapshot download to the files we actually need."""
    if backend == "torch":
        return ["*.pth", "*.pt", "*.model", "*tokens*", "*.txt", "*.json"]
    # onnx: precision-specific files + tokenizers
    pats = ["*.onnx", "*tokens*", "*.txt", "*.json"]
    return pats


def download(entry: ModelEntry, backend: str = "onnx", precision: str = "fp32") -> Path:
    """Snapshot the model repo into the cache and return the local directory."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "huggingface_hub is required to download models. Install it with "
            "`pip install huggingface_hub` (included in the 'onnx' and 'torch' extras)."
        ) from e

    local_dir = config.models_dir() / entry.tag / backend
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=entry.repo,
        local_dir=str(local_dir),
        allow_patterns=_allow_patterns(backend, precision),
    )
    return local_dir


def _precision_suffix(precision: str) -> str:
    return {"fp32": ".onnx", "fp16": ".fp16.onnx", "int8": ".int8.onnx"}[precision]


def resolve_onnx_ctc(root: Path, precision: str) -> Optional[Path]:
    """Find the single CTC ``model.onnx`` (precision-aware) under ``root``."""
    suffix = _precision_suffix(precision)
    candidates = sorted(root.rglob("*.onnx"))
    # Prefer files matching the precision suffix and not an encoder/decoder/joiner part.
    parts = ("encoder-", "decoder-", "joiner-")
    pref = [
        p for p in candidates
        if p.name.endswith(suffix) and not p.name.startswith(parts)
        and (precision != "fp32" or (".fp16.onnx" not in p.name and ".int8.onnx" not in p.name))
    ]
    if pref:
        # 'model.onnx' / 'ctc-model.onnx' style names rank first.
        pref.sort(key=lambda p: (("model" not in p.name.lower()), len(p.name)))
        return pref[0]
    return candidates[0] if candidates else None


def resolve_onnx_transducer(root: Path, precision: str) -> Optional[Dict[str, Path]]:
    """Find encoder/decoder/joiner ONNX files (precision-aware) under ``root``.

    Mirrors the matching logic in ``inference/batch_inference.py``.
    """
    suffix = _precision_suffix(precision)
    for enc in sorted(root.rglob("encoder-*.onnx")):
        if not enc.name.endswith(suffix):
            continue
        if precision == "fp32" and (".fp16.onnx" in enc.name or ".int8.onnx" in enc.name):
            continue
        dec = enc.with_name(enc.name.replace("encoder-", "decoder-"))
        join = enc.with_name(enc.name.replace("encoder-", "joiner-"))
        if dec.exists() and join.exists():
            return {"encoder": enc, "decoder": dec, "joiner": join}
    return None


def resolve_torch_checkpoint(root: Path) -> Optional[Path]:
    """Find a ``.pth`` averaged checkpoint under ``root`` (largest = most likely)."""
    candidates = sorted(root.rglob("*.pth")) + sorted(root.rglob("*.pt"))
    if not candidates:
        return None
    # Prefer an explicitly 'avg' / averaged file; else the largest.
    avg = [p for p in candidates if "avg" in p.name.lower()]
    pool = avg or candidates
    return max(pool, key=lambda p: p.stat().st_size)


def local_model_dir(entry: ModelEntry, backend: str) -> Path:
    return config.models_dir() / entry.tag / backend


# --------------------------------------------------------------------------- #
# CLI: `zipa-cli models ...`
# --------------------------------------------------------------------------- #
def run_models_command(args) -> int:
    if args.models_command == "list":
        return _cmd_list()
    if args.models_command == "info":
        return _cmd_info(args.tag)
    if args.models_command == "download":
        return _cmd_download(args.tag, args.backend, args.precision)
    print(f"unknown models subcommand: {args.models_command}", file=sys.stderr)
    return 2


def _cmd_list() -> int:
    width = max(len(e.tag) for e in _ENTRIES)
    print(f"{'TAG'.ljust(width)}  ARCH        SIZE   STEPS  REPO")
    for e in _ENTRIES:
        cached = local_model_dir(e, "onnx").exists() or local_model_dir(e, "torch").exists()
        mark = "*" if cached else " "
        print(
            f"{mark}{e.tag.ljust(width)}  {e.arch.ljust(10)}  {e.size.ljust(5)}  "
            f"{e.steps // 1000:>4}k  {e.repo}"
        )
    print("\n  (* = present in local cache)")
    return 0


def _cmd_info(tag: str) -> int:
    e = get_entry(tag)
    if e is None:
        print(f"Unknown tag: {tag}\nRun `zipa-cli models list` to see available tags.", file=sys.stderr)
        return 1
    print(f"tag:           {e.tag}")
    print(f"description:   {e.description}")
    print(f"architecture:  {e.arch}")
    print(f"size config:   {e.size}")
    print(f"training steps:{e.steps}")
    print(f"HF repo:       https://huggingface.co/{e.repo}")
    print(f"variants:      ns={e.ns} no_diacritics={e.no_diacritics}")
    for backend in ("onnx", "torch"):
        d = local_model_dir(e, backend)
        print(f"cache [{backend}]: {'present at ' + str(d) if d.exists() else 'not downloaded'}")
    return 0


def _cmd_download(tag: str, backend: str, precision: str) -> int:
    e = get_entry(tag)
    if e is None:
        print(f"Unknown tag: {tag}", file=sys.stderr)
        return 1
    print(f"Downloading {e.tag} [{backend}, {precision}] from {e.repo} ...")
    local = download(e, backend, precision)
    print(f"Done. Cached at: {local}")
    # Report what was resolved so the user can confirm.
    if backend == "onnx":
        if e.arch == "ctc":
            p = resolve_onnx_ctc(local, precision)
            print(f"  CTC model: {p}")
        else:
            parts = resolve_onnx_transducer(local, precision)
            print(f"  Transducer files: {parts}")
    else:
        print(f"  Checkpoint: {resolve_torch_checkpoint(local)}")
    return 0
