"""``zipa-cli`` command-line entry point.

Subcommands
-----------
* ``models list|info|download`` — manage the model registry / local cache.
* ``transcribe`` — convenience decode of a single audio file.
* ``decode`` — batch decode any supported input source to a transcript dataset.
* ``compare`` — align two transcript sets and report match/sub/ins/del + PER/PFER.

Handlers import their heavy dependencies lazily, so ``zipa-cli --help`` and the
registry commands work in a minimal environment.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from . import __version__

INPUT_TYPES = ["auto", "file", "list", "dir", "tsv", "stm", "hf", "manifest", "shar"]
OUTPUT_FORMATS = ["tsv", "jsonl", "manifest", "recogs", "stm", "ctm", "align-json"]
BACKENDS = ["onnx", "torch"]
PRECISIONS = ["fp32", "fp16", "int8"]


def _add_model_args(p: argparse.ArgumentParser) -> None:
    """Arguments shared by ``decode`` and ``transcribe`` for selecting a model."""
    g = p.add_argument_group("model")
    g.add_argument(
        "--model",
        required=True,
        help="Registry tag (e.g. 'zipa-cr-s-300k') or a local path to a model "
        "(.onnx/.pth file, or a directory for an ONNX transducer).",
    )
    g.add_argument(
        "--backend",
        choices=BACKENDS,
        default="onnx",
        help="Inference backend. 'onnx' (default) needs minimal deps; 'torch' "
        "needs torch+icefall+k2 and runs the full PyTorch checkpoints.",
    )
    g.add_argument(
        "--model-type",
        choices=["ctc", "transducer"],
        default=None,
        help="Architecture. Inferred from the registry tag; required when --model "
        "is a raw local path.",
    )
    g.add_argument(
        "--precision",
        choices=PRECISIONS,
        default="fp32",
        help="ONNX precision to use/download (default: fp32).",
    )
    g.add_argument(
        "--tokens",
        default=None,
        help="Path to tokens.txt (ONNX). Defaults to the bundled ipa_simplified/tokens.txt.",
    )
    g.add_argument(
        "--bpe-model",
        default=None,
        help="Path to the sentencepiece model (PyTorch). Defaults to "
        "ipa_simplified/unigram_127.model.",
    )
    g.add_argument(
        "--device",
        default="auto",
        help="torch device: auto|cpu|cuda|cuda:N (PyTorch backend; ONNX uses CPU/GPU "
        "providers automatically).",
    )
    g.add_argument(
        "--zipa-repo",
        default=None,
        help="Path to the ZIPA training repo (for the PyTorch backend imports and "
        "bundled tokenizers). Auto-detected when adjacent.",
    )


def _add_batch_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("batching")
    mx = g.add_mutually_exclusive_group()
    mx.add_argument("--batch-size", type=int, default=None, help="Fixed number of utterances per batch.")
    mx.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Dynamic batching: max pooled audio seconds per batch (lhotse-style).",
    )
    g.add_argument("--num-workers", type=int, default=4, help="Audio/feature loading workers.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zipa-cli",
        description="Batch phonetic decoding with ZIPA zipformer phone-recognition models.",
    )
    parser.add_argument("--version", action="version", version=f"zipa-cli {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # ----- models ----------------------------------------------------------- #
    p_models = sub.add_parser("models", help="List, inspect, and download models.")
    msub = p_models.add_subparsers(dest="models_command", required=True)
    msub.add_parser("list", help="List all known model tags.")
    p_info = msub.add_parser("info", help="Show details for a model tag.")
    p_info.add_argument("tag", help="Registry tag.")
    p_dl = msub.add_parser("download", help="Download a model into the local cache.")
    p_dl.add_argument("tag", help="Registry tag.")
    p_dl.add_argument("--backend", choices=BACKENDS, default="onnx")
    p_dl.add_argument("--precision", choices=PRECISIONS, default="fp32")

    # ----- transcribe ------------------------------------------------------- #
    p_tr = sub.add_parser("transcribe", help="Decode a single audio file (prints phones).")
    p_tr.add_argument("audio", help="Path to an audio file.")
    _add_model_args(p_tr)

    # ----- decode ----------------------------------------------------------- #
    p_dec = sub.add_parser("decode", help="Batch decode an input source to a transcript dataset.")
    p_dec.add_argument("--input", required=True, help="Input path / dataset spec.")
    p_dec.add_argument("--input-type", choices=INPUT_TYPES, default="auto")
    _add_model_args(p_dec)
    _add_batch_args(p_dec)
    p_dec.add_argument("--output", "-o", required=True, help="Output path.")
    p_dec.add_argument("--output-format", choices=OUTPUT_FORMATS, default="tsv")
    p_dec.add_argument("--skip-existing", action="store_true", help="Skip ids already present in --output.")
    p_dec.add_argument(
        "--timestamps",
        action="store_true",
        help="Emit per-phone start/end times (always on for ctm/align-json; adds an "
        "'alignment' field to jsonl). ONNX backend only.",
    )
    # input-type specific options
    p_dec.add_argument("--audio-dir", default=None, help="STM: directory holding the audio files.")
    p_dec.add_argument("--id-col", default=None, help="TSV: id column name or index.")
    p_dec.add_argument("--path-col", default=None, help="TSV: audio-path column name or index.")
    p_dec.add_argument("--ref-col", default=None, help="TSV: optional reference-transcript column.")
    p_dec.add_argument("--hf-dataset", default=None, help="HF dataset name (input-type hf).")
    p_dec.add_argument("--hf-split", default="test", help="HF split (default: test).")
    p_dec.add_argument("--hf-config", default=None, help="HF dataset config/subset name.")
    p_dec.add_argument("--audio-column", default="audio", help="HF audio column (default: audio).")
    p_dec.add_argument("--recordings", default=None, help="Manifest: recordings jsonl(.gz).")
    p_dec.add_argument("--supervisions", default=None, help="Manifest: supervisions jsonl(.gz).")
    p_dec.add_argument("--cuts", default=None, help="Manifest: cuts jsonl(.gz) (may have precomputed feats).")

    # ----- compare ---------------------------------------------------------- #
    p_cmp = sub.add_parser("compare", help="Compare two transcript sets (match/sub/ins/del, PER, PFER).")
    src_a = p_cmp.add_mutually_exclusive_group(required=True)
    src_a.add_argument("--a", dest="hyp_a", help="Transcript file A (tsv/jsonl).")
    src_b = p_cmp.add_mutually_exclusive_group(required=True)
    src_b.add_argument("--b", dest="hyp_b", help="Transcript file B (tsv/jsonl).")
    p_cmp.add_argument("--output", "-o", default=None, help="Write the report here (default: stdout).")
    p_cmp.add_argument("--format", dest="report_format", choices=["txt", "jsonl", "csv"], default="txt")
    p_cmp.add_argument("--no-pfer", action="store_true", help="Skip panphon feature edit distance.")
    p_cmp.add_argument("--top-k", type=int, default=20, help="How many top confusions/utterances to show.")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "models":
        from .registry import run_models_command

        return run_models_command(args)
    if args.command in ("decode", "transcribe"):
        from .pipeline import run_decode_command

        return run_decode_command(args)
    if args.command == "compare":
        from .analysis import run_compare_command

        return run_compare_command(args)

    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
