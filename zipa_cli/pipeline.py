"""The decode pipeline: source -> batch -> features -> recognizer -> writer.

Used by both ``zipa-cli decode`` and ``zipa-cli transcribe``. The model is loaded
exactly once; samples are streamed and written incrementally so arbitrarily large
datasets fit in memory.
"""

from __future__ import annotations

import sys
from typing import List, Optional

import numpy as np

from .data import sources
from .data.batching import batches
from .data.sample import Sample
from .features import TARGET_SR, extract_fbank_batch, load_audio


def _audio_for(sample: Sample) -> np.ndarray:
    """Return a mono 16 kHz array for a sample, honouring segment start/end."""
    if sample.array is not None:
        return np.asarray(sample.array, dtype=np.float32)
    if not sample.audio_path:
        raise ValueError(f"Sample {sample.id!r} has neither array nor audio_path.")
    audio = load_audio(sample.audio_path, TARGET_SR)
    if sample.start is not None and sample.end is not None:
        a = max(0, int(round(sample.start * TARGET_SR)))
        b = min(len(audio), int(round(sample.end * TARGET_SR)))
        audio = audio[a:b]
    return audio


def _features_for_batch(batch: List[Sample]):
    """Build ``(features_padded, feat_lens)`` for a batch.

    If every sample carries precomputed fbank features we pad those directly;
    otherwise we load audio and extract fbank on the fly.
    """
    import torch

    if all(s.has_precomputed_features() for s in batch):
        feats = [
            torch.as_tensor(np.asarray(s.features, dtype=np.float32)) for s in batch
        ]
        feat_lens = np.array([f.shape[0] for f in feats], dtype=np.int64)
        padded = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True)
        return padded, feat_lens

    audio_arrays = [_audio_for(s) for s in batch]
    return extract_fbank_batch(audio_arrays)


def run_decode(args, recognizer, writer, skip_ids: Optional[set] = None, timed: bool = False) -> int:
    """Core loop shared by decode/transcribe. Returns the number of utterances written."""
    from tqdm import tqdm

    skip_ids = skip_ids or set()
    sample_iter = sources.iter_samples(args)

    written = 0
    failed = 0
    pbar = tqdm(unit="utt", desc="decoding")
    for batch in batches(
        sample_iter,
        batch_size=getattr(args, "batch_size", None),
        max_duration=getattr(args, "max_duration", None),
    ):
        batch = [s for s in batch if s.id not in skip_ids]
        if not batch:
            continue
        try:
            features_padded, feat_lens = _features_for_batch(batch)
            if timed:
                hyps = recognizer.infer_features_timed(features_padded, feat_lens)
            else:
                hyps = recognizer.infer_features(features_padded, feat_lens)
        except Exception as e:  # keep going on bad batches
            failed += len(batch)
            print(f"\n[zipa-cli] batch failed ({len(batch)} utts): {e}", file=sys.stderr)
            pbar.update(len(batch))
            continue
        for sample, hyp in zip(batch, hyps):
            if timed:
                writer.write_timed(sample, hyp)
            else:
                writer.write(sample, hyp)
            written += 1
        pbar.update(len(batch))
    pbar.close()
    if failed:
        print(f"[zipa-cli] {failed} utterances failed and were skipped.", file=sys.stderr)
    return written


def run_decode_command(args) -> int:
    from .loader import load_recognizer
    from .output import get_writer, read_existing_ids

    recognizer = load_recognizer(args)

    # `transcribe` is decode of a single file, printed to stdout.
    if args.command == "transcribe":
        sample = Sample(id="utt", audio_path=args.audio)
        features_padded, feat_lens = extract_fbank_batch([_audio_for(sample)])
        hyps = recognizer.infer_features(features_padded, feat_lens)
        print(" ".join(hyps[0]))
        return 0

    skip_ids = (
        read_existing_ids(args.output, args.output_format) if args.skip_existing else set()
    )
    if skip_ids:
        print(f"[zipa-cli] --skip-existing: {len(skip_ids)} ids already present.")

    from .output import TIMED_FORMATS

    timed = args.output_format in TIMED_FORMATS or getattr(args, "timestamps", False)

    mode = "a" if (args.skip_existing and skip_ids) else "w"
    writer = get_writer(args.output_format, args.output, model_name=str(args.model))
    if mode == "a":
        writer._fh.close()
        writer._fh = open(writer.path, "a", encoding="utf-8")
    try:
        n = run_decode(args, recognizer, writer, skip_ids, timed=timed)
    finally:
        writer.close()
    print(f"[zipa-cli] wrote {n} transcripts to {args.output} ({args.output_format}).")
    return 0
