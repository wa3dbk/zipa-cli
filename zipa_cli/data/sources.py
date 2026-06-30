"""Readers that turn each input format into a stream of :class:`Sample`.

Supported ``--input-type`` values:

* ``file``     single audio file
* ``list``     a text file of audio paths (one per line)
* ``dir``      a directory searched recursively for audio
* ``tsv``      a TSV with id / path (+ optional reference) columns (CommonVoice-style)
* ``stm``      ``filename speaker channel start end [text]`` segments + ``--audio-dir``
* ``hf``       a HuggingFace dataset (``--hf-dataset``)
* ``manifest`` lhotse/icefall manifests (``--cuts`` or ``--recordings``/``--supervisions``)
* ``shar``     a directory of lhotse shar shards (``cuts.*.jsonl.gz`` + ``recording.*.tar``)
* ``auto``     inferred from the path / flags
"""

from __future__ import annotations

import gzip
import os
from pathlib import Path
from typing import Iterator, List, Optional

from .sample import Sample

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".wma")


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def iter_samples(args) -> Iterator[Sample]:
    input_type = args.input_type
    if input_type == "auto":
        input_type = _detect_type(args)
        print(f"[zipa-cli] auto-detected input type: {input_type}")

    if input_type == "file":
        return _from_file(args.input)
    if input_type == "list":
        return _from_list(args.input)
    if input_type == "dir":
        return _from_dir(args.input)
    if input_type == "tsv":
        return _from_tsv(args)
    if input_type == "stm":
        return _from_stm(args)
    if input_type == "hf":
        return _from_hf(args)
    if input_type == "manifest":
        return _from_manifest(args)
    if input_type == "shar":
        return _from_shar(args)
    raise ValueError(f"Unknown input type: {input_type!r}")


def _detect_type(args) -> str:
    if getattr(args, "hf_dataset", None):
        return "hf"
    if getattr(args, "recordings", None) or getattr(args, "cuts", None):
        return "manifest"
    p = Path(args.input)
    if p.is_dir():
        # A directory of shar shards vs a directory of audio.
        if any(p.glob("cuts.*.jsonl.gz")) or any(p.glob("cuts*.jsonl.gz")):
            return "shar"
        return "dir"
    suf = "".join(p.suffixes).lower()
    name = p.name.lower()
    if name.endswith(".tsv") or name.endswith(".csv"):
        return "tsv"
    if name.endswith(".stm"):
        return "stm"
    if ".jsonl" in suf:
        return "manifest"
    if p.suffix.lower() in AUDIO_EXTS:
        return "file"
    if p.suffix.lower() in (".txt", ".list"):
        return "list"
    return "file"


# --------------------------------------------------------------------------- #
# Simple file-based sources
# --------------------------------------------------------------------------- #
def _from_file(path: str) -> Iterator[Sample]:
    p = Path(path)
    yield Sample(id=p.stem, audio_path=str(p))


def _from_list(path: str) -> Iterator[Sample]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ap = Path(line)
            yield Sample(id=ap.stem, audio_path=str(ap))


def _from_dir(path: str) -> Iterator[Sample]:
    root = Path(path)
    files: List[Path] = []
    for ext in AUDIO_EXTS:
        files.extend(root.rglob(f"*{ext}"))
    for ap in sorted(files):
        rel = ap.relative_to(root).with_suffix("")
        yield Sample(id=str(rel).replace(os.sep, "/"), audio_path=str(ap))


# --------------------------------------------------------------------------- #
# TSV (CommonVoice-style)
# --------------------------------------------------------------------------- #
def _col_index(spec: Optional[str], header: Optional[List[str]], default: int) -> int:
    if spec is None:
        return default
    if spec.isdigit():
        return int(spec)
    if header and spec in header:
        return header.index(spec)
    raise ValueError(f"Column {spec!r} not found in header {header}")


def _from_tsv(args) -> Iterator[Sample]:
    import csv

    path = Path(args.input)
    delim = "," if path.suffix.lower() == ".csv" else "\t"
    base_dir = path.parent
    with open(path, "r", encoding="utf-8", newline="") as f:
        sample_chunk = f.read(4096)
        f.seek(0)
        has_header = _looks_like_header(sample_chunk, delim)
        reader = csv.reader(f, delimiter=delim)
        header = None
        if has_header:
            header = next(reader)
        id_i = _col_index(args.id_col, header, 0)
        path_i = _col_index(args.path_col, header, 1)
        ref_i = _col_index(args.ref_col, header, None) if args.ref_col else None
        for n, row in enumerate(reader):
            if not row:
                continue
            audio = row[path_i]
            ap = Path(audio)
            if not ap.is_absolute():
                ap = base_dir / ap
            sid = row[id_i] if id_i < len(row) else f"utt{n}"
            ref = row[ref_i] if (ref_i is not None and ref_i < len(row)) else None
            yield Sample(id=sid, audio_path=str(ap), ref=ref)


def _looks_like_header(chunk: str, delim: str) -> bool:
    first = chunk.splitlines()[0] if chunk else ""
    cols = first.split(delim)
    # Heuristic: a header has no obvious audio path in it.
    return not any(c.lower().endswith(AUDIO_EXTS) for c in cols)


# --------------------------------------------------------------------------- #
# STM segments + audio dir
# --------------------------------------------------------------------------- #
def _find_audio(audio_dir: Path, name: str) -> Optional[Path]:
    cand = audio_dir / name
    if cand.exists():
        return cand
    for ext in AUDIO_EXTS:
        c = audio_dir / f"{name}{ext}"
        if c.exists():
            return c
    matches = list(audio_dir.glob(f"{name}.*"))
    return matches[0] if matches else None


def _from_stm(args) -> Iterator[Sample]:
    if not args.audio_dir:
        raise ValueError("STM input requires --audio-dir pointing at the audio files.")
    audio_dir = Path(args.audio_dir)
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";;"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            filename, speaker, channel, start, end = parts[:5]
            try:
                start_f, end_f = float(start), float(end)
            except ValueError:
                continue
            ref = " ".join(parts[5:]) if len(parts) > 5 else None
            ap = _find_audio(audio_dir, filename)
            if ap is None:
                print(f"[zipa-cli] warning: no audio for STM entry {filename!r} in {audio_dir}")
                continue
            sid = f"{filename}-{start_f:.3f}-{end_f:.3f}"
            yield Sample(
                id=sid, audio_path=str(ap), start=start_f, end=end_f,
                channel=int(channel) if channel.isdigit() else 0,
                speaker=speaker, ref=ref, duration=end_f - start_f,
            )


# --------------------------------------------------------------------------- #
# HuggingFace dataset
# --------------------------------------------------------------------------- #
def _from_hf(args) -> Iterator[Sample]:
    from datasets import load_dataset

    from ..features import to_mono_16k

    name = args.hf_dataset or args.input
    ds = load_dataset(name, args.hf_config, split=args.hf_split, streaming=True)
    audio_col = args.audio_column
    for n, row in enumerate(ds):
        audio = row[audio_col]
        array = audio["array"]
        sr = audio["sampling_rate"]
        arr16 = to_mono_16k(array, sr)
        sid = str(row.get("id") or row.get("path") or audio.get("path") or f"utt{n}")
        ref = row.get("sentence") or row.get("text") or row.get("transcription")
        yield Sample(id=sid, array=arr16, sampling_rate=16000, ref=ref,
                     duration=len(arr16) / 16000.0)


# --------------------------------------------------------------------------- #
# lhotse / icefall manifests (recordings+supervisions, or cuts) and shar
# --------------------------------------------------------------------------- #
def _cut_to_sample(cut) -> Sample:
    from ..features import to_mono_16k

    ref = None
    sups = getattr(cut, "supervisions", None)
    if sups:
        ref = " ".join(s.text for s in sups if getattr(s, "text", None)) or None

    features = None
    array = None
    if getattr(cut, "has_features", False):
        try:
            features = cut.load_features()
        except Exception:
            features = None
    if features is None:
        audio = cut.load_audio()  # (channels, samples)
        array = to_mono_16k(audio, int(cut.sampling_rate))
    return Sample(
        id=cut.id,
        array=array,
        sampling_rate=16000 if array is not None else None,
        features=features,
        ref=ref,
        duration=float(cut.duration),
    )


def _from_manifest(args) -> Iterator[Sample]:
    from lhotse import CutSet, load_manifest_lazy

    if args.cuts:
        cuts = load_manifest_lazy(args.cuts)
    elif args.recordings and args.supervisions:
        recordings = load_manifest_lazy(args.recordings)
        supervisions = load_manifest_lazy(args.supervisions)
        cuts = CutSet.from_manifests(recordings=recordings, supervisions=supervisions)
    elif args.input:
        cuts = load_manifest_lazy(args.input)
    else:
        raise ValueError(
            "Manifest input requires --cuts, or --recordings together with --supervisions."
        )
    for cut in cuts:
        yield _cut_to_sample(cut)


def _from_shar(args) -> Iterator[Sample]:
    from glob import glob

    from lhotse import CutSet

    root = args.input
    cuts = sorted(glob(os.path.join(root, "cuts*.jsonl.gz")))
    recordings = sorted(glob(os.path.join(root, "recording*.tar")))
    if not cuts:
        raise ValueError(f"No shar cut shards (cuts*.jsonl.gz) found in {root}")
    fields = {"cuts": cuts}
    if recordings:
        fields["recording"] = recordings
    cutset = CutSet.from_shar(fields=fields)
    for cut in cutset:
        yield _cut_to_sample(cut)
