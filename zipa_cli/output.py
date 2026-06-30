"""Streaming transcript writers.

Each writer consumes ``(Sample, phones)`` pairs and serialises them so large
datasets never have to be buffered in memory. Formats:

* ``tsv``      ``id<TAB>p h o n e s`` — directly reusable downstream;
* ``jsonl``    one rich JSON record per utterance;
* ``manifest`` a lhotse SupervisionSet (phones in ``text``), round-trips into k2;
* ``recogs``   the ``hyp=/ref=`` format that ``scripts/evaluate.py`` parses;
* ``stm``      ``file spk chan start end phones`` segments.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from .data.sample import Sample


class BaseWriter:
    def __init__(self, path: str, model_name: str = ""):
        self.path = Path(path)
        self.model_name = model_name
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8")

    def write(self, sample: Sample, phones: List[str]) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def write_timed(self, sample: Sample, timed: List) -> None:
        """Write a sample whose phones carry ``(token, start_s, end_s)`` timings.

        The default drops the timings; timed-aware writers override this.
        """
        self.write(sample, [t[0] for t in timed])

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class TsvWriter(BaseWriter):
    def write(self, sample: Sample, phones: List[str]) -> None:
        self._fh.write(f"{sample.id}\t{' '.join(phones)}\n")


class JsonlWriter(BaseWriter):
    def _base_record(self, sample: Sample, phones: List[str]) -> dict:
        rec = {
            "id": sample.id,
            "phones": phones,
            "text": " ".join(phones),
            "model": self.model_name,
        }
        if sample.audio_path:
            rec["audio"] = sample.audio_path
        if sample.start is not None:
            rec["start"] = sample.start
            rec["end"] = sample.end
        if sample.speaker is not None:
            rec["speaker"] = sample.speaker
        if sample.ref is not None:
            rec["ref"] = sample.ref
        return rec

    def write(self, sample: Sample, phones: List[str]) -> None:
        self._fh.write(json.dumps(self._base_record(sample, phones), ensure_ascii=False) + "\n")

    def write_timed(self, sample: Sample, timed: List) -> None:
        rec = self._base_record(sample, [t[0] for t in timed])
        offset = sample.start or 0.0
        rec["alignment"] = [
            {"p": tok, "start": round(s + offset, 4), "end": round(e + offset, 4)}
            for tok, s, e in timed
        ]
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


class CtmWriter(BaseWriter):
    """NIST CTM: ``recording channel start dur token`` (one line per phone)."""

    def write(self, sample: Sample, phones: List[str]) -> None:
        # Without timings we cannot emit a meaningful CTM; emit zero-width marks.
        rec_id = Path(sample.audio_path).stem if sample.audio_path else sample.id
        offset = sample.start or 0.0
        for tok in phones:
            self._fh.write(f"{rec_id} {sample.channel or 0} {offset:.3f} 0.000 {tok}\n")

    def write_timed(self, sample: Sample, timed: List) -> None:
        rec_id = Path(sample.audio_path).stem if sample.audio_path else sample.id
        offset = sample.start or 0.0
        for tok, s, e in timed:
            self._fh.write(
                f"{rec_id} {sample.channel or 0} {s + offset:.3f} {max(e - s, 0.0):.3f} {tok}\n"
            )


class AlignJsonWriter(BaseWriter):
    """One JSON line per utterance with timed phones — consumed by the web viewer."""

    def write(self, sample: Sample, phones: List[str]) -> None:
        self.write_timed(sample, [(p, 0.0, 0.0) for p in phones])

    def write_timed(self, sample: Sample, timed: List) -> None:
        offset = sample.start or 0.0
        phones = [
            {"p": tok, "start": round(s + offset, 4), "end": round(e + offset, 4)}
            for tok, s, e in timed
        ]
        duration = sample.duration
        if duration is None and phones:
            duration = phones[-1]["end"]
        rec = {
            "id": sample.id,
            "model": self.model_name,
            "audio": sample.audio_path,
            "duration": duration,
            "phones": phones,
        }
        if sample.ref is not None:
            rec["ref"] = sample.ref
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


class RecogsWriter(BaseWriter):
    """Emit hyp (and ref when available) lines compatible with scripts/evaluate.py."""

    def write(self, sample: Sample, phones: List[str]) -> None:
        self._fh.write(f"{sample.id}\thyp={phones!r}\n")
        if sample.ref is not None:
            ref_tokens = sample.ref.split()
            self._fh.write(f"{sample.id}\tref={ref_tokens!r}\n")


class StmWriter(BaseWriter):
    def write(self, sample: Sample, phones: List[str]) -> None:
        filename = Path(sample.audio_path).stem if sample.audio_path else sample.id
        speaker = sample.speaker or "spk"
        channel = sample.channel if sample.channel is not None else 0
        start = sample.start if sample.start is not None else 0.0
        end = sample.end if sample.end is not None else (sample.duration or 0.0)
        self._fh.write(f"{filename} {speaker} {channel} {start:.3f} {end:.3f} {' '.join(phones)}\n")


class ManifestWriter(BaseWriter):
    """Write a lhotse SupervisionSet with the phones as the supervision text.

    Falls back to a plain jsonl of supervision-like dicts if lhotse is missing.
    """

    def __init__(self, path: str, model_name: str = ""):
        super().__init__(path, model_name)
        self._segments = []
        try:
            from lhotse import SupervisionSegment  # noqa: F401

            self._have_lhotse = True
        except Exception:
            self._have_lhotse = False

    def write(self, sample: Sample, phones: List[str]) -> None:
        text = " ".join(phones)
        start = sample.start if sample.start is not None else 0.0
        duration = (
            (sample.end - sample.start)
            if (sample.start is not None and sample.end is not None)
            else (sample.duration or 0.0)
        )
        if self._have_lhotse:
            from lhotse import SupervisionSegment

            self._segments.append(
                SupervisionSegment(
                    id=sample.id,
                    recording_id=Path(sample.audio_path).stem if sample.audio_path else sample.id,
                    start=start,
                    duration=duration,
                    channel=sample.channel or 0,
                    text=text,
                    speaker=sample.speaker,
                    custom={"phones": phones, "model": self.model_name},
                )
            )
        else:
            self._segments.append(
                {
                    "id": sample.id,
                    "recording_id": Path(sample.audio_path).stem if sample.audio_path else sample.id,
                    "start": start,
                    "duration": duration,
                    "text": text,
                    "custom": {"phones": phones, "model": self.model_name},
                }
            )

    def close(self) -> None:
        if self._have_lhotse:
            from lhotse import SupervisionSet

            # We already opened a file handle; close it and let lhotse write.
            self._fh.close()
            SupervisionSet.from_segments(self._segments).to_file(str(self.path))
        else:
            for seg in self._segments:
                self._fh.write(json.dumps(seg, ensure_ascii=False) + "\n")
            self._fh.close()


_WRITERS = {
    "tsv": TsvWriter,
    "jsonl": JsonlWriter,
    "recogs": RecogsWriter,
    "stm": StmWriter,
    "manifest": ManifestWriter,
    "ctm": CtmWriter,
    "align-json": AlignJsonWriter,
}

# Formats that inherently require per-phone timings.
TIMED_FORMATS = {"ctm", "align-json"}


def get_writer(fmt: str, path: str, model_name: str = "") -> BaseWriter:
    if fmt not in _WRITERS:
        raise ValueError(f"Unknown output format: {fmt!r}")
    return _WRITERS[fmt](path, model_name)


def read_existing_ids(path: str, fmt: str) -> set:
    """Best-effort set of ids already written (for --skip-existing)."""
    p = Path(path)
    if not p.exists():
        return set()
    ids = set()
    try:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                if fmt == "jsonl":
                    try:
                        ids.add(json.loads(line)["id"])
                    except Exception:
                        pass
                else:
                    ids.add(line.split("\t")[0].split()[0])
    except Exception:
        return set()
    return ids
