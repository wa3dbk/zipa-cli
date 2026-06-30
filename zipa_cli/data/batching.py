"""Group a stream of :class:`Sample` into batches.

Two modes:

* **fixed** ``batch_size`` — N utterances per batch, in arrival order;
* **dynamic** ``max_duration`` — accumulate until the summed audio seconds would
  exceed the budget (lhotse-style), which keeps GPU memory bounded across very
  uneven utterance lengths.

In dynamic mode we sort within a bounded look-ahead buffer so similar-length
utterances batch together (less padding) without materialising the whole
dataset.
"""

from __future__ import annotations

from typing import Iterable, Iterator, List, Optional

from .sample import Sample


def _duration_of(s: Sample) -> float:
    if s.duration is not None:
        return float(s.duration)
    if s.start is not None and s.end is not None:
        return float(s.end - s.start)
    if s.array is not None and s.sampling_rate:
        return len(s.array) / float(s.sampling_rate)
    # Unknown: probe the file header cheaply if we can.
    if s.audio_path:
        try:
            import soundfile as sf

            info = sf.info(s.audio_path)
            return info.frames / float(info.samplerate)
        except Exception:
            return 0.0
    return 0.0


def batches(
    samples: Iterable[Sample],
    batch_size: Optional[int] = None,
    max_duration: Optional[float] = None,
    sort_buffer: int = 256,
) -> Iterator[List[Sample]]:
    """Yield batches of samples according to ``batch_size`` or ``max_duration``."""
    if max_duration is not None:
        yield from _dynamic_batches(samples, max_duration, sort_buffer)
        return

    bs = batch_size or 8
    batch: List[Sample] = []
    for s in samples:
        batch.append(s)
        if len(batch) >= bs:
            yield batch
            batch = []
    if batch:
        yield batch


def _dynamic_batches(
    samples: Iterable[Sample], max_duration: float, sort_buffer: int
) -> Iterator[List[Sample]]:
    buf: List[Sample] = []

    def flush_buffer_into_batches(items: List[Sample]) -> Iterator[List[Sample]]:
        items.sort(key=_duration_of)
        batch: List[Sample] = []
        pooled = 0.0
        for s in items:
            d = _duration_of(s)
            if batch and pooled + d > max_duration:
                yield batch
                batch, pooled = [], 0.0
            batch.append(s)
            pooled += d
        if batch:
            yield batch

    for s in samples:
        buf.append(s)
        if len(buf) >= sort_buffer:
            yield from flush_buffer_into_batches(buf)
            buf = []
    if buf:
        yield from flush_buffer_into_batches(buf)
