"""The unit of work that flows through the decode pipeline.

Every input source (file, list, dir, tsv, stm, hf, manifest, shar) is normalised
to a stream of :class:`Sample`. A sample carries enough to (a) obtain a mono
16 kHz waveform and (b) join the resulting transcript back to its source id.

A sample provides audio one of two ways:

* ``audio_path`` (+ optional ``start`` / ``end`` for a segment of it), or
* ``array`` already-decoded float32 mono @ 16 kHz (HF / lhotse cuts), or
* ``features`` precomputed fbank ``(T, 80)`` (lhotse precomputed cuts).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Sample:
    id: str
    audio_path: Optional[str] = None
    array: Optional[np.ndarray] = None          # mono float32 @ 16 kHz
    sampling_rate: Optional[int] = None
    features: Optional[np.ndarray] = None        # precomputed fbank (T, 80)
    start: Optional[float] = None                # segment start (seconds)
    end: Optional[float] = None                  # segment end (seconds)
    channel: int = 0
    speaker: Optional[str] = None
    ref: Optional[str] = None                    # optional reference transcript
    duration: Optional[float] = None             # seconds, used for batching/sorting

    def has_precomputed_features(self) -> bool:
        return self.features is not None
