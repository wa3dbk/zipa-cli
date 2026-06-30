"""Greedy decoding for CTC and transducer outputs (numpy / ONNX path).

Lifted and cleaned up from ``inference/utils.py`` so behaviour matches the
reference ONNX scripts exactly (blank id 0, max 3 symbols/frame for the
transducer, decoder context size 2).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

BLANK_ID = 0

# Fbank frame shift in seconds (lhotse default, 10 ms).
FRAME_SHIFT_S = 0.01
# Output-frame stride relative to the fbank frame rate, per backend/arch. This
# mirrors the length heuristic in inference/batch_inference.py (CTC ~2x, transducer
# ~4x) and is used to place emitted phones on the audio timeline.
CTC_FRAME_STRIDE = 2
TRANSDUCER_FRAME_STRIDE = 4

# A timed phone: (token, start_seconds, end_seconds).
TimedPhone = Tuple[str, float, float]


def load_tokens(token_file: str) -> Dict[int, str]:
    """Load a ``tokens.txt`` mapping ``index -> token`` (e.g. ipa_simplified/tokens.txt)."""
    tokens: Dict[int, str] = {}
    with open(token_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip("\n").split()
            if not parts:
                continue
            token = parts[0]
            idx = int(parts[1]) if len(parts) > 1 else len(tokens)
            tokens[idx] = token
    return tokens


def ctc_greedy_decode(
    probs: np.ndarray,
    vocab: Dict[int, str],
    lengths: Optional[np.ndarray] = None,
) -> List[List[str]]:
    """Collapse-repeat CTC greedy decode.

    ``probs`` is ``(B, T, V)`` or ``(T, V)``. Always returns a list of token
    lists (one per batch item).
    """
    if probs.ndim == 2:
        probs = probs[np.newaxis, :, :]
    if lengths is None:
        lengths = np.array([probs.shape[1]] * probs.shape[0], dtype=np.int64)

    preds = np.argmax(probs, axis=-1)  # (B, T)
    results: List[List[str]] = []
    for b in range(probs.shape[0]):
        decoded: List[str] = []
        prev_idx = -1
        for t in range(int(lengths[b])):
            idx = int(preds[b, t])
            if idx != BLANK_ID and idx != prev_idx:
                tok = vocab.get(idx, "")
                if tok:
                    decoded.append(tok)
            prev_idx = idx
        results.append(decoded)
    return results


def ctc_greedy_decode_with_times(
    probs: np.ndarray,
    vocab: Dict[int, str],
    lengths: Optional[np.ndarray] = None,
    frame_stride: int = CTC_FRAME_STRIDE,
    frame_shift_s: float = FRAME_SHIFT_S,
) -> List[List[TimedPhone]]:
    """Like :func:`ctc_greedy_decode` but also returns per-phone ``(start, end)``.

    Each emitted phone spans the run of consecutive frames whose argmax equals its
    id; the run boundaries are converted to seconds using
    ``frame_stride * frame_shift_s`` per output frame (see the README's note on the
    ONNX length/stride heuristic).
    """
    if probs.ndim == 2:
        probs = probs[np.newaxis, :, :]
    if lengths is None:
        lengths = np.array([probs.shape[1]] * probs.shape[0], dtype=np.int64)

    sec_per_frame = frame_stride * frame_shift_s
    preds = np.argmax(probs, axis=-1)
    results: List[List[TimedPhone]] = []
    for b in range(probs.shape[0]):
        vlen = int(lengths[b])
        p = preds[b]
        timed: List[TimedPhone] = []
        t = 0
        while t < vlen:
            idx = int(p[t])
            if idx == BLANK_ID:
                t += 1
                continue
            j = t
            while j + 1 < vlen and int(p[j + 1]) == idx:
                j += 1
            tok = vocab.get(idx, "")
            if tok:
                timed.append((tok, round(t * sec_per_frame, 4), round((j + 1) * sec_per_frame, 4)))
            t = j + 1
        results.append(timed)
    return results


def transducer_greedy_decode_with_times(
    encoder_out: np.ndarray,
    decoder_model,
    joiner_model,
    vocab: Dict[int, str],
    lengths: Optional[np.ndarray] = None,
    max_sym_per_frame: int = 3,
    context_size: int = 2,
    frame_stride: int = TRANSDUCER_FRAME_STRIDE,
    frame_shift_s: float = FRAME_SHIFT_S,
) -> List[List[TimedPhone]]:
    """Like :func:`transducer_greedy_decode` but returns per-phone ``(start, end)``.

    A phone's start is the encoder frame at which it was emitted; its end is the
    next emitted phone's start (or the utterance end for the last phone).
    """
    if encoder_out.ndim == 2:
        encoder_out = encoder_out[np.newaxis, :, :]
    if lengths is None:
        lengths = np.array([encoder_out.shape[1]] * encoder_out.shape[0], dtype=np.int64)

    sec_per_frame = frame_stride * frame_shift_s
    results: List[List[TimedPhone]] = []
    for b in range(encoder_out.shape[0]):
        T = int(lengths[b])
        enc_seq = encoder_out[b, :T, :]
        emitted: List[Tuple[str, int]] = []  # (token, frame)
        decoder_input = np.zeros((1, context_size), dtype=np.int64)
        dec_out = decoder_model.run(None, {"y": decoder_input})[0]
        for t in range(T):
            enc_frame = enc_seq[t : t + 1, :]
            for _ in range(max_sym_per_frame):
                joiner_out = joiner_model.run(
                    None, {"encoder_out": enc_frame, "decoder_out": dec_out}
                )[0]
                pred = int(np.argmax(joiner_out, axis=-1).item())
                if pred == BLANK_ID:
                    break
                tok = vocab.get(pred, "")
                if tok:
                    emitted.append((tok, t))
                decoder_input[0, 0] = decoder_input[0, 1]
                decoder_input[0, 1] = pred
                dec_out = decoder_model.run(None, {"y": decoder_input})[0]

        timed: List[TimedPhone] = []
        for i, (tok, frame) in enumerate(emitted):
            start = frame * sec_per_frame
            if i + 1 < len(emitted):
                end = max(emitted[i + 1][1] * sec_per_frame, (frame + 1) * sec_per_frame)
            else:
                end = T * sec_per_frame
            timed.append((tok, round(start, 4), round(end, 4)))
        results.append(timed)
    return results


def transducer_greedy_decode(
    encoder_out: np.ndarray,
    decoder_model,
    joiner_model,
    vocab: Dict[int, str],
    lengths: Optional[np.ndarray] = None,
    max_sym_per_frame: int = 3,
    context_size: int = 2,
) -> List[List[str]]:
    """Greedy RNN-T decode over ONNX decoder/joiner sessions.

    ``encoder_out`` is ``(B, T, D)`` or ``(T, D)``. Returns a list of token lists.
    """
    if encoder_out.ndim == 2:
        encoder_out = encoder_out[np.newaxis, :, :]
    if lengths is None:
        lengths = np.array([encoder_out.shape[1]] * encoder_out.shape[0], dtype=np.int64)

    results: List[List[str]] = []
    for b in range(encoder_out.shape[0]):
        enc_seq = encoder_out[b, : int(lengths[b]), :]
        decoded: List[str] = []
        decoder_input = np.zeros((1, context_size), dtype=np.int64)
        dec_out = decoder_model.run(None, {"y": decoder_input})[0]

        for t in range(enc_seq.shape[0]):
            enc_frame = enc_seq[t : t + 1, :]
            for _ in range(max_sym_per_frame):
                joiner_out = joiner_model.run(
                    None, {"encoder_out": enc_frame, "decoder_out": dec_out}
                )[0]
                pred = int(np.argmax(joiner_out, axis=-1).item())
                if pred == BLANK_ID:
                    break
                tok = vocab.get(pred, "")
                if tok:
                    decoded.append(tok)
                decoder_input[0, 0] = decoder_input[0, 1]
                decoder_input[0, 1] = pred
                dec_out = decoder_model.run(None, {"y": decoder_input})[0]
        results.append(decoded)
    return results
