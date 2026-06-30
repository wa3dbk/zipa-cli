# zipa-cli

Batch **phonetic decoding** for the [ZIPA](https://aclanthology.org/2025.acl-long.961/)
family of zipformer phone-recognition models. One command-line tool to run greedy
IPA decoding over many input formats — single files, file lists, directories,
CommonVoice-style TSVs, STM segment files, HuggingFace datasets, and
lhotse/icefall manifests — using either the **minimal-dependency ONNX backend**
(default) or the **full PyTorch backend**, with a built-in model registry,
multiple transcript output formats, and a two-model comparison/analysis mode.

The model is loaded **once** per run and inputs are streamed, so you can decode
arbitrarily large datasets on a GPU without reloading or running out of memory.

> ⚠️ **Caution — ONNX CTC frame-length heuristic.** For the **ONNX CTC** path the
> output length is approximated as `feat_lens // 2` (and `feat_lens // 4` for the
> ONNX transducer), carried verbatim from the reference `inference/batch_inference.py`
> and clipped to the actual session output. The model's true `subsampling_factor`
> is **4**, but the CR-CTC head emits at roughly half that stride, hence `//2`. This
> only affects how many trailing frames are decoded and the time stride used for
> `--timestamps`; the **PyTorch** backend instead uses the encoder's *exact*
> `encoder_out_lens` and is the ground truth. **Validate the ONNX CTC output and
> timecodes against a known reference on your first real run.** See
> [ONNX length heuristic](#caution-onnx-length-heuristic) for details.

---

## Installation

The CLI lives next to the ZIPA training repo (the `zipa/` directory that contains
`zipformer_crctc/`, `zipformer_transducer/`, and `ipa_simplified/`). From the
parent directory:

### Minimal install (ONNX backend — recommended)
Decode with the exported ONNX models. No `torch`/`icefall`/`k2` required.

```bash
pip install -e ".[onnx]"
```

This pulls `onnxruntime soundfile librosa lhotse huggingface_hub sentencepiece`.

### Full install (PyTorch backend — large-scale GPU)
Run the original `.pth` checkpoints. Also needs **icefall + k2** installed
separately, matching your torch/cuda versions (see the ZIPA README).

```bash
pip install -e ".[torch]"
# then install icefall + k2 per https://icefall.readthedocs.io
```

### Optional extras
```bash
pip install -e ".[hf]"        # HuggingFace dataset input
pip install -e ".[analysis]"  # panphon PFER in compare mode
pip install -e ".[all]"       # everything
```

### Configuration
| Env var | Meaning | Default |
|---|---|---|
| `ZIPA_CLI_CACHE` | where downloaded models are cached | `~/.cache/zipa-cli` |
| `ZIPA_REPO` | path to the ZIPA training repo (for PyTorch imports + bundled tokenizers) | auto-detected (sibling `zipa/`) |

You can also pass `--zipa-repo`, `--tokens`, and `--bpe-model` explicitly.

---

## Models

`zipa-cli` knows all 12 released averaged checkpoints by short tag.

```bash
zipa-cli models list
zipa-cli models info  zipa-cr-l-500k
zipa-cli models download zipa-cr-s-300k --backend onnx --precision fp16
```

| Tag pattern | Arch | Example |
|---|---|---|
| `zipa-t-{s,l}-{300k,500k}` | transducer | `zipa-t-l-500k` |
| `zipa-cr-{s,l}-{300k,500k}` | crctc (ctc) | `zipa-cr-s-300k` |
| `zipa-cr-ns-{s,l}-{700k,800k}` | crctc Ns | `zipa-cr-ns-l-800k` |
| `zipa-cr-ns-nd-{s,l}-…` | crctc Ns, no diacritics | `zipa-cr-ns-nd-l-780k` |

Anywhere a model is needed you may pass **a tag** (auto-downloaded and cached) or
**a local path** (an `.onnx` file / transducer directory, or a `.pth` checkpoint).

---

## Repository layout

| Path | Contents |
|---|---|
| `zipa_cli/` | the installable package (CLI, backends, sources, writers, analysis) |
| `tests/` | pytest suite (`pytest` from the repo root) |
| `docs/` | comprehensive LaTeX user manual (`cd docs && make`) |
| `tutorials/` | runnable Jupyter notebooks (quickstart, datasets/manifests, compare+visualize) |
| `web/` | client-side waveform/spectrogram + multi-model phone-tier viewer |
| `zipa/` | the upstream ZIPA training repo (models, tokenizers, reference scripts) |

## Quick start

```bash
# Single file (prints phones)
zipa-cli transcribe sample.wav --model zipa-cr-s-300k

# A directory of audio on GPU, dynamic batching, to a TSV
zipa-cli decode --input audio_dir/ --input-type dir \
    --model zipa-cr-l-500k --backend onnx \
    --max-duration 600 -o transcripts.tsv

# Compare two models' outputs
zipa-cli decode --input audio_dir/ --model zipa-cr-s-300k -o small.tsv
zipa-cli decode --input audio_dir/ --model zipa-cr-l-500k -o large.tsv
zipa-cli compare --a small.tsv --b large.tsv
```

---

## Input sources (`--input-type`)

| Type | What it is | Key flags |
|---|---|---|
| `file` | one audio file | — |
| `list` | text file of audio paths (one per line) | — |
| `dir` | directory searched recursively for audio | — |
| `tsv` | `id`/`path` (+ optional ref) columns, CommonVoice-style | `--id-col --path-col --ref-col` |
| `stm` | `file spk chan start end [text]` segments | `--audio-dir` |
| `hf` | a HuggingFace dataset | `--hf-dataset --hf-split --hf-config --audio-column` |
| `manifest` | lhotse/icefall manifests | `--cuts` **or** `--recordings --supervisions` |
| `shar` | directory of lhotse shar shards | — |
| `auto` | inferred from the path/flags (default) | — |

Examples:

```bash
# CommonVoice TSV (segid -> clip)
zipa-cli decode --input cv/test.tsv --input-type tsv \
    --id-col path --path-col path --ref-col sentence \
    --model zipa-cr-l-500k -o cv.tsv

# STM segments + an audio directory
zipa-cli decode --input show.stm --input-type stm --audio-dir audio/ \
    --model zipa-cr-l-500k --output-format stm -o show.hyp.stm

# HuggingFace dataset (streaming)
zipa-cli decode --input-type hf --hf-dataset mozilla-foundation/common_voice_17_0 \
    --hf-config en --hf-split test --audio-column audio \
    --model zipa-cr-l-500k -o cv17_en.jsonl --output-format jsonl

# icefall manifest (recordings + supervisions)
zipa-cli decode --input-type manifest \
    --recordings data/recordings.jsonl.gz --supervisions data/supervisions.jsonl.gz \
    --model zipa-cr-l-500k -o hyp.manifest.jsonl --output-format manifest

# Precomputed fbank cuts (feature extraction is skipped automatically)
zipa-cli decode --input-type manifest --cuts data/cuts.jsonl.gz \
    --model zipa-cr-l-500k -o hyp.tsv

# lhotse shar shards (cuts.*.jsonl.gz + recording.*.tar)
zipa-cli decode --input data-shar/ --input-type shar \
    --model zipa-cr-l-500k -o hyp.tsv
```

---

## Output formats (`--output-format`)

| Format | Description |
|---|---|
| `tsv` (default) | `id<TAB>p h o n e s` — easy to join downstream |
| `jsonl` | rich record per utterance (`id, phones, text, audio, start, end, ref, model`) |
| `manifest` | lhotse `SupervisionSet`; phones in `text` + `custom.phones` — round-trips into k2/lhotse |
| `recogs` | `hyp=/ref=` lines parsed by `zipa/scripts/evaluate.py` (free PFER eval) |
| `stm` | `file spk chan start end phones` segments |
| `ctm` | NIST CTM: `recording channel start dur phone` (timed) |
| `align-json` | one JSON line per utterance with timed phones — feeds the web viewer (timed) |

When the input carries a reference transcript (TSV `--ref-col`, STM text, manifest
supervisions), it is preserved into `jsonl`/`recogs` for evaluation.

`--skip-existing` appends to an existing output, skipping ids already present —
handy for resuming a long run.

---

## Timestamps & alignment

`ctm` and `align-json` always emit per-phone `start`/`end` times; adding
`--timestamps` also attaches an `alignment` array to `jsonl` output.

```bash
# Per-phone CTM
zipa-cli decode --input utt.wav --model zipa-cr-l-500k \
    --output-format ctm -o utt.ctm

# Alignment JSON for the web viewer
zipa-cli decode --input audio_dir/ --input-type dir --model zipa-cr-l-500k \
    --output-format align-json -o aligned.json
```

Timings are derived from the greedy alignment. **ONNX timings are approximate**
(stride ≈ 20 ms/frame for CTC, 40 ms for the transducer); the PyTorch backend uses
exact encoder lengths. Timestamped decoding currently requires the **ONNX backend**.
See [Caution: ONNX length heuristic](#caution-onnx-length-heuristic).

---

## Web viewer for phoneticians

A zero-install, client-side viewer in [`web/`](web/) renders the waveform (toggle to
a spectrogram) with one **phone tier per model**, positioned by timecode — ideal for
eyeballing where two models insert/substitute/delete phones.

```bash
# 1. produce align-json for each model
zipa-cli decode --input utt.wav --model zipa-cr-l-500k --output-format align-json -o large.json
zipa-cli decode --input utt.wav --model zipa-cr-s-300k --output-format align-json -o small.json
# 2. serve the viewer and open it
cd web && python -m http.server 8000   # http://localhost:8000
```

Load the audio file and both JSONs; entries sharing an utterance id stack as tiers.
See [`web/README.md`](web/README.md).

---

## Batching & device

* `--batch-size N` — fixed utterances per batch, **or**
* `--max-duration S` — dynamic, lhotse-style: cap pooled audio seconds per batch
  (utterances are length-sorted within a buffer to minimise padding).
* `--num-workers N` — audio/feature loading workers.
* `--device auto|cpu|cuda|cuda:N` — PyTorch backend device. The ONNX backend uses
  the CUDA execution provider automatically when available.
* `--precision fp32|fp16|int8` — ONNX precision to download/use.

---

## Compare / analysis mode

```bash
zipa-cli compare --a modelA.tsv --b modelB.tsv [--format txt|jsonl|csv] [-o report.txt]
```

Treats **A as reference**, **B as hypothesis**, joins by id, and reports:

* corpus + per-utterance **Match / Substitution / Insertion / Deletion** counts,
* **PER** = `(S+I+D)/N_ref`,
* **PFER** (panphon articulatory feature edit distance; install `[analysis]`),
* most frequent substitution pairs and inserted/deleted phones,
* the most-disagreeing utterances.

Inputs may be `.tsv` or `.jsonl` transcripts produced by `decode`.

---

## Caution: ONNX length heuristic

The two backends compute the number of decodable output frames differently, **by
design**, and this matters if you rely on the ONNX CTC output or on `--timestamps`.

| Backend | Output-length source | Time stride per output frame |
|---|---|---|
| **ONNX CTC** | `feat_lens // 2`, clipped to the session output (from `inference/batch_inference.py`) | ≈ `0.02 s` (2 × the 10 ms fbank shift) |
| **ONNX transducer** | `feat_lens // 4`, clipped | ≈ `0.04 s` (4 × the fbank shift) |
| **PyTorch (both)** | the encoder's exact `encoder_out_lens` | exact |

Why `//2` for CTC when `subsampling_factor` is 4? The CR-CTC head emits at roughly
half the encoder stride, so the CTC log-probs come out at ~2× the encoder frame
rate, i.e. one output frame per ~2 fbank frames. The heuristic is an approximation
of the *trailing* valid length only — it does not change which tokens are emitted
for the bulk of the utterance, but it can clip or extend the last few frames and it
sets the time-stride constant used to place phones on the timeline.

**Recommendation:** on your first real ONNX CTC run, spot-check the transcript and
the `--timestamps` alignment against a known reference (or against the PyTorch
backend, which is exact). If you need precise timings, prefer `--backend torch`.

## Other notes on faithfulness to the reference scripts

* Features are 80-dim lhotse kaldi fbank (`dither=0.0, snip_edges=False`), mono,
  16 kHz — identical to `inference/utils.py`.
* Decoding is greedy everywhere (blank id 0; transducer context size 2, ≤3
  symbols/frame), matching the ZIPA inference code.
* Tokenizers are bound per backend automatically: `tokens.txt` for ONNX,
  `unigram_127.model` (sentencepiece) for PyTorch.
* Architecture/size come from the **registry**, not from string-matching the
  checkpoint path.
