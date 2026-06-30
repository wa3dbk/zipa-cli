# ZIPA Phone Viewer (web)

A zero-install, **100% client-side** web interface for phoneticians and linguists
to visualise ZIPA model output: load a waveform (toggle to a spectrogram view) and
inspect the decoded phonemes — from **one or several models at once** — laid out on
the time axis with per-phone timecodes.

```
┌───────────────────────────────────────────────┐
│  ~~~~~~~ waveform / spectrogram ~~~~~~~~        │   ◀ toggle view
│  |    |   time ruler   |    |    |             │
├───────────────────────────────────────────────┤
│ ● zipa-cr-l-500k   [h][ə][l][oʊ] ...           │   ◀ tier per model
│ ● zipa-cr-s-300k   [h][ɛ][l][o ] ...           │     (click a phone to seek)
└───────────────────────────────────────────────┘
```

## What it shows

* waveform **and** spectrogram (toggle button) rendered with
  [wavesurfer.js](https://wavesurfer.xyz/) + a time ruler;
* one **tier per model**, each phone a clickable block positioned by its
  `start`/`end` time; hover shows the exact interval, click seeks the audio there;
* a synced playhead across all tiers;
* the reference transcript (if present in the data) under each tier.

## How to use

1. **Produce alignment data** with the CLI (`align-json`, one file per model):

   ```bash
   zipa-cli decode --input utt.wav --model zipa-cr-l-500k \
       --output-format align-json -o large.align.json
   zipa-cli decode --input utt.wav --model zipa-cr-s-300k \
       --output-format align-json -o small.align.json
   ```

   Each file is JSON-lines; every line holds one utterance:

   ```json
   {"id": "utt", "model": "zipa-cr-l-500k", "audio": "utt.wav", "duration": 5.86,
    "phones": [{"p": "h", "start": 0.10, "end": 0.16}, {"p": "ə", "start": 0.16, "end": 0.22}]}
   ```

   > A `jsonl` file produced with `--timestamps` also works — the viewer reads the
   > per-phone times from its `alignment` field.

2. **Open the viewer.** It uses ES-module CDN imports, so serve the folder over
   HTTP (opening `index.html` from `file://` will block the imports):

   ```bash
   cd web && python -m http.server 8000
   # then browse to http://localhost:8000
   ```

3. In the page: pick the **audio file**, pick one or more **alignment JSON** files,
   choose the **utterance id**, and press play (or space). Models that contain the
   selected id are overlaid as separate tiers.

## Comparing models

Load several `align-json` files at once (or one file containing several models'
runs concatenated). All entries sharing an utterance `id` are stacked as tiers so
you can eyeball insertions/substitutions/deletions between models. For quantitative
agreement use `zipa-cli compare`.

## Notes on timing accuracy

Phone times come from the model's greedy alignment. For the **ONNX** backend the
frame stride is approximate (~20 ms/frame for CTC, ~40 ms for the transducer); the
**PyTorch** backend uses exact encoder lengths. See the project README,
*“Caution: ONNX length heuristic.”* Treat ONNX timecodes as approximate.

## Files

| File | Purpose |
|---|---|
| `index.html` | layout + controls |
| `app.js` | loading, parsing `align-json`, wavesurfer setup, tier rendering |
| `styles.css` | dark theme |

No build step, no bundler, no backend.
