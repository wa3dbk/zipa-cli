// ZIPA Phone Viewer — client-side waveform/spectrogram + multi-model phone tiers.
//
// Loads an audio file and one or more `align-json` transcripts produced by
//   zipa-cli decode --output-format align-json
// (also accepts jsonl produced with --timestamps, which stores times under
// `alignment`). Everything runs in the browser; nothing is uploaded.

import WaveSurfer from "https://unpkg.com/wavesurfer.js@7/dist/wavesurfer.esm.js";
import Spectrogram from "https://unpkg.com/wavesurfer.js@7/dist/plugins/spectrogram.esm.js";
import Timeline from "https://unpkg.com/wavesurfer.js@7/dist/plugins/timeline.esm.js";

const PALETTE = ["#5fb3ff", "#3ddc97", "#ffb454", "#ff6b9d", "#b69bff", "#6ee7d8"];

const els = {
  audioFile: document.getElementById("audioFile"),
  alignFiles: document.getElementById("alignFiles"),
  uttSelect: document.getElementById("uttSelect"),
  playBtn: document.getElementById("playBtn"),
  viewToggle: document.getElementById("viewToggle"),
  zoom: document.getElementById("zoom"),
  cursorTime: document.getElementById("cursorTime"),
  status: document.getElementById("status"),
  waveform: document.getElementById("waveform"),
  spectrogram: document.getElementById("spectrogram"),
  tiers: document.getElementById("tiers"),
};

// id -> [ {model, phones:[{p,start,end}], ref} ]
const utterances = new Map();
let ws = null;
let spectrogramOn = false;

function setStatus(msg) {
  els.status.textContent = msg || "";
}

function initWaveSurfer() {
  if (ws) ws.destroy();
  ws = WaveSurfer.create({
    container: els.waveform,
    height: 120,
    waveColor: "#5b6b80",
    progressColor: "#5fb3ff",
    cursorColor: "#ffffff",
    normalize: true,
    plugins: [
      Timeline.create({ container: "#timeline" }),
      Spectrogram.create({ container: els.spectrogram, labels: true, height: 160 }),
    ],
  });

  ws.on("timeupdate", (t) => {
    els.cursorTime.textContent = `${t.toFixed(3)} s`;
    updatePlayheads(t);
  });
  ws.on("play", () => (els.playBtn.textContent = "⏸ Pause"));
  ws.on("pause", () => (els.playBtn.textContent = "▶ Play"));
  ws.on("ready", () => {
    setStatus(`audio loaded · ${ws.getDuration().toFixed(2)} s`);
    renderTiers();
  });
}

// ---- file loading ---------------------------------------------------------
els.audioFile.addEventListener("change", (e) => {
  const f = e.target.files[0];
  if (!f) return;
  if (!ws) initWaveSurfer();
  ws.load(URL.createObjectURL(f));
  setStatus("decoding audio…");
});

els.alignFiles.addEventListener("change", async (e) => {
  const files = [...e.target.files];
  for (const f of files) {
    const text = await f.text();
    parseAlignText(text);
  }
  refreshUttSelect();
  renderTiers();
  setStatus(`${utterances.size} utterance id(s) loaded from ${files.length} file(s)`);
});

function parseAlignText(text) {
  const trimmed = text.trim();
  let records = [];
  if (trimmed.startsWith("[")) {
    records = JSON.parse(trimmed);
  } else {
    for (const line of trimmed.split("\n")) {
      if (line.trim()) records.push(JSON.parse(line));
    }
  }
  for (const rec of records) addRecord(rec);
}

function addRecord(rec) {
  const id = String(rec.id);
  // align-json stores phones under `phones`; jsonl --timestamps under `alignment`.
  const phones = rec.phones || rec.alignment || [];
  const entry = {
    model: rec.model || "model",
    phones: phones.map((p) => ({ p: p.p, start: +p.start, end: +p.end })),
    ref: rec.ref || null,
  };
  if (!utterances.has(id)) utterances.set(id, []);
  utterances.get(id).push(entry);
}

function refreshUttSelect() {
  const cur = els.uttSelect.value;
  els.uttSelect.innerHTML = "";
  for (const id of utterances.keys()) {
    const opt = document.createElement("option");
    opt.value = id;
    const nModels = utterances.get(id).length;
    opt.textContent = `${id}  (${nModels} model${nModels > 1 ? "s" : ""})`;
    els.uttSelect.appendChild(opt);
  }
  if ([...utterances.keys()].includes(cur)) els.uttSelect.value = cur;
}

els.uttSelect.addEventListener("change", renderTiers);

// ---- tier rendering -------------------------------------------------------
function currentDuration() {
  if (ws && ws.getDuration()) return ws.getDuration();
  // fall back to max phone end across selected utterance
  const entries = utterances.get(els.uttSelect.value) || [];
  let m = 0;
  for (const e of entries) for (const p of e.phones) m = Math.max(m, p.end);
  return m || 1;
}

function renderTiers() {
  els.tiers.innerHTML = "";
  const id = els.uttSelect.value;
  if (!id || !utterances.has(id)) return;
  const dur = currentDuration();

  utterances.get(id).forEach((entry, i) => {
    const color = PALETTE[i % PALETTE.length];
    const tier = document.createElement("div");
    tier.className = "tier";

    const head = document.createElement("div");
    head.className = "tier-head";
    head.innerHTML =
      `<span class="tier-swatch" style="background:${color}"></span>` +
      `<span class="tier-name">${escapeHtml(entry.model)}</span>` +
      `<span class="tier-meta">${entry.phones.length} phones</span>`;
    tier.appendChild(head);

    const track = document.createElement("div");
    track.className = "tier-track";
    track.dataset.tier = i;

    for (const ph of entry.phones) {
      const left = (ph.start / dur) * 100;
      const width = Math.max(((ph.end - ph.start) / dur) * 100, 0.3);
      const block = document.createElement("div");
      block.className = "phone";
      block.style.left = left + "%";
      block.style.width = width + "%";
      block.style.borderLeftColor = color;
      block.innerHTML =
        `${escapeHtml(ph.p)}<span class="tip">${escapeHtml(ph.p)} · ` +
        `${ph.start.toFixed(2)}–${ph.end.toFixed(2)}s</span>`;
      block.addEventListener("click", () => {
        if (ws) ws.setTime(ph.start);
      });
      track.appendChild(block);
    }

    const playhead = document.createElement("div");
    playhead.className = "playhead";
    playhead.style.left = "0%";
    track.appendChild(playhead);

    tier.appendChild(track);

    const textLine = document.createElement("div");
    textLine.className = "tier-text";
    textLine.textContent = entry.phones.map((p) => p.p).join(" ");
    tier.appendChild(textLine);

    if (entry.ref) {
      const refLine = document.createElement("div");
      refLine.className = "tier-text ref-text";
      refLine.textContent = "ref: " + entry.ref;
      tier.appendChild(refLine);
    }

    els.tiers.appendChild(tier);
  });
}

function updatePlayheads(t) {
  const dur = currentDuration();
  const pct = (t / dur) * 100;
  document.querySelectorAll(".tier-track .playhead").forEach((ph) => {
    ph.style.left = pct + "%";
  });
}

// ---- toolbar --------------------------------------------------------------
els.playBtn.addEventListener("click", () => ws && ws.playPause());

els.viewToggle.addEventListener("click", () => {
  spectrogramOn = !spectrogramOn;
  els.spectrogram.style.display = spectrogramOn ? "block" : "none";
  els.waveform.style.display = spectrogramOn ? "none" : "block";
  els.viewToggle.textContent = spectrogramOn ? "Waveform view" : "Spectrogram view";
});

els.zoom.addEventListener("input", (e) => {
  if (ws) {
    try { ws.zoom(Number(e.target.value)); } catch (_) {}
  }
});

document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && ws) {
    e.preventDefault();
    ws.playPause();
  }
});

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

setStatus("Load an audio file and alignment JSON(s) to begin.");
