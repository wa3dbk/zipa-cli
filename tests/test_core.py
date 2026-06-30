"""Lightweight regression tests for zipa-cli core logic.

These cover the pieces that don't need the heavy ONNX/torch/lhotse stack: source
readers, batching, output writers, greedy decode, and the compare alignment.
Run with ``python -m pytest zipa_cli/tests`` (only numpy is required) or directly
with ``python zipa_cli/tests/test_core.py``.
"""

from __future__ import annotations

import os
import tempfile
import types

import numpy as np

from zipa_cli import decode, output
from zipa_cli.analysis import EditCounts, align, compare, load_hyps
from zipa_cli.data import batching, sources
from zipa_cli.data.sample import Sample


def _make_audio_tree(d):
    for name in ["a.wav", "b.flac", "sub/c.mp3"]:
        p = os.path.join(d, name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").close()


def test_dir_source():
    with tempfile.TemporaryDirectory() as d:
        _make_audio_tree(d)
        ns = types.SimpleNamespace(input=d, input_type="dir")
        ids = sorted(s.id for s in sources.iter_samples(ns))
        assert ids == ["a", "b", "sub/c"]


def test_tsv_source_with_header():
    with tempfile.TemporaryDirectory() as d:
        tsv = os.path.join(d, "cv.tsv")
        open(tsv, "w").write("client\tpath\tsentence\nu1\ta.wav\thi\nu2\tb.flac\tyo\n")
        ns = types.SimpleNamespace(
            input=tsv, input_type="tsv", id_col="0", path_col="path", ref_col="sentence"
        )
        rows = list(sources.iter_samples(ns))
        assert [r.id for r in rows] == ["u1", "u2"]
        assert rows[0].ref == "hi"


def test_stm_source():
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "a.wav"), "w").close()
        stm = os.path.join(d, "x.stm")
        open(stm, "w").write("a 1 0 0.0 1.5 ref text\na 1 0 1.5 3.0 more\n")
        ns = types.SimpleNamespace(input=stm, input_type="stm", audio_dir=d)
        rows = list(sources.iter_samples(ns))
        assert rows[0].start == 0.0 and rows[0].end == 1.5
        assert rows[0].ref == "ref text"
        assert abs(rows[0].duration - 1.5) < 1e-9


def test_batching_fixed_and_dynamic():
    samps = [Sample(id=str(i), duration=d) for i, d in enumerate([1, 5, 2, 8, 1, 1, 3])]
    fixed = list(batching.batches(iter(samps), batch_size=3))
    assert [len(b) for b in fixed] == [3, 3, 1]
    dyn = list(batching.batches(iter(list(samps)), max_duration=6))
    for b in dyn:
        assert sum(x.duration for x in b) <= 6 or len(b) == 1


def test_writers_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        s = Sample(id="u1", audio_path=f"{d}/a.wav", ref="r e f", start=0.0, end=1.5,
                   duration=1.5, speaker="spk")
        op = os.path.join(d, "out.tsv")
        with output.get_writer("tsv", op, "m") as w:
            w.write(s, ["p", "h", "o"])
        assert open(op).read().strip() == "u1\tp h o"
        # recogs format is literal_eval-parseable (matches scripts/evaluate.py)
        opr = os.path.join(d, "out.recogs")
        with output.get_writer("recogs", opr, "m") as w:
            w.write(s, ["p", "h", "o"])
        lines = open(opr).read().splitlines()
        assert lines[0] == "u1\thyp=['p', 'h', 'o']"
        assert lines[1] == "u1\tref=['r', 'e', 'f']"


def test_ctc_greedy_decode():
    V = 5
    logits = np.zeros((1, 6, V), dtype=np.float32)
    for t, a in enumerate([0, 2, 2, 0, 3, 3]):
        logits[0, t, a] = 10.0
    vocab = {0: "<blk>", 1: "a", 2: "b", 3: "c", 4: "d"}
    assert decode.ctc_greedy_decode(logits, vocab) == [["b", "c"]]


def test_align_all_ops():
    counts, ops = align(["a", "b", "c", "d"], ["a", "x", "c"])
    assert counts.match == 2 and counts.sub == 1 and counts.delete == 1 and counts.ins == 0
    kinds = [o[0] for o in ops]
    assert kinds == ["match", "sub", "match", "del"]


def test_ctc_timed_decode():
    V = 5
    logits = np.zeros((1, 6, V), dtype=np.float32)
    for t, a in enumerate([0, 2, 2, 0, 3, 3]):
        logits[0, t, a] = 10.0
    vocab = {0: "<blk>", 1: "a", 2: "b", 3: "c", 4: "d"}
    timed = decode.ctc_greedy_decode_with_times(logits, vocab)
    # stride 2 * 0.01s = 0.02s/frame: 'b' over frames 1..3, 'c' over 4..6
    assert timed == [[("b", 0.02, 0.06), ("c", 0.08, 0.12)]]


def test_align_json_writer_offsets():
    import json
    with tempfile.TemporaryDirectory() as d:
        s = Sample(id="seg", audio_path=f"{d}/a.wav", start=10.0, end=10.12, duration=0.12)
        op = os.path.join(d, "a.json")
        with output.get_writer("align-json", op, "m") as w:
            w.write_timed(s, [("b", 0.02, 0.06)])
        rec = json.loads(open(op).read())
        assert rec["phones"][0]["start"] == 10.02  # segment offset applied
        assert rec["phones"][0]["end"] == 10.06


def test_ctm_writer():
    with tempfile.TemporaryDirectory() as d:
        s = Sample(id="u1", audio_path=f"{d}/a.wav", duration=0.12)
        op = os.path.join(d, "a.ctm")
        with output.get_writer("ctm", op, "m") as w:
            w.write_timed(s, [("b", 0.02, 0.06), ("c", 0.08, 0.12)])
        lines = open(op).read().splitlines()
        assert lines[0] == "a 0 0.020 0.040 b"


def test_compare_summary():
    a = {"u1": ["a", "b", "c", "d"], "u2": ["p", "q"]}
    b = {"u1": ["a", "x", "c"], "u2": ["p", "q", "r"]}
    res = compare(a, b, with_pfer=False)
    t = res.total
    assert t.match == 4 and t.sub == 1 and t.ins == 1 and t.delete == 1
    assert abs(t.per - 0.5) < 1e-9
    assert res.sub_pairs[("b", "x")] == 1


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
