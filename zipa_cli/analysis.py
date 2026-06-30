"""Two-model comparison: token alignment, edit operations, PER, and PFER.

``zipa-cli compare --a A.tsv --b B.tsv`` joins two transcript sets by utterance id
and, treating set **A as the reference** and **B as the hypothesis**, reports:

* corpus and per-utterance **Match / Substitution / Insertion / Deletion** counts;
* **PER** (phone error rate, ``(S+I+D)/N_ref``);
* **PFER** (panphon articulatory feature edit distance), the same metric used by
  ``scripts/evaluate.py`` (optional; needs ``panphon``);
* the most frequent substitution pairs and inserted/deleted phones;
* the worst-disagreeing utterances.

Transcripts may be ``.tsv`` (``id<TAB>p h o n e s``) or ``.jsonl`` (``{"id","phones"|"text"}``).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_hyps(path: str) -> Dict[str, List[str]]:
    """Load ``id -> [phones]`` from a tsv or jsonl transcript file."""
    p = Path(path)
    out: Dict[str, List[str]] = {}
    is_jsonl = p.suffix == ".jsonl" or p.suffix == ".json"
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if is_jsonl:
                rec = json.loads(line)
                phones = rec.get("phones")
                if phones is None:
                    phones = (rec.get("text") or "").split()
                out[str(rec["id"])] = list(phones)
            else:
                parts = line.split("\t")
                sid = parts[0]
                text = parts[1] if len(parts) > 1 else ""
                out[sid] = text.split()
    return out


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #
@dataclass
class EditCounts:
    match: int = 0
    sub: int = 0
    ins: int = 0
    delete: int = 0
    ref_len: int = 0

    def add(self, other: "EditCounts") -> None:
        self.match += other.match
        self.sub += other.sub
        self.ins += other.ins
        self.delete += other.delete
        self.ref_len += other.ref_len

    @property
    def errors(self) -> int:
        return self.sub + self.ins + self.delete

    @property
    def per(self) -> float:
        return self.errors / self.ref_len if self.ref_len else 0.0


def align(ref: List[str], hyp: List[str]) -> Tuple[EditCounts, List[Tuple[str, str, str]]]:
    """Levenshtein alignment of ``ref`` vs ``hyp``.

    Returns ``(counts, ops)`` where each op is ``(kind, ref_tok, hyp_tok)`` and
    ``kind`` is one of ``match|sub|ins|del`` ('ins' = extra token in hyp).
    """
    n, m = len(ref), len(hyp)
    # dp[i][j] = edit distance between ref[:i] and hyp[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,        # deletion (ref token dropped)
                dp[i][j - 1] + 1,        # insertion (extra hyp token)
                dp[i - 1][j - 1] + cost, # match / substitution
            )

    # Backtrace
    ops: List[Tuple[str, str, str]] = []
    counts = EditCounts(ref_len=n)
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            ops.append(("match", ref[i - 1], hyp[j - 1])); counts.match += 1; i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("sub", ref[i - 1], hyp[j - 1])); counts.sub += 1; i -= 1; j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append(("ins", "", hyp[j - 1])); counts.ins += 1; j -= 1
        else:
            ops.append(("del", ref[i - 1], "")); counts.delete += 1; i -= 1
    ops.reverse()
    return counts, ops


# --------------------------------------------------------------------------- #
# PFER (panphon feature edit distance)
# --------------------------------------------------------------------------- #
def compute_pfer(ref: List[str], hyp: List[str], dst) -> float:
    r = "".join(ref).replace("g", "ɡ")
    h = "".join(hyp).replace("g", "ɡ")
    return float(dst.feature_edit_distance(h, r))


# --------------------------------------------------------------------------- #
# Comparison driver
# --------------------------------------------------------------------------- #
@dataclass
class CompareResult:
    total: EditCounts = field(default_factory=EditCounts)
    n_utts: int = 0
    only_in_a: int = 0
    only_in_b: int = 0
    sub_pairs: Counter = field(default_factory=Counter)
    ins_tokens: Counter = field(default_factory=Counter)
    del_tokens: Counter = field(default_factory=Counter)
    per_utt: List[Tuple[str, int, int]] = field(default_factory=list)  # (id, errors, ref_len)
    pfers: List[float] = field(default_factory=list)


def compare(a: Dict[str, List[str]], b: Dict[str, List[str]], with_pfer: bool = True) -> CompareResult:
    res = CompareResult()
    res.only_in_a = len(set(a) - set(b))
    res.only_in_b = len(set(b) - set(a))
    shared = sorted(set(a) & set(b))

    dst = None
    if with_pfer:
        try:
            import panphon.distance

            dst = panphon.distance.Distance()
        except Exception:
            dst = None

    for sid in shared:
        counts, ops = align(a[sid], b[sid])
        res.total.add(counts)
        res.n_utts += 1
        res.per_utt.append((sid, counts.errors, counts.ref_len))
        for kind, rt, ht in ops:
            if kind == "sub":
                res.sub_pairs[(rt, ht)] += 1
            elif kind == "ins":
                res.ins_tokens[ht] += 1
            elif kind == "del":
                res.del_tokens[rt] += 1
        if dst is not None:
            res.pfers.append(compute_pfer(a[sid], b[sid], dst))
    return res


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _format_txt(res: CompareResult, top_k: int) -> str:
    t = res.total
    lines = []
    lines.append("=" * 60)
    lines.append("ZIPA model comparison  (A = reference, B = hypothesis)")
    lines.append("=" * 60)
    lines.append(f"shared utterances : {res.n_utts}")
    lines.append(f"only in A         : {res.only_in_a}")
    lines.append(f"only in B         : {res.only_in_b}")
    lines.append("")
    lines.append(f"reference phones  : {t.ref_len}")
    lines.append(f"matches           : {t.match}")
    lines.append(f"substitutions     : {t.sub}")
    lines.append(f"insertions        : {t.ins}")
    lines.append(f"deletions         : {t.delete}")
    lines.append(f"PER (S+I+D)/N     : {t.per:.4f}")
    if res.pfers:
        import numpy as np

        arr = np.array(res.pfers)
        q75, q25 = np.percentile(arr, [75, 25])
        lines.append("")
        lines.append(f"PFER mean         : {arr.mean():.3f}")
        lines.append(f"PFER median       : {float(np.median(arr)):.3f}")
        lines.append(f"PFER IQR          : {q75 - q25:.3f}")
        lines.append(f"PFER std          : {arr.std():.3f}")

    lines.append("")
    lines.append(f"-- top {top_k} substitutions (A -> B) --")
    for (rt, ht), c in res.sub_pairs.most_common(top_k):
        lines.append(f"  {rt!r:>6} -> {ht!r:<6}  {c}")
    lines.append(f"-- top {top_k} insertions (only in B) --")
    for tok, c in res.ins_tokens.most_common(top_k):
        lines.append(f"  {tok!r:>6}  {c}")
    lines.append(f"-- top {top_k} deletions (only in A) --")
    for tok, c in res.del_tokens.most_common(top_k):
        lines.append(f"  {tok!r:>6}  {c}")

    lines.append("")
    lines.append(f"-- top {top_k} most-disagreeing utterances --")
    worst = sorted(res.per_utt, key=lambda x: (-x[1], x[0]))[:top_k]
    for sid, errs, rl in worst:
        rate = errs / rl if rl else 0.0
        lines.append(f"  {sid}  errors={errs} ref_len={rl} rate={rate:.3f}")
    return "\n".join(lines) + "\n"


def _format_jsonl(res: CompareResult) -> str:
    t = res.total
    summary = {
        "type": "summary",
        "shared_utts": res.n_utts,
        "only_in_a": res.only_in_a,
        "only_in_b": res.only_in_b,
        "ref_phones": t.ref_len,
        "match": t.match,
        "sub": t.sub,
        "ins": t.ins,
        "del": t.delete,
        "per": t.per,
    }
    if res.pfers:
        import numpy as np

        summary["pfer_mean"] = float(np.mean(res.pfers))
        summary["pfer_median"] = float(np.median(res.pfers))
    rows = [json.dumps(summary, ensure_ascii=False)]
    for (rt, ht), c in res.sub_pairs.most_common():
        rows.append(json.dumps({"type": "sub", "ref": rt, "hyp": ht, "count": c}, ensure_ascii=False))
    for tok, c in res.ins_tokens.most_common():
        rows.append(json.dumps({"type": "ins", "tok": tok, "count": c}, ensure_ascii=False))
    for tok, c in res.del_tokens.most_common():
        rows.append(json.dumps({"type": "del", "tok": tok, "count": c}, ensure_ascii=False))
    return "\n".join(rows) + "\n"


def _format_csv(res: CompareResult) -> str:
    rows = ["id,errors,ref_len,rate"]
    for sid, errs, rl in res.per_utt:
        rate = errs / rl if rl else 0.0
        rows.append(f"{sid},{errs},{rl},{rate:.4f}")
    return "\n".join(rows) + "\n"


def run_compare_command(args) -> int:
    a = load_hyps(args.hyp_a)
    b = load_hyps(args.hyp_b)
    if not (set(a) & set(b)):
        print("[zipa-cli] no shared utterance ids between A and B.", file=sys.stderr)
        return 1
    res = compare(a, b, with_pfer=not args.no_pfer)

    if args.report_format == "txt":
        text = _format_txt(res, args.top_k)
    elif args.report_format == "jsonl":
        text = _format_jsonl(res)
    else:
        text = _format_csv(res)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"[zipa-cli] wrote comparison report to {args.output}")
    else:
        sys.stdout.write(text)
    return 0
