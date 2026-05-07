"""N-gram + memorization leakage probes for the V2.5 ship-rule eval.

Two checks:

1. **5-gram MinHash overlap.** For each test prompt, compute the 5-gram
   shingle set, MinHash to a 128-perm sketch, and compare against the
   pre-built sketches over the training corpora (MedReason 32K +
   medical-o1-reasoning-SFT 25K). Threshold 0.7 Jaccard ⇒ flag.

2. **Memorization probe.** For a sampled subset, split the prompt at the
   midpoint, send first half + a "continue verbatim" instruction to the
   model under test, compare against the held-out second half via Levenshtein
   ratio. Threshold > 0.85 ⇒ flag (the model verbatim-memorized this row).

Both probes are stdlib-only. The MinHash uses Python's deterministic
hashing — we replace built-in `hash()` (salted) with `hashlib.blake2b` for
reproducibility across processes.

Outputs:
    LEAKAGE-AUDIT.md — human-readable summary
    leakage-flags.jsonl — one record per flagged item
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def shingles(tokens: list[str], n: int = 5) -> set[str]:
    if len(tokens) < n:
        return set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _stable_hash(s: str, seed: int) -> int:
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8, person=seed.to_bytes(8, "little"))
    return int.from_bytes(h.digest(), "big")


def minhash(shingle_set: set[str], n_perm: int = 128) -> list[int]:
    if not shingle_set:
        return [0] * n_perm
    sigs: list[int] = []
    for seed in range(n_perm):
        m = min(_stable_hash(s, seed) for s in shingle_set)
        sigs.append(m)
    return sigs


def jaccard_estimate(sig_a: list[int], sig_b: list[int]) -> float:
    if len(sig_a) != len(sig_b) or not sig_a:
        return 0.0
    matches = sum(1 for a, b in zip(sig_a, sig_b, strict=True) if a == b)
    return matches / len(sig_a)


@dataclass
class LeakageHit:
    item_id: str
    benchmark: str
    train_source: str
    jaccard: float
    test_prompt_excerpt: str

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "benchmark": self.benchmark,
            "train_source": self.train_source,
            "jaccard": self.jaccard,
            "test_prompt_excerpt": self.test_prompt_excerpt,
        }


def build_train_sketches(
    train_files: list[Path],
    *,
    n_perm: int = 128,
    n: int = 5,
) -> list[tuple[str, list[int]]]:
    """Build (source_label, minhash) for each line in each training file.

    Each file is treated as JSONL where each row has a `text` or `prompt`
    field; falls back to the whole line if neither.
    """
    sketches: list[tuple[str, list[int]]] = []
    for tf in train_files:
        if not tf.exists():
            continue
        label_base = tf.stem
        with tf.open("r") as fh:
            for i, raw in enumerate(fh):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                    if isinstance(obj, dict):
                        text = obj.get("text") or obj.get("prompt") or obj.get("question") or raw
                    else:
                        text = raw
                except json.JSONDecodeError:
                    text = raw
                toks = tokenize(text if isinstance(text, str) else json.dumps(text))
                sh = shingles(toks, n=n)
                if not sh:
                    continue
                sketches.append((f"{label_base}#{i}", minhash(sh, n_perm=n_perm)))
    return sketches


def scan_test_prompts(
    *,
    test_items: list[dict],
    train_sketches: list[tuple[str, list[int]]],
    benchmark: str,
    threshold: float = 0.7,
    n_perm: int = 128,
    n: int = 5,
    prompt_field: str = "prompt",
    id_field: str = "item_id",
) -> list[LeakageHit]:
    """Score each test prompt vs every training sketch; emit hits."""
    hits: list[LeakageHit] = []
    for item in test_items:
        prompt = item.get(prompt_field) or ""
        item_id = str(item.get(id_field, ""))
        toks = tokenize(prompt)
        sh = shingles(toks, n=n)
        if not sh:
            continue
        sig = minhash(sh, n_perm=n_perm)
        for src_label, train_sig in train_sketches:
            j = jaccard_estimate(sig, train_sig)
            if j >= threshold:
                excerpt = prompt[:200].replace("\n", " ")
                hits.append(LeakageHit(item_id, benchmark, src_label, j, excerpt))
                break  # one hit per test item is enough to flag
    return hits


def levenshtein_ratio(a: str, b: str) -> float:
    """Stdlib Levenshtein-distance-derived similarity ratio in [0, 1]."""
    if not a and not b:
        return 1.0
    la, lb = len(a), len(b)
    if max(la, lb) == 0:
        return 1.0
    # Two-row DP for memory efficiency.
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    dist = prev[lb]
    return 1.0 - (dist / max(la, lb))


def memorization_probe(
    *,
    test_items: list[dict],
    completions_by_id: dict[str, str],
    threshold: float = 0.85,
    prompt_field: str = "prompt",
    id_field: str = "item_id",
) -> list[dict]:
    """For each item, compare model continuation vs the held-out second half.

    `completions_by_id` is the raw model continuation when given the first
    half of `prompt`. Caller is responsible for generating those (via
    sovereign serve endpoint, NOT cloud).
    """
    flags: list[dict] = []
    for item in test_items:
        prompt = item.get(prompt_field) or ""
        item_id = str(item.get(id_field, ""))
        if not prompt or item_id not in completions_by_id:
            continue
        mid = len(prompt) // 2
        held_out_second_half = prompt[mid:]
        continuation = completions_by_id[item_id] or ""
        ratio = levenshtein_ratio(held_out_second_half, continuation[: len(held_out_second_half)])
        if ratio >= threshold:
            flags.append(
                {
                    "item_id": item_id,
                    "ratio": ratio,
                    "expected_excerpt": held_out_second_half[:200],
                    "got_excerpt": continuation[:200],
                }
            )
    return flags


def write_audit(
    *,
    out_md: Path,
    out_jsonl: Path,
    overlap_hits: list[LeakageHit],
    memorization_hits: list[dict],
    n_test_items: int,
    threshold_jaccard: float,
    threshold_memorize: float,
) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w") as fh:
        for h in overlap_hits:
            fh.write(json.dumps({"kind": "ngram_overlap", **h.to_dict()}) + "\n")
        for m in memorization_hits:
            fh.write(json.dumps({"kind": "memorization", **m}) + "\n")
    by_bench: dict[str, int] = {}
    for h in overlap_hits:
        by_bench[h.benchmark] = by_bench.get(h.benchmark, 0) + 1
    md_lines = [
        "# V2.5 Ship-Rule Eval — Leakage Audit",
        "",
        f"- N test items scanned: {n_test_items}",
        f"- 5-gram MinHash threshold (Jaccard): {threshold_jaccard}",
        f"- Memorization threshold (Levenshtein ratio): {threshold_memorize}",
        f"- Overlap hits: {len(overlap_hits)}",
        f"- Memorization hits: {len(memorization_hits)}",
        "",
        "## Overlap hits by benchmark",
        "",
    ]
    if not by_bench:
        md_lines.append("_None._")
    else:
        for k in sorted(by_bench):
            md_lines.append(f"- `{k}`: {by_bench[k]}")
    md_lines.extend(
        [
            "",
            "## Sample overlap hits (first 10)",
            "",
        ]
    )
    for h in overlap_hits[:10]:
        md_lines.append(
            f"- `{h.benchmark}` item `{h.item_id}` ↔ `{h.train_source}` "
            f"(jaccard {h.jaccard:.3f}): {h.test_prompt_excerpt[:120]}…"
        )
    md_lines.extend(
        [
            "",
            "## Memorization hits (first 10)",
            "",
        ]
    )
    if not memorization_hits:
        md_lines.append("_None._")
    else:
        for m in memorization_hits[:10]:
            md_lines.append(
                f"- item `{m['item_id']}` ratio={m['ratio']:.3f}; expected "
                f"`{m['expected_excerpt'][:80]}…` got `{m['got_excerpt'][:80]}…`"
            )
    md_lines.extend(
        [
            "",
            "## Disposition",
            "",
            "Any non-empty hit set requires manual review BEFORE publishing the",
            "ship-rule decision. n-gram overlap with thresholds <0.85 may be",
            "boilerplate (rubric phrasing); >0.85 is presumptively contaminated.",
            "Memorization-probe hits are presumptive contamination at any rate.",
        ]
    )
    out_md.write_text("\n".join(md_lines) + "\n")
