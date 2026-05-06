#!/usr/bin/env python3
"""HealthBench-Hard leakage check vs MedReason + medical-o1 training corpora."""
from datasketch import MinHashLSH, MinHash
from datasets import load_dataset
import json, sys

NUM_PERM = 128; N = 5; THRESH = 0.7

def ngrams(text):
    if not text:
        return set()
    toks = text.lower().split()
    if len(toks) < N:
        return set()
    return {" ".join(toks[i:i+N]) for i in range(len(toks) - N + 1)}

def mh(sh):
    m = MinHash(num_perm=NUM_PERM)
    for s in sh:
        m.update(s.encode("utf8"))
    return m

lsh = MinHashLSH(threshold=THRESH, num_perm=NUM_PERM)
train_minhashes = {}
train_corpus_of = {}
train_texts = {}
n = 0

corpora = [
    ("UCSC-VLAA/MedReason", None,
     lambda x: f'{x.get("question","")} {x.get("reasoning","")} {x.get("response","")}'),
    ("FreedomIntelligence/medical-o1-reasoning-SFT", "en",
     lambda x: f'{x.get("Question","")} {x.get("Complex_CoT","")} {x.get("Response","")}'),
]

for hf_id, cfg, fn in corpora:
    if cfg:
        ds = load_dataset(hf_id, name=cfg, split="train", streaming=True)
    else:
        ds = load_dataset(hf_id, split="train", streaming=True)
    nn = 0
    for item in ds:
        text = fn(item)
        sh = ngrams(text)
        if not sh:
            continue
        m = mh(sh)
        iid = f"{hf_id}_{nn}"
        lsh.insert(iid, m)
        train_minhashes[iid] = m
        train_corpus_of[iid] = hf_id
        train_texts[iid] = text[:200]
        nn += 1
        if nn >= 5000:
            break
    print(f"[train] {hf_id} -> {nn}", file=sys.stderr)
    n += nn

print(f"[train] total {n}", file=sys.stderr)

try:
    ds = load_dataset("Tonic/Health-Bench-Eval-OSS-2025-07", split="hard", streaming=True)
except Exception as e:
    print(f"[hb] FAIL load: {e}", file=sys.stderr)
    sys.exit(1)

n_eval = 0; n_flag = 0; flagged = []
for i, item in enumerate(ds):
    if i >= 1000:
        break
    # try common HB shapes: 'prompt' (list of messages) | 'messages' | 'input'
    msgs = item.get("prompt") or item.get("messages") or item.get("input")
    if isinstance(msgs, list):
        text = " ".join(
            (m.get("content", "") if isinstance(m, dict) else str(m)) for m in msgs
        )
    elif isinstance(msgs, str):
        text = msgs
    else:
        text = json.dumps(item)[:5000]
    sh = ngrams(text)
    if not sh:
        continue
    m_hash = mh(sh)
    n_eval += 1
    hits = lsh.query(m_hash)
    if hits:
        confirmed = []
        for h in hits[:5]:
            if h in train_minhashes:
                j = m_hash.jaccard(train_minhashes[h])
                if j >= THRESH:
                    confirmed.append({
                        "train_id": h,
                        "jaccard": float(j),
                        "snippet": train_texts.get(h, "")[:200],
                    })
        if confirmed:
            n_flag += 1
            flagged.append({
                "eval_id": f"HB_{i}",
                "eval_snippet": text[:200],
                "matches": confirmed,
            })

pct = n_flag / n_eval * 100 if n_eval else 0.0
print(json.dumps({
    "name": "HealthBench-Hard",
    "n_eval": n_eval,
    "n_flag": n_flag,
    "pct": round(pct, 3),
    "first_5_flagged": flagged[:5],
}, indent=2))
