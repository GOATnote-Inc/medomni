#!/usr/bin/env python3
"""E4 — Data-leakage check via 5-gram MinHashLSH overlap (Jaccard >= 0.7).

Run on lobster (where datasketch + datasets are available).

Strategy:
- Build a MinHashLSH index from training-corpora text.
- For each eval-test item, query the index; flag if any training match has
  Jaccard >= 0.7 (estimated from MinHash with num_perm=128).
- Report per-benchmark % flagged.

Designed to be light on disk + memory:
- Stream the training datasets via HF datasets `streaming=True` to avoid
  pre-downloading multi-GB shards.
- Cap items at SAMPLE_LIMIT_TRAIN per training corpus and SAMPLE_LIMIT_EVAL
  per eval to stay tractable.
- Use 5-grams over whitespace tokens (cheap, paper-standard).
- Save raw flagged-items to leak_raw_<benchmark>.jsonl for spot-check.

NOTE: This is a SAMPLE-based check. The headline % is a sample estimate.
For T+24h we should run full-population with the same script.
"""
import argparse, json, sys, time, hashlib, os
from pathlib import Path
from collections import Counter

# Strategic sample sizes. Tuned for ~10-15 min runtime on lobster while
# ship-rule eval is co-tenant (don't compete heavily).
SAMPLE_LIMIT_TRAIN = 5000   # per training corpus
SAMPLE_LIMIT_EVAL = 1000    # per eval benchmark
NUM_PERM = 128
NGRAM_N = 5
JACCARD_THRESHOLD = 0.7

def ngrams(text, n=NGRAM_N):
    if not text or not isinstance(text, str):
        return set()
    toks = text.lower().split()
    if len(toks) < n:
        # for short text, use char-trigrams (mild fallback)
        s = text.lower()
        return {s[i:i+n*3] for i in range(len(s) - n*3 + 1)} if len(s) >= n*3 else {s}
    return {" ".join(toks[i:i+n]) for i in range(len(toks) - n + 1)}

def make_minhash(shingles, num_perm=NUM_PERM):
    from datasketch import MinHash
    m = MinHash(num_perm=num_perm)
    for sh in shingles:
        m.update(sh.encode("utf8"))
    return m

def stream_train_text():
    """Yield (corpus_name, item_id, text) tuples from training corpora."""
    from datasets import load_dataset
    corpora = [
        ("MedReason", "UCSC-VLAA/MedReason", None, lambda x: f"{x.get('question','')} {x.get('reasoning','')} {x.get('response','')}"),
        ("medical-o1-SFT", "FreedomIntelligence/medical-o1-reasoning-SFT", "en", lambda x: f"{x.get('Question','')} {x.get('Complex_CoT','')} {x.get('Response','')}"),
    ]
    for name, hf_id, config, text_fn in corpora:
        sys.stderr.write(f"[train] streaming {hf_id} (cfg={config})...\n")
        try:
            ds = load_dataset(hf_id, name=config, split="train", streaming=True)
        except Exception as e:
            sys.stderr.write(f"[train] FAIL {hf_id}: {e}\n")
            continue
        n = 0
        for item in ds:
            text = text_fn(item)
            if not text or not text.strip():
                continue
            yield (name, f"{name}_{n}", text)
            n += 1
            if n >= SAMPLE_LIMIT_TRAIN:
                break
        sys.stderr.write(f"[train] {hf_id} streamed {n} items\n")

def stream_eval_text(spec):
    """spec: dict with keys hf_id, split, config, text_fn_name."""
    from datasets import load_dataset
    name = spec["name"]
    sys.stderr.write(f"[eval] loading {spec['hf_id']} split={spec['split']}\n")
    try:
        if spec.get("config"):
            ds = load_dataset(spec["hf_id"], spec["config"], split=spec["split"], streaming=True)
        else:
            ds = load_dataset(spec["hf_id"], split=spec["split"], streaming=True)
    except Exception as e:
        sys.stderr.write(f"[eval] FAIL {spec['hf_id']}: {e}\n")
        return
    text_fn = spec["text_fn"]
    n = 0
    for item in ds:
        text = text_fn(item)
        if not text or not text.strip():
            continue
        yield (name, f"{name}_{n}", text, item)
        n += 1
        if n >= SAMPLE_LIMIT_EVAL:
            break
    sys.stderr.write(f"[eval] {spec['hf_id']} streamed {n} items\n")

def main():
    from datasketch import MinHashLSH
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/leak_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Build LSH on training corpora.
    lsh = MinHashLSH(threshold=JACCARD_THRESHOLD, num_perm=NUM_PERM)
    train_minhashes = {}  # id -> minhash (kept for retrieval verification)
    train_texts = {}      # id -> text snippet (first 200 chars, for examples)
    train_corpus_of = {}  # id -> corpus name
    t0 = time.time()
    n_train = 0
    for corpus_name, item_id, text in stream_train_text():
        sh = ngrams(text)
        if not sh:
            continue
        m = make_minhash(sh)
        lsh.insert(item_id, m)
        train_minhashes[item_id] = m
        train_texts[item_id] = text[:200]
        train_corpus_of[item_id] = corpus_name
        n_train += 1
        if n_train % 1000 == 0:
            sys.stderr.write(f"  [train] inserted {n_train} ({time.time()-t0:.1f}s)\n")
    sys.stderr.write(f"[train] LSH built: {n_train} items in {time.time()-t0:.1f}s\n")

    # 2) Eval benchmarks to query.
    eval_specs = [
        {
            "name": "MedQA-USMLE-4opt",
            "hf_id": "GBaker/MedQA-USMLE-4-options",
            "split": "test",
            "config": None,
            "text_fn": lambda x: f"{x.get('question','')} {x.get('options',{})}",
        },
        {
            "name": "PubMedQA",
            "hf_id": "qiaojin/PubMedQA",
            "split": "train",  # PubMedQA-L test split shipped under 'train' in pqa_labeled subset
            "config": "pqa_labeled",
            "text_fn": lambda x: f"{x.get('question','')} {' '.join(x.get('context',{}).get('contexts',[]) if isinstance(x.get('context',{}),dict) else [])}",
        },
        {
            "name": "MedXpertQA-Text",
            "hf_id": "TsinghuaC3I/MedXpertQA",
            "split": "test",
            "config": "Text",
            "text_fn": lambda x: f"{x.get('question','')} {x.get('options','')}",
        },
        # HealthBench-Hard: not on HF Hub directly. We'll attempt via openai/simple-evals
        # if the manifest is local; else skip with a NOTE in the output.
    ]

    results = {}
    for spec in eval_specs:
        name = spec["name"]
        flagged = []
        n_eval = 0
        n_flag = 0
        try:
            for _name, item_id, text, raw_item in stream_eval_text(spec):
                sh = ngrams(text)
                if not sh:
                    continue
                m = make_minhash(sh)
                hits = lsh.query(m)
                n_eval += 1
                if hits:
                    # Actual Jaccard verification (LSH approx; verify exact MinHash Jaccard)
                    confirmed = []
                    for h in hits[:10]:
                        if h in train_minhashes:
                            j = m.jaccard(train_minhashes[h])
                            if j >= JACCARD_THRESHOLD:
                                confirmed.append({
                                    "train_id": h,
                                    "train_corpus": train_corpus_of.get(h, "?"),
                                    "jaccard_estimated": float(j),
                                    "train_text_snippet": train_texts.get(h, "")[:200],
                                })
                    if confirmed:
                        n_flag += 1
                        flagged.append({
                            "eval_id": item_id,
                            "eval_text_snippet": text[:200],
                            "matches": confirmed,
                        })
        except Exception as e:
            sys.stderr.write(f"[eval] {name} ERROR: {e}\n")
            results[name] = {"error": str(e), "n_eval": n_eval, "n_flag": n_flag}
            continue
        pct = (n_flag / n_eval * 100) if n_eval else 0.0
        results[name] = {
            "n_eval": n_eval,
            "n_flag": n_flag,
            "pct_overlap_at_jaccard_>=_0.7": round(pct, 3),
            "first_5_flagged": flagged[:5],
        }
        # write raw flagged items for spot-check
        (out_dir / f"leak_raw_{name}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in flagged)
        )
        sys.stderr.write(f"[eval] {name}: {n_flag}/{n_eval} = {pct:.3f}%\n")

    # 3) HealthBench-Hard - check local manifest if exists
    hb_path = Path("/home/ubuntu/medomni/corpus/pins/healthbench-hard-1000.yaml")
    if hb_path.exists():
        try:
            import yaml
            data = yaml.safe_load(hb_path.read_text())
            examples = data.get("examples", [])[:SAMPLE_LIMIT_EVAL]
            flagged = []
            n_eval = 0; n_flag = 0
            for i, ex in enumerate(examples):
                msgs = ex.get("messages", [])
                text = " ".join(m.get("content", "") for m in msgs if m.get("role") == "user")
                if not text.strip():
                    continue
                sh = ngrams(text)
                if not sh:
                    continue
                m_hash = make_minhash(sh)
                hits = lsh.query(m_hash)
                n_eval += 1
                if hits:
                    confirmed = []
                    for h in hits[:10]:
                        if h in train_minhashes:
                            j = m_hash.jaccard(train_minhashes[h])
                            if j >= JACCARD_THRESHOLD:
                                confirmed.append({
                                    "train_id": h,
                                    "train_corpus": train_corpus_of.get(h, "?"),
                                    "jaccard_estimated": float(j),
                                    "train_text_snippet": train_texts.get(h, "")[:200],
                                })
                    if confirmed:
                        n_flag += 1
                        flagged.append({
                            "eval_id": f"HealthBench-Hard_{i}",
                            "eval_text_snippet": text[:200],
                            "matches": confirmed,
                        })
            pct = (n_flag / n_eval * 100) if n_eval else 0.0
            results["HealthBench-Hard"] = {
                "n_eval": n_eval,
                "n_flag": n_flag,
                "pct_overlap_at_jaccard_>=_0.7": round(pct, 3),
                "first_5_flagged": flagged[:5],
            }
            (out_dir / "leak_raw_HealthBench-Hard.jsonl").write_text(
                "\n".join(json.dumps(r) for r in flagged)
            )
            sys.stderr.write(f"[eval] HealthBench-Hard: {n_flag}/{n_eval} = {pct:.3f}%\n")
        except Exception as e:
            sys.stderr.write(f"[eval] HealthBench-Hard ERROR: {e}\n")
            results["HealthBench-Hard"] = {"error": str(e), "note": "local manifest read failed"}
    else:
        results["HealthBench-Hard"] = {
            "note": "manifest not found at /home/ubuntu/medomni/corpus/pins/healthbench-hard-1000.yaml; "
                    "run from laptop or check repo path",
            "skipped": True,
        }

    summary = {
        "config": {
            "ngram_n": NGRAM_N,
            "num_perm": NUM_PERM,
            "jaccard_threshold": JACCARD_THRESHOLD,
            "sample_limit_train_per_corpus": SAMPLE_LIMIT_TRAIN,
            "sample_limit_eval_per_benchmark": SAMPLE_LIMIT_EVAL,
        },
        "n_train_inserted": n_train,
        "wall_time_s": int(time.time() - t0),
        "results": results,
    }
    (out_dir / "leak_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
