#!/usr/bin/env python3
"""B300 throughput sweep harness.

Posts a fixed prompt set to vLLM /v1/chat/completions at varying concurrency
levels. Records TTFT and end-to-end latency from streamed responses.

Usage:
  python3 bench.py --batch 4 --prompts prompts.json --out throughput-b4.csv
  python3 bench.py --batch 4 --prompts prompts.json --out /dev/null --duration 300

Drop in /tmp/b300-profile/ inside catfish.
"""
import argparse, json, time, statistics, sys, os, csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "nemotron"
MAX_TOKENS = 1024


def stream_one(prompt, idx):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "stream": True,
    }).encode("utf-8")
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    out_tokens = 0
    last_chunk = b""
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            for line in resp:
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    break
                if not payload:
                    continue
                try:
                    obj = json.loads(payload)
                except Exception:
                    continue
                delta = obj.get("choices", [{}])[0].get("delta", {})
                if delta.get("content"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    out_tokens += 1
                last_chunk = payload
    except Exception as e:
        return {"idx": idx, "error": str(e), "e2e": time.perf_counter() - t0}
    e2e = time.perf_counter() - t0
    return {"idx": idx, "ttft": ttft, "e2e": e2e, "tokens": out_tokens}


def run_batch(prompts, batch_size, label):
    """Fire one wave of batch_size requests in parallel; return per-request metrics."""
    results = []
    t_wave = time.perf_counter()
    with ThreadPoolExecutor(max_workers=batch_size) as ex:
        futs = {}
        for i in range(batch_size):
            p = prompts[i % len(prompts)]
            futs[ex.submit(stream_one, p, i)] = i
        for f in as_completed(futs):
            results.append(f.result())
    wave_dur = time.perf_counter() - t_wave
    return results, wave_dur


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round(p / 100.0 * (len(xs) - 1)))))
    return xs[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--out", default="-")
    ap.add_argument("--waves", type=int, default=3, help="independent batched waves")
    ap.add_argument("--duration", type=float, default=0.0, help="if >0, loop run_batch for this many seconds")
    args = ap.parse_args()

    with open(args.prompts) as f:
        prompts = json.load(f)

    all_res = []
    waves_run = 0
    t_start = time.perf_counter()

    if args.duration > 0:
        while time.perf_counter() - t_start < args.duration:
            res, _ = run_batch(prompts, args.batch, f"wave{waves_run}")
            all_res.extend(res)
            waves_run += 1
    else:
        for w in range(args.waves):
            res, _ = run_batch(prompts, args.batch, f"wave{w}")
            all_res.extend(res)
            waves_run += 1

    total_dur = time.perf_counter() - t_start
    ok = [r for r in all_res if "error" not in r]
    errs = [r for r in all_res if "error" in r]

    ttfts = [r["ttft"] for r in ok if r.get("ttft") is not None]
    e2es = [r["e2e"] for r in ok]
    total_tokens = sum(r["tokens"] for r in ok)
    tps = total_tokens / total_dur if total_dur > 0 else 0.0

    summary = {
        "batch": args.batch,
        "waves": waves_run,
        "n_ok": len(ok),
        "n_err": len(errs),
        "wall_s": round(total_dur, 3),
        "total_tokens": total_tokens,
        "tokens_per_sec": round(tps, 2),
        "ttft_p50": round(pct(ttfts, 50), 4) if ttfts else None,
        "ttft_p95": round(pct(ttfts, 95), 4) if ttfts else None,
        "e2e_p50": round(pct(e2es, 50), 4) if e2es else None,
        "e2e_p95": round(pct(e2es, 95), 4) if e2es else None,
    }

    if args.out == "-":
        print(json.dumps(summary, indent=2))
    else:
        # CSV row append
        new = not os.path.exists(args.out)
        with open(args.out, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(list(summary.keys()))
            w.writerow([summary[k] for k in summary])
        print(json.dumps(summary))


if __name__ == "__main__":
    main()
