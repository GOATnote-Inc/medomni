#!/usr/bin/env python3
"""Phase 2.2 — judge-throughput micro-benchmark.

Compares two OpenAI-compatible /chat/completions endpoints serving the same
Qwen2.5-7B-Instruct judge model:

  - vllm-BF16 on B300:8003 (Phase 2.1 baseline, accessed via tunnel)
  - TRT-LLM-FP8 on prism:9003 (Phase 2.2 target, accessed via tunnel)

Reports tokens/sec at concurrency=1 and concurrency=8.

Usage:
    python scripts/bench_judge_throughput.py \\
        --url http://127.0.0.1:9003/v1 \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --label prism-trtllm-fp8 \\
        --concurrencies 1,8 \\
        --requests-per-c 8 \\
        --max-tokens 256

Both URLs must be 127.0.0.1/localhost (sovereign-stack constraint).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import httpx


JUDGE_PROMPT = (
    "You are a strict clinical-rubric grader. Given a single question, an "
    "AI-generated answer, and a list of binary criteria, return a JSON list "
    "of {criterion_id: <id>, met: true|false, evidence: <quoted span>}.\n\n"
    "Question: A 67-year-old woman post-menopausal with a 5-year ASCVD risk "
    "of 8% asks whether tamoxifen with a Mirena IUD is appropriate primary "
    "breast cancer prevention. List the rubric findings.\n\n"
    "Answer: Tamoxifen is approved for risk reduction in women with elevated "
    "5-year breast-cancer risk (Gail >= 1.67%). Mirena is a levonorgestrel "
    "IUD. The combination is studied in Cochrane 2020 (low-grade evidence, "
    "no signal of harm in 5-year endometrial outcomes). Discuss VTE risk "
    "(2-3x baseline), endometrial monitoring frequency, and aromatase "
    "inhibitor alternatives.\n\n"
    "Criteria:\n"
    "C1: Mentions Gail/BCRAT risk threshold (1.67% 5-yr)\n"
    "C2: Discusses VTE risk\n"
    "C3: Mentions endometrial monitoring\n"
    "C4: Lists AI alternatives (anastrozole, exemestane)\n"
    "C5: References primary literature\n"
    "C6: FKGL <= 8 if patient register\n"
    "C7: Includes when-to-call-physician guidance\n"
    "C8: Acknowledges uncertainty\n\n"
    "Return JSON only."
)


async def _one_request(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    max_tokens: int,
    timeout_s: float,
) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": JUDGE_PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    t0 = time.perf_counter()
    try:
        resp = await client.post(url + "/chat/completions", json=body, timeout=timeout_s)
        resp.raise_for_status()
        d = resp.json()
        ttft_or_full_s = time.perf_counter() - t0
        completion_text = d["choices"][0]["message"].get("content") or ""
        usage = d.get("usage") or {}
        out_tok = usage.get("completion_tokens") or len(completion_text.split())
        return {
            "ok": True,
            "latency_s": ttft_or_full_s,
            "completion_tokens": out_tok,
            "in_tokens": usage.get("prompt_tokens"),
        }
    except Exception as e:
        return {"ok": False, "err": str(e), "latency_s": time.perf_counter() - t0}


async def _run_concurrency(url: str, model: str, n: int, concurrency: int, max_tokens: int, timeout_s: float) -> dict:
    async with httpx.AsyncClient() as client:
        # Warmup: 1 sequential to load engine.
        await _one_request(client, url, model, max_tokens=max_tokens, timeout_s=timeout_s)
        sem = asyncio.Semaphore(concurrency)

        async def _bound():
            async with sem:
                return await _one_request(client, url, model, max_tokens=max_tokens, timeout_s=timeout_s)

        t0 = time.perf_counter()
        results = await asyncio.gather(*[_bound() for _ in range(n)])
        wall_s = time.perf_counter() - t0

    ok_results = [r for r in results if r.get("ok")]
    failed = len(results) - len(ok_results)
    total_out_tokens = sum(r["completion_tokens"] for r in ok_results)
    latencies = [r["latency_s"] for r in ok_results]
    return {
        "concurrency": concurrency,
        "requests_total": n,
        "requests_ok": len(ok_results),
        "requests_failed": failed,
        "wall_s": wall_s,
        "total_completion_tokens": total_out_tokens,
        "throughput_tok_per_s": (total_out_tokens / wall_s) if wall_s > 0 else 0.0,
        "throughput_req_per_s": (len(ok_results) / wall_s) if wall_s > 0 else 0.0,
        "p50_latency_s": statistics.median(latencies) if latencies else None,
        "p95_latency_s": (sorted(latencies)[int(0.95 * len(latencies))] if len(latencies) >= 5 else None),
        "mean_latency_s": (sum(latencies) / len(latencies)) if latencies else None,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True, help="OpenAI-compatible base URL, e.g. http://127.0.0.1:9003/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--label", required=True, help="label for this run, e.g. prism-trtllm-fp8")
    p.add_argument("--concurrencies", default="1,8")
    p.add_argument("--requests-per-c", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--timeout-s", type=float, default=180.0)
    p.add_argument("--out", default=None, help="optional JSON output path")
    args = p.parse_args()

    if not args.url.startswith(("http://127.0.0.1", "http://localhost")):
        print(f"FAIL: --url must be 127.0.0.1/localhost; got {args.url}", file=sys.stderr)
        return 1

    cs = [int(c) for c in args.concurrencies.split(",")]
    runs = []
    for c in cs:
        print(f"[bench] {args.label}: concurrency={c} n={args.requests_per_c} ...", flush=True)
        r = asyncio.run(_run_concurrency(args.url, args.model, args.requests_per_c, c, args.max_tokens, args.timeout_s))
        runs.append(r)
        print(json.dumps(r, indent=2))
    payload = {
        "label": args.label,
        "url": args.url,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "runs": runs,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"[bench] artifact: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
