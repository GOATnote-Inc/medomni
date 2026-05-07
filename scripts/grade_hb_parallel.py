#!/usr/bin/env python3
"""Parallel HealthBench rubric grading via ThreadPoolExecutor.

Drop-in replacement for the serial grader.grade_jsonl() for healthbench-hard.
Issues all 10 rubric calls per item in parallel (10x speedup per item).
"""

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from openai import OpenAI

GPT_MODEL = "gpt-4.1"


def grade_one_rubric(client, response: str, criterion: str, model: str) -> dict:
    from _healthbench_grader_bridge import GRADER_TEMPLATE  # type: ignore

    conversation = f"assistant: {response}"
    rubric_block = f"[points] {criterion}"
    prompt = GRADER_TEMPLATE.replace("<<conversation>>", conversation).replace(
        "<<rubric_item>>", rubric_block
    )
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_completion_tokens=512,
            )
            text = (resp.choices[0].message.content or "").strip()
            if "```" in text:
                text = re.sub(r"^```(?:json)?\s*|\s*```\s*$", "", text, flags=re.MULTILINE)
            obj = json.loads(text)
            return {
                "criteria_met": bool(obj.get("criteria_met", False)),
                "explanation": str(obj.get("explanation", "")),
                "judge_model": model,
            }
        except Exception as e:
            if attempt == 2:
                return {
                    "criteria_met": None,
                    "explanation": f"judge_error: {type(e).__name__}: {e}",
                    "judge_model": model,
                }
            time.sleep(1.5 * (attempt + 1))
    return {"criteria_met": None, "explanation": "exhausted", "judge_model": model}


def grade_record(client, record: dict, model: str, executor: ThreadPoolExecutor) -> dict:
    from _healthbench_grader_bridge import RubricItem, calculate_score  # type: ignore

    rubric = record.get("rubric") or []
    if not rubric:
        record["graded"] = {"score": None, "n_rubrics": 0, "judge_log": []}
        return record

    items = [
        RubricItem(criterion=r["criterion"], points=float(r["points"]), tags=r.get("tags", []))
        for r in rubric
    ]
    response = record.get("response", "")

    futures = {
        i: executor.submit(grade_one_rubric, client, response, r["criterion"], model)
        for i, r in enumerate(rubric)
    }
    judge_log = [None] * len(rubric)
    for i, f in futures.items():
        judge_log[i] = f.result()
    grading_results = [
        {"criteria_met": bool(out.get("criteria_met") or False)} for out in judge_log
    ]
    score = calculate_score(items, grading_results)
    record["graded"] = {"score": score, "n_rubrics": len(items), "judge_log": judge_log}
    return record


def main():
    if len(sys.argv) != 3:
        print("usage: grade_hb_parallel.py <in.jsonl> <out.jsonl>", file=sys.stderr)
        return 2
    in_p = Path(sys.argv[1])
    out_p = Path(sys.argv[2])

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY missing", file=sys.stderr)
        return 3

    client = OpenAI()
    # 10 rubrics per item — pool of 16 covers parallelism + some headroom for retries.
    executor = ThreadPoolExecutor(max_workers=16)

    out_p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    sums_v0 = 0.0
    sums_v25 = 0.0
    n_v0 = 0
    n_v25 = 0
    t_start = time.time()
    with in_p.open() as fh, out_p.open("w") as out_fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            graded = grade_record(client, rec, GPT_MODEL, executor)
            out_fh.write(json.dumps(graded) + "\n")
            out_fh.flush()
            n += 1
            score = graded.get("graded", {}).get("score")
            arm = rec.get("arm")
            if score is not None:
                if arm == "v0":
                    sums_v0 += score
                    n_v0 += 1
                elif arm == "v25":
                    sums_v25 += score
                    n_v25 += 1
            if n % 10 == 0:
                elapsed = time.time() - t_start
                print(
                    f"[hbgrade] {n} done, elapsed={elapsed:.0f}s rate={n / elapsed:.2f}/s",
                    flush=True,
                )

    summary = {
        "benchmark": "healthbench-hard",
        "n_graded": n,
        "v0_mean": sums_v0 / n_v0 if n_v0 else None,
        "v25_mean": sums_v25 / n_v25 if n_v25 else None,
        "n_v0": n_v0,
        "n_v25": n_v25,
        "primary_grader": GPT_MODEL,
    }
    summary_path = out_p.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[hbgrade] DONE summary: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
