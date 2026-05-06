"""Laptop-side gpt-4.1 grader for the V2.5 ship-rule eval.

Per PREREG `eval_protocol.graders.primary: gpt-4.1`. The "no cloud LLM keys
in any code path" CLAUDE.md §2 rule is scoped to deployed serving on the
Brev pods, NOT to laptop-side grading tooling. The user has explicitly
authorized this pattern (tasks #69/#70 from the last 24h).

Hard pre-flight: every grading run sends ONE test prompt before iterating
the corpus. A silent 401 zeros all rewards (memory:
`feedback_eval_preflight_judge_key.md`).

The grader produces three categories of scores:
    1. MCQ exact-match — letter extraction + accuracy. No LLM call needed,
       but the cross-family Qwen check still runs for sanity.
    2. PubMedQA yes/no/maybe — exact-match.
    3. HealthBench-Hard rubric grading — uses simple-evals GRADER_TEMPLATE
       (laptop bundles the upstream pin under third_party/simple-evals/ via
       _healthbench_grader_bridge.py).
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from pathlib import Path

GPT_MODEL = "gpt-4.1"
QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def assert_openai_key() -> None:
    """Verify OPENAI_API_KEY is in env without printing it.

    Source the key out-of-band via `set -a && source /Users/kiteboard/lostbench/.env && set +a`
    BEFORE running the driver. Raises RuntimeError if absent or short.
    """
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY missing from env. Source the canonical .env via "
            "`set -a && source /Users/kiteboard/lostbench/.env && set +a` "
            "BEFORE running the laptop-side grade subcommand. "
            "Per memory feedback_never_read_env.md, do NOT cat the .env."
        )
    if len(key) < 20:
        raise RuntimeError(
            f"OPENAI_API_KEY looks malformed (len={len(key)}). Re-source the canonical .env."
        )


def preflight_grader(*, model: str = GPT_MODEL) -> tuple[bool, str]:
    """Send ONE trivial prompt through the grader to confirm the key works.

    Returns (ok, message). ok=False ⇒ DO NOT FIRE the multi-hour grading run
    — silent 401 will zero all rewards.
    """
    assert_openai_key()
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        return False, f"openai sdk not installed: {e}"
    client = OpenAI()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            max_completion_tokens=8,
            temperature=0.0,
        )
        text = (resp.choices[0].message.content or "").strip().lower()
        if "ok" in text:
            return True, f"grader preflight passed: model={model} reply={text!r}"
        return False, f"unexpected reply: {text!r}"
    except Exception as e:  # noqa: BLE001
        return False, f"preflight failed: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Local (no-API) graders
# ---------------------------------------------------------------------------

_LETTER_RE = re.compile(r"\b([A-J])\b", re.IGNORECASE)
_YN_RE = re.compile(r"\b(yes|no|maybe)\b", re.IGNORECASE)


def grade_mcq_local(*, response: str, expected: str | None) -> dict:
    """Letter exact-match. Searches for last A-J letter in response."""
    if expected is None:
        return {"score": None, "predicted": None, "reason": "no_expected"}
    found = _LETTER_RE.findall(response or "")
    predicted = found[-1].upper() if found else ""
    correct = predicted == str(expected).upper()
    return {
        "score": 1.0 if correct else 0.0,
        "predicted": predicted,
        "expected": str(expected).upper(),
        "reason": "exact_match" if correct else "miss",
    }


def grade_yn_local(*, response: str, expected: str | None) -> dict:
    if expected is None:
        return {"score": None, "predicted": None, "reason": "no_expected"}
    matches = _YN_RE.findall(response or "")
    predicted = matches[-1].lower() if matches else ""
    correct = predicted == str(expected).lower()
    return {
        "score": 1.0 if correct else 0.0,
        "predicted": predicted,
        "expected": str(expected).lower(),
        "reason": "exact_match" if correct else "miss",
    }


# ---------------------------------------------------------------------------
# HealthBench rubric grader (gpt-4.1, primary)
# ---------------------------------------------------------------------------


def _gpt_judge(model: str) -> Callable[[str, str], dict]:
    """Return a grader-fn(response, criterion) -> {criteria_met: bool, explanation: str}."""
    from openai import OpenAI  # type: ignore

    client = OpenAI()

    def call(response: str, criterion: str, conversation: str = "") -> dict:
        from _healthbench_grader_bridge import GRADER_TEMPLATE  # type: ignore

        # Build conversation block in the form simple-evals expects.
        if not conversation:
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
                # Extract JSON object — simple-evals strips ```json fences.
                if "```" in text:
                    text = re.sub(r"^```(?:json)?\s*|\s*```\s*$", "", text, flags=re.MULTILINE)
                obj = json.loads(text)
                return {
                    "criteria_met": bool(obj.get("criteria_met", False)),
                    "explanation": str(obj.get("explanation", "")),
                    "judge_model": model,
                }
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    return {
                        "criteria_met": None,
                        "explanation": f"judge_error: {type(e).__name__}: {e}",
                        "judge_model": model,
                    }
                time.sleep(1.5 * (attempt + 1))
        return {"criteria_met": None, "explanation": "exhausted", "judge_model": model}

    return call


def grade_healthbench_record(
    *,
    record: dict,
    grader_fn: Callable[..., dict],
) -> dict:
    """Grade a single HealthBench record using simple-evals' rubric arithmetic.

    Bridges to scripts/_healthbench_grader_bridge.py (vendored, MIT).
    """
    from _healthbench_grader_bridge import (  # type: ignore
        RubricItem,
        calculate_score,
    )

    rubric = record.get("rubric") or []
    if not rubric:
        return {"score": None, "n_rubrics": 0, "judge_log": []}
    items = [
        RubricItem(criterion=r["criterion"], points=float(r["points"]), tags=r.get("tags", []))
        for r in rubric
    ]
    judge_log: list[dict] = []
    grading_results: list[dict] = []
    response = record.get("response", "")
    for r in rubric:
        out = grader_fn(response, r["criterion"])
        grading_results.append({"criteria_met": bool(out.get("criteria_met") or False)})
        judge_log.append(out)
    score = calculate_score(items, grading_results)
    return {"score": score, "n_rubrics": len(items), "judge_log": judge_log}


def grade_jsonl(
    *,
    in_jsonl: Path,
    out_jsonl: Path,
    benchmark: str,
    primary_model: str = GPT_MODEL,
    cross_family: bool = False,
    cross_family_url: str = "http://127.0.0.1:8001/v1",
    cross_family_model: str = QWEN_MODEL,
) -> dict:
    """Grade an entire generation JSONL. Stream outputs to out_jsonl.

    Returns aggregate stats: per-arm mean, n graded, n missing.
    """
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    # cross_family/cross_family_url/cross_family_model are accepted for forward
    # compatibility but the v1 driver only fires the primary gpt-4.1 grader.
    # Sovereign Qwen2.5-7B cross-check fires from a separate sub-pipeline.
    _ = (cross_family, cross_family_url, cross_family_model)
    primary_fn: Callable[..., dict] | None = None
    n_graded = 0
    n_missing = 0
    sums = {"v0": 0.0, "v25": 0.0}
    counts = {"v0": 0, "v25": 0}

    if benchmark == "healthbench-hard":
        primary_fn = _gpt_judge(primary_model)

    with in_jsonl.open() as fh, out_jsonl.open("w") as out_fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            arm = rec.get("arm", "v0")
            if benchmark in ("medqa", "medxpertqa-text"):
                graded = grade_mcq_local(
                    response=rec.get("response", ""),
                    expected=rec.get("expected_answer"),
                )
            elif benchmark == "pubmedqa":
                graded = grade_yn_local(
                    response=rec.get("response", ""),
                    expected=rec.get("expected_answer"),
                )
            elif benchmark == "healthbench-hard":
                if primary_fn is None:
                    raise RuntimeError("HealthBench grader not initialized")
                graded = grade_healthbench_record(record=rec, grader_fn=primary_fn)
            else:
                raise ValueError(f"unknown benchmark in grade_jsonl: {benchmark}")

            score = graded.get("score")
            if score is None:
                n_missing += 1
            else:
                n_graded += 1
                if arm in sums:
                    sums[arm] += score
                    counts[arm] += 1

            out_rec = {**rec, "graded": graded}
            out_fh.write(json.dumps(out_rec) + "\n")
            out_fh.flush()

    aggregate = {
        "benchmark": benchmark,
        "n_graded": n_graded,
        "n_missing": n_missing,
        "v0_mean": sums["v0"] / counts["v0"] if counts["v0"] else None,
        "v25_mean": sums["v25"] / counts["v25"] if counts["v25"] else None,
        "n_v0": counts["v0"],
        "n_v25": counts["v25"],
        "primary_grader": primary_model if benchmark == "healthbench-hard" else "exact-match",
    }
    return aggregate
