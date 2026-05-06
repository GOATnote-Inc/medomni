#!/usr/bin/env python3
"""Clinical-safety auto-review for PRs touching clinical-content paths.

Invoked by .github/workflows/clinical-skill-review.yml on every PR open/sync.
Track #3 of the Cherny-pattern initiative — Boris Cherny reports 100% of
Anthropic PRs are auto-reviewed by Claude before human review. We scope that
discipline here to clinical-content paths (skills, system prompt, corpus pins).

Inputs (env): ANTHROPIC_API_KEY, GH_TOKEN, PR_NUMBER, PR_BASE_REF.
Outputs: posts a review comment via `gh pr comment`. Exits 1 on BLOCK.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import anthropic

MODEL = "claude-opus-4-7"
MAX_DIFF_CHARS = 60_000  # truncate diff to keep prompt + cost bounded

CLINICAL_PATHS = (
    "web/lib/agent/skills/",
    "mvp/medomni-inference/skills/",
    "mvp/medomni-inference/system_prompt_v1.md",
    "corpus/pins/",
)

RUBRIC = """\
You are a clinical-safety reviewer for medomni, a sovereign medical-LLM stack.
This PR touches clinical-content paths: agent skill markdown, the V_final
system prompt, or pinned eval fixtures. These files steer model behavior at
runtime; they are NOT ordinary code.

Apply this rubric, in order:

1. HALLUCINATION RISK — does the diff add a confident clinical claim
   (drug dose, threshold, criteria, guideline number) that is NOT obviously
   verifiable from a primary source visible in the diff?
2. TOOL-CALL AMBIGUITY — if the diff instructs the model to call tools, are
   the trigger conditions, argument shapes, and expected outputs unambiguous?
3. SAFETY-RULE CONTRADICTION — does the diff weaken or contradict an
   existing safety rule (refusal patterns, abstain semantics, PHI handling,
   escalation criteria)?
4. PHI GUARDRAILS — does the diff add a flow where user PHI could plausibly
   be logged, exfiltrated, or persisted without an explicit guard?
5. UNVERIFIED CITATIONS — does the diff cite a guideline (ACC/AHA, NICE,
   UpToDate, etc.) without a verifiable identifier (year, version, URL)?

VERDICTS:
- PASS — no rubric item triggers; diff is safe to merge.
- FLAG — at least one rubric item triggers but the issue is recoverable
  in a follow-up PR (e.g. add a citation, tighten a phrase). Merge OK,
  open a follow-up.
- BLOCK — at least one rubric item triggers AND the issue could ship a
  clinical-safety regression to runtime. Must be fixed before merge.

Return ONLY a fenced JSON block:
```json
{
  "verdict": "PASS" | "FLAG" | "BLOCK",
  "summary": "one sentence",
  "findings": [
    {"rubric": "hallucination|tool_call|safety_rule|phi|citation",
     "severity": "low|med|high",
     "file": "<path>",
     "note": "<one-line concern>"}
  ]
}
```
After the JSON, you MAY add a short rationale paragraph for the review
comment. The JSON is parsed; everything else is rendered as-is.
"""


def run(cmd: list[str], check: bool = True) -> str:
    out = subprocess.run(cmd, capture_output=True, text=True, check=check)
    return out.stdout


def get_clinical_diff(base_ref: str) -> tuple[str, list[str]]:
    run(["git", "fetch", "origin", base_ref, "--depth", "50"], check=False)
    files_raw = run(
        ["git", "diff", "--name-only", f"origin/{base_ref}...HEAD"],
        check=False,
    )
    all_files = [f for f in files_raw.splitlines() if f.strip()]
    clinical = [f for f in all_files if any(f.startswith(p) for p in CLINICAL_PATHS)]
    if not clinical:
        return "", []
    diff = run(
        ["git", "diff", f"origin/{base_ref}...HEAD", "--", *clinical],
        check=False,
    )
    if len(diff) > MAX_DIFF_CHARS:
        diff = diff[:MAX_DIFF_CHARS] + f"\n\n[diff truncated at {MAX_DIFF_CHARS} chars]\n"
    return diff, clinical


def parse_verdict(text: str) -> tuple[str, dict, str]:
    """Extract JSON verdict + remaining rationale."""
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not m:
        m = re.search(r"(\{[^{}]*\"verdict\"[^{}]*\})", text, re.DOTALL)
    if not m:
        return "FLAG", {"summary": "could not parse model output", "findings": []}, text
    try:
        blob = json.loads(m.group(1))
    except json.JSONDecodeError:
        return "FLAG", {"summary": "model returned invalid JSON", "findings": []}, text
    verdict = str(blob.get("verdict", "FLAG")).upper()
    if verdict not in {"PASS", "FLAG", "BLOCK"}:
        verdict = "FLAG"
    rationale = text[m.end() :].strip()
    return verdict, blob, rationale


def post_comment(pr: str, body: str) -> None:
    p = Path("/tmp/clinical_review_comment.md")
    p.write_text(body)
    subprocess.run(
        ["gh", "pr", "comment", pr, "--body-file", str(p)],
        check=False,
    )


def main() -> int:
    pr = os.environ["PR_NUMBER"]
    base_ref = os.environ.get("PR_BASE_REF", "main")

    diff, files = get_clinical_diff(base_ref)
    if not diff:
        print(f"No clinical-content files changed; skipping. (paths: {CLINICAL_PATHS})")
        return 0

    print(f"Reviewing {len(files)} clinical-content file(s):")
    for f in files:
        print(f"  - {f}")

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        messages=[
            {
                "role": "user",
                "content": (
                    f"{RUBRIC}\n\n---\n## Files changed (clinical-content paths only)\n"
                    + "\n".join(f"- {f}" for f in files)
                    + f"\n\n---\n## Diff\n```diff\n{diff}\n```\n"
                ),
            }
        ],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    verdict, blob, rationale = parse_verdict(text)

    findings = blob.get("findings", []) or []
    summary = blob.get("summary", "(no summary)")
    body = [
        f"### clinical-skill-review (auto): **{verdict}**",
        "",
        f"_Model: `{MODEL}` · Track #3 of the Cherny-pattern initiative · "
        f"reviewed {len(files)} clinical-content file(s)._",
        "",
        f"**Summary.** {summary}",
        "",
    ]
    if findings:
        body.append("**Findings:**")
        body.append("")
        body.append("| rubric | severity | file | note |")
        body.append("|---|---|---|---|")
        for f in findings[:20]:
            body.append(
                f"| {f.get('rubric', '?')} | {f.get('severity', '?')} | "
                f"`{f.get('file', '?')}` | {f.get('note', '')} |"
            )
        body.append("")
    if rationale:
        body.append("**Rationale.**")
        body.append("")
        body.append(rationale)
        body.append("")
    if verdict == "BLOCK":
        body.append(
            "> **This PR is BLOCKED by the clinical-safety gate.** "
            "Resolve the findings above and push a fix; the workflow re-runs on sync."
        )
    elif verdict == "FLAG":
        body.append("> Merge OK; please open a follow-up PR addressing the findings.")

    post_comment(pr, "\n".join(body))
    print(f"Verdict: {verdict}")
    return 1 if verdict == "BLOCK" else 0


if __name__ == "__main__":
    sys.exit(main())
