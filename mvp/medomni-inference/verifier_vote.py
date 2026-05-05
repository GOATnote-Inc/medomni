"""V_final inference wrapper — Best-of-K + PRM-min + claim-audit.

Per `findings/2026-05-05-process-supervision-verifiability/SPEC.md`
(iter-38 4-agent synthesis) and `findings/2026-05-05-improvement-
dimensions-roadmap/CARD.md`. Med-PRM EMNLP 2025: +13.5pp on MedQA-class.
MedS3 arxiv 2501.12051: +6.45 across 11 benchmarks via min-step-score.

Latency budget: 2 s clinical limit. K=4 with batched verifier scoring
fits on B300 catfish; K=8 needs MCTS-style early termination.

NOT YET WIRED into /api/agent. Integration point at V_final ship time:
1. Replace single-shot V_final call with verifier_vote.generate(...)
2. Gate to mutating tool calls (Skill /handoff) + final answer turns;
   pass through /calc and /open turns to single-shot for latency.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any


# ─────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────


@dataclass
class VerifierVoteResult:
    """Result of a verifier-vote inference turn."""

    chosen_response: str  # the winner
    chain_min_score: float  # PRM min-step-score across the chosen chain
    abstained: bool  # True if winner emitted <abstain/> or claim-audit blocked
    rejected_count: int  # K candidates considered; rejected_count = K - 1
    claim_audit: dict[str, Any]  # {claim_text: (evidence_id, NLI_label)}
    latency_ms_p50: float


async def generate(
    user_message: str,
    fhir_bundle: dict | None = None,
    *,
    base_endpoint: str = "http://catfish:8000/v1",
    prm_endpoint: str = "http://lobster:8001/v1/score-step",  # 8B Med-PRM
    k_samples: int = 4,
    temperature: float = 0.7,
    abstain_threshold: float = 0.3,  # PRM min-step-score below this → abstain
) -> VerifierVoteResult:
    """Generate K candidates, score with PRM, return min-step-score winner.

    Per MedS3 algorithm 1: chain_score = min(prm_step_scores) per chain;
    rank chains by chain_score descending; winner = ranked[0]. If winner's
    min-step-score < abstain_threshold → emit <abstain/>.

    Claim-audit pass: decompose winner.answer into atomic claims via
    MedScore; verify each against fhir_bundle + retrieved evidence via
    NLI; hard-block on `contradict` AND safety-critical claim.
    """
    # 1. Sample K chains from V_final base
    chains = await _sample_k_chains(
        base_endpoint, user_message, fhir_bundle, k=k_samples, temperature=temperature
    )

    # 2. Score each chain step-by-step with PRM
    scored = await asyncio.gather(*[_score_chain_with_prm(prm_endpoint, c) for c in chains])

    # 3. MedS3 min-step-score selector
    ranked = sorted(scored, key=lambda c: c["min_score"], reverse=True)
    winner = ranked[0]

    if winner["min_score"] < abstain_threshold:
        return VerifierVoteResult(
            chosen_response="<abstain/>",
            chain_min_score=winner["min_score"],
            abstained=True,
            rejected_count=k_samples - 1,
            claim_audit={},
            latency_ms_p50=winner["latency_ms"],
        )

    # 4. Claim-audit on the winner's answer (only the RESPOND step, not CoT)
    final_answer = _extract_respond_block(winner["text"])
    claims = _medscore_decompose(final_answer)
    audit = await _claim_audit(claims, fhir_bundle, prm_endpoint)

    # 5. Hard-block on safety-critical contradictions
    safety_violations = [c for c, v in audit.items() if v["label"] == "contradict" and _is_safety_critical(c)]
    if safety_violations:
        return VerifierVoteResult(
            chosen_response="<abstain/>",
            chain_min_score=winner["min_score"],
            abstained=True,
            rejected_count=k_samples - 1,
            claim_audit=audit,
            latency_ms_p50=winner["latency_ms"],
        )

    return VerifierVoteResult(
        chosen_response=winner["text"],
        chain_min_score=winner["min_score"],
        abstained=False,
        rejected_count=k_samples - 1,
        claim_audit=audit,
        latency_ms_p50=winner["latency_ms"],
    )


# ─────────────────────────────────────────────────────────────────────
# Helpers (TODO: implement at V_final ship time)
# ─────────────────────────────────────────────────────────────────────


async def _sample_k_chains(endpoint: str, msg: str, bundle: dict | None, *, k: int, temperature: float) -> list[dict]:
    """K parallel completions from V_final via vLLM /chat/completions."""
    # TODO: replace with httpx.AsyncClient calls to base_endpoint.
    # Each chain returned as {"text": str, "step_texts": list[str], "latency_ms": float}.
    raise NotImplementedError("wire to vLLM at V_final ship time")


async def _score_chain_with_prm(prm_endpoint: str, chain: dict) -> dict:
    """Per-step PRM scoring returning min-step-score and per-step scores."""
    # TODO: POST each step_text + retrieved guideline snippet to PRM endpoint;
    # PRM returns 0.0-1.0 per step; min across steps = chain score.
    raise NotImplementedError("wire to Med-PRM after V3 PRM training lands")


def _extract_respond_block(chain_text: str) -> str:
    """Pull the final RESPOND block out of the plan-then-act chain."""
    m = re.search(r"<plan>(.*?)</plan>\s*(?P<after>.*)", chain_text, flags=re.DOTALL)
    if not m:
        return chain_text
    return m.group("after").strip()


def _medscore_decompose(answer_text: str) -> list[str]:
    """MedScore atomic-claim decomposition (arxiv 2505.18452)."""
    # TODO: real MedScore via finetuned 7B decomposer. For now, sentence-split.
    sentences = re.split(r"(?<=[.!?])\s+", answer_text)
    return [s.strip() for s in sentences if len(s) > 12]


async def _claim_audit(claims: list[str], bundle: dict | None, prm_endpoint: str) -> dict[str, dict]:
    """Per-claim NLI against retrieved evidence + FHIR bundle.

    Returns {claim_text: {label: 'entailed|neutral|contradicted',
                          evidence_id: str|None,
                          score: float}}
    """
    # TODO: per-claim retrieval (PrimeKG-anchored + FHIR lookup) + NLI verifier.
    return {c: {"label": "neutral", "evidence_id": None, "score": 0.5} for c in claims}


def _is_safety_critical(claim: str) -> bool:
    """Heuristic: dosage/contraindication/dose-response claims are safety-critical."""
    pat = re.compile(
        r"\b(\d+(\.\d+)?\s*m?(g|cg|mg|kg|ml|mL|IU|U))|"
        r"contraindicat|black[- ]box|maximum dose|fatal|lethal|"
        r"interact|allergy|hypersensitiv",
        flags=re.IGNORECASE,
    )
    return bool(pat.search(claim))


# ─────────────────────────────────────────────────────────────────────
# Smoke test (run after V_final ships)
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    print("verifier_vote.py — V_final inference wrapper SCAFFOLD", file=sys.stderr)
    print("Status: NOT YET WIRED — populate _sample_k_chains, _score_chain_with_prm, _claim_audit", file=sys.stderr)
    print("Integration: see findings/2026-05-05-process-supervision-verifiability/SPEC.md", file=sys.stderr)
