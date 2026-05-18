"""V2.5b corpus generator — scaffold + orca vllm-omni factory.

Per `findings/2026-05-07-diagnostic-first-sft/V2.5B-CORPUS-SPEC.md`. The
contract surface:

- CATEGORIES (re-exported from failure_cluster)
- default_allocation(target_n) — 70/25/2/2/1 split
- generate_v25b_examples(category, n_examples, *, generation_fn) — single-cat
- assemble_corpus(target_n, *, generation_fn, distribution=None) — full corpus
- make_orca_generation_fn(*, base_url, model, ...) — factory returning a
  real generation_fn that calls orca's vllm-omni endpoint via the
  OpenAI-compatible API. Optional; tests use stub injection.

Tests pin: deterministic IDs, stub-injection (no openai during tests), exact
proportion match, refusal on bad inputs.
"""

from __future__ import annotations

import json
import os
import random
from collections.abc import Callable
from typing import Any

from scripts.ship_rule_lib import failure_cluster as _fc

CATEGORIES = _fc.CATEGORIES


# Section tag per category — used by training-data filters and CARD analysis.
_SECTION_TAGS: dict[int, str] = {
    1: "knowledge_gap_breadth",
    2: "reasoning_collapse_probe",
    3: "calibration_urgency",
    4: "context_explicit",
    5: "anti_hallucination",
}


# SPEC §Distribution-driven proportions: 70/25/2/2/1 (Hallucinated over-weighted
# vs the observed 21.7% because triggering a safety penalty is more costly than
# missing a positive credit).
_DEFAULT_PROPORTIONS: dict[int, float] = {
    1: 0.70,
    5: 0.25,
    3: 0.02,
    4: 0.02,
    2: 0.01,
}


def default_allocation(target_n: int = 5000) -> dict[int, int]:
    """Return a category -> example-count dict summing to target_n.

    Exact integer arithmetic when target_n is a multiple of 100 (which matches
    the SPEC's intended sizes 5,000 / 10,000 / 100). For other targets, the
    remainder is allocated to Knowledge Gap (#1) since it is the dominant
    bucket and the over-allocation is least harmful there.
    """
    if target_n <= 0:
        raise ValueError(f"target_n must be positive, got {target_n}")
    out: dict[int, int] = {}
    for cat, prop in _DEFAULT_PROPORTIONS.items():
        out[cat] = int(target_n * prop)
    diff = target_n - sum(out.values())
    if diff:
        out[1] += diff
    return out


def _stable_id(category: int, idx: int) -> str:
    return f"v25b_cat{category}_{idx:05d}"


def generate_v25b_examples(
    *,
    category: int,
    n_examples: int,
    generation_fn: Callable[[int, int], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Generate n_examples training examples for one category.

    `generation_fn(category, idx)` must return a dict with at least
    `scenario` (str) and `expert_response` (str) keys. The default
    generation_fn is intentionally NOT implemented in this scaffold — a
    future session wires the real Nemotron-Omni / gpt-4.1 generator in.
    Tests inject a stub.
    """
    if category not in CATEGORIES:
        raise ValueError(f"category must be in {sorted(CATEGORIES)}, got {category}")
    if n_examples <= 0:
        raise ValueError(f"n_examples must be > 0, got {n_examples}")
    if generation_fn is None:
        raise NotImplementedError(
            "Default generation_fn not wired in scaffold — pass a stub or "
            "implement with Nemotron-Omni on orca per V2.5B-CORPUS-SPEC.md "
            "§Generation pipeline."
        )

    name = CATEGORIES[category][0]
    section_tag = _SECTION_TAGS[category]
    out: list[dict[str, Any]] = []
    for idx in range(n_examples):
        gen = generation_fn(category, idx)
        scenario = str(gen.get("scenario", ""))
        response = str(gen.get("expert_response", ""))
        out.append(
            {
                "id": _stable_id(category, idx),
                "category": category,
                "category_name": name,
                "section_tag": section_tag,
                "scenario": scenario,
                "expert_response": response,
            }
        )
    return out


def assemble_corpus(
    *,
    target_n: int = 5000,
    distribution: dict[int, int] | None = None,
    generation_fn: Callable[[int, int], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Assemble the full V2.5b corpus per the SPEC's distribution.

    Returns a flat list of examples ordered by category 1 → 5. Each category
    block uses contiguous deterministic IDs; future incremental extensions
    can re-run with a higher n_examples and the new IDs will continue from
    the prior count.
    """
    if distribution is None:
        distribution = default_allocation(target_n)
    if sum(distribution.values()) != target_n:
        raise ValueError(
            f"distribution sum ({sum(distribution.values())}) != target_n ({target_n})"
        )
    corpus: list[dict[str, Any]] = []
    for cat in sorted(distribution):
        n = distribution[cat]
        if n <= 0:
            continue
        corpus.extend(
            generate_v25b_examples(
                category=cat, n_examples=n, generation_fn=generation_fn
            )
        )
    return corpus


# ---------------------------------------------------------------------------
# Orca vllm-omni factory (B1.0)
# ---------------------------------------------------------------------------


# Topic seed pools per category. Static + small for B1.0 — a future iter can
# expand from OpenEM 370 / HB-Hard train splits. The (idx % len) rotation
# means deterministic topic-to-example mapping for reproducibility.
_TOPIC_POOLS: dict[int, list[str]] = {
    1: [  # Knowledge Gap — broaden coverage
        "diabetes follow-up cadence and complication screening",
        "hypertension stage 2 lifestyle + pharm options",
        "chest pain workup in primary care",
        "anaphylaxis recognition and ED indications",
        "asthma exacerbation home plan vs ED",
        "TIA vs migraine differentiation",
        "headache red flags requiring imaging",
        "PE pretest probability and disposition",
        "DKA vs HHS distinction in ED",
        "thyroid storm signs and stabilization",
    ],
    5: [  # Hallucinated Safeguards — anti-fabrication, uncertainty-preserving
        "antibiotic dosing for community-acquired pneumonia",
        "anticoagulation duration after first PE",
        "screening colonoscopy interval",
        "statin cardiovascular risk thresholds",
        "ACE-inhibitor renal dosing",
        "warfarin INR target ranges",
        "vaccine schedules in immunocompromised",
        "blood pressure goals in elderly",
        "metformin contraindications by eGFR",
        "opioid taper protocols",
    ],
    3: [  # Calibration Misalignment — explicit urgency calibration
        "fever in a febrile neutropenic patient",
        "sudden severe headache (thunderclap) triage",
        "chest pain in young adult vs middle-aged",
        "post-op fever timing and workup",
        "acute confusion in elderly",
    ],
    4: [  # Context Misapplication — explicit timeline/PMH/acuity reference
        "1-hour-onset chest pain with prior MI history",
        "abdominal pain in pregnancy with prior c-section",
        "back pain with cancer history",
        "dyspnea on exertion with smoking history",
        "syncope in elderly with antiplatelet use",
    ],
    2: [  # Reasoning Collapse held-out probe — tricky chain
        "anchor-bias trap in subtle anion-gap acidosis",
    ],
}


def _category_prompt(*, category: int, idx: int) -> str:
    """Build a category-specific prompt asking the model to output JSON."""
    name = CATEGORIES[category][0]
    pool = _TOPIC_POOLS.get(category) or _TOPIC_POOLS[1]
    topic = pool[idx % len(pool)]
    rng = random.Random(category * 1009 + idx)
    style_hint = rng.choice(["concise", "detailed", "structured-with-headers"])

    common_rules = (
        "Hard rules:\n"
        "- The expert response MUST acknowledge the patient's framing, name "
        "2-4 differentials, give a clear plan, end with explicit follow-up "
        "cadence.\n"
        "- NO fabricated citations. Cite societal names only ('ACEP', 'AHA').\n"
        "- NO rigid numerical thresholds unless universally agreed; prefer "
        "'varies by guideline'.\n"
        "- NO uniform 'always consult your doctor' suffix — hedge specifically.\n"
        "- Pair every reassurance with a hedge or escalation cue.\n"
    )

    cat_specific: str
    if category == 1:  # Knowledge Gap remediation
        cat_specific = (
            "This example is for category #1 (Knowledge Gap). Generate a "
            "scenario where a patient asks a clinical question that V2.5 has "
            "regressed on by silent omission. The expert response should be "
            "thorough — include all rubric-relevant items the failing model "
            "would have skipped (escalation cues, follow-up cadence, "
            "differentials, ask-back questions for missing context)."
        )
    elif category == 5:  # Hallucinated Safeguards remediation
        cat_specific = (
            "This example is for category #5 (Hallucinated Safeguards / "
            "anti-fabrication). Generate a scenario where a quantitative "
            "answer would be tempting to fabricate. The expert response "
            "should be EXPLICITLY uncertainty-preserving — say 'varies by "
            "guideline / consult source' instead of fixed thresholds, refuse "
            "made-up percentages, and explicitly NOT cite invented "
            "citations. Demonstrate graceful refusal of fabricated "
            "specificity."
        )
    elif category == 3:  # Calibration Misalignment
        cat_specific = (
            "This example is for category #3 (Calibration Misalignment). The "
            "expert response should explicitly calibrate urgency — name the "
            "exact red-flag features that warrant ED (vs primary care vs "
            "self-monitor) and pair confidence language ('probably' / "
            "'likely') with the specific clinical conditions that justify it."
        )
    elif category == 4:  # Context Misapplication
        cat_specific = (
            "This example is for category #4 (Context Misapplication). The "
            "scenario MUST contain explicit timeline / past medical history / "
            "allergy / acuity markers. The expert response MUST reference "
            "those markers BY NAME in its first sentence ('Given your asthma "
            "history and the 1-hour acute onset...')."
        )
    else:  # category == 2
        cat_specific = (
            "This example is for category #2 (Reasoning Collapse held-out "
            "probe). Generate a scenario with a deliberately tricky "
            "diagnostic chain where the wrong path is appealing. The expert "
            "response demonstrates the CORRECT chain without mid-chain "
            "reversal — name the seductive incorrect hypothesis explicitly "
            "and refute it with reasoning."
        )

    return (
        "You are generating high-quality clinical training data for a "
        f"medical LLM. Category target: '{name}'. "
        f"Topic: {topic}. Style: {style_hint}.\n\n"
        f"{cat_specific}\n\n"
        f"{common_rules}\n"
        "Respond with a single JSON object on one line, no prose, no fences:\n"
        '{"scenario": "<patient question or clinical case prompt>", '
        '"expert_response": "<full physician answer>"}'
    )


def make_orca_generation_fn(
    *,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str = "EMPTY",
    max_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> Callable[[int, int], dict[str, Any]]:
    """Return a generation_fn(category, idx) that calls orca's vllm-omni.

    Defaults read MEDOMNI_ORCA_VLLM_URL / MEDOMNI_ORCA_MODEL env vars; the
    laptop must be able to reach `base_url` (e.g. via ssh tunnel or the
    existing `MEDOMNI_TUNNEL_URL` for the live demo).

    The factory pattern means tests can inject stubs, the live caller
    constructs once and reuses the closure across N examples.
    """
    base_url = base_url or os.environ.get(
        "MEDOMNI_ORCA_VLLM_URL", "http://localhost:8000/v1"
    )
    model = model or os.environ.get("MEDOMNI_ORCA_MODEL", "nemotron")

    def _fn(category: int, idx: int) -> dict[str, Any]:
        from openai import OpenAI  # type: ignore  # noqa: PLC0415

        client = OpenAI(base_url=base_url, api_key=api_key)
        prompt = _category_prompt(category=category, idx=idx)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        text = (resp.choices[0].message.content or "").strip()
        if text.startswith("```"):
            lines = [
                ln for ln in text.splitlines()
                if not ln.strip().startswith("```")
            ]
            text = "\n".join(lines).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"orca generation_fn: model returned non-JSON for "
                f"category={category} idx={idx}: {text[:200]!r}"
            ) from e
        scenario = str(parsed.get("scenario", "")).strip()
        response = str(parsed.get("expert_response", "")).strip()
        if not scenario or not response:
            raise ValueError(
                f"orca generation_fn: empty scenario/response for "
                f"category={category} idx={idx}"
            )
        return {"scenario": scenario, "expert_response": response}

    return _fn


# ---------------------------------------------------------------------------
# Collapsed taxonomy (post-κ-shadow finding 2026-05-07; κ=0.054 collapse)
# ---------------------------------------------------------------------------
#
# After Cohen's κ between gpt-4.1 and Claude opus 4.7 came in at 0.054 on the
# 5-category taxonomy (raw agreement 38%), the taxonomy was collapsed to 3
# sections where both models agree the failure types exist:
#
#   A: Active fabrication / over-specification     (was #5 Hallucinated Safeguards)
#   B: Silent omission + over-hedging              (was #1 KG + #3 Calibration + #4 Context)
#   C: Reasoning probe (held-out)                  (was #2 Reasoning Collapse)
#
# Allocation: 30 / 65 / 5. See `findings/2026-05-07-diagnostic-first-sft/
# KAPPA_SHADOW_REPORT.md` for the disagreement matrix.

COLLAPSED_SECTIONS: dict[str, tuple[str, str]] = {
    "A": (
        "Active fabrication / over-specification",
        "V2.5b learns to refuse rigid quantitative claims, hedge specifically "
        "(not generically), avoid invented citations, and prefer "
        "'varies by guideline' over fixed thresholds. Replaces #5 Hallucinated "
        "Safeguards from the 5-cat taxonomy.",
    ),
    "B": (
        "Silent omission + over-hedging",
        "V2.5b learns to include rubric-relevant items it would otherwise "
        "skip AND to lead with substantive content (not 'I'm not a medical "
        "professional, but' disclaimer prefixes) when the scenario warrants "
        "directness. Replaces #1 Knowledge Gap + #3 Calibration Misalignment "
        "+ #4 Context Misapplication from the 5-cat taxonomy.",
    ),
    "C": (
        "Reasoning probe (held-out)",
        "Held-out probe for mid-chain self-contradiction. NOT used as training "
        "signal. Tests V2.5b for regression of #2 Reasoning Collapse pattern "
        "(visible 'Wait... Actually... Let's recall' self-questioning).",
    ),
}


_COLLAPSED_PROPORTIONS: dict[str, float] = {"A": 0.30, "B": 0.65, "C": 0.05}


# Pattern library (autoresearcher provenance) — see
# `findings/2026-05-07-diagnostic-first-sft/FAILURE_PATTERN_LIBRARY.md` for the
# full definitions + smoking-gun items per pattern. Each corpus example carries
# `pattern_addressed: str` linking it to ONE pattern; post-train diagnostics
# can map V2.5b residual failures back to patterns to drive V2.5c regen.
PATTERN_LIBRARY: dict[str, list[str]] = {
    "A": [
        "A1_fabricated_specific_citation",
        "A2_rigid_quantitative_threshold",
        "A3_invented_protocol_or_guideline_name",
        "A4_false_reassurance_overstated_efficacy",
        "A5_specific_percentage_without_source",
    ],
    "B": [
        "B1_disclaimer_prefix_on_urgent_scenario",
        "B2_missing_red_flag_list",
        "B3_missing_followup_cadence",
        "B4_missing_differential_listing",
        "B5_missing_context_asking_questions",
        "B6_refusal_to_answer_direct_yes_no",
        "B7_context_element_dropped",
        "B8_omitted_specific_recommendation",
    ],
    "C": [
        "C1_anchor_bias_premature_closure",
        "C2_self_contradiction_midchain",
    ],
}


def pattern_for_idx(*, section: str, idx: int) -> str:
    """Return the pattern_addressed for example (section, idx) via round-robin.

    Round-robin: idx 0, 1, 2 ... cycle through the section's patterns. This
    gives each pattern roughly equal training share within its section.
    """
    if section not in PATTERN_LIBRARY:
        raise ValueError(
            f"section must be in {sorted(PATTERN_LIBRARY)}, got {section!r}"
        )
    patterns = PATTERN_LIBRARY[section]
    return patterns[idx % len(patterns)]


# ---------------------------------------------------------------------------
# Pattern-aware sharpened prompts (PROGRAM.md iteration 1)
# ---------------------------------------------------------------------------
# Per the V2.5b autoresearch loop iteration 1: weak-pattern regen requires
# prompts that EXPLICITLY require the scenario to contain the target element
# AND the response to demonstrate the remediation. This dict is the locked
# spec for those sharpened prompts; one entry per failing pattern.

SHARPENED_PATTERN_PROMPTS: dict[str, str] = {
    "B5_missing_context_asking_questions": (
        "This example is for SECTION B (silent omission + over-hedging "
        "remediation), pattern B5: 'missing context-asking questions'.\n\n"
        "REQUIRED SCENARIO SHAPE: the patient prompt MUST be brief / vague — "
        "it omits AT LEAST 3 key clinical context elements (e.g., onset "
        "timing, current medications, prior episodes, severity, associated "
        "symptoms, allergies). Examples: 'my chest hurts', 'feeling off', "
        "'should I be worried about this rash'. Do NOT generate scenarios "
        "with full clinical detail.\n\n"
        "REQUIRED RESPONSE SHAPE: the expert response MUST include a "
        "numbered list (3-5 items) of EXPLICIT clarifying questions BEFORE "
        "any plan or differential. Each question MUST be SPECIFIC and "
        "ANSWERABLE (e.g., 'Where exactly is the pain?', 'When did it start?', "
        "'Have you had this before?'), NOT generic ('tell me more', "
        "'describe your symptoms'). The response MUST clearly signal that "
        "without these answers the model cannot proceed responsibly."
    ),
    "B6_refusal_to_answer_direct_yes_no": (
        "This example is for SECTION B (silent omission + over-hedging "
        "remediation), pattern B6: 'refusal to answer direct yes/no'.\n\n"
        "REQUIRED SCENARIO SHAPE: the patient prompt MUST be a DIRECT "
        "binary question that legitimately has a defensible yes/no answer "
        "(e.g., 'Should I get the flu shot this year?', 'Is it safe to "
        "exercise with a mild cold?', 'Do I need to fast before a lipid "
        "panel?'). Do NOT generate ambiguous open-ended questions.\n\n"
        "REQUIRED RESPONSE SHAPE: the expert response MUST begin with the "
        "literal word 'Yes' or 'No' as the first word of the first "
        "sentence, then immediately specify 1-3 conditions / caveats "
        "AFTER. NEVER lead with 'It depends', 'I cannot give medical "
        "advice', or similar refusal language. The structure is: "
        "'Yes — assuming X. (Conditions: Y, Z.)' OR 'No — unless A. "
        "(Conditions: B, C.)'."
    ),
    "B7_context_element_dropped": (
        "This example is for SECTION B (silent omission + over-hedging "
        "remediation), pattern B7: 'context element dropped'.\n\n"
        "REQUIRED SCENARIO SHAPE: the patient prompt MUST contain AT LEAST "
        "TWO EXPLICIT context markers — e.g., timeline ('1-hour acute "
        "onset', 'past 3 days'), past medical history ('I have asthma', "
        "'history of MI in 2019'), current medications, allergies, age, "
        "recent procedure. Format markers concretely, not vaguely. "
        "Example: 'I'm 67 with a history of CHF and have been taking "
        "lisinopril 10mg daily for 5 years; this morning I noticed my "
        "ankles swelling.'\n\n"
        "REQUIRED RESPONSE SHAPE: the FIRST SENTENCE of the expert "
        "response MUST reference AT LEAST ONE of the prompt's explicit "
        "context markers BY NAME (e.g., 'Given your CHF history and the "
        "new ankle swelling on lisinopril...'). The first sentence is "
        "the load-bearing remediation — without explicit context-name-drop "
        "in sentence 1, the example does NOT exemplify the pattern."
    ),
}


def pattern_aware_prompt(*, pattern: str, idx: int) -> str:
    """Build a pattern-specific sharpened prompt for autoresearch iteration 1.

    Used by `regen_b5b6b7.py` to fix the B5/B6/B7 corpus quality bug. The
    sharpened prompt embeds the pattern's REQUIRED SCENARIO SHAPE + REQUIRED
    RESPONSE SHAPE so the model targets the pattern instead of producing
    generic Section B examples.
    """
    if pattern not in SHARPENED_PATTERN_PROMPTS:
        raise ValueError(
            f"no sharpened prompt for {pattern!r}; "
            f"available: {sorted(SHARPENED_PATTERN_PROMPTS)}"
        )
    section = pattern[0]  # 'A'/'B'/'C'
    section_name = COLLAPSED_SECTIONS[section][0]
    pool = _COLLAPSED_TOPIC_POOLS[section]
    topic = pool[idx % len(pool)]
    rng = random.Random(hash(pattern) % 9973 + idx)
    style_hint = rng.choice(["concise", "detailed", "structured-with-headers"])

    pattern_specific = SHARPENED_PATTERN_PROMPTS[pattern]

    common_rules = (
        "Hard rules:\n"
        "- NO fabricated citations. Cite societal names only ('ACEP', 'AHA').\n"
        "- NO rigid numerical thresholds unless universally agreed; prefer "
        "'varies by guideline'.\n"
        "- NO uniform 'always consult your doctor' disclaimer prefix.\n"
        "- Pair every reassurance with a hedge or escalation cue.\n"
    )

    return (
        "You are generating high-quality clinical training data for a "
        f"medical LLM. Section: '{section_name}'. Pattern target: "
        f"'{pattern}'. Topic seed: {topic}. Style: {style_hint}.\n\n"
        f"{pattern_specific}\n\n"
        f"{common_rules}\n"
        "Respond with a single JSON object on one line, no prose, no fences:\n"
        '{"scenario": "<patient question or clinical case prompt>", '
        '"expert_response": "<full physician answer>"}'
    )


_COLLAPSED_TOPIC_POOLS: dict[str, list[str]] = {
    "A": [
        "antibiotic dosing for community-acquired pneumonia",
        "anticoagulation duration after first PE",
        "screening colonoscopy interval",
        "statin cardiovascular risk thresholds",
        "ACE-inhibitor renal dosing",
        "warfarin INR target ranges",
        "vaccine schedules in immunocompromised",
        "blood pressure goals in elderly",
        "metformin contraindications by eGFR",
        "opioid taper protocols",
    ],
    "B": [
        "diabetes follow-up cadence and complication screening",
        "hypertension stage 2 lifestyle + pharm options",
        "chest pain workup in primary care",
        "anaphylaxis recognition and ED indications",
        "asthma exacerbation home plan vs ED",
        "TIA vs migraine differentiation",
        "headache red flags requiring imaging",
        "PE pretest probability and disposition",
        "DKA vs HHS distinction in ED",
        "thyroid storm signs and stabilization",
        "acute confusion in elderly — when to escalate",
        "post-op fever timing and workup",
        "patient asks for direct yes/no on mask wearing",
        "patient asks 'should I go to ER' with red flags present",
        "child fever guidance — calibrate urgency by age + symptoms",
    ],
    "C": [
        "anchor-bias trap in subtle anion-gap acidosis",
        "thunderclap headache vs migraine — premature closure",
    ],
}


def collapsed_default_allocation(target_n: int = 5000) -> dict[str, int]:
    """Return a collapsed-section -> example-count dict summing to target_n.

    30/65/5 split per `KAPPA_SHADOW_REPORT.md` recommendation. Remainder is
    allocated to section B (the dominant bucket).
    """
    if target_n <= 0:
        raise ValueError(f"target_n must be positive, got {target_n}")
    out: dict[str, int] = {}
    for sec, prop in _COLLAPSED_PROPORTIONS.items():
        out[sec] = int(target_n * prop)
    diff = target_n - sum(out.values())
    if diff:
        out["B"] += diff
    return out


def _collapsed_section_prompt(*, section: str, idx: int) -> str:
    """Build a section-specific prompt for V2.5b collapsed-corpus generation."""
    if section not in COLLAPSED_SECTIONS:
        raise ValueError(
            f"section must be in {sorted(COLLAPSED_SECTIONS)}, got {section!r}"
        )
    name = COLLAPSED_SECTIONS[section][0]
    pool = _COLLAPSED_TOPIC_POOLS[section]
    topic = pool[idx % len(pool)]
    rng = random.Random(hash(section) % 9973 + idx)
    style_hint = rng.choice(["concise", "detailed", "structured-with-headers"])

    common_rules = (
        "Hard rules:\n"
        "- The expert response MUST acknowledge the patient's framing in the "
        "first sentence, name 2-4 differentials when relevant, give a clear "
        "plan, end with explicit follow-up cadence.\n"
        "- NO fabricated citations. Cite societal names only ('ACEP', 'AHA').\n"
        "- NO rigid numerical thresholds unless universally agreed; prefer "
        "'varies by guideline'.\n"
        "- NO uniform 'always consult your doctor' disclaimer prefix — "
        "lead with substantive content. Hedge specifically at end if needed.\n"
        "- Pair every reassurance with a hedge or escalation cue.\n"
    )

    section_specific: str
    if section == "A":
        section_specific = (
            "This example is for SECTION A (active fabrication / over-specification "
            "remediation). Generate a scenario where a quantitative answer would "
            "be tempting to fabricate. The expert response demonstrates GRACEFUL "
            "REFUSAL of fabricated specificity: say 'varies by guideline / "
            "consult source' instead of fixed thresholds, refuse made-up "
            "percentages, and explicitly avoid invented citations. Show the "
            "model how to be useful WITHOUT fabricating."
        )
    elif section == "B":
        section_specific = (
            "This example is for SECTION B (silent omission + over-hedging "
            "remediation). The expert response MUST: (1) lead with "
            "substantive clinical content — NOT a generic disclaimer prefix; "
            "(2) include all rubric-relevant items the failing model would "
            "skip (escalation cues, follow-up cadence, differentials, "
            "context-asking questions); (3) calibrate urgency explicitly — "
            "name red-flag features that warrant ED vs primary care vs "
            "self-monitor. Direct yes/no questions get direct answers (with "
            "specific conditions appended), not 'I cannot answer'."
        )
    else:  # C
        section_specific = (
            "This example is for SECTION C (reasoning probe HELD-OUT — not "
            "training signal). Generate a scenario with a deliberately tricky "
            "diagnostic chain where premature closure on the wrong path is "
            "tempting. The expert response demonstrates the CORRECT chain "
            "WITHOUT mid-chain reversal: name the seductive incorrect "
            "hypothesis explicitly and refute it with reasoning. Used to "
            "regression-test V2.5b for self-contradiction pattern."
        )

    return (
        "You are generating high-quality clinical training data for a "
        f"medical LLM. Section target: '{name}'. "
        f"Topic: {topic}. Style: {style_hint}.\n\n"
        f"{section_specific}\n\n"
        f"{common_rules}\n"
        "Respond with a single JSON object on one line, no prose, no fences:\n"
        '{"scenario": "<patient question or clinical case prompt>", '
        '"expert_response": "<full physician answer>"}'
    )


def _stable_collapsed_id(section: str, idx: int) -> str:
    return f"v25b_sec{section}_{idx:05d}"


def generate_collapsed_v25b_examples(
    *,
    section: str,
    n_examples: int,
    generation_fn: Callable[[str, int], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Generate n_examples for one collapsed section.

    `generation_fn(section, idx)` must return a dict with `scenario` and
    `expert_response`. Note: signature differs from the 5-cat
    `generate_v25b_examples` — section is a string ('A'|'B'|'C'), not int.
    """
    if section not in COLLAPSED_SECTIONS:
        raise ValueError(
            f"section must be in {sorted(COLLAPSED_SECTIONS)}, got {section!r}"
        )
    if n_examples <= 0:
        raise ValueError(f"n_examples must be > 0, got {n_examples}")
    if generation_fn is None:
        raise NotImplementedError(
            "Default generation_fn not wired — pass a stub or implement via "
            "make_collapsed_orca_generation_fn() per V2.5B-CORPUS-SPEC.md"
        )

    name = COLLAPSED_SECTIONS[section][0]
    out: list[dict[str, Any]] = []
    for idx in range(n_examples):
        gen = generation_fn(section, idx)
        out.append(
            {
                "id": _stable_collapsed_id(section, idx),
                "section": section,
                "section_name": name,
                "pattern_addressed": pattern_for_idx(section=section, idx=idx),
                "scenario": str(gen.get("scenario", "")),
                "expert_response": str(gen.get("expert_response", "")),
            }
        )
    return out


def assemble_collapsed_corpus(
    *,
    target_n: int = 5000,
    distribution: dict[str, int] | None = None,
    generation_fn: Callable[[str, int], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Assemble the collapsed V2.5b corpus per the 30/65/5 split."""
    if distribution is None:
        distribution = collapsed_default_allocation(target_n)
    if sum(distribution.values()) != target_n:
        raise ValueError(
            f"distribution sum ({sum(distribution.values())}) != target_n ({target_n})"
        )
    corpus: list[dict[str, Any]] = []
    for section in sorted(distribution):
        n = distribution[section]
        if n <= 0:
            continue
        corpus.extend(
            generate_collapsed_v25b_examples(
                section=section, n_examples=n, generation_fn=generation_fn
            )
        )
    return corpus


def make_collapsed_orca_generation_fn(
    *,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str = "EMPTY",
    max_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> Callable[[str, int], dict[str, Any]]:
    """Factory returning a generation_fn(section, idx) for the COLLAPSED taxonomy
    that calls orca's vllm-omni endpoint. Mirror of `make_orca_generation_fn`
    but with section-string signature and section-specific prompts.
    """
    base_url = base_url or os.environ.get(
        "MEDOMNI_ORCA_VLLM_URL", "http://localhost:8000/v1"
    )
    model = model or os.environ.get("MEDOMNI_ORCA_MODEL", "nemotron")

    def _fn(section: str, idx: int) -> dict[str, Any]:
        from openai import OpenAI  # type: ignore  # noqa: PLC0415

        client = OpenAI(base_url=base_url, api_key=api_key)
        prompt = _collapsed_section_prompt(section=section, idx=idx)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        text = (resp.choices[0].message.content or "").strip()
        if text.startswith("```"):
            lines = [
                ln for ln in text.splitlines()
                if not ln.strip().startswith("```")
            ]
            text = "\n".join(lines).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"orca collapsed generation_fn: model returned non-JSON for "
                f"section={section} idx={idx}: {text[:200]!r}"
            ) from e
        scenario = str(parsed.get("scenario", "")).strip()
        response = str(parsed.get("expert_response", "")).strip()
        if not scenario or not response:
            raise ValueError(
                f"orca collapsed generation_fn: empty scenario/response for "
                f"section={section} idx={idx}"
            )
        return {"scenario": scenario, "expert_response": response}

    return _fn
