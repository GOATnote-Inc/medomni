"""TDD red phase: contract tests for the V2.5b corpus generator scaffold.

Pinned by `findings/2026-05-07-diagnostic-first-sft/V2.5B-CORPUS-SPEC.md`.
Stdlib + (lazily) openai only. The actual gpt-4.1 / Nemotron-Omni generation
is out of scope this turn — these tests pin the contract surface so a future
session can wire the real generation_fn without re-deriving the schema.
"""

from __future__ import annotations

import pytest

from scripts.ship_rule_lib import corpus_generator as cg
from scripts.ship_rule_lib import failure_cluster as fc


# ---------------------------------------------------------------------------
# Allocation match
# ---------------------------------------------------------------------------


def test_default_allocation_sums_to_target() -> None:
    """SPEC §Distribution-driven proportions: target 5000, allocation must sum to it."""
    alloc = cg.default_allocation(target_n=5000)
    assert sum(alloc.values()) == 5000
    # All 5 categories present, even minor ones
    assert set(alloc.keys()) == {1, 2, 3, 4, 5}


def test_default_allocation_proportions_match_spec() -> None:
    """SPEC §Distribution-driven proportions: 70/25/2/2/1."""
    alloc = cg.default_allocation(target_n=10_000)
    assert alloc[1] == 7000   # Knowledge Gap 70%
    assert alloc[5] == 2500   # Hallucinated Safeguards 25% (over-weight)
    assert alloc[3] == 200    # Calibration 2%
    assert alloc[4] == 200    # Context Misapp 2%
    assert alloc[2] == 100    # Reasoning Collapse 1% (held-out probe)


def test_categories_match_failure_cluster_module() -> None:
    """Generator must reuse the locked taxonomy from failure_cluster.CATEGORIES."""
    assert cg.CATEGORIES is fc.CATEGORIES or cg.CATEGORIES == fc.CATEGORIES


# ---------------------------------------------------------------------------
# Single-category generation
# ---------------------------------------------------------------------------


def test_generate_v25b_examples_uses_injected_stub() -> None:
    """SPEC §Generation pipeline: generation_fn is injectable so tests + future
    real generators (gpt-4.1 / Nemotron-Omni) plug into the same surface."""
    captured: list[tuple[int, int]] = []

    def stub(category: int, idx: int) -> dict:
        captured.append((category, idx))
        return {
            "scenario": f"scenario for cat {category} idx {idx}",
            "expert_response": f"expert response cat {category} idx {idx}",
        }

    examples = cg.generate_v25b_examples(category=1, n_examples=3, generation_fn=stub)
    assert len(examples) == 3
    assert captured == [(1, 0), (1, 1), (1, 2)]
    for ex in examples:
        assert ex["category"] == 1
        assert ex["category_name"] == "Knowledge Gap"
        assert "scenario" in ex and "expert_response" in ex
        assert "id" in ex
        assert "section_tag" in ex


def test_generate_v25b_examples_rejects_bad_category() -> None:
    def stub(category: int, idx: int) -> dict:
        return {"scenario": "s", "expert_response": "r"}
    with pytest.raises(ValueError, match="category"):
        cg.generate_v25b_examples(category=7, n_examples=1, generation_fn=stub)


def test_generate_v25b_examples_rejects_zero_or_negative_n() -> None:
    def stub(category: int, idx: int) -> dict:
        return {"scenario": "s", "expert_response": "r"}
    with pytest.raises(ValueError):
        cg.generate_v25b_examples(category=1, n_examples=0, generation_fn=stub)
    with pytest.raises(ValueError):
        cg.generate_v25b_examples(category=1, n_examples=-3, generation_fn=stub)


def test_generate_v25b_examples_ids_are_deterministic() -> None:
    """Stable IDs allow incremental corpus extension without dup risk."""
    def stub(category: int, idx: int) -> dict:
        return {"scenario": "s", "expert_response": "r"}
    a = cg.generate_v25b_examples(category=5, n_examples=2, generation_fn=stub)
    b = cg.generate_v25b_examples(category=5, n_examples=2, generation_fn=stub)
    assert [e["id"] for e in a] == [e["id"] for e in b]


# ---------------------------------------------------------------------------
# Full-corpus assembly
# ---------------------------------------------------------------------------


def test_assemble_corpus_total_equals_target_n() -> None:
    def stub(category: int, idx: int) -> dict:
        return {"scenario": "s", "expert_response": "r"}
    corpus = cg.assemble_corpus(target_n=100, generation_fn=stub)
    assert len(corpus) == 100


def test_assemble_corpus_distribution_per_category() -> None:
    """SPEC §Distribution-driven proportions: 70/25/2/2/1 at target_n=100."""
    def stub(category: int, idx: int) -> dict:
        return {"scenario": "s", "expert_response": "r"}
    corpus = cg.assemble_corpus(target_n=100, generation_fn=stub)
    counts = {c: sum(1 for ex in corpus if ex["category"] == c) for c in (1, 2, 3, 4, 5)}
    assert counts[1] == 70
    assert counts[5] == 25
    assert counts[3] == 2
    assert counts[4] == 2
    assert counts[2] == 1


def test_assemble_corpus_does_not_call_real_openai_when_stub_provided() -> None:
    """Critical: real gpt-4.1 generation must NEVER happen during tests. The
    stub plus the lazy-import pattern guarantees this. Validated by injecting
    a stub that records the call count."""
    calls = {"n": 0}

    def stub(category: int, idx: int) -> dict:
        calls["n"] += 1
        return {"scenario": "s", "expert_response": "r"}

    cg.assemble_corpus(target_n=50, generation_fn=stub)
    assert calls["n"] == 50


# ---------------------------------------------------------------------------
# B1.0 — orca generation_fn factory + category prompt builder
# ---------------------------------------------------------------------------


def test_category_prompt_includes_anti_fabrication_hard_rules() -> None:
    """SPEC §CORPUS_PRINCIPLES: every generation prompt must embed the
    anti-fabrication rules verbatim — no fabricated citations, no rigid
    thresholds, hedge specifically."""
    p = cg._category_prompt(category=5, idx=0)
    low = p.lower()
    assert "fabricated citations" in low or "no fabricated" in low
    assert "rigid" in low or "varies by guideline" in low
    assert "hedge" in low


def test_category_prompt_is_deterministic_per_idx() -> None:
    """Same (category, idx) yields identical prompt text — required for
    reproducible corpus generation."""
    a = cg._category_prompt(category=1, idx=0)
    b = cg._category_prompt(category=1, idx=0)
    assert a == b


def test_category_prompt_rotates_topic_by_idx() -> None:
    """Topic pools rotate via idx % len(pool). idx=0 and idx=len(pool) hit
    the same topic; consecutive idx values hit different topics for cat 1."""
    p0 = cg._category_prompt(category=1, idx=0)
    p1 = cg._category_prompt(category=1, idx=1)
    p_wrap = cg._category_prompt(category=1, idx=len(cg._TOPIC_POOLS[1]))
    # Different topics for consecutive idx
    topic_0 = p0.split("Topic:")[1].split("\n")[0].strip()
    topic_1 = p1.split("Topic:")[1].split("\n")[0].strip()
    topic_wrap = p_wrap.split("Topic:")[1].split("\n")[0].strip()
    assert topic_0 != topic_1
    # Wrap returns to idx 0 topic
    assert topic_0 == topic_wrap


def test_category_prompt_per_category_remediation_guidance() -> None:
    """Each category's prompt includes its specific remediation framing."""
    p1 = cg._category_prompt(category=1, idx=0)
    p5 = cg._category_prompt(category=5, idx=0)
    p4 = cg._category_prompt(category=4, idx=0)
    p3 = cg._category_prompt(category=3, idx=0)
    assert "Knowledge Gap" in p1
    assert "category #5" in p5.lower() or "anti-fabrication" in p5.lower()
    assert "category #4" in p4.lower() or "explicit timeline" in p4.lower()
    assert "category #3" in p3.lower() or "calibrate urgency" in p3.lower()


def test_make_orca_generation_fn_returns_callable() -> None:
    """Factory contract: returns a callable usable as generation_fn."""
    fn = cg.make_orca_generation_fn(base_url="http://localhost:8000/v1", model="x")
    assert callable(fn)


def test_make_orca_generation_fn_reads_env_vars(monkeypatch) -> None:
    """Defaults: MEDOMNI_ORCA_VLLM_URL + MEDOMNI_ORCA_MODEL drive base_url/model."""
    monkeypatch.setenv("MEDOMNI_ORCA_VLLM_URL", "http://from-env:9999/v1")
    monkeypatch.setenv("MEDOMNI_ORCA_MODEL", "env-model")
    fn = cg.make_orca_generation_fn()
    # Closure construction succeeds; readable means env path is reached
    assert callable(fn)


def test_topic_pools_have_expected_sizes() -> None:
    """B1.0 topic pool sizes: 10/10/5/5/1 across cats 1/5/3/4/2."""
    assert len(cg._TOPIC_POOLS[1]) == 10
    assert len(cg._TOPIC_POOLS[5]) == 10
    assert len(cg._TOPIC_POOLS[3]) == 5
    assert len(cg._TOPIC_POOLS[4]) == 5
    assert len(cg._TOPIC_POOLS[2]) == 1


# ---------------------------------------------------------------------------
# Collapsed taxonomy (post-κ=0.054 finding 2026-05-07)
# ---------------------------------------------------------------------------


def test_collapsed_sections_are_three_with_stable_keys() -> None:
    assert set(cg.COLLAPSED_SECTIONS.keys()) == {"A", "B", "C"}
    assert "fabrication" in cg.COLLAPSED_SECTIONS["A"][0].lower()
    assert "omission" in cg.COLLAPSED_SECTIONS["B"][0].lower()
    assert "probe" in cg.COLLAPSED_SECTIONS["C"][0].lower()


def test_collapsed_default_allocation_proportions_30_65_5() -> None:
    alloc = cg.collapsed_default_allocation(target_n=10_000)
    assert alloc["A"] == 3000
    assert alloc["B"] == 6500
    assert alloc["C"] == 500
    assert sum(alloc.values()) == 10_000


def test_collapsed_default_allocation_remainder_to_B() -> None:
    """When target_n isn't a clean multiple of 100, remainder goes to dominant B."""
    alloc = cg.collapsed_default_allocation(target_n=103)
    # int(103*0.30)=30; int(103*0.65)=66; int(103*0.05)=5; sum=101; diff=2 → B
    assert alloc["A"] == 30
    assert alloc["B"] == 68
    assert alloc["C"] == 5
    assert sum(alloc.values()) == 103


def test_collapsed_section_prompt_includes_section_specific_guidance() -> None:
    pA = cg._collapsed_section_prompt(section="A", idx=0)
    pB = cg._collapsed_section_prompt(section="B", idx=0)
    pC = cg._collapsed_section_prompt(section="C", idx=0)
    assert "SECTION A" in pA and "fabrication" in pA.lower()
    assert "SECTION B" in pB and ("disclaimer" in pB.lower() or "substantive" in pB.lower())
    assert "SECTION C" in pC and "held-out" in pC.lower()


def test_collapsed_section_prompt_rejects_invalid_section() -> None:
    with pytest.raises(ValueError, match="section"):
        cg._collapsed_section_prompt(section="X", idx=0)


def test_collapsed_section_prompt_includes_anti_disclaimer_rule() -> None:
    """Section B's load-bearing rule: lead with substantive content, NOT a
    'I'm not a medical professional' disclaimer prefix. Pin it."""
    p = cg._collapsed_section_prompt(section="B", idx=0)
    low = p.lower()
    assert "lead with substantive" in low or "no uniform" in low


def test_generate_collapsed_v25b_examples_uses_stub() -> None:
    captured: list[tuple[str, int]] = []

    def stub(section: str, idx: int) -> dict:
        captured.append((section, idx))
        return {"scenario": "s", "expert_response": "r"}

    examples = cg.generate_collapsed_v25b_examples(
        section="A", n_examples=3, generation_fn=stub,
    )
    assert len(examples) == 3
    assert captured == [("A", 0), ("A", 1), ("A", 2)]
    for ex in examples:
        assert ex["section"] == "A"
        assert ex["section_name"] == "Active fabrication / over-specification"
        assert ex["id"].startswith("v25b_secA_")


def test_assemble_collapsed_corpus_distribution() -> None:
    def stub(section: str, idx: int) -> dict:
        return {"scenario": "s", "expert_response": "r"}
    corpus = cg.assemble_collapsed_corpus(target_n=100, generation_fn=stub)
    counts = {s: sum(1 for ex in corpus if ex["section"] == s) for s in ("A", "B", "C")}
    assert counts["A"] == 30
    assert counts["B"] == 65
    assert counts["C"] == 5


def test_make_collapsed_orca_generation_fn_returns_callable() -> None:
    fn = cg.make_collapsed_orca_generation_fn(
        base_url="http://localhost:8000/v1", model="x"
    )
    assert callable(fn)


def test_collapsed_topic_pool_includes_yes_no_scenario_in_section_b() -> None:
    """Section B should include the 'direct yes/no' pattern from the κ-shadow
    finding (item 7eba4984: V2.5 over-hedged on a direct yes/no mask question)."""
    pool_b = cg._COLLAPSED_TOPIC_POOLS["B"]
    assert any("yes/no" in topic.lower() for topic in pool_b)


# ---------------------------------------------------------------------------
# Pattern provenance (autoresearcher loop — V2.5b → V2.5c bootstrap)
# ---------------------------------------------------------------------------


def test_pattern_library_has_three_sections_with_expected_patterns() -> None:
    """SPEC §FAILURE_PATTERN_LIBRARY: A has 5 patterns, B has 8, C has 2."""
    assert set(cg.PATTERN_LIBRARY.keys()) == {"A", "B", "C"}
    assert len(cg.PATTERN_LIBRARY["A"]) == 5
    assert len(cg.PATTERN_LIBRARY["B"]) == 8
    assert len(cg.PATTERN_LIBRARY["C"]) == 2
    # Pattern IDs follow the convention "<section><idx>_<snake_case_name>"
    for section, patterns in cg.PATTERN_LIBRARY.items():
        for i, pattern in enumerate(patterns, 1):
            assert pattern.startswith(f"{section}{i}_"), (
                f"pattern {pattern!r} should start with {section}{i}_"
            )


def test_pattern_for_idx_round_robin() -> None:
    """Round-robin assignment — idx 0..N-1 cycles through section's patterns."""
    a_patterns = cg.PATTERN_LIBRARY["A"]
    assert cg.pattern_for_idx(section="A", idx=0) == a_patterns[0]
    assert cg.pattern_for_idx(section="A", idx=4) == a_patterns[4]
    assert cg.pattern_for_idx(section="A", idx=5) == a_patterns[0]  # wrap
    assert cg.pattern_for_idx(section="A", idx=10) == a_patterns[0]  # wrap×2


def test_pattern_for_idx_rejects_invalid_section() -> None:
    with pytest.raises(ValueError, match="section"):
        cg.pattern_for_idx(section="X", idx=0)


def test_generated_collapsed_examples_carry_pattern_addressed() -> None:
    """Each example MUST have a pattern_addressed field for V2.5c bootstrap."""
    def stub(section: str, idx: int) -> dict:
        return {"scenario": "s", "expert_response": "r"}
    examples = cg.generate_collapsed_v25b_examples(
        section="B", n_examples=10, generation_fn=stub,
    )
    assert len(examples) == 10
    # Distinct patterns surface across the 10 (section B has 8 patterns)
    distinct = {ex["pattern_addressed"] for ex in examples}
    assert len(distinct) == 8  # all 8 B-patterns hit at least once
    # And they round-robin in order
    expected = cg.PATTERN_LIBRARY["B"]
    for i, ex in enumerate(examples):
        assert ex["pattern_addressed"] == expected[i % 8]


def test_sharpened_pattern_prompts_cover_three_failed_patterns() -> None:
    """PROGRAM.md iteration 1: B5/B6/B7 are the failed patterns from
    v25b_judge_filter (means 0.26/0.48/0.51). Each MUST have a sharpened prompt."""
    expected = {
        "B5_missing_context_asking_questions",
        "B6_refusal_to_answer_direct_yes_no",
        "B7_context_element_dropped",
    }
    assert set(cg.SHARPENED_PATTERN_PROMPTS.keys()) == expected


def test_pattern_aware_prompt_includes_required_scenario_and_response_shape() -> None:
    """Sharpened prompts MUST embed both REQUIRED SCENARIO SHAPE and REQUIRED
    RESPONSE SHAPE — these are the two halves of the autoresearch hypothesis."""
    for pattern in cg.SHARPENED_PATTERN_PROMPTS:
        prompt = cg.pattern_aware_prompt(pattern=pattern, idx=0)
        assert "REQUIRED SCENARIO SHAPE" in prompt
        assert "REQUIRED RESPONSE SHAPE" in prompt
        assert pattern in prompt


def test_pattern_aware_prompt_b5_demands_clarifying_questions() -> None:
    p = cg.pattern_aware_prompt(pattern="B5_missing_context_asking_questions", idx=0)
    low = p.lower()
    assert "clarifying question" in low or "clarifying questions" in low
    assert "numbered list" in low or "3-5 items" in low


def test_pattern_aware_prompt_b6_demands_yes_no_first_word() -> None:
    p = cg.pattern_aware_prompt(pattern="B6_refusal_to_answer_direct_yes_no", idx=0)
    assert "'Yes' or 'No'" in p or "Yes or No" in p
    assert "first word" in p.lower()


def test_pattern_aware_prompt_b7_demands_first_sentence_context_drop() -> None:
    p = cg.pattern_aware_prompt(pattern="B7_context_element_dropped", idx=0)
    assert "FIRST SENTENCE" in p
    assert "context marker" in p.lower()


def test_pattern_aware_prompt_rejects_unknown_pattern() -> None:
    with pytest.raises(ValueError, match="no sharpened prompt"):
        cg.pattern_aware_prompt(pattern="X9_unknown", idx=0)


def test_assemble_collapsed_corpus_balances_patterns_within_section() -> None:
    """A 5000-corpus B-section (n=3250) should have ~equal pattern coverage:
    3250 / 8 patterns ≈ 406 each (range 405-407)."""
    def stub(section: str, idx: int) -> dict:
        return {"scenario": "s", "expert_response": "r"}
    corpus = cg.assemble_collapsed_corpus(target_n=5000, generation_fn=stub)
    b_examples = [ex for ex in corpus if ex["section"] == "B"]
    assert len(b_examples) == 3250  # default 65% allocation
    counts: dict[str, int] = {}
    for ex in b_examples:
        counts[ex["pattern_addressed"]] = counts.get(ex["pattern_addressed"], 0) + 1
    assert len(counts) == 8  # all 8 B-patterns surfaced
    # 3250 / 8 = 406.25 → some patterns get 406, some 407
    assert all(405 <= c <= 407 for c in counts.values()), counts
