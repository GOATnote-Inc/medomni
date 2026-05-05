# Clinical Prompt Engineering for V_final Inference (medomni)

**Date:** 2026-05-05
**Author:** Research Agent A (4-agent improvement-dimensions synthesis)
**Target:** Inference-time wrappers around `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning` post-V_final, evaluated on Stanford **MedAgentBench** (300 FHIR-environment EHR tasks; SOTA = GPT-4.1 + memory + planning prompt = **98.0%** per MedAgentBench v2; Claude 3.5 Sonnet baseline = 69.67%).

## TL;DR

Three techniques dominate the 2025-Q4 / 2026-Q1 SOTA on clinical reasoning + agentic EHR work and are stackable on top of V_final's trained skills:

1. **Plan-then-act system prompt + few-shot good/bad tool-call exemplars** (MedAgentBench v2 recipe). +~28 pts on MedAgentBench (Claude 69.67% → GPT-4.1 91-98%). Cheap, this is the lowest-hanging fruit and aligns directly with V2.7 tool-call SFT.
2. **Self-consistency with verifier-weighted voting (Med-PRM-style PRM as judge)**. +13.5% over base on five medical QA benchmarks (Med-PRM, EMNLP 2025). Plug-and-play; verifier need not be the same model.
3. **Medprompt-lite: k-NN dynamic exemplar retrieval + choice-shuffle ensembling**. +11.3 pts on MedQA over zero-shot GPT-4. Note 2025 update: o1-style reasoners *lose* points from few-shot — but ensembling still gains. Since Nemotron-3-Nano-Omni-30B is a reasoner, **drop few-shot, keep ensembling**.

## Techniques (ranked by expected lift on MedAgentBench)

### 1. Plan-then-act + few-shot tool-call exemplars (MedAgentBench v2 recipe)
Force the model to emit a numbered plan before any tool call; include 2-4 paired good/bad tool-call examples in the system prompt. **Expected lift on MedAgentBench: +20-28 pts** (this is the v2 paper's headline). **Cost:** ~600 system-prompt tokens (cached). **Latency:** zero — first-call only.

### 2. Self-consistency + verifier voting (Med-PRM, arxiv 2506.11474)
Sample K=8 reasoning paths at T=0.7; have a small medical PRM (or a second pass of V_final in "verifier mode") score each step against retrieved clinical guideline snippets; pick the highest-scored trajectory rather than majority vote. **Expected lift: +13.5% on medical QA, +6-10 pts on agentic tasks.** **Cost:** 8× generation + 8× verifier passes. **Latency:** parallelizable; ~3-4× wall-clock. Can be gated to high-stakes turns only (mutating tool calls, final differential).

### 3. Medprompt-style ensembling **without** few-shot (the 2025 reasoner update)
Choice-shuffle / phrasing-shuffle ensembling across K=5 samples; pick the answer least sensitive to the shuffle. Skip k-NN exemplars — they hurt reasoners (Microsoft, arxiv 2411.03590). **Expected lift: +5-8 pts on MedQA-style, +3-6 on MedAgentBench.** **Cost:** 5× generation. Most useful for the multiple-choice / classification fragments of MedAgentBench.

### 4. ToT/MedKGI for differential diagnosis turns
Tree-of-Thoughts branching anchored to a knowledge-graph or guideline-retrieved hypothesis frontier (MedKGI, arxiv 2512.24181; AMIE-DDx top-10 = 59.1% vs 33.6% unassisted clinicians, Nature 2025). **Expected lift on DDx subtasks: +15-25 pts top-10 accuracy.** **Cost:** 10-30× tokens. Gate to DDx-shaped turns only.

### 5. Tool-call vs CoT: lean tool-call by default
FHIR-AgentEval (PMC12919212, 2026) shows ReAct-style tool-call loops + memory consistently beat plain CoT on EHR tasks; "memory" specifically reduces incorrect-tool and resource-type-confusion failures. Since V2.7 already trains tool-calling, **the prompt's job is to *not* suppress it** — explicitly forbid "answer from memory" on data-bound questions. **Expected lift: prevents -10 to -20 pt regression** more than it adds.

### 6. FactScore-style claim verification on outputs
Decompose the final answer into atomic claims; verify each against a FHIR-fetched evidence set or guideline RAG (arxiv 2502.14765, 2511.12817). **Expected lift on hallucination metrics: -30-50% unsupported-claim rate.** **Cost:** 1× extra verifier pass per response.

### 7. Skills as structured prompting layer (Anthropic merged slash-commands → Skills, 2026-01-24)
Ship `/differential`, `/calc`, `/citation`, `/handoff` as Skills (Skill = markdown frontmatter + body, loaded only on intent match). Not a benchmark lift per se — it's a context-budget and consistency win that makes (1)-(6) easier to A/B and version. **Expected lift: indirect; reduces context bloat ~40%, enables per-task prompt versioning.** **Cost:** zero at inference.

## Integration into V_final inference path

Default request flow (vLLM serving Nemotron-3-Nano-Omni-30B):

```
user_msg
  → router (cheap classifier: ddx? calc? data-fetch? open-ended?)
    → /differential skill   → ToT-lite (K=5 branches, 1 verifier pass)
    → /calc skill           → tool-call-only, no CoT ensemble
    → /handoff (EHR write)  → MedAgentBench-v2 system prompt + plan-then-act + verifier vote (K=4)
    → /open                 → single-shot V_final
  → FactScore claim-verifier pass (only if output cites EHR data)
  → response
```

System-prompt template: MedAgentBench-v2-style header (plan, 2 good/2 bad tool-call exemplars, "no answers from memory on data questions") + Skill-specific body. Cache the header (Anthropic prompt-caching pattern; cache-warm < 5min).

## Concrete top-3 to ship as inference wrappers

1. **`mvp/medomni-inference/system_prompt_v1.md`** — MedAgentBench-v2 plan-then-act header + 4 paired tool-call exemplars. Ship first; this alone is the bulk of the lift.
2. **`mvp/medomni-inference/verifier_vote.py`** — K=8 sample, Med-PRM-style verifier scoring (use a 7B medical PRM or V_final-as-judge prompt), gated to mutating tool calls + final answers.
3. **`mvp/medomni-inference/skills/`** — `/differential` (ToT-lite for DDx), `/calc` (tool-only), `/handoff` (FHIR-write with verifier gate). Skills format = Anthropic 2026-01-24 spec (frontmatter + body, intent-loaded).

Defer: full Medprompt k-NN exemplars (hurts reasoners), full FactScore pipeline (build after baseline lands).

## References

- Microsoft Medprompt: arxiv 2303.13375; 2411.03590 (Medprompt → o1 runtime study, Nov 2025).
- Med-PaLM 2 ensemble refinement: arxiv 2305.09617.
- MedAgentBench: arxiv 2501.14654; v2 PSB 2026 (Eric Chen, MIT) — GPT-4.1 91.0% no-mem / 98.0% with memory.
- Med-PRM: arxiv 2506.11474 (EMNLP 2025); github.com/eth-medical-ai-lab/Med-PRM.
- AMIE differential diagnosis: Nature 2025 (s41586-025-08869-4); top-10 59.1% vs clinician 33.6%.
- MedKGI iterative DDx: arxiv 2512.24181.
- Tree of Thoughts: arxiv 2305.10601.
- FHIR-AgentEval: PMC12919212 (2026) — ReAct + memory beats CoT on EHR.
- Medical fact-check: arxiv 2502.14765, 2511.12817.
- Anthropic Skills: 2026-01-24 slash-commands→Skills merge.
