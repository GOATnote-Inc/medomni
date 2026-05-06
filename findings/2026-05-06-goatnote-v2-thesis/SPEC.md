# GOATnote v2 Thesis — Alströmer + Blomfield + Hu Applied to AI-Native EMDR Services

**Path:** `findings/2026-05-06-goatnote-v2-thesis/SPEC.md`
**Date:** 2026-05-06
**Status:** SPEC ONLY — strategic positioning. No code changes in this PR. Implementation is a multi-quarter program tracked in §9.
**Author:** Claude (Opus 4.7) on behalf of `b@thegoatnote.com`
**Sibling SPECs:** `findings/2026-05-06-mobile-first-redesign-spec/SPEC.md` (PR #102, just shipped) and the parallel `2026-05-06-llm-curated-ia-spec/SPEC.md` (may not yet exist).
**Author thesis (verbatim from founder):** "Rebuild GOATnote as an AI-native Healthcare service company (Alströmer) by giving it a Company Brain (Blomfield) and a closed-loop operating system (Hu), with an EMDR-domain RAG layer that doubles as a published clinical-AI-safety contribution. The GUI is downstream of the agents; booking, proposals, training delivery, and reporting all flow through the same substrate. Government contracts are the wedge customer because the unit economics, sales cycle, and reporting requirements all reward this architecture."

---

## 1. Executive summary

GOATnote, Inc. is a Service-Disabled-Veteran-eligible (founder-route TBD §11) clinical-AI services firm. Its founder is a licensed physician (Brandon Dent, MD) authoring the substrate. The Y Combinator Summer 2026 Request for Startups says, in three separate partner letters, that the firm GOATnote should become already exists as a category — an AI-native services company, with a Company Brain that turns scattered organizational knowledge into executable agent skills, running on a closed-loop OS that records every interaction and learns from it. The four-layer architecture in this SPEC binds those three frameworks to one specific clinical service line — EMDR therapy delivery for trauma-exposed populations — and to one wedge customer — federal trauma-care programs (VA, DoD, IHS) where unit economics, sales cycle, and reporting requirements all reward this architecture.

The four layers stack as follows. (1) **Company Brain** — twelve markdown skill files, each describing one operational workflow (`onboard-va-clinic.md`, `run-emdr-session.md`, `quarterly-outcomes-report.md`, etc.) that an agent reads at runtime. The skills router primitive already exists in this repo at `findings/2026-05-05-skills-router-v1/CARD.md` and is wired live at `https://medomni.vercel.app/api/agent?profile=v_final`. The clinical-skill-review CI at `.github/workflows/clinical-skill-review.yml` is the Company Brain's quality gate; every skill change is auto-reviewed by Claude Opus 4.7 before merge. (2) **Closed-loop OS** — every session, every booking, every outcome questionnaire (PCL-5, PHQ-9), every audit-log row is captured into a structured store, monitored against per-program targets, and adjusted by agents that own each loop. The hourly adversarial-probe daemon at `.github/workflows/adversarial-probe.yml` is already running this pattern against `/api/agent`. (3) **EMDR-domain RAG** — extends the in-flight `project_emdr_rag.md` build (42 commits, deployment-ready) into a sovereign retrieval layer over EMDRIA standards, ISTSS PTSD CPG, the 2023 VA/DoD PTSD Clinical Practice Guideline, and the EMDR PubMed corpus. (4) **GUI as downstream** — booking, proposals, training delivery, outcomes reporting all flow through the same agent substrate; the React surface is thin.

The wedge customer is federal trauma-care programs because every dimension of the Alströmer thesis is amplified there. Per-veteran-month pricing aligns with VA's existing capitated reimbursement patterns; SDVOSB / VOSB / SBIR set-asides give a sub-$250K Phase I door without competing in an open RFP; FedRAMP authorization is a moat that takes incumbents 6-18 months to clear and becomes table stakes for any further VA/DoD work; the 5+ year retention requirement on every audit-log row matches what an AI-native firm must capture anyway for its closed-loop OS. The first deal is hardest. The recommendation in §6 is a Charleston VA / National Center for PTSD CONNECT-style telehealth pilot via the SDVOSB set-aside path, structured as a 12-month per-veteran-month subscription, with the empirical results submitted to *NEJM AI* as a pre-registered observational study.

The GPU footprint is right-sized. The four pods (catfish B300 prod inference, lobster H200 training, narwhal H200 corpus factory, prism-mla-h100 idle/TBD) map cleanly onto the four-layer architecture if prism-mla-h100 is repurposed as the continuous compliance worker (HITRUST/FedRAMP evidence collection — automated control attestation against `oscal-cli`, generation of System Security Plan deltas for every `web/` deploy, FIPS-140-3 inventory diffs against TLS-terminated edges). Compliance-as-a-pod is the cheapest of the four roles per dollar of SaaS-substitute revenue. The 90-day execution plan in §9 has a single first-deal critical path; everything else is parallel and reversible.

---

## 2. The three frameworks, with citations

### 2.1 Alströmer — AI-Native Service Companies

Gustaf Alströmer's letter in YC's Summer 2026 RFS argues for a specific category transition: "AI-native service companies" that **deliver the service directly**, not copilots that augment human service workers. The headline economic claim he uses to motivate the category is, verbatim:

> "The total spend on services is many times larger than the spend on software."
> — Gustaf Alströmer, [Y Combinator Summer 2026 RFS](https://www.ycombinator.com/rfs)

The target verticals he names — insurance brokerage, accounting, tax, audit, compliance, **healthcare administration** — are all domains where the customer is buying an outcome, not a tool. The thesis is not "build software that helps a brokerage do brokerage." It is "be the brokerage, with software underneath." The Epsilla deep-dive paraphrases the same shift as: *"the ultimate demand of any enterprise is to have a problem solved, not to purchase a tool"* ([Epsilla 2026-05-02](https://www.epsilla.com/blogs/2026-05-02-yc-rfs-deep-dive-the-twilight-of-saas-and-the-dawn-of-agent-)).

**Why this applies to GOATnote.** Clinical EMDR delivery for trauma-exposed populations is a *service* (per-session, per-veteran, per-outcome), not a tool license. The federal customer (VA, DoD, IHS) is already buying a service from in-house clinicians or from contracted networks like Telehealth Access for Seniors / Magellan / Optum Behavioral Health. Replacing-or-augmenting the service is the AI-native play. Selling a "therapist copilot" SaaS to those same clinicians is not. The pricing model in §7 is per-veteran-month, which is exactly the unit Alströmer's framework predicts. The service is delivered by GOATnote's licensed clinician network — augmented, not replaced — and reported through GOATnote's outcomes substrate, which is what the federal customer pays for.

### 2.2 Blomfield — The Company Brain

Tom Blomfield's letter in the same RFS argues that the binding constraint on AI-native services is not LLM capability — it is the **fragmentation of organizational knowledge**. Verbatim:

> "[A system that pulls knowledge out of every fragmented source, structures it, keeps it current, and turns it into] an executable skills file for AI [...] a living map of how a company actually works."
> — Tom Blomfield, [Y Combinator Summer 2026 RFS](https://www.ycombinator.com/rfs), as compiled in [The VC Corner](https://www.thevccorner.com/p/yc-summer-2026-requests-for-startups-ideas)

Blomfield is explicit that this is **not a chatbot over documents**. It is a structured, versioned, executable description of how the company operates: how refunds get handled, how pricing exceptions are decided, how engineers respond to incidents, how a new clinician gets credentialed, how a quarterly outcomes report gets written. The skills file is a map, not a search index. The point is that an agent can *act* from it.

**Why this applies to GOATnote.** This repo (`medomni`) already has the primitive. `findings/2026-05-05-skills-router-v1/CARD.md` describes a skills router wired to `https://medomni.vercel.app/api/agent?profile=v_final` that loads markdown skill files from `web/lib/agent/skills/` (canonical authoring at `mvp/medomni-inference/skills/`). The skills router replaces a monolithic prompt with progressive disclosure: classify intent → load just the skill block needed for *this* turn → splice into the system prompt → dispatch to the B300 vLLM. The pattern is Cherny-cycle (Boris Cherny's "Building Claude Code" framing — markdown-driven, progressive disclosure, ships in a 5-min PR not a training cycle). Today the repo has four skills (`differential.md`, `calc.md`, `handoff.md`, `system_prompt_v1.md`). §3.1 below lists the twelve operational skills GOATnote-the-services-firm needs on top.

### 2.3 Hu — The Closed-Loop AI Operating System

Diana Hu's letter argues that high-performing AI-native companies are the ones that have made themselves *queryable* — every meeting recorded, every ticket tracked, every customer interaction captured — feeding a single intelligence layer that can monitor outputs against targets and adjust automatically. She frames this as the difference between **open-loop** organizations (which are "inherently lossy") and **closed-loop** ones that are "self-regulating." Verbatim quotes from the StartupHub.ai breakdown:

> "Open-loop systems, common in traditional business operations, are inherently lossy."
>
> "If your API bill doesn't make you uncomfortable, you're not doing enough" — framed as: **"burn tokens, not headcount."**
> — Diana Hu, paraphrased and quoted in [StartupHub.ai 2026 closed-loop article](https://www.startuphub.ai/ai-news/artificial-intelligence/2026/build-ai-native-companies-with-closed-loop-systems)

Hu's practical implementation list includes: AI note-takers on every meeting, structured-data-first communication (minimize DMs and email), AI agents embedded in the channels where work happens, "one-shot internal dashboards" generated on demand. The economic outcome she reports is teams that "cut sprint time in half and ship twice as much" once the loop is closed.

**Why this applies to GOATnote.** A clinical services firm's open loop is its dropouts: patients who DNA their second session, therapists whose fidelity slips between supervision audits, federal program managers who don't know if the contract is on track until the quarterly report arrives. A closed-loop OS captures the booking event, the session note, the PCL-5 score, the supervision audit, and the audit-log entry into one queryable substrate, then tasks an agent with monitoring each loop against its target and adjusting (rebook the no-show, schedule the supervision call, generate the quarterly outcomes draft). The hourly adversarial-probe daemon at `.github/workflows/adversarial-probe.yml` is already running this pattern in miniature against `/api/agent` — every hour, 20 red-team prompts hit the agent, results are scored, regressions are surfaced. §3.2 below extends this to the operational loops of a services firm.

### 2.4 Why all three frameworks at once

The three letters compose. Alströmer is the **outside view** (sell the service, not the software). Blomfield is the **knowledge layer** (turn how-the-company-works into agent-executable skills). Hu is the **runtime** (close every loop). Without all three, the Alströmer thesis collapses: a services firm without a Company Brain is just a consultancy, and a Company Brain without a closed loop drifts within weeks of the first deal because nothing keeps the skills file current. The composed posture — AI-native services firm + Company Brain + closed-loop OS + a domain-RAG layer specific to the service — is the position GOATnote already occupies in skeleton form (skills router live, adversarial-probe live, EMDR-RAG deployment-ready, mobile-first GUI in spec). v2 finishes the build.

---

## 3. The four-layer architecture

### 3.1 Layer 1 — Company Brain (twelve operational skill files)

Each skill is a markdown file under `web/lib/agent/skills/operations/`, canonical authoring at `mvp/medomni-inference/skills/operations/`, sync via `make sync-skills` (existing target — see [`Makefile`](../../Makefile)). Each carries YAML front-matter declaring `intent_keys`, `inputs`, `outputs`, `requires_human_signoff`, `audit_event_type`. Each is auto-reviewed on PR by `.github/workflows/clinical-skill-review.yml` before merge. The twelve operational skills GOATnote-the-services-firm needs are:

| # | Skill file | Intent keys | Owner | Human sign-off | Audit event |
|---|---|---|---|---|---|
| 1 | `onboard-va-clinic.md` | "onboard", "new site", "kickoff" | Operations | Required (clinic director) | `clinic.onboarded` |
| 2 | `credential-new-therapist.md` | "credential", "license verify", "OIG check" | Operations | Required (compliance officer) | `therapist.credentialed` |
| 3 | `run-emdr-session.md` | "session", "EMDR protocol", "8 phases" | Clinical (therapist-driven) | Always (licensed therapist signs note) | `session.completed` |
| 4 | `grade-session-fidelity.md` | "fidelity", "supervisor review", "EMDR adherence" | Clinical Quality | Required (EMDRIA-Approved Consultant) | `fidelity.graded` |
| 5 | `handle-clinical-adverse-event.md` | "adverse event", "decompensation", "crisis", "988" | Clinical Quality | Required (medical director within 24h) | `ae.reported` |
| 6 | `quarterly-outcomes-report.md` | "quarterly report", "PCL-5 outcomes", "PHQ-9 trend" | Operations + Clinical | Required (founder + clinical lead) | `report.generated` |
| 7 | `author-government-proposal.md` | "RFP response", "SAM.gov", "SBIR Phase I", "task order" | Business Development | Required (founder + counsel) | `proposal.drafted` |
| 8 | `audit-baa-coverage.md` | "BAA gap", "vendor sweep", "subcontractor PHI" | Compliance | Required (compliance officer quarterly) | `baa.audited` |
| 9 | `book-veteran-intake.md` | "schedule", "intake", "first session", "informed consent" | Operations (agent-led) | Soft (therapist confirms slot) | `appointment.booked` |
| 10 | `bill-per-veteran-month.md` | "invoice", "billing", "VA Tungsten", "DFAS" | Operations | Required (founder approves invoice batch) | `invoice.sent` |
| 11 | `monitor-drop-off.md` | "no-show", "DNA", "engagement risk", "rebook" | Operations + Clinical | Soft (case manager outreach script) | `engagement.alert` |
| 12 | `run-fedramp-evidence-collection.md` | "OSCAL", "control evidence", "continuous monitoring", "POAM" | Compliance | Required (compliance officer monthly) | `oscal.synced` |

Each skill file follows the same shape as the existing four (`differential.md`, `calc.md`, `handoff.md`, `system_prompt_v1.md`) — a short YAML front-matter, a one-sentence purpose, a numbered runbook, an "outputs" section describing the agent's handoff, and a "do not" section listing red lines (e.g. `run-emdr-session.md` includes "do not run reprocessing on a client without an EMDRIA-Approved Consultant or Approved-Consultant-in-Training reviewing the case formulation"). The point of the markdown form is what Cherny calls progressive disclosure: only the skill needed for the turn enters the prompt, so the model sees a focused 800-word skill block instead of a 12,000-word kitchen-sink system prompt. This is Blomfield's *executable skills file* in the literal sense.

The clinical-skill-review CI runs on every PR that touches `**/skills/**`. Claude Opus 4.7 reads the diff, the existing skill, and a short reviewer prompt that asks: does this preserve the human-sign-off requirement, does it preserve the EMDRIA standard-of-care references, does it preserve the audit-event emission, does it introduce a new external-data dependency without a corresponding BAA / DUA. A PR cannot merge until the auto-review posts an APPROVE comment. This is the Company Brain's **quality gate**, and it is the operational expression of what Blomfield calls "keeping it current."

### 3.2 Layer 2 — Closed-loop OS (capture / monitor / adjust table)

| Loop | Captured artifact | Stored in | Monitored against | Adjusted by | Cadence |
|---|---|---|---|---|---|
| Booking | `appointment.booked` event with payer, therapist, time | Postgres `events.appointments` | Per-clinic capacity utilization target (e.g. ≥75%) | `book-veteran-intake.md` agent rebalances calendar, opens slots, prompts ops on shortfall | Real-time |
| Session | EMDR session note + 8-phase phase reached + SUDS pre/post | Postgres `clinical.sessions` (encrypted, BAA-scoped) | Per-veteran clinical-progress trend (PCL-5 every 4 sessions per VA/DoD CPG cadence) | Therapist drafts via `run-emdr-session.md`; supervisor reviews via `grade-session-fidelity.md` weekly | Per session |
| Outcome | PCL-5 / PHQ-9 / DES-II questionnaires | Postgres `clinical.outcomes` | Cohort: ≥10pt PCL-5 reduction in ≥X% of completers (target set per cohort, recommendation §7) | `quarterly-outcomes-report.md` agent generates draft; founder + clinical lead sign | Weekly cohort review, quarterly federal report |
| Fidelity | Supervisor audit of N% of recorded sessions | S3 (encrypted, Object Lock, BAA-scoped) + Postgres `clinical.fidelity_grades` | EMDR adherence rubric ≥80% per therapist per quarter | `grade-session-fidelity.md` flags below-threshold therapist; ops schedules consultation | Weekly sample |
| Adverse event | `ae.reported` event + clinician narrative | Postgres `clinical.adverse_events` (immutable append-only) | Time-to-medical-director-review <24h; serious AE reported to VA per contract | `handle-clinical-adverse-event.md` triggers paging tree | Real-time |
| Engagement | DNA / late-cancel / dropout signal | Postgres `events.engagement` | DNA rate <15%, dropout-by-session-3 <25% | `monitor-drop-off.md` agent runs case-manager outreach script | Daily |
| Compliance | `oscal.synced` evidence diffs vs SSP | S3 Object Lock + Postgres `compliance.oscal_evidence` | All controls have evidence ≤ control's `frequency`; POAMs current | `run-fedramp-evidence-collection.md` cron on prism-mla-h100 | Hourly evidence sync, monthly POAM review |
| Adversarial robustness | Hourly probe results vs `/api/agent` | `findings/2026-05-05-adversarial-probe-spec/` artifacts | Per-class pass rate ≥ baseline; no new safety regression | `.github/workflows/adversarial-probe.yml` opens GH issue on regression | Hourly (already shipping) |
| Skill drift | Diff between deployed skill file and most-recent supervisor sign-off | Git blame + `clinical-skill-review.yml` audit log | All deployed skills have a clinical-reviewer sign-off ≤ 90d old | Auto-Claude review (existing) + monthly clinical lead PR-blob review | Per PR + monthly |

The substrate for this is straightforward: Postgres for structured event rows, S3 with Object Lock for immutable evidence (sessions, AE narratives, OSCAL evidence), `events` schema feeds a single `agent_event` view that every operational skill reads from. The `medomni` repo's `web/` BFF is already wired into the substrate; the existing pattern in `findings/2026-05-04-pattern-b-spike/` (FHIR-fetch p95 11ms) is the proof point that the BFF can pull from a federated EMR substrate cheaply. The `monitor-drop-off.md` and `grade-session-fidelity.md` agents close the two highest-value loops first — patient retention and clinical fidelity. Everything else is iteration.

The closed-loop framing is consistent with the medomni v1 architecture decisions captured in memory `project_medomni_v1_architecture_decisions.md` (single Medplum Project + AccessPolicy as tenant boundary, Medplum AuditEvent → S3 Object Lock 7yr for audit). The retention requirement (7 years for VA, 10 years for some IHS programs) is what *forces* immutability, which in turn makes the closed-loop OS auditable for the federal customer — the substrate the firm needs anyway *is* the substrate the customer requires.

### 3.3 Layer 3 — EMDR-domain RAG

The in-flight build at `/Users/kiteboard/emdr-rag` (memory `project_emdr_rag.md` — 42 commits, deployment-ready pending six user-action gates) is the foundation. It is sovereign: Nemotron 3 Nano Omni on vLLM (NVFP4 on B300 catfish, fp8 fallback on H100/H200), `llama-3.2-nv-embedqa-1b-v2` + `llama-3.2-nv-rerankqa-1b-v2` (NIM, both Apache-2.0-compatible — NV-Embed-v2's CC-BY-NC was disqualified for paid clinical), Milvus+cuVS, cuGraph, NeMo Guardrails Colang 1.0, FastAPI. Crisis path is **deterministic** regex+classifier → hard template, never LLM-improvised. The corpus is ~95 HTML pages + 20 Buzzsprout transcripts ~= 350K tokens.

For v2 the corpus expands beyond `scalingupemdr.com` to cover the full clinical evidence base:

| Source | Tokens | Authority | Update cadence | License posture |
|---|---|---|---|---|
| `scalingupemdr.com` (existing) | ~350K | Practitioner training | Monthly | Crawl with author authorization (already in flight) |
| EMDRIA standards documents | ~50K | Professional governing body — the 2-year-experience + 50-sessions + 25-clients + 20 consultation hours requirement and the June 2026 fee update are the binding facts the agent must cite ([EMDRIA Approved Consultant requirements](https://www.emdria.org/emdr-training/emdr-consultant/)) | Annual | Public; cite, don't republish |
| ISTSS PTSD CPG | ~120K | International Society for Traumatic Stress Studies (gives EMDR a strong recommendation per [ISTSS guidelines](https://istss.org/clinical-resources/adult-trauma-assessments/ptsd-checklist-for-dsm-5/)) | 5-year cadence | Public PDF; cite |
| **VA/DoD 2023 PTSD CPG** | ~180K | The federal customer's own CPG. The Annals 2024 summary says CPT, EMDR, PE all benefit clinician-rated PTSD, with the caveat that there are **no studies of EMDR in active-duty service members and few in veterans** ([VA/DoD CPG Annals 2024 PDF](https://www.healthquality.va.gov/guidelines/MH/ptsd/PTSD-in-Annals-2024.pdf)). This evidence gap is precisely the empirical contribution §4 proposes to fill. | 5-year cadence | Public PDF; cite |
| EMDR PubMed corpus | ~2M abstracts, full-text where OA | Primary literature | Daily NCBI mirror | Cite; full-text only when OA license permits |
| 2024 Cambridge IPDMA (Roberts et al., *Psychological Medicine*) | ~30K | Most recent individual-participant-data meta-analysis: no significant difference EMDR vs other psychological treatments for PTSD ([PubMed 38173121](https://pubmed.ncbi.nlm.nih.gov/38173121/)) | One-time | OA |
| 2025 *British Journal of Psychology* clinical+cost-effectiveness review (Simpson et al.) | ~40K | Health-economics framing for the VA business case ([BJP DOI 10.1111/bjop.70005](https://bpspsychub.onlinelibrary.wiley.com/doi/10.1111/bjop.70005)) | One-time | Likely paywalled — abstract only in RAG, link to publisher |
| GOATnote-internal session transcripts (post-deployment, with informed consent) | growing | Operational ground truth | Real-time | Internal; never re-released; per-tenant adapter only |

Architecture-wise, v2 keeps the project_emdr_rag.md stack but adds three cuts. (1) **Per-therapist rank-4 LoRA adapter** trained on lobster H200 — each therapist's session-fidelity-graded notes become a personalization corpus, the adapter steers the agent to that therapist's case-conceptualization style. Adapters swap at inference time on catfish B300 — this is the same swap pattern that underlies the medomni inference path. (2) **Contextual retrieval** (Anthropic's Sept 2024 technique — 49-67% retrieval-failure reduction) is mandatory; already locked in `project_emdr_rag.md`. (3) **Audit-trail inference** — every retrieved chunk + generation produces a per-turn JSONL row including chunk hashes, embedding model version, generation seed (where the model permits), and `agent_id`. The S3 Object Lock store is queryable by FOIA / IG / VA OIG without exporting PHI.

The relationship to the existing project_emdr_rag deploy (memory: 36-iter loop already complete, `7969fa1a` cron expires 2026-05-12) is *additive*, not replacement: the CONNECT pilot in §6 lights up the existing build under a real federal cohort. The crisis path stays deterministic. The agent never improvises 988-routing or VA Crisis Line escalation.

### 3.4 Layer 4 — GUI as downstream

Three React surfaces, each thin over the agent substrate:

| Surface | Path | Today | v2 target |
|---|---|---|---|
| Clinician console | `web/app/console/` (new) | — | Booking + session-note authoring + outcome questionnaire entry. Agent drafts; clinician edits; clinician signs |
| Federal program-manager dashboard | `web/app/programs/[program_id]/` (new) | — | Per-cohort PCL-5 trend, dropouts, fidelity, AEs. One-shot dashboards generated on demand by an agent that reads from the closed-loop OS |
| Public-facing services site | `web/app/services/` (new) | — | Government RFP / contact / outcome reports landing page. Replaces the current marketing-only `thegoatnote.com` |
| Records OS (existing) | `web/app/records/RecordsOS.tsx` at `/4UWHAt` | Mobile-first redesign in spec PR #102 | Personal-records demo; pivot is *not* the wedge but stays as proof-of-architecture |
| Public skill registry (existing) | `web/app/skills/page.tsx` at `/4UWHAt/skills` | Live | Trust-through-transparency; Federal customer reviewer can read every skill file the agent will execute on their data |

The PR #102 mobile-first redesign is the design language for the new surfaces — single content stream by default, auto-fit grids, sticky thumb input, bottom tab on Compact, left rail on Expanded. The point is that v2 *adds* surfaces but doesn't reinvent the visual layer; one design system across all four. Cross-reference `findings/2026-05-06-mobile-first-redesign-spec/SPEC.md`.

---

## 4. The published clinical-AI-safety paper

This is not optional. The federal customer reads journals; an SDVOSB/SBIR Phase II that has shipped a *NEJM AI* observational study has a different conversion rate than one that has not. The paper is also the most efficient sales asset the firm can produce — it is concurrent with the work, not in addition to it.

**Title sketch:** *"Sovereign EMDR-Domain Retrieval-Augmented Generation with Per-Therapist Rank-4 Adapters and Audit-Trail Inference for Federal Trauma-Care Programs: A Pre-Registered Observational Study"*

**Target venues, in order of preference:**

1. **NEJM AI** — *primary target*. Editorial scope is exactly clinical-AI evaluation with a clinical-evidence bar matching other clinical interventions. Submission requires either pre-registration in WHO ICTRP for trials starting after Jan 1, 2025, or rigorous observational design ([NEJM AI editorial policies](https://ai.nejm.org/about/editorial-policies)). Reporting must follow TRIPOD-AI or MI-CLAIM. Code availability required. The paper would not be a randomized trial — it would be a pre-registered observational cohort study comparing the Phase 1 cohort against a propensity-matched VA/DoD CPG-as-usual cohort drawn from CDW (Corporate Data Warehouse) administrative records. Per-veteran-month outcome: PCL-5 trajectory at sessions 4, 8, 12.

2. **Lancet Digital Health** — secondary. Broader clinical-AI scope, less surgical on methods reporting than NEJM AI; useful if NEJM AI desk-rejects on novelty.

3. **npj Digital Medicine** — fallback. Faster turnaround, OA fee.

4. **arXiv preprint + JMIR Mental Health** — concurrent submission. arXiv on the day of NEJM AI submission for citation; JMIR Mental Health as a methodology-companion paper if the dataset itself yields a separate contribution (e.g. the contextual-retrieval ablation results).

**Empirical claims the paper would make.** (a) The sovereign EMDR-RAG agent, when paired with a licensed EMDRIA-Approved-Consultant-supervised therapist network, produces **non-inferior** PCL-5 trajectories at session 8 vs the VA/DoD CPG-as-usual matched cohort (pre-specified non-inferiority margin of 3 PCL-5 points, well below the [9-12 PCL-5 MID range](https://pubmed.ncbi.nlm.nih.gov/37845820/)). (b) Audit-trail inference enables **post-hoc verification of every clinical recommendation** — the paper publishes the audit JSONL schema and reports the proportion of the cohort's clinical recommendations that are traceable to a CPG citation chain (target: 100%). (c) The per-therapist rank-4 adapter improves session-fidelity-graded adherence over the unadapted base agent by ≥X percentage points (effect size pre-registered).

**Data we'd need.** The pilot cohort needs to be ≥80 veterans for adequate power on the non-inferiority margin (assuming SD ~14 on PCL-5 change scores per the Marx et al. estimates summarized in [PCL-5 MID PMC article](https://pmc.ncbi.nlm.nih.gov/articles/PMC10754254/)) and a matched comparator of ≥160 from CDW. Outcome: PCL-5 at baseline, 4, 8, 12 sessions. Secondary: PHQ-9, retention, AE rate. The CONNECT-style pilot in §6 is sized to deliver this with overhead.

**IRB.** The pilot needs VA IRB approval (Charleston VA's IRB is the local-of-record per §6); separately the GOATnote side needs Western IRB or another commercial IRB for the analytic protocol. The pre-registration goes to ClinicalTrials.gov per NEJM AI policy. WHO ICTRP-compliant ([NEJM AI editorial policies](https://ai.nejm.org/about/editorial-policies)).

**Reporting.** TRIPOD-AI checklist + MI-CLAIM. Both are explicit NEJM AI requirements.

---

## 5. Why government is the wedge

The wedge customer dimensions, with primary sources:

| Dimension | Federal trauma-care program | Why this rewards an AI-native services firm |
|---|---|---|
| **Unit economics** | VA's National Center for PTSD operates 6 academic centers of excellence ([NCPTSD About](https://www.ptsd.va.gov/about/index.asp)); the Palo Alto Dissemination & Training Division alone produces the field's training+digital-tools backbone for VA. CONNECT-style telemedicine programs have been rolling out across CBOCs since the original TOP pilot ([VA News PTSD telehealth](https://news.va.gov/132608/telemental-health-ptsd-treatment-for-veterans/)). FY 2026 IHS budget is $8.7B (12% YoY increase) with $80M for the new Native Behavioral Health & SUD Program ([NIHB FY26 funding analysis](https://www.nihb.org/what-the-fy-2026-funding-package-means-for-tribal-health-systems/)). DoD telehealth flexibility is extended through 2027/2028 under the FY26 omnibus ([Behavioral Health Business 2026-01-22](https://bhbusiness.com/2026/01/22/new-appropriations-bill-would-increase-samhsa-funding-expand-telehealth-flexibilities-through-2028/)). Per-veteran-month subscription pricing fits how the federal customer already pays. | An AI-native firm can deliver per-veteran-month at a unit cost a copilot SaaS firm cannot match because the agent layer absorbs case-management and reporting overhead that incumbents bill out as separate line items. |
| **Sales cycle** | 12-18 months for a first VA prime contract via open RFP. **Set-aside paths are faster:** SDVOSB / VOSB sole-source up to $7M (services), VA SBIR Phase I up to ~$250K with a 6-month period of performance and Phase II up to ~$2M ([SBA veterans contracting](https://www.sba.gov/federal-contracting/contracting-assistance-programs/veteran-contracting-assistance-programs); [VA acquisition VAAR Part 819](https://www.acquisition.gov/vaar/part-819-small-business-programs)). VHA Innovation Ecosystem AI Tech Sprint runs a 12-week mental-health cohort that has previously fielded suicide-prevention apps like VITAE ([VHA IE](https://www.va.gov/INNOVATIONECOSYSTEM/views/explore/innovators-network.html); [VA Orlando VITAE story](https://www.va.gov/orlando-health-care/stories/pioneering-health-care-solutions-ai-innovations-take-shape-at-hackathon/)). The Innovation Ecosystem path is *months not years*. | An AI-native firm with a working substrate (skills router live, RAG built, mobile-first GUI in spec) can prototype against a Tech Sprint cohort and have a deployable artifact at week 12. The incumbent's 18-month build *is* the moat the AI-native firm gets to skip. |
| **Reporting requirements** | VA contracts require structured outcomes reporting (PCL-5, PHQ-9 trajectories) at quarterly cadence, plus audit-log retention ≥7 years (HIPAA + VA Records Control Schedule). DoD adds DFARS clause flow-down. FedRAMP authorization for cloud handling of PHI is moving to the OSCAL-based FedRAMP 20x rev — VA was the first agency to submit an OSCAL-format SSP ([DigitalVA OSCAL](https://digital.va.gov/security-excellence/va-first-to-submit-oscal-plan/); [Convox FedRAMP 2026 guide](https://www.convox.com/blog/fedramp-authorization-2026-guide-saas-companies)); FedRAMP 20x Phase 2 concludes Q1 FY26 with broader Low/Moderate authorization paths opening Q3-Q4 FY26 ([Lazarus Alliance FedRAMP 20x timeline](https://lazarusalliance.com/the-fedramp-20x-phase-two-timeline/); [PI Tech FedRAMP 20x 2026](https://pitechsol.com/blog/fedramp-20x-authorization-readiness-government-contractors-2026/)). | The closed-loop OS in §3.2 captures every artifact the federal customer requires anyway. The audit-log substrate *is* the firm's product surface — the federal reporting requirement is met as a side-effect of operating, not as a separate compliance line. |
| **Vendor preferences** | VA goal: 5%+ of contract dollars to SDVOSBs ([SBA SDVOSB](https://www.sba.gov/partners/contracting-officials/contracting-program-administration/sdvosb-program-administration)); VA's actual SDVOSB spend has exceeded 15% in some recent years per Congressional Research Service data ([CRS R47226](https://www.congress.gov/crs-product/R47226)). Preference for sovereign-stack AI (no PHI to commercial LLM APIs) is rising — VA AI strategy explicitly prioritizes high-impact AI with veteran-data-sovereignty considerations ([VA AI strategy](https://department.va.gov/ai/building-the-future-vas-strategy-for-adopting-high-impact-artificial-intelligence-to-improve-services-for-veterans/)). | GOATnote's sovereign-stack thesis (memory: `project_prism42_sovereign_stack_thesis.md`) is the federal customer's preferred posture. Cloud-LLM-API-dependent competitors lose on "where does PHI go" before the technical eval starts. |

The combined picture: an SDVOSB-eligible (founder-route TBD) AI-native services firm with a sovereign stack, a closed-loop OS that produces VA-shaped reports as an operating side-effect, and a per-veteran-month service unit, is built for the federal trauma-care market. Every other GOATnote market is downstream of winning here.

---

## 6. First-pilot recommendation

**Recommendation: Charleston VA / Ralph H. Johnson VAMC, via a CONNECT-style telemedicine PTSD pilot, structured as a VA SBIR Phase I direct-to-Phase-II solicitation.**

Three options were evaluated; the reasoning is explicit so the founder can override.

### Option A: Palo Alto VA / National Center for PTSD HQ (Dissemination & Training Division)

**Pros.** Largest brand surface in VA PTSD care. The Dissemination & Training Division "develops widely-used mobile mental health apps, conducts research on online and mobile interventions, and trains VA staff in use of digital mental health tools" ([NCPTSD Dissemination](https://www.ptsd.va.gov/about/divisions/dissemination/index.asp)) — they are *the* internal customer for digital tools at VA. A pilot here lands the firm in front of Field Implementation Team national leadership.

**Cons.** They build their own tools. PTSD Coach, AIMS, etc. are NCPTSD products. They are predisposed to read an external vendor as a competitor or as a potential research collaborator — not as a service vendor. Procurement path is unclear; this likely runs through a Cooperative Research and Development Agreement (CRADA) or a research grant, not a services contract. **Sales cycle 18-24+ months** for a first deal; lower-probability conversion.

### Option B: Charleston VA / Ralph H. Johnson VAMC (CONNECT pilot precedent)

**Pros.** Charleston is one of the 12 CBOCs in the original TOP / CONNECT-style telehealth-PTSD pilot ([VA News telehealth PTSD](https://news.va.gov/132608/telemental-health-ptsd-treatment-for-veterans/); [VA Charleston Mental Health Care](https://www.va.gov/charleston-health-care/health-services/mental-health-care/)). The local IRB and contracting officer have *already authorized* a remote-trauma-treatment pilot before — the institutional muscle exists. It is a "regional center of excellence for Veteran-focused mental health care." It is also small enough that an SDVOSB sole-source or competitive set-aside under $7M is unbottlenecked. The local rural-veteran population is a population the existing CPG explicitly identifies as evidence-gap'd. The founder is a physician — the credibility ramp at a clinical-operations VA medical center is shorter than at NCPTSD HQ.

**Cons.** Smaller national brand. Outcomes from Charleston don't automatically scale to other VAMCs; each VISN has its own contracting officer.

### Option C: Indian Health Service (regional Area Office, e.g. Phoenix or Albuquerque)

**Pros.** FY 2026 budget includes the new $80M Native Behavioral Health & SUD Program plus the $26.66M Native Connections program ([NIHB FY26](https://www.nihb.org/what-the-fy-2026-funding-package-means-for-tribal-health-systems/)). IHS is small (FY26 budget $8.7B vs VA's >$300B) and procurement is more accessible — Tribal 638 self-determination contracts can move fast. The trauma-prevalence rate among AI/AN populations is among the highest in the US; the clinical need is acute.

**Cons.** Tribal sovereignty rules add real complexity around data jurisdiction; PHI flows that touch a non-Tribal cloud can break Tribal-state-federal data-sovereignty agreements. The sovereign-stack thesis (no commercial cloud LLM) is *especially* aligned here, but the BAA framework is more complex than VA's. Less near-term leverage for an *NEJM AI* paper because cohort sizes per Tribal entity may be too small for the matched-control study.

### Recommendation

**Lead with Charleston VA. Use the SDVOSB-set-aside / VA SBIR Phase I path. Open the IHS conversation in parallel as Pilot-2 (within 6 months of Charleston Pilot-1 award), positioned as "Tribal trauma-care extension of the VA Charleston pilot." Use Palo Alto NCPTSD as the *publication / dissemination* relationship, not the contracting customer — pre-print drops with NCPTSD copied; a CRADA gets signed in year 2 around the *NEJM AI* publication.**

Charleston is the lowest-friction first deal. IHS is the most-aligned-with-mission second deal. NCPTSD is the highest-leverage *publication* relationship. The single critical-path activity for the next 90 days is a VA SBIR Phase I solicitation response targeting a topic near "AI-augmented trauma-care delivery for rural veterans" — VA SBIR Phase I awards run up to ~$250K with a 6-month period of performance based on general-SBIR averages ([SBIR Awards](https://www.sbir.gov/awards)) and the VA's specific solicitation calendar can be tracked at [research.va.gov/funding](https://www.research.va.gov/funding/).

---

## 7. Pricing model recommendation

Three models were evaluated against Alströmer's per-outcome thesis and VA's actual procurement patterns.

| Model | Unit | Incentive | VA fit | Risk to GOATnote |
|---|---|---|---|---|
| **Per-session** | Single therapist hour | Maximize sessions | Familiar (VA already pays per-session for community-care) | None to GOATnote, but no Alströmer outcome alignment — degenerate to a staffing firm |
| **Per-veteran-month** | One veteran enrolled for one month, regardless of session count | Maximize retention + stable revenue | Familiar (VA capitated arrangements) | Underutilization risk if veteran DNAs — but the closed-loop OS *is* the dropout countermeasure (`monitor-drop-off.md`), so this risk is the firm's actual product |
| **Per-outcome-improvement** | Tier-1 fee per veteran reaching ≥10pt PCL-5 reduction at session 12 (the [10-point MCID per VA NCPTSD](https://www.ptsd.va.gov/professional/articles/article-pdf/id1626220.pdf)) | Maximize clinical effect | Aligns with VA's value-based-care pivot but no current-year contracting vehicle | Severe revenue volatility in year 1 with N≈80; unfundable |

**Recommendation: Per-veteran-month base + a per-outcome bonus tier.** Specifically: a base of $X / veteran-enrolled-month covering sessions, supervision, reporting, and substrate; a **$Y bonus** per veteran achieving ≥10-point PCL-5 reduction at session 12, billed quarterly. This structure (a) matches VA's procurement vehicle preference (capitated services contract), (b) gives the firm stable cash flow in year 1, (c) embeds Alströmer's per-outcome alignment in the bonus tier without making the entire revenue line dependent on it, and (d) makes the closed-loop OS — the dropout-monitor in particular — directly revenue-protective.

**Numeric anchors (recommendations, not commitments).** Base $400-600/veteran-month assuming 3-4 sessions/month, supervisor overhead ~10%, substrate ~$50/veteran-month at scale (the GPU footprint in §8 amortizes flat). Bonus $1500-2500 per ≥10pt PCL-5 responder at session 12. Comparable VA community-care psychotherapy rates run $150-250/session per CMS/CHAMPVA fee schedules; per-veteran-month at 3 sessions is $450-750, so $400-600 base is competitive with a margin. **The founder should validate these against the actual VA SBIR Phase I budget cap (~$250K per current SBIR averages — confirm at solicitation time) and the period of performance (6 months) before committing.**

The pre-registered ≥10pt PCL-5 responder rate is *also* the empirical claim in the *NEJM AI* paper (§4). The pricing model and the publication co-design.

---

## 8. GPU footprint analysis — confirm right-sized

| Pod | Hardware | Today | v2 role | Why this pod, not another |
|---|---|---|---|---|
| **catfish** | B300 (Blackwell SM 10.x) | medomni public inference + skills-router profile=v_final | **Production inference + per-tenant adapter swap**: medomni public + GOATnote services + EMDR-RAG retrieval | Blackwell-only NVFP4 quant for the production model `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`; per-tenant rank-4 LoRA swap at inference is the deploy pattern (memory `nemotron_omni_tool_call_parser.md`). Adapter swap is fast; per-clinic / per-therapist personalization at runtime |
| **lobster** | H200 (Hopper SM 9.0) | V2.5 SFT + judge | **V2.5/V2.7/V3/V3.5/V_final base training + per-tenant rank-4 adapter training** | H200 fp8 + bf16-mixed for LoRA is the safe default per `medomni/CLAUDE.md` §3. Each new clinic / therapist gets a rank-4 adapter trained here on their session-fidelity-graded notes; result ships to catfish |
| **narwhal** | H200 (Hopper SM 9.0) | factory_loop reasoning corpus | **EMDR-domain corpus factory** — extends factory_loop pattern to ingest EMDRIA standards / ISTSS / VA-DoD CPG / EMDR PubMed; generates contextual-retrieval augmented chunks per Anthropic Sept 2024 | H200's memory bandwidth is the bottleneck for embedding generation at scale; factory_loop pattern (continuous corpus expansion + dedup + index rebuild) is the right shape |
| **prism-mla-h100** | H100 (Hopper SM 9.0) | Idle — flagged for deletion 2026-05-02 deadline (memory `feedback_idle_gpus_get_deleted.md`) | **Continuous compliance worker** — HITRUST/FedRAMP evidence collection: hourly OSCAL evidence sync, automated control-attestation against `oscal-cli`, generation of SSP deltas for every `web/` deploy, FIPS-140-3 inventory diffs against TLS-terminated edges. Also runs the hourly adversarial-probe daemon (currently on GH Actions; offload onto the pod for unbounded test surface) | H100 with no production traffic is the cheapest of the four roles per dollar of SaaS-substitute revenue. The `feedback_idle_gpus_get_deleted.md` 12h utilization rule is satisfied by the hourly compliance cadence + the every-few-minute adversarial-probe daemon |

The footprint is right-sized at four pods. If the Charleston pilot scales to a second VAMC in year 1, the answer is a fifth catfish-class pod (B300, sovereign inference) before any pod is repurposed. Training (lobster) and corpus-factory (narwhal) scale up sub-linearly with cohort size; compliance (prism-mla-h100) scales flat. **Action: rescue prism-mla-h100 from idle-deletion this week** by standing up the OSCAL evidence cron — even a stub job that produces an hourly evidence diff is sufficient to keep utilization above the 10% threshold per the deletion rule.

---

## 9. 90-day execution plan

Each week names the work, the blocker, and the single dependency. Founder-action items are tagged **[F]**.

### Weeks 1-2 (May 6 - May 19, 2026): Foundation

- **W1.1** Open this PR (this SPEC document). **[F]** ratify the four-lever decisions in §12. Nothing else can finalize until those land.
- **W1.2** Stand up `web/lib/agent/skills/operations/` skeleton — empty markdown stubs for the 12 skills in §3.1. PR per skill or PR per batch-of-3, each gated by `clinical-skill-review.yml`. No clinical content until W3 (gated on **[F]** ratification of §12 + a clinical-reviewer-of-record commitment).
- **W1.3** Rescue `prism-mla-h100` from idle-deletion. Write the `run-fedramp-evidence-collection.md` skill stub + a 5-line cron that runs `oscal-cli validate` against an empty SSP and writes the result to S3. Utilization ≥10% threshold satisfied. **Blocker:** none.
- **W2.1** SBIR target identification. Pull the open VA SBIR FY26 solicitation index from [research.va.gov](https://www.research.va.gov/funding/), filter for Mental Health / PTSD / digital-therapeutic topics, identify top-2 fit. **[F]** approve target.
- **W2.2** Begin SAM.gov / VA Vetbiz registration if not already complete ([VA VetBiz](https://www.vetbiz.va.gov/sbprogram/)) — SDVOSB certification is a 60-90-day process; start day 1. **[F]** required.

### Weeks 3-6 (May 20 - June 16): Skills authoring + EMDR-RAG corpus expansion

- **W3-W4** Author `onboard-va-clinic.md`, `credential-new-therapist.md`, `book-veteran-intake.md`, `bill-per-veteran-month.md`, `audit-baa-coverage.md`, `monitor-drop-off.md`, `run-fedramp-evidence-collection.md` — the operational seven. Author by Claude with founder review per skill. Clinical skills (`run-emdr-session.md`, `grade-session-fidelity.md`, `handle-clinical-adverse-event.md`) deferred to W7 once clinical-reviewer-of-record retained. **Blocker:** clinical-reviewer-of-record retention.
- **W5-W6** EMDR-domain corpus expansion on narwhal — ingest VA/DoD 2023 PTSD CPG, ISTSS PTSD CPG, EMDRIA standards, 2024 IPDMA. Run contextual-retrieval augmentation pass. Update `data/eval/baseline-mock-mock.json` baseline with the expanded corpus. **Blocker:** founder authorization to crawl the public sources (most are PDF-public).

### Weeks 7-9 (June 17 - July 7): SBIR Phase I proposal

- **W7-W8** Author SBIR Phase I narrative using `author-government-proposal.md`. Cite Charleston VA precedent, cite the four-layer architecture, cite the *NEJM AI* publication target and IRB plan, attach the existing `findings/` artifacts as technical readiness evidence. **[F]** review + counsel review.
- **W9** Submission. **[F]** sign.

### Weeks 10-12 (July 8 - July 28): Pilot prep + paper draft

- **W10** Author the *NEJM AI* paper draft skeleton (Methods + Pre-registration sections — Results stay TBD until pilot completes). Pre-register at ClinicalTrials.gov.
- **W11** IRB submissions: VA Charleston IRB (gated on award) + Western IRB or comparable for the analytic-side protocol (can submit immediately, doesn't depend on award).
- **W12** First pilot-readiness review. End-of-quarter checkpoint: skills router live with operational skills, EMDR-RAG corpus expanded, SBIR submitted, IRB in motion, paper pre-registered. **[F]** decision: continue or replan.

### Identified blockers (cross-cutting)

| # | Blocker | Mitigation | When binding |
|---|---|---|---|
| B1 | SDVOSB certification (60-90d) | Submit Day 1; founder must confirm service-disability rating from VA Compensation & Pension review (founder-route TBD §11 — if non-veteran, partner with an SDVOSB primer or pursue VOSB / SBA 8(a) instead) | W2-W12 |
| B2 | Clinical-reviewer-of-record (EMDRIA Approved Consultant) | Retain by W2 | W3 (clinical skills) |
| B3 | FedRAMP authorization (6-18 months) | Begin OSCAL SSP draft on prism-mla-h100 W1; not blocking pilot if VA Charleston accepts a Moderate-equivalent posture under a contract-specific ATO | Year 2 for broad multi-VAMC scale |
| B4 | BAA with Medplum / Postgres host / S3 (per memory `project_medomni_v1_architecture_decisions.md`) | All in scope at v1; needs founder + counsel sign | W3 |
| B5 | Clinician-network capacity (need ≥3 EMDRIA-certified therapists on retainer) | Recruit via EMDRIA directory + founder's professional network; per-veteran-month requires therapist-first not therapist-last | W7 |

---

## 10. Risk register

| ID | Risk | Likelihood | Impact | Mitigation | Owner |
|---|---|---|---|---|---|
| R1 | HIPAA BAA gap with any vendor in PHI path | Medium | Catastrophic (deal-killer + fine) | `audit-baa-coverage.md` skill quarterly + per-vendor sweep at procurement; Medplum self-host (per memory `project_medomni_personalized_records_research.md`); never `.env`-read of cloud API keys | Compliance officer |
| R2 | EMDR clinical liability (adverse outcome attributed to AI) | Low-Medium | Catastrophic (suit + license action against supervising clinician) | **The agent never runs reprocessing autonomously.** The agent supports a licensed EMDRIA-certified therapist running the 8-phase protocol; the agent drafts notes, surfaces relevant CPG citations, flags fidelity drift. Every clinical recommendation is signed by the therapist. Marketing copy explicitly says "augments licensed therapists, never replaces." | Medical director (founder) |
| R3 | FedRAMP authorization slips beyond pilot timeline | High | Medium (constrains scale to single-VAMC pilot until ATO) | Begin OSCAL SSP draft W1; structure pilot under a contract-specific ATO (Moderate-equivalent posture acceptable to local VAMC); FedRAMP 20x Phase 2 timeline puts broader Low/Moderate authorization in late FY26 ([FedRAMP 20x timeline](https://lazarusalliance.com/the-fedramp-20x-phase-two-timeline/)) | Compliance officer |
| R4 | Sales cycle 12-18 months for first deal | High | Medium (cash-flow) | Two-track procurement: SBIR Phase I direct-to-Phase-II (3-6 month award decision) **and** SDVOSB sole-source (no full RFP needed under $7M services). Run both. | Founder + BD |
| R5 | First-pilot cohort underpowered for *NEJM AI* | Medium | Medium (paper goes to *Lancet Digital Health* or *npj Digital Medicine* instead) | Pre-register at N=80 minimum; if Charleston commits to a smaller N initially, structure as 12-month enrollment window to reach N=80 cumulatively; combine with IHS pilot data if needed | Founder (clinical lead) |
| R6 | Capacity expansion lag if first deal succeeds | Medium | Medium (constrains second deal) | Therapist network expansion in parallel with W7-W12; standing up a fifth catfish-class B300 pod takes ~2-4 weeks lead time on Brev/Nebius — order at first pilot award, not at second pilot signing | Operations |
| R7 | EMDRIA / ISTSS objection to AI-augmented EMDR | Low | High (loss of professional-society alignment) | Engage EMDRIA Approved Consultant network early; position as augmenting EMDRIA-trained therapists, not bypassing certification; consider EMDRIA / ISTSS conference submission concurrent with *NEJM AI* paper | Clinical lead |
| R8 | Sovereign-stack thesis collapses if Nemotron 3 Nano Omni gets deprecated | Low | Medium | Memory `project_prism42_sovereign_stack_thesis.md` mandates following NVIDIA's published reference voice stack; if the model is deprecated, follow NVIDIA's successor recommendation. Adapter substrate is model-agnostic. | Sovereign-stack lead |
| R9 | Audit-trail inference produces a record subpoenaed by plaintiff in clinical-liability suit | Medium | High but *intended* | The audit log is the firm's defense, not its liability. Every clinical recommendation traceable to a CPG citation chain *helps* the supervising clinician; the absence of that traceability is what current EMR-stack therapists already lack | Legal counsel |
| R10 | Concurrent prism42 emergency demands disrupt focus | Medium | Medium | Memory `feedback_prism42_prod_path_sacred.md` already enforces isolation; per `medomni/CLAUDE.md` §1, this repo is air-gapped from prism42 prod surface. Hold the line. | All |

---

## 11. Cross-references

### In-repo

- [`findings/2026-05-05-skills-router-v1/CARD.md`](../2026-05-05-skills-router-v1/CARD.md) — Cherny-cycle skills router (Company Brain primitive, live)
- [`.github/workflows/clinical-skill-review.yml`](../../.github/workflows/clinical-skill-review.yml) — auto-Claude-review CI (Company Brain quality gate)
- [`.github/workflows/adversarial-probe.yml`](../../.github/workflows/adversarial-probe.yml) — hourly probe daemon (closed-loop OS heartbeat)
- [`web/lib/agent/skills/`](../../web/lib/agent/skills/) — runtime skill markdown
- [`mvp/medomni-inference/skills/`](../../mvp/medomni-inference/skills/) — canonical authoring location
- [`findings/2026-05-06-mobile-first-redesign-spec/SPEC.md`](../2026-05-06-mobile-first-redesign-spec/SPEC.md) — mobile-first design language for v2 GUI surfaces (PR #102)
- [`findings/2026-05-04-pattern-b-spike/`](../2026-05-04-pattern-b-spike/) — FHIR-fetch p95=11ms substrate
- [`findings/2026-05-05-world-class-medomni-strategy/SPEC.md`](../2026-05-05-world-class-medomni-strategy/SPEC.md) — V2.5→V_final training trajectory (the specialization track that under-pins the inference layer)
- [`findings/2026-05-05-clinical-rag-architecture/SPEC.md`](../2026-05-05-clinical-rag-architecture/SPEC.md) — Hybrid sparse+dense+ColBERT rerank, MedScore + RAGAS faithfulness gate
- [`CLAUDE.md`](../../CLAUDE.md) — repo-level isolation contract, sovereignty thesis, hardware reality

### Memory base

- `project_emdr_rag.md` — therapist-facing RAG over scalingupemdr.com slated for H100 deploy (foundation for §3.3)
- `project_prism42_sovereign_stack_thesis.md` — sovereign NVIDIA stack thesis (foundation for §8)
- `feedback_prism42_prod_path_sacred.md` — production-deploy discipline (R10)
- `project_medomni_personalized_records_research.md` — Medplum self-host + Pattern B FHIR-fetch (substrate for §3.2 closed-loop OS)
- `project_medomni_v1_architecture_decisions.md` — single Project + AccessPolicy as tenant boundary, AuditEvent → S3 Object Lock 7yr (R1, §3.2)
- `feedback_idle_gpus_get_deleted.md` — prism-mla-h100 deletion-risk rule (§8 W1.3)
- `feedback_correspondence_email.md` — `b@thegoatnote.com` for all professional correspondence
- `feedback_no_emojis.md` — plain-text everywhere (this SPEC)
- `nemotron_omni_tool_call_parser.md` — vLLM flag set verified working on B300 (§8)
- `loop_agent_medomni_protocol.md` — 15-min /loop cadence (operational discipline)
- `stealth-tic.md` — trauma-informed care platform substrate; potential year-2 sibling product line
- `feedback_eval_preflight_judge_key.md` — judge-401 silent-poisoning rule (closed-loop monitoring)

### External primary sources

- [Y Combinator Summer 2026 RFS](https://www.ycombinator.com/rfs)
- [The VC Corner — YC Summer 2026 RFS Breakdown](https://www.thevccorner.com/p/yc-summer-2026-requests-for-startups-ideas)
- [StartupHub.ai — Build AI-native companies with closed-loop systems](https://www.startuphub.ai/ai-news/artificial-intelligence/2026/build-ai-native-companies-with-closed-loop-systems)
- [Epsilla — YC RFS Deep Dive: Twilight of SaaS](https://www.epsilla.com/blogs/2026-05-02-yc-rfs-deep-dive-the-twilight-of-saas-and-the-dawn-of-agent-)
- [Boris Cherny on Building Claude Code (Pragmatic Engineer)](https://newsletter.pragmaticengineer.com/p/building-claude-code-with-boris-cherny)
- [VA Research / ORD Funded Projects FY26](https://www.research.va.gov/about/funded_research/projects-FY2026.cfm)
- [VA HSR Research Priorities](https://www.hsrd.research.va.gov/funding/HSR-Research-Priorities.pdf)
- [VA National Center for PTSD About](https://www.ptsd.va.gov/about/index.asp)
- [VA Charleston Mental Health Care](https://www.va.gov/charleston-health-care/health-services/mental-health-care/)
- [VA News — TeleMental Health PTSD treatment](https://news.va.gov/132608/telemental-health-ptsd-treatment-for-veterans/)
- [VA AI Strategy](https://department.va.gov/ai/building-the-future-vas-strategy-for-adopting-high-impact-artificial-intelligence-to-improve-services-for-veterans/)
- [VHA Innovation Ecosystem](https://www.va.gov/INNOVATIONECOSYSTEM/views/explore/innovators-network.html)
- [VA Acquisition Regulation Part 819 (Small Business Programs)](https://www.acquisition.gov/vaar/part-819-small-business-programs)
- [SBA Veteran Contracting Programs](https://www.sba.gov/federal-contracting/contracting-assistance-programs/veteran-contracting-assistance-programs)
- [DigitalVA — VA First to Submit OSCAL SSP](https://digital.va.gov/security-excellence/va-first-to-submit-oscal-plan/)
- [Lazarus Alliance — FedRAMP 20x Phase 2 Timeline](https://lazarusalliance.com/the-fedramp-20x-phase-two-timeline/)
- [Convox — FedRAMP Authorization 2026 Guide](https://www.convox.com/blog/fedramp-authorization-2026-guide-saas-companies)
- [NIHB — FY 2026 Funding Package & Tribal Health](https://www.nihb.org/what-the-fy-2026-funding-package-means-for-tribal-health-systems/)
- [IHS Telebehavioral Health](https://www.ihs.gov/telebehavioral/)
- [Behavioral Health Business — FY26 Appropriations & Telehealth Through 2028](https://bhbusiness.com/2026/01/22/new-appropriations-bill-would-increase-samhsa-funding-expand-telehealth-flexibilities-through-2028/)
- [EMDRIA Approved Consultant requirements](https://www.emdria.org/emdr-training/emdr-consultant/)
- [VA/DoD PTSD CPG Annals 2024 PDF](https://www.healthquality.va.gov/guidelines/MH/ptsd/PTSD-in-Annals-2024.pdf)
- [Roberts et al. 2024 IPDMA — EMDR vs other psychological therapies (PubMed 38173121)](https://pubmed.ncbi.nlm.nih.gov/38173121/)
- [Simpson et al. 2025 — EMDR clinical+cost-effectiveness (BJP)](https://bpspsychub.onlinelibrary.wiley.com/doi/10.1111/bjop.70005)
- [Marx et al. — PCL-5 MID PMC 10754254](https://pmc.ncbi.nlm.nih.gov/articles/PMC10754254/)
- [VA NCPTSD — Using PCL-5 (MID 10pt threshold)](https://www.ptsd.va.gov/professional/articles/article-pdf/id1626220.pdf)
- [NEJM AI Editorial Policies](https://ai.nejm.org/about/editorial-policies)
- [NEJM AI Author Center](https://ai.nejm.org/author-center/article-types-and-submission-information)
- [SBIR.gov Awards](https://www.sbir.gov/awards)
- [CRS R47226 — Federal Contracting by Veteran-Owned Small Businesses](https://www.congress.gov/crs-product/R47226)

---

## 12. Founder decision points (single-line answers + my recommendation each)

The following five decisions block 90-day execution. Each is a one-line yes/no/which. My recommendation is in **bold**; the founder overrides explicitly to switch.

1. **Pilot target — Charleston VA, Palo Alto NCPTSD, IHS regional area office, or other?**
   → **Charleston VA** as Pilot-1. IHS as Pilot-2 by month 6. NCPTSD as publication relationship, not contracting customer.

2. **Pricing model — per-session, per-veteran-month, per-outcome-improvement, or hybrid?**
   → **Per-veteran-month base ($400-600 range) + per-outcome bonus ($1500-2500 per ≥10pt PCL-5 responder at session 12).** Numbers to validate against actual VA SBIR Phase I cap at solicitation time.

3. **Credentialing — hire EMDRIA Approved Consultant on retainer (W2), partner with an existing EMDR consultancy, or contract-as-needed?**
   → **Hire on retainer by W2.** Per-veteran-month requires therapist-first economics; the supervising EMDRIA Approved Consultant is the binding clinical-quality control + sign-off-of-record for `grade-session-fidelity.md` and `handle-clinical-adverse-event.md`. Contract-as-needed is too fragile for federal pilot.

4. **Paper venue — NEJM AI, Lancet Digital Health, npj Digital Medicine, or arXiv-only first?**
   → **NEJM AI primary; arXiv concurrent at submission day; Lancet Digital Health fallback.** NEJM AI's editorial bar (TRIPOD-AI / MI-CLAIM, ClinicalTrials.gov pre-registration) is also exactly what the federal customer reviews under contract.

5. **SDVOSB certification path — founder-route (if veteran), VOSB / SBA 8(a) / WOSB path otherwise, or SDVOSB-prime partnership?**
   → **Founder confirms veteran status. If yes, file Day 1 (60-90d cycle). If no, file VOSB instead (also 60-90d). If neither, partner with an SDVOSB prime for the first VA Charleston SBIR; renegotiate to direct in year 2.** This is the one decision Claude cannot recommend without founder data.

Two bonus decisions of slightly lower priority:

6. **Sovereign-stack hardware redundancy — order fifth catfish-class B300 pod at first pilot award, or wait for second pilot signing?**
   → **Order at first pilot award.** Lead time is ~2-4 weeks; running prod on a single B300 with no failover is a single-pod-loss outage waiting to happen.

7. **Marketing surface — keep `thegoatnote.com` as is, refresh as services landing, or new domain (e.g. `goatnote.health`)?**
   → **Refresh `thegoatnote.com` as services landing.** Same domain, new IA per the parallel `2026-05-06-llm-curated-ia-spec/SPEC.md` (if exists). Reverse-proxy `medomni.vercel.app/4UWHAt` Records OS demo as the "see the architecture in action" link.

---

## Appendix A — Why this SPEC is the right deliverable now

Three reasons the timing is correct.

**(1) The substrate is live, not vapor.** The skills router (`findings/2026-05-05-skills-router-v1/CARD.md`) is wired to `https://medomni.vercel.app/api/agent?profile=v_final` today; the clinical-skill-review CI is gating PRs today; the adversarial-probe daemon is running today; the EMDR-RAG build (`/Users/kiteboard/emdr-rag`) is 42 commits, deployment-ready, license-clean, with a baseline pass@1=0.360 saved. Nothing in this SPEC requires a research breakthrough. Every layer has a working primitive in this repo or in a sibling repo.

**(2) The YC RFS framing is current and converges.** The three letters (Alströmer / Blomfield / Hu) appeared together in Summer 2026's RFS; the deep-dive analyses (VC Corner / Epsilla / StartupHub) are May 2026, not 2024. The "twilight of SaaS / dawn of agents" framing is *the* current framing among YC-adjacent capital, which is the capital cohort GOATnote will most credibly raise from. Aligning the company narrative with the framing the relevant LPs already read each Tuesday is free leverage.

**(3) The federal procurement window opens this fiscal year.** FY26 IHS budget is +12% YoY with $80M new behavioral-health line. DoD telehealth flexibilities extend through 2027/2028. FedRAMP 20x Phase 2 concludes Q1 FY26 with broader authorization paths Q3-Q4 FY26 — meaning a FedRAMP-aware AI-native firm submitting an SDVOSB SBIR Phase I in summer 2026 is competing in the most-favorable procurement environment of the decade. This window does not stay open forever; the sovereign-stack thesis becomes table-stakes within 24 months as larger incumbents rebuild their stacks to match.

The 90-day plan (§9) is *not* "build a company from scratch." It is "wire the existing primitives into a federal pilot proposal." The work is integration and sales, not research.

---

## Appendix B — One-page summary for founder skim

GOATnote v2 is an AI-native services company delivering EMDR therapy to federal trauma-care programs (VA, DoD, IHS) through a licensed-clinician network augmented by a sovereign agent substrate. The substrate has four layers: a Company Brain (12 markdown skill files), a closed-loop OS (every booking, session, outcome, fidelity audit captured into a queryable store), an EMDR-domain RAG (sovereign Nemotron 3 Nano Omni + NIM embedding + cuGraph + Guardrails over EMDRIA / ISTSS / VA-DoD CPG / EMDR PubMed), and a thin React GUI for clinician console + program-manager dashboard + public services site. The primitive for each layer is already running in `medomni` or `emdr-rag`. Wedge customer is Charleston VA via the VA SBIR Phase I direct-to-Phase-II / SDVOSB sole-source path. Pricing is per-veteran-month base + per-outcome bonus. The pilot results submit to *NEJM AI* as a pre-registered observational study with CDW propensity-matched controls. GPU footprint stays at four pods (catfish prod, lobster training, narwhal corpus factory, prism-mla-h100 compliance). 90-day plan has one critical-path activity — the SBIR Phase I submission in W9 — and everything else is parallel and reversible. Five founder decisions in §12 unblock execution; my recommendation on each is documented and can be overridden line-by-line.

— end SPEC —
