# GOATnote v2 thesis — Emergency Medicine correction

**Status:** SPEC, awaiting founder ratification
**Date:** 2026-05-06
**Author:** Brandon Dent, MD (b@thegoatnote.com)
**Supersedes:** `findings/2026-05-06-goatnote-v2-thesis/SPEC.md` (PR #104) **at the domain layer only.** The architectural skeleton is preserved verbatim.

---

## 0. TL;DR

PR #104 ("GOATnote v2 thesis") authored a SPEC for AI-native EMDR (Eye-Movement Desensitization and Reprocessing) services. The original instruction contained a typo. The intended domain is **EM = Emergency Medicine**. This document corrects the domain, retains the entire architecture, and re-points the go-to-market, paper venue, pilot site, credentialing, and pricing to EM. The repo's existing asset base — `healthcraft`, `openem-corpus`, `lostbench`, `radslice`, `safeshift`, and the narwhal `factory_loop` — is already EM-shaped. This is a retitle, not a pivot.

---

## 1. The retitle

The user's original "EMDR-domain RAG layer" was a typo. The correct domain is **EM = Emergency Medicine**. This SPEC documents what changes vs what survives.

| Axis | EMDR (PR #104, superseded) | EM (correct) |
|---|---|---|
| Domain | Eye-Movement Desensitization and Reprocessing (trauma-focused psychotherapy) | Emergency Medicine (acute, undifferentiated patient care) |
| RAG corpus | scalingupemdr.com + EMDRIA practice library + ISTSS PTSD CPG | ALiEM, EM:RAP, ACEP Clinical Policies, Tintinalli, Rosen, IDSA / AHA / ACC EM-relevant CPGs, UpToDate EM modules, NHAMCS, NEDS, OpenEM-corpus (370 conditions, already in repo) |
| Customer | VA PTSD outpatient programs | VA EDs, DoD MTF EDs, IHS EDs, county / safety-net EDs, deployed-medicine units |
| Volume | ~50K-100K active EMDR cases nationally | ~155M ED visits/year US; ~6M VA-ED visits/year alone |
| Pilot site | Charleston VA / Ralph H. Johnson VAMC PTSD program | High-volume VA ED — **Houston VA Medical Center ED** (DoD overlap, busy, AI-pilot-friendly) |
| Credentialing | EMDRIA Approved Consultant on retainer | Board-certified Emergency Physician (ABEM/AOBEM) on retainer |
| Pricing | Per-veteran-month + outcome bonus on PCL-5 ≥10pt reduction | Per-encounter ($5–$15) + outcome bonus on missed-diagnosis-rate reduction |
| Paper venue | NEJM AI primary | *Annals of Emergency Medicine* primary; NEJM AI parallel; *JACEP Open* fallback |
| Empirical claims | Cost reduction + audit-trail (since EMDR efficacy is non-inferior question) | Door-to-disposition time, missed-diagnosis rate, triage accuracy, handoff communication quality, documentation-time delta |
| N | 80 PCL-5 outcomes | 1,000–10,000 ED encounters (much higher statistical power) |
| Regulatory framing | Behavioral-health software (low FDA risk) | Clinical Decision Support (must stay non-autonomous to qualify for 21st Century Cures Act CDS exemption) |
| Liability | Therapist-of-record retains | Physician-of-record retains; AI is decision-support only |

---

## 2. What survives from PR #104 (architecture, verbatim)

The full architectural skeleton transfers without modification. The five load-bearing pillars from PR #104 remain:

### 2.1 Alströmer thesis — sell the service, not the software

GOATnote sells an **outcome service** to government / health-system buyers, not a SaaS license. The skill-encoded operating system is the asset, not a product SKU. Buyers receive: clinical decision support output + a structured CARD-style audit trail + a named EM physician of record + service-level outcomes. The technology is implementation detail; the buyer is buying answers, not infrastructure.

### 2.2 Blomfield Company Brain — executable skills file

Every operational behavior of GOATnote is captured in an executable Skills directory (Anthropic Skills format) so that any session — human, agent, or auditor — can replay the company's standard operating procedure without tribal knowledge. The 12-skill enumeration from PR #104 transfers, with EM-specific renaming:

| PR #104 skill | EM-corrected skill |
|---|---|
| `run-emdr-session.md` | `run-em-encounter.md` |
| `author-protocol-fidelity-card.md` | `author-encounter-card.md` |
| `score-pcl5-followup.md` | `score-encounter-outcome.md` (door-to-dispo, missed-dx, ED-1/ED-2 metrics) |
| `escalate-suicidal-ideation.md` | `escalate-acute-decompensation.md` (sepsis, STEMI, stroke, PE) |
| `author-government-proposal.md` | unchanged |
| `submit-irb-amendment.md` | unchanged |
| `clinical-skill-review.md` | unchanged |
| `adversarial-probe.md` | unchanged |
| `audit-monthly-encounters.md` | unchanged |
| `train-per-tenant-lora.md` | unchanged (per-physician scope) |
| `rotate-pod-fleet.md` | unchanged |
| `respond-to-fda-inquiry.md` | unchanged |

Skills are versioned, lint-checked by the auto-Claude-review CI, and rebuilt on every merge.

### 2.3 Hu closed-loop OS — every interaction queryable, system self-adjusts

Every encounter (request → retrieval → reasoning → tool calls → final output) is captured as a structured trace, scored by an LLM judge against the encounter rubric, and fed back into:
- **Retrieval index** (corpus gaps surface as judge "no-evidence-found" labels)
- **LoRA training queue** (physician-specific patterns extracted from rated traces)
- **Adversarial probe set** (failure cases get promoted to the regression suite)
- **CARD audit log** (compliance posture is a query, not a manual report)

### 2.4 Government wedge unchanged

Unit economics, sales cycle, and reporting requirements all reward the architecture:
- VA / DoD / IHS / county-ED contracts pay annually, allow multi-year IDIQ
- SBIR Phase I / II → Phase III sole-source pathway is well-trod
- Federal compliance (FedRAMP, ATO, BAA, AuditEvent retention, 7-yr Object Lock) maps cleanly onto Hu-loop infrastructure
- SDVOSB sole-source ceiling ($4M for non-construction services) is achievable for a single-pilot deal
- The same skill-pack that runs the clinical service runs the compliance reporting

### 2.5 Sovereign NVIDIA stack thesis (4-pod fleet)

No third-party LLM API. All inference, training, and continuous evaluation live on a 4-pod NVIDIA fleet under direct GOATnote control, satisfying federal sovereignty preferences and the BAA boundary.

| Pod | GPU | Role (EM-corrected) |
|---|---|---|
| catfish | B300 | Production EM inference + per-physician LoRA hot-swap on encounter open |
| lobster | H200 | Continuous training: V2.5 → V2.7 → V3 → V3.5 → V_final + per-physician rank-4 adapter training |
| narwhal | H200 | EM-domain corpus factory (extends `factory_loop` with EM seeds; 62K+ items, 4,531 curated via Option-C judge_filter) |
| prism-mla-h100 | H100 | Continuous compliance worker: nightly AuditEvent rollups, federal audit-pack generation, FedRAMP-style log retention |

### 2.6 Per-tenant LoRA (rank-4, ~10MB each, hot-swap)

Each physician (and, layered above, each patient) gets a rank-4 LoRA adapter trained on their accumulated encounter corpus. ~10MB per adapter; hot-swappable on B300 in <500ms; never co-mingles training data across tenants. The adapter is the unit of personalization, the unit of audit isolation, and the unit of revocation.

### 2.7 Auto-Claude-review CI

`.github/workflows/clinical-skill-review.yml` runs an Opus 4.7 review against every PR touching `findings/`, `skills/`, `corpus/`, or `schemas/`. Required checks: clinical-skill-review (PASS), no-PHI-in-diff, no-secret-leak, schema-lint. Non-admin pushes / PRs blocked until the auto-review approves.

### 2.8 Adversarial probe daemon

Hourly cron hits production `/api/agent` endpoint with a rotating set of hard-cases drawn from:
- The `lostbench` MTR-001..078 emergency-triage scenarios
- The `safeshift` 23 scenarios (15 clinical)
- The `healthcraft` 195 eval tasks (2,255 binary criteria, 515 safety-critical)
- A growing pool of **promoted failures** from production traces

Any regression breaks the build and pages the on-call engineer.

---

## 3. Why EM is a stronger fit than EMDR for the existing asset base

This is the load-bearing insight: **the repo already has EM-shaped artifacts.** The EMDR framing was a typo; the codebase is already an EM platform.

### 3.1 `healthcraft` is an EM training environment

From `~/.claude/projects/-Users-kiteboard/memory/MEMORY.md`:
> **HEALTHCRAFT** ... Emergency Medicine RL Training Environment adapting Corecraft architecture (arXiv:2602.16179v5). 195 eval tasks, 2,255 binary criteria (515 safety-critical), 6 categories. V8 results (authoritative, 2026-03-15): Claude Opus 4.6 Pass@1 24.8% [21.5–28.4] / Pass@3 37.9% / Pass^3 13.8% / reward 0.634 / safety-fail 27.5%. GPT-5.4 Pass@1 12.6% [10.2–15.6] / Pass@3 24.6% / Pass^3 3.1% / reward 0.546 / safety-fail 34.0%.

This is already a publication-quality EM benchmark, with NeurIPS 2026 D&B whitepaper in dual-build.

### 3.2 `openem-corpus` is the RAG substrate

370 EM conditions, FHIR R4-mapped, 80 physician-reviewed (tier A), schema v2.0. LanceDB index already built (`data/index/openem.lance/`). All five GOATnote repos already pin `openem>=0.2.0` as a shared dependency. The EM RAG layer doesn't need to be built; it needs to be indexed.

### 3.3 `lostbench` is the EM eval harness

MTR-001 through MTR-078 are explicitly **Emergency Medical Triage** retrieval scenarios. CEIS 3-layer grading, failure classes A/B/C/D, ERS/CRS with Wilson CI. 78 emergency + 43 adversarial + 15 defer + 3 crisis + 279 coverage-cycle seeds. Already wired to a `lostbench ceis run` CLI with resume + cache.

### 3.4 `radslice` is EM imaging

330 tasks across 133 OpenEM conditions, 65 LostBench cross-refs. EM-imaging-adjacent (chest pain, abdominal pain, head trauma — the bread and butter of ED imaging).

### 3.5 `safeshift` is EM safety scenarios

23 scenarios (15 clinical, 8 robotic). Public Apache 2.0. The 15 clinical scenarios skew acute-care.

### 3.6 narwhal `factory_loop` skews EM

62K+ generated items, 4,531 curated via Option-C judge_filter (PR #94). Seed-question distribution skews EM presentations (chest pain, dyspnea, altered mental status, abdominal pain, trauma activation).

### 3.7 Conclusion

This is not a pivot. The retitle aligns the domain label with what is already built, indexed, evaluated, and partly-published.

---

## 4. Why EM is a stronger business fit for government

### 4.1 ED throughput is the most-tracked CMS metric

CMS ED measures (ED-1 median time from arrival to ED departure for admitted patients, ED-2 admit-decision-to-departure, OP-18 median time from arrival to ED departure for discharged patients, ED-22 left-without-being-seen, OP-22 patient left without being seen) are reported quarterly, public, and tied to reimbursement adjustments. AI augmentation has a **measurable, audit-traceable** impact on every one of these metrics.

### 4.2 Safety-critical outcomes are well-defined and quantifiable

Missed STEMI, missed sepsis (per CMS SEP-1), missed stroke (door-to-needle / door-to-CT), missed PE, missed aortic dissection, missed ectopic. Each has insurance-grade quantification methodology. AI-augmentation impact on each is measurable in 90-day windows.

### 4.3 VA EDs are chronically understaffed; political demand is strong

VA ED access is a recurring Congressional oversight topic. Bipartisan support for VA AI-augmentation pilots is durable across administrations. VA Office of Research and Development (ORD) has standing funding lines for ED quality improvement.

### 4.4 DoD active-duty deployed-medicine angle is high-priority

Forward deployment, austere environments, ship-board medicine, and remote duty stations all need EM decision support more than they need EMDR therapy. DoD JAIC / CDAO / DHA AI portfolios all explicitly call out forward-deployed clinical decision support.

### 4.5 Federal contracting officers know EM contracts

TeamHealth, USACS, Envision, and Vituity all have decades of VA / DoD / IHS ED staffing contracts. The contracting pattern is well-trodden; the procurement officers know how to score and award these. This dramatically de-risks the sales cycle compared to a novel EMDR service category.

---

## 5. The EM-domain RAG layer

Replaces the EMDR-domain RAG layer of PR #104. Same architecture, EM scope.

### 5.1 Corpus sources

**Primary clinical:**
- ALiEM (Academic Life in Emergency Medicine) — open
- EM:RAP transcripts (license required)
- ACEP Clinical Policies — open
- *Tintinalli's Emergency Medicine* (licensed)
- *Rosen's Emergency Medicine* (licensed)
- UpToDate EM modules (institutional license)

**Guidelines:**
- IDSA (sepsis, abx stewardship, CAP, UTI, SSTI)
- AHA / ACC (ACS, STEMI, NSTEMI, A-fib, HF)
- ACEP Clinical Policies (chest pain, syncope, headache, abd pain, etc.)
- AAP / PEM guidelines (pediatric EM)
- ATLS / ACS-COT (trauma)

**Population data:**
- NHAMCS (National Hospital Ambulatory Medical Care Survey, ED component)
- NEDS (Nationwide ED Sample, HCUP)
- VA Corporate Data Warehouse (under DUA)

**In-repo:**
- `openem-corpus` (370 conditions, 80 physician-reviewed)
- `healthcraft` task corpus (195 tasks, 2,255 binary criteria)

### 5.2 Retrieval architecture (unchanged from PR #104)

- LanceDB index (`openem.lance/` already exists, fresh as of 2026-03-13)
- nx-cugraph traversal for condition / symptom / drug / contraindication graph hops
- NeMo Guardrails Colang refusal layer for out-of-scope clinical claims (pediatric dosing without weight, OB without LMP, etc.)
- Reranker: `nv-rerankqa-mistral-4b-v3` via NIM

### 5.3 Personalization layers (unchanged from PR #104)

- **Per-physician LoRA** (replaces per-therapist LoRA): rank-4, ~10MB, trained on the physician's last 1,000 encounters
- **Per-patient LoRA** (from PR #103, layered on top): rank-4, ~10MB, trained on the patient's longitudinal record once they have ≥5 encounters in-system
- Both swap in <500ms on B300 at encounter open

---

## 6. Pilot site recommendation

**Primary: Houston VA Medical Center ED (Michael E. DeBakey VAMC).**

Rationale:
- High volume (~75K ED visits / yr at peak)
- Texas medical referral region has large active-duty population → DoD overlap
- Historically AI-pilot-friendly per VA ORD
- Established ED research infrastructure
- SDVOSB-friendly contracting officers in VISN 16

Path:
- VA SBIR Phase I (NIH/VA solicitation) authored via the `author-government-proposal.md` skill
- SDVOSB sole-source approach (parallel) for sub-$4M pilot
- IRB submission for retrospective ED-encounter validation study (no PHI off-prem; all inference on-prem at VA)

**Backup pilots:**
- Boise VA Medical Center ED (smaller volume, easier first deal, supportive ORD relationship)
- Sacramento VA (population health focus, integrates with VA Innovators Network)

---

## 7. Credentialing

**Board-certified Emergency Physician (ABEM or AOBEM) on retainer by week 2.**

This is *easier* than the EMDR equivalent — many EM physicians already moonlight or consult, and the hourly pattern ($250–$500/hr) is well-established. Recruit via:
- ACEP Career Center
- Direct outreach to academic ED faculty looking for non-clinical income
- SAEM Industry Advisory Council network

The physician of record:
- Reviews and signs off on every CARD before it goes to compliance
- Holds the prescriptive authority and decision-of-record liability
- Co-authors the *Annals of EM* paper
- Carries malpractice tail coverage paid by GOATnote

---

## 8. Pricing model

**Base: per-encounter $5–$15** (banded by complexity / acuity, ESI 1–2 vs 3 vs 4–5).
**Outcome bonus:** ≥30% reduction in missed-diagnosis rate at 6 months, paid as a true-up against baseline.
**Platform fee:** aligned to VA Choice / Community Care Network rate methodology so contracting officers can score it against existing schedules.

Reference contract pattern: TeamHealth / USACS group-staffing contracts, but for decision support rather than physician staffing. Floor: $50K / month per ED (covers 5–10K encounters); ceiling: $250K / month per ED (high-acuity with full LoRA personalization).

---

## 9. GPU footprint update

Same 4-pod fleet (Section 2.5 above). EM-specific roles called out explicitly here so the operational team can re-baseline:

| Pod | GPU | EM role | Idle-risk mitigation |
|---|---|---|---|
| catfish | B300 | Production EM inference + per-physician adapter swap | Always-on; idle < 1% of time once a single pilot is live |
| lobster | H200 | V2.5 → V_final continues; per-physician rank-4 adapter training | Training queue from Hu loop keeps utilization >70% |
| narwhal | H200 | EM-domain corpus factory; extends `factory_loop` with EM seeds | Seed pump from Hu-loop "no-evidence-found" labels |
| prism-mla-h100 | H100 | Continuous compliance worker — nightly AuditEvent rollups, federal audit-pack generation | Daily compliance pipeline; backfill on idle |

Per `feedback_idle_gpus_get_deleted.md`, no pod is allowed to sit at <10% utilization for >12h without an explicit "OK to idle?" from the user.

---

## 10. 90-day execution plan revision

Week-by-week, EM-specific milestones. Each week's deliverable is a CARD-style artifact in `findings/`.

**Week 1**
- Board-certified EM physician on retainer (LOI signed; tail coverage scoped)
- VA SBIR Phase I solicitation tracked; draft outline authored via `author-government-proposal.md`
- `openem-corpus` LanceDB index re-checked, EM corpus expansion plan drafted

**Week 2**
- SBIR Phase I full draft submitted (or held for next solicitation window if missed)
- Houston VA initial contact (VISN 16 SBIR coordinator)
- SDVOSB sole-source memo drafted (parallel path)

**Week 3**
- MOU draft with Houston VA ED service line chief
- IRB protocol drafted: retrospective ED-encounter validation (no PHI off-prem)
- EM-domain RAG index rebuilt with full corpus (Section 5.1)

**Week 4**
- IRB submitted
- BAA template aligned with VA Office of General Counsel template
- Adversarial probe set re-baselined against EM corpus

**Weeks 5–6**
- IRB approved (target)
- First 2 EM physicians onboarded; per-physician LoRA training pipeline tested end-to-end
- `evaluator.py` re-grading pipeline validated against `healthcraft` V8 baseline

**Weeks 7–9**
- First ED decision-support deployment (1–2 physicians, observation-only mode; no patient-facing recommendations yet)
- Hu-loop traces flowing; first 100 encounters analyzed
- Compliance worker on prism-mla-h100 generating daily audit packs

**Weeks 10–12**
- First 1,000 encounter analyses
- CARD authored covering door-to-disposition delta, missed-diagnosis rate, triage-accuracy delta, handoff-quality scoring
- *Annals of Emergency Medicine* paper draft circulated internally
- Phase II SBIR pre-application draft

---

## 11. Risk register

### 11.1 Malpractice

ED is the highest-malpractice-risk specialty in US medicine. **AI augmentation must be framed exclusively as decision support, never as autonomous decision-making.** The physician sign-off-of-record is binding. The Skills directory enforces this in the `run-em-encounter.md` workflow: every output marks the physician as the decision-of-record, and outputs are formatted to support — not replace — clinical judgment.

### 11.2 FDA SaMD pathway

The 21st Century Cures Act §3060 carves out CDS that meets four criteria (does not analyze a medical image / signal / pattern; provides recommendations rather than specific directives; reasoning is transparent to the clinician; clinician can independently review the basis). GOATnote-EM stays inside this carve-out by:
- Not analyzing waveforms / images directly (that is `radslice`'s scope and a separate regulatory question)
- Surfacing CARDs with full reasoning trace and corpus citations
- Always presenting outputs as recommendations, not orders

If GOATnote ever moves to autonomous-mode (e.g. discharge instructions without physician review), it triggers Class II 510(k) — the typical EM CDS pathway — and the V_final model card needs an FDA pre-submission package.

### 11.3 HIPAA + FedRAMP

HIPAA BAA: scoped to GOATnote LLC + each pilot site. FedRAMP authorization timeline is 6–12 months; pilot proceeds under VA-authorized environment hosting (catfish + lobster + narwhal + prism-mla-h100 all need ATO inheritance through the pilot site or via a sponsor). Compliance worker on prism-mla-h100 produces the artifact pack.

### 11.4 Capacity expansion

Single ED at 75K visits/yr × $10/encounter = $750K/yr per pilot. 10 pilots = $7.5M ARR — within the GPU footprint. Beyond 10 pilots, the fleet must scale (target: $10M ARR / 4-pod baseline; add a second 4-pod cluster at $30M ARR).

### 11.5 Judge poisoning / hallucination ceiling

Per `project_v9_overlay_kappa_findings.md` — judge agreement 76.1%, κ 0.402 on `healthcraft` V9 overlay; judge hallucination is the binding ceiling. Fix path: tighten attestation contract for the EM judge (analogous to `intent_rescue_reason` from V9), and require dual-judge agreement on safety-critical encounters. CARD explicitly lists "judge-agreement" as a quality dimension.

### 11.6 Documentation-time delta could regress

If the system's draft note is worse than the physician's solo note, documentation time will *increase*. Mitigation: the per-physician LoRA captures the physician's note voice; the rank-4 adapter is trained on accepted-edit deltas, so the system's draft converges on the physician's final note within ~50 encounters.

---

## 12. Cross-references

### 12.1 In-repo

- `findings/2026-05-06-goatnote-v2-thesis/SPEC.md` (PR #104, **superseded at the domain layer**)
- `findings/2026-05-06-mobile-first-redesign-spec/SPEC.md` (PR #102, unchanged)
- `findings/2026-05-06-llm-curated-ia-spec/SPEC.md` (PR #103, unchanged — per-patient LoRA layered on top of per-physician)
- `findings/2026-05-05-skills-router-v1/CARD.md` (Cherny-cycle skills router; runs the EM skill pack)
- `findings/2026-05-05-clinical-skill-review-ci/` (auto-Claude-review CI)
- `findings/2026-05-05-adversarial-probe-spec/` (adversarial probe daemon)
- `findings/2026-05-05-clinical-rag-architecture/` (RAG architecture; reused with EM corpus)
- `findings/2026-05-04-pattern-b-spike/` (FHIR-fetch p95=11ms, ships)

### 12.2 Memory

- `healthcraft.md`, `healthcraft-details.md` (EM RL training environment, V8 results)
- `openem-corpus` entry (370 conditions, FHIR R4)
- `lostbench` entry (CEIS, MTR-001..078)
- `radslice-details.md` (EM-imaging tasks)
- `safeshift` entry (clinical scenarios)
- `project_medomni_v1_architecture_decisions.md` (FHIR / AccessPolicy / AuditEvent)
- `project_v9_overlay_kappa_findings.md` (judge hallucination ceiling)
- `nemotron_omni_tool_call_parser.md` (vLLM tool-call wiring on B300)

### 12.3 External

- ACEP Clinical Policies — https://www.acep.org/patient-care/policies/clinical-policies
- ALiEM — https://www.aliem.com/
- IDSA Practice Guidelines — https://www.idsociety.org/practice-guideline/
- AHA / ACC Guidelines — https://professional.heart.org/en/guidelines-and-statements
- 21st Century Cures Act §3060 (CDS exemption) — https://www.fda.gov/medical-devices/software-medical-device-samd
- VA SBIR — https://www.va.gov/osdbu/sb/sbir.asp
- Annals of Emergency Medicine — https://www.annemergmed.com/
- JACEP Open — https://onlinelibrary.wiley.com/journal/26884712

---

## 13. Single founder-decision asks (revised)

The four asks from PR #104, re-pointed to EM:

1. **Pilot:** Houston VA Medical Center ED via VA SBIR Phase I + SDVOSB sole-source parallel. Backup: Boise VA. **Approve / counter / defer.**
2. **Pricing:** per-encounter $5–$15 (banded by ESI) + missed-diagnosis-rate outcome bonus (≥30% reduction at 6 months) + platform fee aligned to VA CCN. **Approve / counter / defer.**
3. **Credentialing:** Board-certified EM physician on retainer by week 2; recruit via ACEP Career Center + SAEM Industry Advisory Council. **Approve / counter / defer.**
4. **Paper:** *Annals of Emergency Medicine* primary; NEJM AI parallel; *JACEP Open* fallback. **Approve / counter / defer.**

---

## 14. Supersession notice

This document supersedes PR #104 at the **domain layer only.** The architectural skeleton (Sections 2.1–2.8 above) remains in force. Any architecture-layer references in PR #104 remain canonical; any domain-layer references in PR #104 (EMDR, scalingupemdr.com, EMDRIA, PCL-5, Charleston VAMC PTSD, ISTSS) are **deprecated** and replaced by the EM-correct equivalents in this SPEC.

If PR #104 is merged before this SPEC, this SPEC stands as the corrected domain layer atop it. If this SPEC is merged first, PR #104 should be re-tagged as "architecture-only superseded by EM correction" prior to merge.

— END SPEC —
