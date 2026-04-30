# MedOmni — repo establishment plan

**Status**: research synthesis + decisions + scaffolding manifest. Drives the migration `prism42-nemotron-med` (private local) → `github.com/GOATnote-Inc/medomni`.
**Date**: 2026-04-29.
**Org casing (verified)**: `GOATnote-Inc` per `gh api orgs/GOATnote-Inc` (case-preserving — both `GOATnote-Inc` and `goatnote-inc` resolve, but the canonical login string is `GOATnote-Inc`).

---

## 1. Frontier-lab repo norms (April 2026 snapshot)

Synthesis of patterns observed across NVIDIA-AI-Blueprints, Anthropic, OpenAI, Hugging Face, DeepSeek, Mistral, Stanford CRFM, and the broader 2026 ML-repo landscape.

| Pattern | Who does it | Why |
|---|---|---|
| **Apache-2.0 license** | NVIDIA-AI-Blueprints/rag, mistralai/mistral-finetune, stanford-crfm/helm, huggingface/transformers, openai/openai-agents-python | Patent grant protects contributors; default permissive license for any repo with corporate contributors. ([Endor Labs license guide](https://www.endorlabs.com/learn/open-source-licensing-simplified-a-comparative-overview-of-popular-licenses)) |
| **MIT license** | openai/whisper, openai/CLIP, single-author research releases | Lighter weight, no patent grant; common for "research code, take it or leave it" releases. ([whisper LICENSE](https://github.com/openai/whisper/blob/main/LICENSE)) |
| **Custom OpenRAIL-derived** | deepseek-ai/DeepSeek-V3 | Lab-internal license, model weights only — code stays MIT/Apache. Not appropriate for code-only repos. |
| **`pyproject.toml` + `uv.lock`** | NVIDIA-AI-Blueprints/rag, openai/openai-agents-python, openai/openai-python | `uv` rapidly displacing poetry/pip-tools in 2025-26 frontier labs; 10-100× faster resolver, single binary, drop-in PEP 621. ([uv docs](https://docs.astral.sh/uv/concepts/projects/init/), [Sarah Glasmacher 2026 ML setup](https://www.sarahglasmacher.com/how-i-set-up-a-machine-learning-project-with-uv-and-pyproject-toml/)) |
| **Architecture-first README** | NVIDIA-AI-Blueprints/rag, NVIDIA/TensorRT-LLM | Diagram + component list at top; `Get Started` section second. Trains the reader to think in components. |
| **Results-first README** | OpenAI HealthBench, eval-publication repos, papers-with-code mirrors | Lead with the headline number; architecture follows. Best for benchmark-shaped repos. |
| **Quickstart-first README** | huggingface/transformers, openai/openai-python | Five-line `pip install` + minimal example before anything else. Best for SDKs. |
| **GPU pytest markers** | huggingface/transformers (`@require_torch_gpu`, `@require_torch_multigpu`), NVIDIA NeMo | Skip-decorator pattern; CI laptop-runners filter `-m "not gpu"`. ([HF testing docs](https://huggingface.co/docs/transformers/en/testing)) |
| **Self-hosted GPU runner** | huggingface/transformers (push-only), NVIDIA-AI-Blueprints (workbench-tagged) | Slow tests on a maintained GPU box; commits to `main` only. Not used for forks/PR runs (cost). |
| **`detect-secrets` baseline + gitleaks in CI** | DEV community pattern, GitGuardian-recommended, what we already use | Dual-layer: pre-commit (fast, false-negative tolerant) + CI (canonical, fail-closed). ([snyk 2025 secrets state](https://snyk.io/articles/state-of-secrets/)) |
| **release-please + Conventional Commits** | googleapis/*, increasingly Anthropic SDK, openai-python | Auto-changelog, auto-semver, auto-tag. ([release-please](https://github.com/googleapis/release-please)) |
| **Manual `CHANGELOG.md` (Keep-a-Changelog)** | DeepSeek, Mistral, Qwen | Human-curated narrative beats parsed-commit-history when the audience is researchers. |
| **CITATION.cff** | NVIDIA NeMo, stanford-crfm/helm, huggingface/transformers | Machine-readable citation; we already have one. |
| **Reproducibility manifest** | stanford-crfm/helm (per-leaderboard reproducibility doc), NVIDIA Nemotron-3 reproducibility recipe | Manifest-locked evals; we are slightly ahead with the 9-layer SHA256 emitter. ([HELM reproducibility](https://crfm-helm.readthedocs.io/)) |
| **Branch protection: required CI + no force-push** | Every Anthropic/OpenAI/NVIDIA repo at scale | Standard. Two-reviewer rule reserved for repos with > 5 maintainers. |
| **Issue + PR templates in `.github/`** | Every NVIDIA-AI-Blueprints repo, all HF repos | Bug + feature + reproducibility-issue triad. |
| **`SECURITY.md` with disclosure email** | All large labs | `security@<org>` or `security-<repo>@<org>`. |
| **`CODE_OF_CONDUCT.md` Contributor Covenant 2.1** | Modal across all labs | Boilerplate; signals professional intent. |

The 2026 modal pattern for a research-grade ML repo with a single corporate maintainer and ambitions of partner adoption: **Apache-2.0 + uv + architecture-first README + GPU markers + dual-layer secret scanning + release-please + Conventional Commits**. That stack is what NVIDIA-AI-Blueprints/rag converged on, and it's the closest reference architecture for MedOmni.

---

## 2. Decisions for `GOATnote-Inc/medomni`

| # | Question | Decision | One-line rationale |
|---|---|---|---|
| 1 | Monorepo vs polyrepo | **Monorepo** | Inference + retrieval + eval + corpus tightly coupled; same pattern as DeepSeek-V3, Qwen3, NVIDIA RAG Blueprint. |
| 2 | Dependency manager | **`uv` + `pyproject.toml` + `uv.lock`** (drop `requirements.txt`) | Frontier-lab modal in 2026; 10-100× resolver speed; single-binary; existing `pyproject.toml` already PEP 621. |
| 3 | Data / weight storage | **HF Hub for weights (revision-pinned in manifest), S3-compatible for fixtures, no Git LFS in code repo** | Manifest pin = reproducibility; LFS rot is real (HELM moved off it); weights too large for any repo storage. |
| 4 | Reproducibility | **9-layer manifest emitter (existing) + CARD.md per run + manifest-verify CI gate** | Already ahead of the published patterns; document it as differentiator. |
| 5 | CI for GPU tests | **Laptop-runners only for v0.1.0 public scaffold; GPU runner deferred until partner-funded** | Cost-aware; `pytest -m "not gpu and not integration"` on GitHub-hosted; GPU tests run by maintainer locally + recorded in CARDs. |
| 6 | Branch protection | **Required: lint + test + manifest-verify + secrets-scan; admin bypass `enforce_admins=false`; no force-push; no deletions** | Solo-dev-with-CI-gate; matches all 5 GOATnote repos per memory. |
| 7 | Pre-commit stack | **ruff + ruff-format + detect-secrets + check-added-large-files + check-yaml + check-merge-conflict + detect-private-key + custom no-cloud-llm-keys hook + custom no-prod-url-leak hook** | Already configured in this repo; carry forward verbatim. |
| 8 | License | **Apache-2.0** | Patent grant matters for medical-AI methods; ad-revenue + NVIDIA-partnership both compatible; no GPL friction with downstream consumers. |
| 9 | Commit conventions | **Conventional Commits, manual `CHANGELOG.md` (Keep-a-Changelog) for v0.1.0–v1.0.0, evaluate `release-please` at v1.1** | Human-curated changelog while audience is researchers + reviewers; auto-changelog after the project shape stabilizes. |
| 10 | README structure | **Results-first** (headline 0.385 mean / +41% lift / sovereign-stack badge) **then architecture-first** (10-stage pipeline diagram) **then quickstart** | Hybrid optimized for the NVIDIA committee + nurse-pilot reviewer + GitHub drive-by reader; results lead because the lift is the moat. |
| 11 | Test taxonomy | **`@pytest.mark.gpu`, `@pytest.mark.integration`, `@pytest.mark.h100`, `@pytest.mark.h200`, `@pytest.mark.training`** (already configured in `pyproject.toml`) — CI runs `not gpu and not integration` | Matches HF transformers convention; existing markers unchanged. |
| 12 | `.github/` | **Workflows: lint, test, manifest-verify, secrets-scan; ISSUE_TEMPLATE: bug, feature, reproducibility-issue; PULL_REQUEST_TEMPLATE; CODEOWNERS; SECURITY.md** | Modal frontier-lab `.github/` directory; nothing exotic. |
| 13 | Secret detection | **`detect-secrets` baseline (existing) + `gitleaks` in CI + `no-cloud-llm-keys.sh` custom hook + `no-prod-url-leak.sh` custom hook** | Layered defense per RunPod-PTY-echo + .env-no-read incidents; gitleaks adds CI-side canonical pass. |
| 14 | Public vs private | **Private at first push; flip public after (a) physician-advisor sign-off on persona-graph nurse content + (b) red-team pass on cited-subgraph cite rail + (c) LICENSE + SECURITY review** | OpenEvidence stayed private; Hippocratic stayed private; NVIDIA Blueprints public. We need the SOTA gate (HealthBench Hard ≥ 0.55) before going public to avoid "shipped at 0.385" misread. **Phase 1: private repo with seven-day NVIDIA-committee-only access via `gh repo edit --visibility public --collaborators`. Phase 2: public after gate.** |

---

## 3. Initial scaffolding inventory

Files added (or refreshed) for the public repo's first commit. Existing files marked `(present)`.

```
LICENSE                                 NEW    Apache-2.0 full text (~11KB)
README.md                               NEW    Results-first + architecture-first + quickstart hybrid (replaces existing private README)
CONTRIBUTING.md                         NEW    Style + tests + manifest discipline + commit hygiene
SECURITY.md                             NEW    Disclosure email + scope + RunPod-PTY-echo + .env-no-read durable doctrine
CODE_OF_CONDUCT.md                      NEW    Contributor Covenant 2.1
CHANGELOG.md                            NEW    Keep-a-Changelog format, v0.1.0 first entry
SECURITY-INCIDENTS.md                   NEW    2026-04-29 HF_TOKEN PTY-echo incident postmortem
CITATION.cff                            (present, refresh "name" + "url" for medomni)
pyproject.toml                          (present, refresh project.name → "medomni" + add uv config block)
.gitignore                              (present, no change)
.gitattributes                          NEW    Empty; placeholder for future LFS lines
.pre-commit-config.yaml                 (present, no change)
.secrets.baseline                       (present, no change)
.env.example                            (present, no change)
.github/workflows/lint.yml              NEW    ruff + ruff format
.github/workflows/test.yml              NEW    pytest -m "not gpu and not integration"
.github/workflows/manifest-verify.yml   NEW    re-emit manifest, byte-equal check
.github/workflows/secrets-scan.yml      NEW    gitleaks + detect-secrets verify
.github/ISSUE_TEMPLATE/bug.yml          NEW    Bug report template
.github/ISSUE_TEMPLATE/feature.yml      NEW    Feature request template
.github/ISSUE_TEMPLATE/repro.yml        NEW    Reproducibility issue template (manifest hash + run ID + CARD link)
.github/PULL_REQUEST_TEMPLATE.md        NEW    Checklist (manifest, no-secrets, methodology gate)
.github/CODEOWNERS                      NEW    @bGOATnote on critical paths
docs/README.md                          DEFER  Sketched only; copy from findings/research/2026-04-29-medomni-v1-northstar/INDEX.md after migration
```

Total **NEW** files: 14. Files already present and reused: 7. Files explicitly deferred to migration step: docs/ tree (sketched in §4).

---

## 4. Migration plan (`prism42-nemotron-med` → `GOATnote-Inc/medomni`)

The repo is already a fresh-import-no-shared-history per CLAUDE.md provenance line ("squash-import, no history" from `github.com/GOATnote-Inc/prism42` HEAD `e02e62dd...` on 2026-04-28). Two migration paths:

### Path A — single squash commit, fresh public history (RECOMMENDED)

Same pattern as `prism42` → `prism42` public repo (per memory `project_prism42_public.md`). The provenance line in CLAUDE.md and CITATION.cff retains the lineage; the public commit history starts clean.

```bash
# (user authorization required for every step below)

# 1. Create the public-private remote
gh repo create GOATnote-Inc/medomni \
  --private \
  --description "Sovereign nurse-first medical-LLM stack on NVIDIA's open-component stack — held-out 0.385 mean, manifest-locked reproducibility" \
  --homepage "https://github.com/GOATnote-Inc/medomni"

# 2. Create a fresh init dir (separate from prism42-nemotron-med)
mkdir -p /tmp/medomni-fresh && cd /tmp/medomni-fresh
git init -b main
git config user.email "b@thegoatnote.com"
git config user.name "Brandon Dent"

# 3. Copy the staged scaffolding only (no git history)
rsync -av --exclude=".git" --exclude=".venv" --exclude=".pytest_cache" --exclude=".ruff_cache" \
  --exclude="results/" --exclude="data/embeddings/" --exclude="data/lazygraph/" \
  --exclude="data/raw_pdfs/" --exclude="trt_engines/" --exclude="*.engine" \
  --exclude="nemo/checkpoints/" --exclude="reproducibility/captured/" \
  --exclude=".env" --exclude="third_party/" \
  /Users/kiteboard/prism42-nemotron-med/ /tmp/medomni-fresh/

# 4. Stage by name (NEVER -A or .)
git add README.md LICENSE CONTRIBUTING.md SECURITY.md CODE_OF_CONDUCT.md \
        CHANGELOG.md SECURITY-INCIDENTS.md CITATION.cff \
        pyproject.toml .gitignore .gitattributes .pre-commit-config.yaml \
        .secrets.baseline .env.example \
        Makefile CLAUDE.md DEMO.md \
        .github/ scripts/ tests/ corpus/ configs/ docs/ \
        mla/ findings/ data/seed_kg/

# 5. Pre-commit run (must pass clean before commit)
pip install pre-commit detect-secrets ruff
pre-commit run --all-files

# 6. Squash commit
git commit -m "$(cat <<'EOF'
feat: initial public scaffold — MedOmni v0.1.0

Sovereign nurse-first medical-LLM stack on NVIDIA's open-component stack
(CUDA 13.2 + RAPIDS 26.04 + cuVS + nx-cugraph + NeMo Guardrails +
NeMo Curator + TensorRT-LLM + vLLM).

Headline result: held-out 6-fixture mean 0.385, +41% over v0 baseline,
deterministic across N=3 seeded trials, manifest sha256 560baccbb706
byte-stable.

Provenance: derived from prism42-nemotron-med private local repo at
HEAD <REDACTED-LOCAL-SHA>; squash-imported, no shared history. The
medical-LLM eval harness was lifted from public prism42 with zero
prod-surface entanglement.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"

# 7. Add remote and push
git remote add origin https://github.com/GOATnote-Inc/medomni.git
git push -u origin main

# 8. Branch protection (modeled on memory § "Branch Protection 2026-03-06")
gh api -X PUT repos/GOATnote-Inc/medomni/branches/main/protection \
  -F required_status_checks.strict=true \
  -F 'required_status_checks.contexts[]=lint' \
  -F 'required_status_checks.contexts[]=test' \
  -F 'required_status_checks.contexts[]=manifest-verify' \
  -F 'required_status_checks.contexts[]=secrets-scan' \
  -F enforce_admins=false \
  -F required_pull_request_reviews.required_approving_review_count=0 \
  -F restrictions= \
  -F allow_force_pushes=false \
  -F allow_deletions=false
```

### Path B — keep `prism42-nemotron-med` as-is, push as MedOmni's private staging

Skip the rsync step; just rename the local repo, change remote, push. Faster but less hygenic — the local has 200+ commits of session-driven research that does not need to be in the public history.

**Recommendation: Path A.** Worth the hour; keeps the public history readable and `git log` useful to potential collaborators.

### Docs migration (deferred to commit 2)

After commit 1 lands, restructure as:

```
docs/
├── README.md                  # ← findings/research/2026-04-29-medomni-v1-northstar/INDEX.md (renamed)
├── SPEC.md                    # ← findings/research/2026-04-29-medomni-v1-northstar/SPEC.md
├── POSITIONING.md             # ← findings/research/2026-04-29-medomni-v1-northstar/POSITIONING.md
├── REVENUE-MODEL.md           # ← findings/research/2026-04-29-medomni-v1-northstar/REVENUE-MODEL.md
├── DEMO-RUNBOOK.md            # ← findings/research/2026-04-29-medomni-v1-northstar/DEMO-RUNBOOK.md
├── PITCH-DECK-OUTLINE.md      # ← findings/research/2026-04-29-medomni-v1-northstar/PITCH-DECK-OUTLINE.md
├── methodology.md             # ← findings/research/2026-04-29-medomni-v0/methodology-status.md
└── manifests/                 # promoted captured manifests, hand-curated
```

The original `findings/` tree stays in the repo as research provenance (frontier labs do this — see HELM `docs/heim.md` history). Don't delete; just promote the Cardinal documents.

---

## 5. Branch protection + CI plan, first 90 days

**Day 0 — repo creation, private**

- Branch protection: required status checks `lint`, `test`, `manifest-verify`, `secrets-scan`. `enforce_admins=false`. No force-push. No deletions.
- CODEOWNERS: `@bGOATnote` on `.github/`, `CLAUDE.md`, `scripts/_runpod_ssh.sh`, `scripts/serve_*.sh`, `Makefile`, `LICENSE`, `SECURITY.md`.
- No required reviewers (solo dev + automated gates).

**Day 0–14 — internal hardening**

- Run `gitleaks detect --source .` against the full history; surface findings. Should be zero (we're starting fresh from squash).
- `pre-commit run --all-files` green.
- `make test` green on laptop.
- `make manifest-verify ARTIFACT=results/ci-medomni-heldout-consolidated-20260429-173557/heldout.json` green.
- CARD.md links resolve.

**Day 14–30 — partner-only access**

- `gh repo edit GOATnote-Inc/medomni --add-collaborator <NVIDIA-committee-member>` (per email)
- Tag `v0.1.0`; release notes from CHANGELOG.md.

**Day 30–60 — preparation for public**

- Physician-advisor review of persona-graph nurse content (per POSITIONING §5 moat point 5).
- Red-team pass: NR-Labs jailbreak corpus on output-rail; cited-subgraph cite-rail adversarial probes.
- Phase 2.4 PrimeKG integration lands; held-out re-bench targets ≥ 0.45.

**Day 60–90 — public flip (gated)**

- HealthBench Hard re-bench mean ≥ 0.55 → `gh repo edit GOATnote-Inc/medomni --visibility public`.
- Or held below the gate → stay private; ship to invited reviewers only.

The public-flip date is **gated on the SOTA-accuracy bar from POSITIONING §7 rejection #5**, not on a calendar.

---

## 6. First three commits

Each commit is a complete, reviewable atomic unit.

**Commit 1 — `feat: initial public scaffold — MedOmni v0.1.0`** (the one in §4 path A step 6)

- Adds: 14 new scaffolding files (LICENSE, README, CONTRIBUTING, SECURITY, CODE_OF_CONDUCT, CHANGELOG, SECURITY-INCIDENTS, .gitattributes, .github/* (8 files))
- Refreshes: pyproject.toml (project.name → medomni, add uv.lock support stanza), CITATION.cff (name → MedOmni)
- Carries forward: existing CLAUDE.md, .gitignore, .pre-commit-config.yaml, .secrets.baseline, .env.example, scripts/, tests/, corpus/, configs/, mla/, data/seed_kg/, findings/, Makefile, DEMO.md
- Net: ~14 files added, ~2 refreshed, ~30K LOC carried

**Commit 2 — `docs: promote north-star findings to docs/ tree`**

- Moves Cardinal documents from `findings/research/2026-04-29-medomni-v1-northstar/` to `docs/`
- Updates internal links
- Adds `docs/README.md` (renamed INDEX.md)
- Keeps `findings/` as historical provenance

**Commit 3 — `feat: bench-results showcase + comparator artifacts`**

- Promotes `results/ci-medomni-heldout-consolidated-20260429-173557/CARD.md` + `MANIFEST.yaml` to `results/showcase/v0.1.0/`
- Adds `scripts/compare_cards.py` example invocation + golden-output `delta-CONSOLIDATED.md`
- Pins `corpus/clinical-fixtures-heldout/MANIFEST.md` as the held-out evidence base
- Includes the NIM-incident postmortem reference in CHANGELOG (link to SECURITY-INCIDENTS.md)
- Net: ~6 files

After commit 3, the repo's first impression is: results-first README → headline number → cited subgraph → architecture diagram → quickstart → CARD link → manifest verify → SOTA roadmap.

---

## 7. Open user judgement calls

| Topic | Default I assumed | Alternative | Why surface |
|---|---|---|---|
| Repo name casing | `medomni` (lowercase) | `MedOmni` | GitHub URL slugs are lowercase by convention; user may want display-cased. |
| Initial visibility | **Private**, flip public after SOTA gate | Public from day 1 | Headline 0.385 < SPEC §6 gate 0.55; public-now risks misread. |
| License | **Apache-2.0** | MIT, dual community/commercial, source-available | Apache-2.0 patent grant matters for medical-AI methods + downstream NVIDIA contributors. |
| Disclosure email | `security@thegoatnote.com` placeholder | `b@thegoatnote.com` direct | Org email looks more mature; needs to actually exist or auto-forward. |
| `release-please` adoption | **Defer** to v1.1 | Adopt at v0.1 | Manual changelog reads better while project is research-shaped; auto-changelog wins after stabilization. |
| GPU CI runner | **None** for v0.1 | Self-host one B300 worker | Cost-aware default; partner-funded later. |
| Persona-graph schema disclosure | Document in `docs/SPEC.md` | Withhold | Moat point 1 (POSITIONING §5) — graph data is the moat. Schema doc is fine; data files stay in `data/persona_graph/` private. |

These are presented in the report so the user can correct any of them before commit 1 happens.

---

## 8. Sources

- [NVIDIA-AI-Blueprints/rag — reference architecture](https://github.com/NVIDIA-AI-Blueprints/rag)
- [openai/whisper — MIT license example](https://github.com/openai/whisper)
- [openai/openai-agents-python — pyproject + uv example](https://github.com/openai/openai-agents-python/blob/main/pyproject.toml)
- [stanford-crfm/helm — reproducibility pattern](https://github.com/stanford-crfm/helm)
- [huggingface/transformers — testing & GPU markers](https://huggingface.co/docs/transformers/en/testing)
- [deepseek-ai/DeepSeek-V3 — custom OpenRAIL license](https://github.com/deepseek-ai/DeepSeek-V3)
- [uv — Python project init](https://docs.astral.sh/uv/concepts/projects/init/)
- [release-please](https://github.com/googleapis/release-please)
- [gitleaks](https://github.com/gitleaks/gitleaks)
- [Snyk 2025 state of secrets](https://snyk.io/articles/state-of-secrets/)
- [Endor Labs license guide](https://www.endorlabs.com/learn/open-source-licensing-simplified-a-comparative-overview-of-popular-licenses)
- [Contributor Covenant 2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/)
- [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/)
- [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/)
