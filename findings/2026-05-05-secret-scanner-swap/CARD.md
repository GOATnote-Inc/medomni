# Secret-scanner swap — gitleaks → TruffleHog OSS

## Trigger

`gitleaks/gitleaks-action@v2` switched to a paid-license model in 2025. medomni's
`secrets-scan` job in `.github/workflows/lint.yml` had it pinned with
`continue-on-error: true` (already non-blocking) but the failure annotation was
loud and the tool was contributing zero scanning value (no license = no scan).

User directive 2026-05-05: replace with cutting-edge OSS that aligns with
NVIDIA's own conventions; explore via parallel agent research; commitment to
"best in class everywhere".

## Method

Two parallel research agents dispatched in iter-9 of the babysitter /loop:

- **Agent A (broad OSS landscape, May 2026)**: surveyed TruffleHog, Kingfisher,
  Titus, Nosey Parker, detect-secrets, Semgrep CE, gitleaks CLI, gitGraber.
  Verified license + last-release dates against actual repos via WebFetch.
- **Agent B (NVIDIA OSS audit)**: read `.github/workflows/*.yml` from 17 NVIDIA
  + RAPIDS + Triton + vLLM repos to find the de-facto NVIDIA convention.

## Findings

### Agent A — landscape

| Tool | License | Last release | Verifier? | NVIDIA-org adoption |
|---|---|---|---|---|
| **TruffleHog** | AGPL-3.0 (binary) | v3.95.2 (2026-04-21) | **Y** — 700+ live API verifiers | High in `NVIDIA/*` (bionemo-framework, Model-Optimizer, terraform-provider-ngc, nvflow, dsx-github-actions composite) |
| Kingfisher (MongoDB) | Apache-2.0 | v1.98.0 (2026-04-30) | **Y** — 484 of 942 rules | None observed |
| Titus (Praetorian) | Apache-2.0 | v1.1.31 (2026-04-28) | **Y** — 487 rules | None observed |
| detect-secrets (Yelp) | Apache-2.0 | v1.5.0 (2024-05-06; commits Apr 2026) | N (regex+entropy) | Universal in `NVIDIA-NeMo/*` (NeMo, Run, Curator, Evaluator + canonical reusable workflow `FW-CI-templates/_secrets-detector.yml`) |
| Nosey Parker | Apache-2.0 | v0.24.0 (2024-05-08) | N | None — stale, superseded by Titus |
| Semgrep CE secrets | LGPL + Semgrep Rules License v1.0 | active | gated to paid tier | None observed |
| gitleaks CLI | MIT | active | N | Some `NVIDIA/nvcf-*` (will hit same paid issue) |

### Agent B — NVIDIA convention

| Repo | Secret scanner |
|---|---|
| `NVIDIA-NeMo/NeMo` | detect-secrets via `secrets-detector.yml` |
| `NVIDIA-NeMo/Run` | detect-secrets via shared `FW-CI-templates@v0.74.0` |
| `NVIDIA-NeMo/Curator` | detect-secrets via shared `FW-CI-templates@v0.70.0` |
| `NVIDIA-NeMo/Evaluator` | detect-secrets |
| `NVIDIA/bionemo-framework` | TruffleHog (`trufflehog.yml`) |
| `NVIDIA/Model-Optimizer` | TruffleHog (`trufflesecurity/trufflehog@v3.90.5`) |
| `NVIDIA/terraform-provider-ngc` | TruffleHog (`@v3.94.3`) |
| `NVIDIA/nvflow` | TruffleHog |
| `NVIDIA/dsx-github-actions` | TruffleHog (composite action `trufflehog-scan/action.yml`) |
| `NVIDIA/cccl` | Black Duck SCA (commercial, Synopsys) — exception |
| `triton-inference-server/server`, `vllm-project/vllm`, RAPIDS | none observed |

## Decision

**Keep `detect-secrets`. Replace `gitleaks-action@v2` with `trufflesecurity/trufflehog@v3.95.2`.**

Rationale: medomni inherits from BOTH NVIDIA conventions simultaneously —
detect-secrets matches `NVIDIA-NeMo/*` (medomni uses NeMo Framework PEFT,
NeMo Guardrails, NeMo Curator), TruffleHog matches `NVIDIA/bionemo-framework`
(medical/biological NVIDIA flagship — closest cousin to medomni's medical-LLM
positioning). Defense in depth: detect-secrets catches regex/entropy patterns;
TruffleHog actively verifies via API calls. The two have non-overlapping rule
sets and different fidelity tiers.

AGPL-3.0 on TruffleHog binary is not a concern: AGPL obligations attach on
distribution or network-service offering, neither of which apply when a CI
workflow runs the binary against our own code on GitHub-hosted runners.

## Implementation

`.github/workflows/lint.yml` `secrets-scan` job — second step swapped:

```yaml
- name: TruffleHog OSS — verifier-based secret scan
  uses: trufflesecurity/trufflehog@v3.95.2
  with:
    base: ${{ github.event.repository.default_branch }}
    head: HEAD
    extra_args: --results=verified,unknown
```

`--results=verified,unknown` filters out unverified false-positive matches,
which is the main FP class on a public repo.

`SECURITY.md` and `CONTRIBUTING.md` updated to describe the new dual-layer.

## What's NOT in this CARD

- **Pre-commit TruffleHog hook**: optional follow-up. NeMo doesn't use one for
  detect-secrets locally; relies on the CI step. Same pattern fits here.
- **`.secrets.baseline` migration to NeMo-canonical name**: today's workflow
  uses `.detect-secrets.baseline` and falls back to `--all-files` when missing.
  NeMo uses `.secrets.baseline`. Not blocking; rename is cosmetic.
- **OpenSSF Scorecard / SLSA / sigstore cosign**: zero NVIDIA repos audited use
  these. Not adopting in this PR; surface to user as a separate consideration.

## Sources

Agent transcripts archived under `tasks/a3b8f42fd20a2767f.output` (landscape)
and `tasks/ad39139609ae6660b.output` (NVIDIA audit) on the babysitter laptop.
Concrete file URLs cited in agent reports; not duplicated here.

Key URLs:
- https://github.com/trufflesecurity/trufflehog/releases — version pinned `v3.95.2`
- https://github.com/NVIDIA/bionemo-framework/blob/main/.github/workflows/trufflehog.yml
- https://github.com/NVIDIA/Model-Optimizer/blob/main/.github/workflows/code_quality.yml
- https://github.com/NVIDIA/dsx-github-actions/.github/actions/trufflehog-scan/action.yml
- https://github.com/NVIDIA-NeMo/FW-CI-templates/blob/main/.github/workflows/_secrets-detector.yml
- https://gitleaks.io/gitleaks-action/commercial-license.html
