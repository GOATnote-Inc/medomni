# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial public scaffolding for `GOATnote-Inc/medomni`
- Apache 2.0 license
- README, CONTRIBUTING, SECURITY, CODE_OF_CONDUCT
- `.github/` workflows (lint, test, manifest-determinism), CODEOWNERS, issue templates, PR template
- Reproducibility manifest emitter (`scripts/emit_manifest.py`) with byte-deterministic output verified
- Multi-pod health monitor (`scripts/health_check_all_pods.sh`) covering Brev B300/H200/H100 + RunPod prism
- CARD comparator (`scripts/compare_cards.py`) for diff-of-runs with verdict labels
- Held-out chemoprevention fixture set (initial 6 fixtures, citation-disjoint from the original tamoxifen rubric)
- Multimodal demo fixtures (ECG, warfarin pill, CXR pneumonia, lung-crackles audio) with Omni smoke-test results
- v1.0 north-star architecture spec (SPEC.md), positioning brief, revenue model, demo runbook, pitch deck outline
- Methodology status doc documenting the rubric-corpus circularity / same-family-judge red-team finding
- Durable RunPod ssh-proxy PTY-echo guard in `scripts/_runpod_ssh.sh` (post-2026-04-29 incident)

### Results (consolidated v1.0 baseline)

- Held-out 6-fixture mean: **0.385**, deterministic across N=3 seeded trials
- Comparator verdict vs v0 (0.273): **PASS** — significant lift, no major regression (+0.112 absolute, +41% relative)
- Per-axis: instruction_following +0.269, accuracy +0.132, completeness +0.122, context_awareness +0.120
- Reproducibility manifest sha256 byte-stable across re-emit: `560baccbb706...`

### Known limitations

- Held-out PASS gate (≥0.45) not yet met; gap −0.065 closes via Phase 1.6 broader corpus, Phase 1.7 fixture scaling N=6→30, Phase 2.4 PrimeKG factual-graph integration
- vllm[audio] extras pending on B300 omni container (blocks DEMO Scene 3 audio path)
- TRT-LLM-FP8 judge engine on RunPod H100 prism deferred to v0.2 (cp311 wheel + deep_gemm source-build issues)
- Phase 2.2 ↔ B300 tunnel from laptop blocked on RunPod proxy ssh `-L` limitation; user pubkey install in RunPod console UI unblocks

[Unreleased]: https://github.com/GOATnote-Inc/medomni/compare/HEAD...HEAD
