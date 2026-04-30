# Contributing to MedOmni

Thanks for your interest. MedOmni is a medical-AI codebase, which means safety, reproducibility,
and methodology rigor are non-negotiable. This document describes how to contribute in a way that
respects those constraints.

## Code of Conduct

By participating you agree to abide by the [Contributor Covenant 2.1](CODE_OF_CONDUCT.md).

## Before you start

Read these in order, every session:

1. [`CLAUDE.md`](CLAUDE.md) — operating charter for AI agents and humans working in this repo.
   §0 (RunPod PTY-echo doctrine), §1 (isolation contract), §2 (sovereignty) are load-bearing.
2. [`findings/research/2026-04-29-medomni-v1-northstar/SPEC.md`](findings/research/2026-04-29-medomni-v1-northstar/SPEC.md)
   — the architectural north star.
3. [`findings/research/2026-04-29-medomni-v0/methodology-status.md`](findings/research/2026-04-29-medomni-v0/methodology-status.md)
   — the methodology audit. Understand rubric-corpus circularity, same-family judge bias, and N=1
   reporting hygiene before touching the eval harness.

## Development setup

```bash
# Clone + bootstrap
git clone https://github.com/GOATnote-Inc/medomni.git
cd medomni

# Recommended: uv (10-100x faster than pip resolver)
uv venv && uv sync

# OR: classic pip + venv
make venv

# Wire pre-commit hooks (you must do this before your first commit)
make pre-commit-install

# Lint + test (laptop, no GPU required)
make lint
make test
```

## Style

- **Formatting**: `ruff format` — line length 100, target `py310`.
- **Linting**: `ruff check` with rules `E F I W UP B`. Per-file-ignores in `pyproject.toml`.
- **Imports**: `ruff` handles isort. No manual reordering.
- **Docstrings**: Google-style for any new public function. Brevity over completeness.
- **Type hints**: required on new public APIs. Internal helpers may skip.

## Tests

- **Unit tests** — laptop-runnable, no network, no GPU. Default `make test` target.
- **Integration tests** (`@pytest.mark.integration`) — require live Brev pod via SSH.
  Skipped in CI; run by the maintainer locally before merge.
- **GPU tests** (`@pytest.mark.gpu`) — require CUDA + a Hopper or Blackwell GPU.
- **H100 / H200 tests** (`@pytest.mark.h100`, `@pytest.mark.h200`) — pod-specific.
- **Training tests** (`@pytest.mark.training`) — long-running NeMo PEFT LoRA.

CI runs `pytest -m "not integration and not gpu and not h100 and not h200 and not training"` on
GitHub-hosted runners. PRs that need integration coverage must attach a manifest hash from a
local re-bench (see "Manifest discipline" below).

## Manifest discipline (load-bearing)

This is the rule that distinguishes MedOmni from a generic eval harness.

**Every PR that touches the inference path must attach a fresh manifest hash from a re-bench.**

The inference path is everything that affects the answer JSON: serving config, retrieval, rerank,
guardrails, persona prompts, judge config, fixture set. If your PR touches any of those:

```bash
# 1. Run a representative bench
make ci-medomni                              # full closed-loop on configured pods

# 2. Generate the manifest from the artifact
make manifest ARTIFACT=results/<run-id>/heldout.json

# 3. Verify byte-determinism (must pass)
make manifest-verify ARTIFACT=results/<run-id>/heldout.json

# 4. Paste the SHA256 into your PR description under "Manifest hash"
```

PRs without a manifest hash that touch inference will not merge. PRs that don't touch inference
(docs, scripts that don't change the answer surface, test infrastructure) are exempt — say so
in the PR description.

## Commit hygiene

- **Author email**: `b@thegoatnote.com` for project-related commits. (Personal email is for personal
  projects only — see project memory.)
- **Conventional Commits**: type prefix required.
  - `feat:` new feature on the inference path
  - `fix:` bug fix
  - `docs:` documentation only
  - `test:` test infrastructure
  - `refactor:` no behavior change
  - `perf:` performance change with measurement
  - `ci:` workflows / pre-commit
  - `chore:` everything else
- **Stage by name**: never `git add -A` or `git add .`. Stage files explicitly.
  Evaluation artifacts under `results/`, `data/embeddings/`, `data/lazygraph/`, `nemo/checkpoints/`,
  `trt_engines/` accumulate fast and must not be staged.
- **Pre-commit must pass clean**: no `--no-verify`. If pre-commit fails, fix the issue and create a
  new commit (not `--amend`, especially after a hook failure — see CLAUDE.md commit safety).
- **One Co-Authored-By line per Claude commit**:
  ```
  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  ```

## Secrets

The `.env` file is gitignored. The `.env.example` lists the only two secrets MedOmni needs:
`HF_TOKEN` and `BREV_PEM_PATH`. If you find yourself wanting to add `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, etc., **stop and re-read CLAUDE.md §2**. Sovereignty by
construction means zero cloud LLM keys in any code path.

The pre-commit `no-cloud-llm-keys` hook blocks references to those keys outside of explicitly
allowlisted files. The `detect-secrets` baseline + `gitleaks` CI run as defense-in-depth.

If you discover you've staged a secret:

1. **Do not** commit. Roll back the staging area.
2. Rotate the secret immediately (see SECURITY-INCIDENTS.md for the 2026-04-29 precedent).
3. Surface to the maintainer.

## Reporting bugs

Use [`.github/ISSUE_TEMPLATE/bug.yml`](.github/ISSUE_TEMPLATE/bug.yml). Include:
- Manifest hash if reproducible
- CARD.md path of the run
- Pod ID (if pod-side bug)

## Reporting reproducibility issues

Use [`.github/ISSUE_TEMPLATE/repro.yml`](.github/ISSUE_TEMPLATE/repro.yml). Reproducibility issues
take priority over feature work. Provide both manifest hashes (yours + the showcase) and the diff.

## Reporting security vulnerabilities

Don't open a public issue. See [SECURITY.md](SECURITY.md).

## License

By contributing, you agree your contributions are licensed under [Apache-2.0](LICENSE).
