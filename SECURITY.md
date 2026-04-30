# Security Policy

## Reporting a vulnerability

If you discover a security vulnerability in MedOmni, **do not open a public issue**.

Email **security@thegoatnote.com** with:

- A description of the vulnerability
- Steps to reproduce
- Affected versions / components
- Suggested mitigation if you have one

We will acknowledge receipt within 72 hours and aim to provide a fix or remediation plan within
14 days for high-severity issues.

## Scope

In scope:

- The MedOmni codebase under this repository
- The reproducibility manifest emitter (`scripts/emit_manifest.py`)
- The pre-commit and CI gates (`.pre-commit-config.yaml`, `.github/workflows/`)
- The sovereignty contract (CLAUDE.md §2) — any code path that introduces a cloud LLM API call
  is in scope as a security defect
- The persona-tagged-graph schema (`docs/SPEC.md` §5.5)

Out of scope:

- Vulnerabilities in upstream NVIDIA components (CUDA, RAPIDS, NeMo, TensorRT-LLM, vLLM) —
  please report those directly to NVIDIA
- Vulnerabilities in upstream model weights — please report those to Hugging Face / the weight
  publisher
- Theoretical attacks against medical-AI systems in general — the LICENSE includes the standard
  warranty disclaimer; medical decisions remain the responsibility of the credentialed clinician

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | Yes |
| < 0.1.0 | No |

## Operational security doctrine (durable rules)

These rules are extracted from real incidents in this project and the broader GOATnote stack.
They are non-negotiable:

### Never read `.env` value contents directly

Reading `.env` loads its contents into the conversation context, which exposes secrets to LLM
agents and to any logging surface. Use awk to inspect key names only:

```bash
awk -F= '/^[A-Z_]+=/ {print $1, "len:", length($2)}' .env
```

Do not use `cat .env`, `grep -E 'KEY=|TOKEN=|SECRET=' .env`, or any pipeline that interpolates
the value into a shell variable that gets re-exported.

### Never push secrets through SSH proxies that allocate a PTY

The 2026-04-29 incident (see [SECURITY-INCIDENTS.md](SECURITY-INCIDENTS.md)) leaked an HF_TOKEN
through `ssh.runpod.io`'s required PTY allocation. PTYs echo stdin to stdout server-side; the
client cannot disable this. Heredoc, base64, inline-env, and stdin-pipe all leak.

The durable mitigation is `scripts/_runpod_ssh.sh`, which carries a hard secret-grep guard. It
refuses to forward any command body matching `(API_KEY|SECRET|PASSWORD|TOKEN)=` patterns or
specific known-secret prefixes (`hf_*`, `nvapi-*`, `sk-*`, `sk-ant-*`, `xai-*`, `AIza*`, `ghp_*`,
`ghs_*`).

Provision pod secrets via:
- The pod provider's console environment-variable UI, or
- A separate user-controlled SSH session that is not under agent control

Never via a Claude-driven shell.

### Never commit cloud LLM API keys to any path

Per [CLAUDE.md](CLAUDE.md) §2 (sovereignty contract), MedOmni runs with exactly two secrets:
`HF_TOKEN` (Hugging Face read-only) and `BREV_PEM_PATH` (filesystem path, not a value).

`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `XAI_API_KEY`, etc. are blocked by:
- The `no-cloud-llm-keys.sh` pre-commit hook
- The `detect-secrets` baseline
- The `gitleaks` CI workflow
- A grep CI gate on every PR

If you find yourself adding one of those keys to a script or an env-template, the design is
wrong. The judge runs locally on H100. The serve runs locally on H200. External keys defeat the
entire premise.

### Never run secret-printing commands

The 2026-04-27 prism42 incident and the 2026-04-29 medomni incident both involved commands that
printed env-var contents to a shell that was being captured by a logging surface. Do not run:

- `systemctl show ... --property=Environment`
- `cat /proc/*/environ`
- bare `printenv` or `env` (without filter)
- `cat .env`
- `grep -E 'KEY=|TOKEN=|SECRET='` over a file containing values

Use the awk pattern above to read key names without values.

### Pre-commit must pass clean

No `--no-verify`. If a pre-commit hook fails, fix the underlying issue and create a new commit
— not `--amend`. After a hook failure, the commit did not happen, so `--amend` would modify the
previous commit (potentially losing work).

### Branch protection holds

Required status checks: `lint`, `test`, `manifest-verify`, `secrets-scan`. No force-push to
`main`. No deletions. `enforce_admins=false` allows admin bypass for solo-dev sprints; this is
intentional and reviewed quarterly.

## Threat model

MedOmni is a clinical decision-support codebase. Its threat model is shaped by:

1. **Hallucinated medical content** — addressed by the constrained-decoding cite rail
   (`docs/SPEC.md` §5.3 stage 9) and the output-rail Nemotron-Content-Safety-Reasoning-4B with
   clinical policy.
2. **PHI leak** — addressed by sovereignty (no cloud LLM calls), HIPAA-by-construction, and the
   airplane-mode demo. PHI never leaves the pod.
3. **Adversarial jailbreak** — addressed by the input-rail NemoGuard JailbreakDetect.
4. **Methodology-side circularity** — addressed by held-out fixtures with non-overlapping evidence
   base, cross-family judge, and N≥3 seeded trials with deterministic decoding.
5. **Supply-chain compromise** — addressed by revision-pinned weights in the 9-layer manifest,
   `detect-secrets` + `gitleaks` dual-layer secret scanning, and pinned dependency versions in
   `pyproject.toml` + `uv.lock`.

## Disclosure timeline

We follow standard 90-day coordinated disclosure. After acknowledgment:

- Day 0–14: triage + assignment
- Day 14–60: fix development + testing
- Day 60–90: coordinated release
- Day 90+: public disclosure (with credit to the reporter, if they wish)
