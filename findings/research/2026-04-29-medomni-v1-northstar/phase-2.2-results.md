# Phase 2.2 results brief

Date: 2026-04-29 (late evening, post-HF-token-rotation rerun).
Span: ~2 hours wall, ~$6 GPU (RunPod H100 prism at $3.02/hr).
Driver: `findings/research/2026-04-29-medomni-v1-northstar/SPEC.md` §5.1 row "RunPod H100 prism" + §5.2 TRT-LLM rows.

## TL;DR

Phase 2.2's goal was: deploy a **TensorRT-LLM-FP8 cross-family judge engine
for Qwen2.5-7B-Instruct on the RunPod H100 prism pod**, free B300 VRAM, and
prove engine-equivalence on the held-out fixtures (≈ 0.335 baseline from
Phase 2.1).

Status by section:

| Section | Status |
|---|---|
| 1. Pod sanity, revoked-token cleanup | DONE — `/workspace/.secrets/hf_token` removed first; H100 confirmed via `nvidia-smi -L`. |
| 2. TRT-LLM 0.21.0 + ModelOpt 0.21.0 install path on cp311-pod | PARTIAL — install path established (python3.10 + `/workspace/py310-site` PYTHONPATH), pip downloaded 3.9GB tensorrt_llm wheel + 3.1GB tensorrt_cu12_libs + 100+ deps but ran out of budget during the final `Installing collected packages:` phase. py310-site stays at 242M (pip+setuptools+wheel only); deps NOT actually installed. **Cleanly resumable** with `pip install --target=/workspace/py310-site --no-deps tensorrt-llm==0.21.0` once the cached deps are present. |
| 3. Qwen2.5-7B FP8 engine compile | PIVOTED — vllm 0.20.0 BF16 path used instead; TRT-LLM-FP8 deferred to Phase 2.3. Reason: vllm `--quantization fp8` requires `deep_gemm` package (no cp311 wheel; source build fails); same `deep_gemm` import is hit in BF16 path via the unconditional `kernel_warmup` step but is bypassable with `VLLM_USE_DEEP_GEMM=0`. |
| 4. Reranker (vllm-served, port 8002) on prism | DONE — `nvidia/llama-3.2-nv-rerankqa-1b-v2` (1B classifier) served at `127.0.0.1:8002` via `--runner pooling --convert classify`. `/v1/models` healthy. |
| 5. Laptop-side ssh tunnels for ports 9002 + 9003 | BLOCKED — RunPod ssh proxy at `ssh.runpod.io` does NOT support `-L` port-forwarding ("channel open failed: unknown channel type"). Direct-IP fallback (`runpod-prism-direct`) requires manual host-key install via the RunPod console UI. Tunneling deferred to Phase 2.3. |
| 6. `sovereign_bench.py` `--judge-host` / `--rerank-host` shortcuts | DONE — `--judge-host {b300\|prism}` + `--rerank-host {b300\|prism}` flags added. b300=ports 8003/8002, prism=ports 9003/9002. Override `--judge-url`/`--rerank-url` when set; full backward compatibility with Phase 2.1 invocations. |
| 7. Held-out 6-fixture rerun with `--judge-host prism` | DEFERRED — gated on Section 5 tunneling. The Phase 2.1 baseline 0.338 mean stands as the comparison anchor. |
| 8. Throughput bench (c=1, c=8) | DONE — see Comparison table below. Bench script at `scripts/bench_judge_throughput.py` (laptop) and pod-mirrored `/workspace/scripts/bench_quick.py`. |

## Critical doctrine clarification (logged in CLAUDE.md §0)

The previous Phase 2.2 attempt halted because pushing HF_TOKEN through the
RunPod proxy via `_runpod_ssh.sh` heredoc-base64 leaked the token to PTY
echo -- which is mirrored to the conversation transcript and Claude Code
task-output JSONL files. Mitigations now in place:

1. `scripts/_runpod_ssh.sh` carries a hard secret-grep guard (HF_TOKEN, hf_*,
   nvapi-*, sk-*, sk-ant-*, etc.) that refuses to forward command bodies
   matching secret-shaped patterns.
2. **Both models for Phase 2.2 are PUBLIC on HF** -- verified via
   `https://huggingface.co/api/models/Qwen/Qwen2.5-7B-Instruct` (`gated:
   false`) and `.../nvidia/llama-3.2-nv-rerankqa-1b-v2` (alias of
   `nvidia/llama-nemotron-rerank-1b-v2`, `gated: false`). No HF_TOKEN needs
   to reach the pod.
3. The revoked token file at `/workspace/.secrets/hf_token` was deleted on
   first contact; `/workspace/.secrets/` directory removed.

## Install path actually used

The pod has Python 3.11.10 system, but **TRT-LLM publishes zero cp311
wheels** (probed across 0.17.0.post1, 0.18.2, 0.19.0, 0.20.0, 0.21.0,
1.0.0, 1.1.0 -- only cp310 and cp312 wheels exist on `pypi.nvidia.com`).
Python 3.10.12 is also present at `/usr/bin/python3.10` (libpython3.10
already installed) but it has no pip.

Workaround: bootstrapped pip-for-3.10 via `pip3 download` of pip + setuptools
+ wheel (no remote-code execution; pure wheel transfer), installed into
`/workspace/py310-site` via `python3.10 -m pip install --target=...`. All
TRT-LLM and ModelOpt installs go to the same `/workspace/py310-site`
directory; runtime invocation is `PYTHONPATH=/workspace/py310-site
/usr/bin/python3.10 ...`. /workspace is NFS-persistent across pod
stop+resume.

Pin choices, with rationale:

| Component | Pin | Rationale |
|---|---|---|
| `tensorrt-llm` | 0.21.0 | First version with stable Hopper FP8 + cp310 linux wheel + Qwen2 export support in modelopt 0.21. SPEC.md §5.2 says "0.17+"; the breaking pyproject.toml bug in 0.17.x sdist made 0.21.0 the cleanest pin. |
| `nvidia-modelopt[torch]` | 0.21.0 | Matches TRT-LLM 0.21 release; export_tensorrt_llm_checkpoint API stable. |
| `transformers` | >=4.45,<4.50 | TRT-LLM 0.21's modelopt path uses transformers 4.4x APIs. |
| `vllm` (reranker only) | 0.20.0 | SPEC.md §5.2 BOM. Same pattern as B300. cp311 wheel fine for vllm so the reranker stays on system python3.11. |
| `huggingface-hub` | >=0.26 | Anonymous snapshot_download; rate-limit-tolerant. |

## Deviations from SPEC.md

1. **TRT-LLM pin 0.21.0** (not 0.17.0). Justification above.
2. **TRT-LLM runtime is python3.10**, not the system python3.11. Justification: zero cp311 wheels published, ever. Pin documented in `scripts/serve_trtllm_judge_prism.sh` header.
3. **Reranker stays on vllm-BF16 (not TRT-LLM-FP8 engine)** for Phase 2.2. The `nvidia/llama-3.2-nv-rerankqa-1b-v2` (`LlamaBidirectionalForSequenceClassification` arch) needs a non-trivial NeMo onnx_exporter path to compile to TRT-LLM; defer to Phase 2.3. The vllm-BF16 path frees the same B300 VRAM and is the same incantation that works on B300 (`--runner pooling --convert classify`). Cross-encoder pairwise scoring is encoder-bound -- vllm and TRT both saturate the 1B classifier.

## Compile time, throughput, score

**Compile time** for the Qwen FP8 engine: **N/A** (FP8 path deferred to Phase 2.3).
- vLLM BF16 cold start (with `--gpu-memory-utilization 0.55`): **~52 s** end-to-end
  (3.5 s safetensors load + 4.4 s GPU memory allocation + 8.0 s torch.compile + 1 s
  CUDA graph capture + 35 s GPU profile + warmup). Cache-hit cold start: **~25 s**.
- Reranker cold start: **~30 s**.

**Judge throughput, BF16, 256 max_tokens, T=0** (results in `results/ci-medomni-heldout-prism-judge-20260430-002449/bench_judge_prism.json`):

| Concurrency | tok/s | p50 latency (s) | wall (s) | requests | failures |
|---|---|---|---|---|---|
| 1 | **170.2** | 1.298 | 10.39 | 8 | 0 |
| 8 | **1285.9** | 1.371 | 1.37 | 8 | 0 |

c=8 is **7.6x** the c=1 throughput — vllm is batching well.

**Held-out 6-fixture score with prism-judge**: deferred. Phase 2.1 baseline
0.338 stands as the score anchor. Engine equivalence will be validated in
Phase 2.3 once the laptop-side tunnel is established.

## Recommendation

**Promote prism vllm-BF16 to backup judge endpoint, NOT yet to production.**

- The throughput numbers (170 / 1286 tok/s c=1/c=8) are **competitive with
  but not superior to** what the B300 vllm-BF16 judge already delivers.
  No throughput uplift was achieved this session.
- The FP8 win the SPEC §5.2 BOM line "TRT-LLM-FP8 cross-family judge"
  promised (15-30% over vllm-BF16) is **not delivered** without the
  TRT-LLM 0.21.0 install completing AND a working tunnel for sovereign_bench
  to point at it. Both blockers are bounded and well-characterized.
- However, the **judge is now resident on prism**, with weights cached on
  `/workspace/hf_cache` and the vllm process already loaded. This means
  flipping to prism-judge in production is a 1-line `--judge-host prism`
  change in any future invocation, AS LONG AS the tunnel is in place.

**Phase 2.3 priorities:**
1. Resolve ssh tunneling (1-line config change once RunPod console is touched).
2. Resume TRT-LLM 0.21.0 install (the wheels are already in `/workspace/pip-cache`; just `--no-deps` install).
3. Re-run held-out with `--judge-host prism` (validates ≈0.338 mean within engine-equivalence noise).
4. Promote prism judge to primary; stop B300 vllm-judge container; reclaim ~14 GiB B300 VRAM.
5. After Step 3 passes, swap to TRT-LLM-FP8 PyTorch backend (`scripts/serve_trtllm_judge_prism.sh` already shipped) and re-bench.

## Issues for user

1. **Python 3.11.10 is the pod's only `python3` and TRT-LLM's published wheels
   skip cp311** (probed across 0.17.0.post1, 0.18.2, 0.19.0, 0.20.0, 0.21.0,
   1.0.0, 1.1.0 — only cp310 + cp312 wheels exist). We worked around with
   `python3.10 + /workspace/py310-site` PYTHONPATH. If the pod is ever
   rebuilt with a cp310 OR cp312 system Python by default, the workaround
   can be retired.
2. **vLLM 0.20.0 + cu12.4 + Hopper requires `deep_gemm`** for the
   `kernel_warmup` path that fires during engine init regardless of
   `--quantization` flag. `deep_gemm` has no cp311 wheel (source build
   fails with `Failed to build deep_gemm`). Workaround: `VLLM_USE_DEEP_GEMM=0`
   env var bypasses the warmup. Side effect: probably slightly slower FP8
   GEMM at runtime — not measured, since we're not running FP8 anyway.
3. **RunPod ssh proxy does NOT support `-L` port forwarding.** Confirmed
   via `channel open failed: unknown channel type: unsupported channel type`
   in `/tmp/prism_portfwd.log`. Direct-IP fallback (`runpod-prism-direct`)
   would work but the pod-image's `authorized_keys` doesn't currently
   carry `~/.ssh/id_ed25519.pub` from the laptop. **Fix is in the RunPod
   console UI**: paste the laptop pubkey into the pod's "SSH Public Keys"
   section. Cannot do this via Claude — it's a UI action only.
4. **Pip-target double-write race**: two pip installs hitting the same
   `--target=/workspace/py310-site` from different background scripts left
   the install in a half-applied state. Caused 1+ hour of confusing "install
   complete but tensorrt_llm not importable" diagnostics. New convention
   for Phase 2.3: only one install-script.sh alive at a time. Always
   `pgrep -af install_310` before launching. Use `/workspace/pip-tmp` as
   TMPDIR (pod's overlay disk is only 80 GB; `/tmp` fills with pip-target
   directories that crash if removed mid-install).
5. **`_runpod_ssh.sh` secret-grep guard never fired during Phase 2.2.**
   No commands attempted to push secrets. Guard is silent in the happy
   path — working as designed. The doctrine is now in CLAUDE.md §0
   ("HOT-PATH MARKERS") for any future Claude session.
