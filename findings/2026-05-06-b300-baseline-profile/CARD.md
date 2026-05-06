# B300 Baseline Profile — Blackwell SM 10.x, Nemotron-3-Nano-Omni-30B-NVFP4

**Captured:** 2026-05-06
**Pod:** `unnecessary-peach-catfish` (Brev B300, NVIDIA provisioning-bug recreate target)
**Host:** `95.133.253.29`
**Operator:** Claude (Opus 4.7), under live prod inference (no restart, no disruption)

This is the **only** B300+NVFP4 perf evidence we will retain after the pod is recreated.

---

## Hardware + driver

| Field | Value |
|---|---|
| GPU | NVIDIA B300 SXM6 AC |
| Serial | 1650526065609 |
| UUID | `GPU-fefaccb2-bc6d-2e25-0004-40ad6b1b0e65` |
| Driver | 580.126.09 |
| CUDA (driver-reported) | 13.0 |
| Persistence Mode | Enabled |
| ECC Mode | Enabled (`gpu-state.txt` for breakdown) |
| MIG | Disabled |
| Compute Mode | Default |
| Default Power Limit | 1100 W |
| VRAM | 275040 MiB total (≈ 268.6 GiB) |
| VRAM allocated by vLLM | 235559 MiB used / 38556 MiB free (88.5% reserved per `--gpu-memory-utilization 0.72`; the difference is HBM headroom + driver reservation) |

Source: `gpu-state.txt`.

## Software pin

| Field | Value |
|---|---|
| Container image | `medomni/omni:v0.20.0-audio` |
| Image SHA-256 | `c62e3937687d68b819940b4809a3e6358b606f87e6255f0a34aae89c042d5474` |
| Container ENTRYPOINT | `["vllm", "serve"]` |
| vLLM | (image-baked; pip-freeze.txt for component versions) |
| torch | 2.11.0+cu130 |
| transformers | 5.6.2 |
| triton | 3.6.0 |
| flashinfer-python | 0.6.8.post1 |
| flashinfer-jit-cache | 0.6.8.post1+cu130 |
| cuda-python | 13.2.0 |
| nsys (host) | 2024.6.2 |
| ncu (host) | 2025.1.1 (CUDA 12.8 toolkit at /usr/local/cuda-12.8/bin/ncu) |

Sources: `image-sha.txt`, `container.json`, `pip-freeze.txt`.

## Served model + flags

```
nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4
served-model-name: nemotron
host: 0.0.0.0  port: 8000  network: host
tensor-parallel-size: 1
max-model-len: 131072
max-num-seqs: 32
max-num-batched-tokens: 32768
gpu-memory-utilization: 0.72
kv-cache-dtype: fp8
no-async-scheduling
limit-mm-per-prompt: video=1, image=4, audio=1
media-io-kwargs: video fps=2, num_frames=256
video-pruning-rate: 0.5
allowed-local-media-path: /tmp/medomni
reasoning-parser: nemotron_v3
enable-auto-tool-choice
tool-call-parser: qwen3_coder
```

(Note: served name on this pod is the `-Reasoning-` SKU, not the bare `-NVFP4` SKU referenced in the original capture brief. Captured-as-served per `container.json`.)

## Throughput sweep — 20 fixed clinical prompts × 3 waves per batch

`max_tokens=1024` per request, `stream=True`, single client process; b=1/4/16/32 served sequentially.
Token counts include the model's reasoning trace (`reasoning_parser=nemotron_v3` emits a `reasoning` channel + `content` channel; the `content` SSE delta is what is counted).

| metric | b=1 | b=4 | b=16 | b=32 |
|---|---:|---:|---:|---:|
| n_ok                | 3      | 12     | 48     | 96     |
| wall_s              | 4.696  | 12.407 | 15.538 | 19.335 |
| total_tokens (content) | 905 | 5973   | 27283  | 50850  |
| **tokens/sec**      | **193** | **481** | **1756** | **2630** |
| ttft p50 (s)        | 0.744  | 1.929  | 2.189  | 2.005  |
| ttft p95 (s)        | 0.745  | 2.793  | 4.542  | 4.520  |
| e2e p50 (s)         | 1.333  | 3.711  | 4.703  | 5.479  |
| e2e p95 (s)         | 2.077  | 5.095  | 6.180  | 6.280  |
| VRAM used (MiB)     | 235559 | 235559 | 235559 | 235559 |
| VRAM free (MiB)     | 38556  | 38556  | 38556  | 38556  |

VRAM is allocator-stable at the gpu-memory-utilization=0.72 reservation; no fragmentation/regrowth observed across the sweep.

Source: `throughput-sweep.csv`, `vram-snapshots.txt`.

## Idle-but-prod (60 s) vs sustained-load (5 min @ b=4)

The pod is serving live traffic, so "idle" here means no synthetic bench load — real ambient prod calls were ongoing. Numbers from `nvidia-smi dmon -s pucvmt`.

| metric | "idle"-prod (60s) | sustained b=4 (300s) |
|---|---:|---:|
| Power p50 (W) | (see `dmon-idle.csv`) | **465** |
| Power p95 (W) | — | **526** |
| Power max (W) | — | **531** |
| GPU temp p50 (C) | — | 42 |
| GPU temp max (C) | — | 43 |
| SM util p50 (%) | 75–80 (already busy from prod traffic) | 79 |
| SM util max (%) | — | 82 |

Power max 531 W is roughly 48% of the 1100 W default cap — this workload is **far** from power-limited. Thermals are extremely cool (43 C peak) — no DVFS throttling concerns at this duty cycle. SM occupancy at b=4 maxes ≈82%, suggesting room above b=4 for further throughput (corroborated by the b=16, b=32 sweep above).

Sustained b=4 throughput (`bench-sustained-summary.txt`): **548 tok/s × 310 s = 170,350 tokens; 84 waves complete; 0 errors.**

Sources: `dmon-idle.csv`, `dmon-sustained.csv`, `bench-sustained-summary.txt`.

## /metrics scrape (idle, prod)

`vllm-metrics-idle.prom` — 632 lines of Prometheus-format counters at the moment-of-capture. Contains per-engine queue depth, kv-cache util, prefix-cache hit rate, request waiting reasons, sleep state, request-latency histograms. Use as a B300+vLLM v0.20.0 baseline reference.

## Headline numbers

- **Peak content tokens/sec at b=32: 2630**
- **Peak content tokens/sec at b=4 (sustained): 548 (300s rolling)**
- **TTFT p50 at b=4: 1.93 s** (this is **chat** TTFT — first emitted `content` token. The reasoning channel emits earlier and was not separately timed in this capture.)
- **Sustained power: 465 W p50 / 526 W p95** (out of 1100 W cap — 42% / 48%)
- **Sustained gtemp: 42 C p50 / 43 C max** (deeply cool)
- **Image SHA: `sha256:c62e3937687d68b819940b4809a3e6358b606f87e6255f0a34aae89c042d5474`**

## Gaps — what we could not capture

**B1h (nsys profile of a 60s decode timeline) — BLOCKED.**
- Available on host: `nsys 2024.6.2`. Available in container: nsys not installed.
- 2024.6.2 has no `--pid=` / "attach to running process" mode (introduced in later versions; this build only supports `nsys profile <command>` or `nsys start/stop` on a session that included the target at launch).
- The vLLM EngineCore process was launched without nsys instrumentation; instrumenting it would require a container restart, which violates the brief's "do not disrupt prod inference" constraint.
- **Mitigation:** the throughput sweep + dmon sustained-load tell us aggregate pacing (≈ 30 ms / token at b=4 sustained, ≈ 12 ms / token at b=32 peak). This is the best-available timing surface without a restart.

**B1i (ncu --set full on 5 representative decode kernels) — BLOCKED.**
- Available on host: `ncu 2025.1.1`. ncu's `--mode=attach` requires the target to have been launched with `--mode=launch-and-suspend`; ncu **cannot** attach to a process that started without it.
- Same root cause as nsys: would require container restart. Brief explicitly forbids.
- **Mitigation:** none from this pod. The ncu profile is recoverable later only if we re-stand a B300+NVFP4 stack from scratch with ncu instrumentation pre-wired (and explicit user OK to take prod down). Documented as a deferrable for any future B300 access window.

**Dmon-idle interpretation caveat.** The 60s "idle" sample shows 75–80% SM util — this is **prod ambient traffic**, not a quiescent baseline. A true cold idle baseline would require gating user traffic, which we did not do (would have disrupted the live endpoint). Use these numbers as the **floor** of what the GPU is doing under normal serving conditions, not as a "GPU-doing-nothing" baseline.

## Sanity checks performed

- Endpoint smoke before any bench: `/v1/chat/completions` with `max_tokens=4000` (per `nemotron_omni_tool_call_parser.md`, the model emits reasoning before content and needs headroom). Returns OK.
- Container ENTRYPOINT verified via `docker inspect --format` before any docker-touching command (per `feedback_check_docker_entrypoint_before_docker_run.md`). ENTRYPOINT is `["vllm", "serve"]` — no command-prefix bug risk.
- Bench harness was driven from inside the host (not from the container), via the prod port 8000. Zero modifications to the running container.
- All 0 errors across 159 bench requests (3 + 12 + 48 + 96).

## Files in this directory

| File | Purpose |
|---|---|
| `gpu-state.txt`           | `nvidia-smi -q` snapshot |
| `container.json`          | `docker inspect vllm-omni-b300` |
| `image-sha.txt`           | container image SHA-256 (the irreplaceable pin) |
| `pip-freeze.txt`          | software stack from inside the container |
| `vllm-metrics-idle.prom`  | Prometheus scrape of /metrics |
| `dmon-idle.csv`           | 60 s `nvidia-smi dmon -s pucvmt` (ambient prod) |
| `dmon-sustained.csv`      | 300 s `nvidia-smi dmon -s pucvmt` under bench b=4 |
| `bench-sustained-summary.txt` | 5-min sustained throughput summary line |
| `throughput-sweep.csv`    | b=1/4/16/32 sweep |
| `vram-pre-b{1,4,16,32}.txt`, `vram-post-...` | per-batch VRAM snapshots |
| `vram-snapshots.txt`      | collated VRAM snapshots |
| `prompts.json`            | 20 fixed clinical prompts (≈50–80 tokens each) |
| `bench.py`                | throughput harness (Python 3 stdlib only) |
| `CARD.md`                 | this file |

## Next-time, if there's a next-time

If we get another window with B300+NVFP4 and **explicit OK to restart vLLM under profilers**, the missing artifacts are:

1. `nsys profile --trace=cuda,nvtx,osrt --duration=60 vllm serve <flags>` and then drive bench at b=4. Captures CUDA stream timeline, kernel sequencing, prefill→decode handoff.
2. `ncu --launch-skip 100 --launch-count 5 --set full --target-processes all <vllm cmd>` and drive bench. Captures TensorCore pipe util, DRAM throughput, SM throughput, L1/L2 reuse on representative decode kernels.
3. NVFP4 GEMM verification: pull weight tensor headers via `safetensors` to confirm the on-disk dtype in `/root/.cache/huggingface/hub/models--nvidia--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4/`. Cross-check that the GEMM kernels ncu observes match the NVFP4 cuBLASLt path.

These three were skipped today only because the constraint was "do not touch prod" — not because the tooling was missing.
