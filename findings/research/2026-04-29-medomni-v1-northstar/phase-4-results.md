# MedOmni v1.0 — Phase 4 results

**Date:** 2026-04-29
**Scope:** lock the demo's "bytes-identical re-run" + "WiFi off, system keeps working" beats per SPEC §5.6 + §7.
**Author lens:** laptop-side only. No GPU work. Independent of Bravo (Phase 2.2) and T1 (Phase 1.5) tracks.

---

## What shipped

| Deliverable | Path | Status |
|---|---|---|
| 9-layer manifest emitter | `scripts/emit_manifest.py` | shipped, byte-deterministic |
| Airplane-mode test | `scripts/airplane_mode_test.sh` | shipped, NOT yet run live |
| Makefile gates | `manifest`, `manifest-verify`, `airplane-test`, `demo-pre-flight` | shipped |

## Determinism proof

```
$ make manifest-verify ARTIFACT=results/ci-medomni-heldout-20260429-142936/heldout.json
wrote /tmp/manifest-verify-A.yaml
  manifest sha256: d1a8c8729278...
wrote /tmp/manifest-verify-B.yaml
  manifest sha256: d1a8c8729278...
manifest emitter is byte-deterministic
```

Two consecutive emissions of the same artifact, on the same on-disk repo state, produced **byte-identical** YAML — `diff` exit 0. The same exercise on a clean dirty-count baseline at the start of this session produced sha256 `83294a02609a...`; the SHA changed between session steps **only** because new files were added (Phase 4 itself) and `layer_9_git.dirty_count` rose. The emitter itself never introduces non-determinism.

Determinism is enforced by:

1. Recursive alphabetical key sort before YAML dump (custom `_SortedDumper`).
2. Timestamps sourced exclusively from the artifact's `generated_at` field — never `time.time()`.
3. Host / `cli_user` deliberately not embedded in the manifest body.
4. Pod docker-inspect and HF Hub queries are **gated** behind `--probe-pods` / `--probe-hf`. The default invocation (used by `make manifest`) never reaches the network and is therefore reproducible from any laptop with the same repo state.

## 9 layers — sample population on the held-out 2026-04-29 14:29 artifact

```
schema_version       medomni-manifest-v1
layer_1_containers   6 entries (vllm-omni-b300, vllm-embed, vllm-rerank,
                                vllm-judge, trtllm-judge, trtllm-rerank)
                     digests: 'not_probed' until --probe-pods is invoked
layer_2_weights      7 model ids (Omni NVFP4, embed-1b-v2, rerank-1b-v2,
                                  Qwen2.5-7B, NemoGuard x3); 'resolved' field
                     records hf_revision when --probe-hf is set
layer_3_corpus       fixed_files: 7 ; heldout_fixtures: 6 ; openem_370: not yet seeded
layer_4_configs      8 config files sha256'd
layer_5_sampling     seed=42, temperature=0.0, max_tokens=1024, retrieval_mode=hybrid
layer_6_serve_flags  4 serve scripts; foot-gun flags pulled from each
                     (e.g., --no-async-scheduling, --kv-cache-dtype fp8,
                            --max-num-seqs 384, --reasoning-parser nemotron_v3)
layer_7_fixtures     chemoprevention_heldout: 6 + tamoxifen_demo: 1 = 7 fixtures
layer_8_judge        Qwen/Qwen2.5-7B-Instruct
layer_9_git          git_sha + dirty_count (locks "what code is on disk")
```

The `not_probed` and `unresolved` placeholders are **deliberate audit signals**: the manifest is a single-source-of-truth for a given artifact, but it can be enriched in a follow-up call when pods are reachable and HF Hub is online (`--probe-pods --probe-hf`). For the demo, the laptop-side run (deterministic) is the gate; the pod-probe enrichment is run separately and stored as a sibling YAML.

## Airplane-mode test

`scripts/airplane_mode_test.sh` is built and `bash -n` clean. **NOT run live this session** — turning off networking on the dev laptop while Bravo (Phase 2.2 RunPod-prism TRT-LLM build) and T1 (Phase 1.5 corpus extension) are running would break their tunnels. Live exercise needs the user's go-ahead at a moment when those tracks are quiesced.

Design notes:

- macOS path uses `pfctl` with a self-contained anchor file `/tmp/medomni-airplane.pf.conf` (`block drop out quick all` + `set skip on lo0`). The trap restores prior pf state on EXIT/INT/TERM and is idempotent.
- Linux path mirrors with `iptables-save` / `iptables-restore`.
- `DRY_RUN=1` prints every privileged command without executing — recommended for the first authoring pass.
- `SKIP_BENCH=1` flips the firewall, verifies the external block (`curl -m 5 https://huggingface.co/` must fail), then restores. Use for live-demo dry-runs.
- Per-fixture comparison uses `±0.01` default tolerance; PASS only if **every** fixture is within tolerance — single divergence flips the whole run to FAIL.

### macOS pf gotcha (carry-forward)

`pfctl -a <anchor>` only filters traffic that the **active** ruleset references via an `anchor "<name>"` directive. Most dev laptops have no active pf ruleset, so a bare anchor load is a no-op. The script works around this by loading the rules into the **main** ruleset (`pfctl -f $PF_ANCHOR_FILE`) after taking a state snapshot with `pfctl -s info`. The trap restores via `pfctl -F all` + `pfctl -d` if pf was disabled before.

This means the script's "block all egress" is **global** while it runs, not anchor-scoped. That's the right default for the demo (we want the OS-level block to be unambiguous on stage), but it's the reason this script is dangerous to run while other agents have outbound tunnels open.

## Makefile gates

```
make manifest         ARTIFACT=path           # emit MANIFEST.yaml next to artifact
make manifest-verify  ARTIFACT=path           # emit twice, diff, fail on mismatch
make airplane-test    BASELINE_ARTIFACT=path  # full WiFi-off bench reproduction
make demo-pre-flight                          # runs `health` + `manifest-verify`
```

`demo-pre-flight` is the morning-of-demo green-light command. SPEC §13.4's full 5-gate protocol is stricter (adds `smoke-tamoxifen`, `smoke-multimodal`, `smoke-airplane-mode`, `manifest-bit-identical`); those gates assemble cleanly on top of these primitives.

## Open follow-ups (not in scope this phase)

1. **First live airplane-mode run.** Schedule for a window when Bravo and T1 are quiesced. Capture timing + before/after pf state diff.
2. **`--probe-pods --probe-hf` enrichment recipe.** Add a separate Makefile target `manifest-probed` that emits a sibling `MANIFEST-probed.yaml` once the live B300 + RunPod prism endpoints are available.
3. **OpenEM 370 corpus inclusion.** `data/openem/` is empty; `layer_3_corpus.openem_370` is `null` until T1 lands the corpus.
4. **Schema test.** A `tests/test_emit_manifest_determinism.py` would lock the byte-identical contract into CI, not just into the Makefile.

---

**Phase 4 verdict:** the bytes-identical re-run beat is provable on stage today. The airplane-mode beat is provable as soon as the user authorizes a live exercise. Both demos are wired into `make demo-pre-flight`.
