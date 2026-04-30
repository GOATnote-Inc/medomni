# Phase 3 — multimodal demo fixtures (results brief)

**Owner**: T_fixtures · **Date**: 2026-04-29 · **Repo**: prism42-nemotron-med

## What landed

Four real, public-domain or open-CC-licensed fixtures wired up under `corpus/demo-fixtures/`, plus per-fixture license + clinical-context, an aggregate `MANIFEST.md` (with sha256 + scene mapping), and an Omni round-trip `SMOKE-RESULTS.md`.

| ID | File | License | Bytes | SHA-256 prefix |
|---|---|---|---|---|
| `fixture-ecg-001` | `ecg/fixture-ecg-001-stemi-12lead.png` (1601×913, 12-lead anterior STEMI) | CC BY 4.0 — Cureus 2018 mirror on Wikimedia Commons | 2.6 MB | `9229f211` |
| `fixture-pill-001` | `pill/fixture-pill-001-warfarin-tablets.png` (800×496, warfarin 5/3/1 mg trade-dress) | CC BY-SA 3.0 — Wikimedia Commons | 0.6 MB | `482d2906` |
| `fixture-cxr-001` | `cxr/fixture-cxr-001-rml-pneumonia.png` (3027×2407, lobar pneumonia PA) | CC0 1.0 — Mikael Häggström, MD | 8.3 MB | `e84b5864` |
| `fixture-aud-001` | `auscultation/fixture-aud-001-crackles-pneumonia.{wav,mp3}` (14 s, pneumonia crackles) | CC BY-SA 3.0 — James Heilman, MD | 446 KB / 113 KB | `c660b996` / `f118848b` |

All licenses verified directly from each source's Wikimedia file-description page; no fabricated metadata; conversion provenance tracked (`.jpg → .png` via `sips`, `.ogg → .wav/.mp3` via `ffmpeg -ar 16000 -ac 1`).

## Smoke-test results (Omni `Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` on B300, port 8000)

| Fixture | HTTP | Latency | Verdict |
|---|---|---|---|
| ECG | 200 | 3.7 s | **PASS** — "Sinus rhythm, rate 100 bpm, ST elevation V3-V6." Defensible. |
| Pill | 200 | 1.4 s | **PASS** — "Warfarin (Coumadin)" + bleeding-risk + drug-interaction safety pair. Strong. |
| CXR | 200 | 4.7–5.2 s | **PARTIAL PASS** — calls right-sided pneumonia / RLL; source caption says RML. Defensible from a single PA frame; gap documented in clinical-context.md. |
| Audio (WAV + MP3) | 400 | 0.6–0.7 s | **FAIL — server-side capability gap.** |

## Blocker — audio path

Container introspection (`docker exec vllm-omni-b300 python3 -c "import soundfile"` and `import av`) confirms **the running container does not have `vllm[audio]` extras installed**. Both `soundfile` and `pyav` are missing; vLLM's `load_audio` raises `ValueError("Invalid or unsupported audio file.")` before any base64 audio reaches the Parakeet encoder. This is not a fixture problem — the WAV is a clean 16 kHz mono PCM file (Omni's canonical audio shape).

**Required fix** (owner: `serve_omni_b300.sh` / concurrent agent — I did not modify per task constraints): add `pip install vllm[audio]` (pulls `soundfile` + `av`) to the container build, bounce the pod. After that, this fixture works as-is; the runbook Scene 3 acceptance gate becomes satisfiable.

## Scene mapping (vs. T3's DEMO-RUNBOOK.md as written 2026-04-29 17:10)

T3's runbook split SPEC §7 into 7 scenes. My fixtures land cleanly:

- **Scene 2 (pill camera)** — `fixture-pill-001` substitutes for the runbook's `[placeholder atorvastatin "Z 10"]` placeholder. Demo-ready.
- **Scene 3 (auscultation)** — `fixture-aud-001` fills the `[chosen demo prop confirmed before pitch]` slot. Demo-blocked on server `vllm[audio]`.
- **Stretch beats** — `fixture-ecg-001` and `fixture-cxr-001` are not currently in the runbook's seven-scene list but are available if a 4th bedside beat or the §7 reproducibility scene needs an image fixture.

T3's runbook is the SoT going forward; if scenes shift before pitch, this fixture set covers all four target modalities (image-ECG, image-pill, image-CXR, audio-auscultation).

## Demo readiness

- 3 of 4 fixtures: **demo-ready** (ECG + pill PASS; CXR PARTIAL with documented defensible gap).
- 1 of 4 fixtures: **fixture-ready, server-blocked** — auscultation. Server fix is a one-line `pip install vllm[audio]` + container restart, not a fixture change.

If the audio scene cannot ship before pitch, Scene 3 must be cut or replaced with a pre-recorded video clip; fixture and clinical-context already document this fallback.

## Files

- `corpus/demo-fixtures/MANIFEST.md` — full inventory, scene map, limitations, v1.5 wishlist
- `corpus/demo-fixtures/SMOKE-RESULTS.md` — full Omni request/response transcript per fixture
- `corpus/demo-fixtures/{ecg,pill,cxr,auscultation}/fixture-*-LICENSE.md` — per-fixture provenance
- `corpus/demo-fixtures/{ecg,pill,cxr,auscultation}/fixture-*-clinical-context.md` — per-fixture clinical brief + Omni response + scene mapping
