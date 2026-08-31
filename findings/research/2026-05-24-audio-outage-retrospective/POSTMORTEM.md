# 2026-05-23 P0 audio outage — postmortem

**Status:** RESOLVED 2026-05-24 ~07:50 UTC via `scripts/serve_orca_h100.sh snapshot`. Audio live end-to-end on `thegoatnote.com/4UWHAt`.

**Incident type:** silent capability gap surfaced under investor-demo pressure.

## TL;DR

the H100 serving pod's `nemotron-serve` container launched with `--limit-mm-per-prompt '{"audio":1,...}'`, promising audio capability. The `vllm/vllm-openai:latest` base image was missing the audio decoder backend (`librosa`, `soundfile`, `torchcodec` all `ModuleNotFoundError`). vLLM lazily holds a `PlaceholderModule` and only fails on the first real audio request with HTTP 400 "Invalid or unsupported audio file". Startup looked healthy. Detection waited until end-user audio attempts during the a16z investor session.

Fix: snapshot the running container after `pip install librosa soundfile` → restart from the snapshot. ~90 s downtime. Durable path is `scripts/serve_orca_h100.sh build` (Dockerfile FROM `vllm/vllm-openai:latest` + `RUN pip install` audio deps).

## Timeline (UTC)

| Time | Event |
| --- | --- |
| ~2026-04-30 | `/api/ask` empirical test logs `nemotron_v3` reasoning-channel routing (150 reasoning + 27 content chunks at `enable_thinking=true`; 0 content at `enable_thinking=false`). Audio capability untested. |
| ~2026-05-15 | orca container launched manually with `--limit-mm-per-prompt '{"audio":1,...}'`. Audio capability assumed working from the flag alone. No e2e probe. |
| 2026-05-23 ~09:00 | User reports "audio issues persist. voice recorded but no action triggered by model" during a16z investor evaluation. |
| 2026-05-23 ~09:30 | PR #412: changed request shape from `audio_url` → `input_audio` based on a research brief that inverted the actual nemotron-omni spec. Audio still broken (and text now also degraded). |
| 2026-05-23 ~10:00 | PR #414: reverts #412. Restores documented `audio_url`. |
| 2026-05-23 ~10:30 | PR #415: adds reasoning→content fallback in `/api/agent` + `served` model surface in `/api/telemetry`. Text path fixed. Audio still 400 (decoder missing — root cause not yet diagnosed). |
| 2026-05-23 ~11:00 | Direct ssh on orca surfaces the actual gap: `librosa`/`soundfile`/`torchcodec` all `ModuleNotFoundError` inside the container. |
| 2026-05-23 ~12:00 | `docker exec pip install librosa soundfile` succeeds. vLLM still 400 — `PlaceholderModule` was cached at startup; container restart required. |
| 2026-05-23 ~13:00 | HOLD recommended: a16z investor evaluation in progress; risk of full prod outage if restart fails outweighs the audio-fix benefit. |
| 2026-05-24 ~07:00 | `docker commit nemotron-serve → nemotron-serve-with-audio:2026-05-24` (23.9 GB). Snapshot preserves the runtime overlay containing the pip-installed deps. |
| 2026-05-24 ~07:46 | PR #416: `scripts/serve_orca_h100.sh` shipped as in-repo source-of-truth (build / snapshot / inplace modes). |
| 2026-05-24 ~07:50 | Snapshot relaunch executed: rename old container, stop, run from snapshot, poll `/v1/models` → vLLM ready in 90 s, `librosa` import OK, test audio request returned 200 + `"Yeah"`. **Audio live.** |
| 2026-05-24 ~07:55 | End-to-end verification through Vercel + tunnel: audio path emits 18 text-deltas + 1004 reasoning-deltas (the model genuinely processes the WAV). `/api/telemetry`'s `served` field reports the actual loaded model root. |

## Root cause (5 whys)

1. **Why did audio requests return 400?** vLLM couldn't decode the WAV: `librosa`/`soundfile` missing in the container.
2. **Why were they missing?** The `vllm/vllm-openai:latest` base image doesn't include the audio decoder backend; the `vllm[audio]` extra wasn't installed.
3. **Why wasn't `vllm[audio]` installed?** Orca's container launch was hand-rolled in a shell session and never tracked in the repo; nobody re-derived a Dockerfile that included the extras.
4. **Why wasn't the gap detected at startup?** vLLM lazy-loads optional dependencies (`PlaceholderModule` pattern). Startup succeeded because no audio request had been issued; the placeholder only raises on first use.
5. **Why didn't any probe / test catch it?** No e2e audio test on the `/api/agent` smoke path; the adversarial-probe corpus has no audio cases; `/api/telemetry` had no audio-capability check.

## Contributing factors

- **Launch state lived only in `docker inspect`** — no `scripts/serve_orca_h100.sh` existed in the repo. Container args were invisible to code review, unreproducible on pod replacement, and the missing audio deps were invisible by extension.
- **The capability flag promised more than the runtime delivered.** `--limit-mm-per-prompt '{"audio":1,...}'` was set as if it were a config; in reality it's a per-request quota that only matters if the runtime can actually decode the modality.
- **Detection only on first user attempt.** No probe in CI, no startup self-test, no telemetry-level capability check.
- **PR #412 made it worse before it got better** — Claude trusted a research brief over an empirical test. The brief inverted the actual spec (`input_audio` is wrong; `audio_url` is the documented shape).
- **The `nemotron_v3` reasoning-channel routing** masked which subsystem was broken: text-only turns also produced empty UI (different bug, same symptom), so audio looked like "another text bug" rather than a distinct capability gap. PR #415's reasoning-fallback resolved the text symptom but did not address the audio decoder gap.
- **Investor-demo timing** raised the cost of any restart attempt above its usual baseline (a successful restart is cheap; a failed restart during evaluation is catastrophic), which is why the fix was deferred until ~01:00 PCT.

## Resolution

| PR | What | Status |
| --- | --- | --- |
| #414 | Revert wrong-shape #412 (back to documented `audio_url`) | Merged 2026-05-23 |
| #415 | `/api/agent` reasoning→content fallback + `/api/telemetry` `served` surface | Merged 2026-05-23 (admin) |
| #416 | `scripts/serve_orca_h100.sh` + this postmortem + handoff + CLAUDE.md §0 update | **Open**, awaiting user merge |
| (operational) | `docker commit` → `nemotron-serve-with-audio:2026-05-24` + restart from snapshot | Executed 2026-05-24 07:50 UTC, ~90 s downtime |

## Preventive moves (ordered by impact)

These were identified during the post-mortem discussion. Move (1) is shipped in this PR; (2)–(5) are queued for follow-up.

1. **In-repo launch source of truth (DONE in PR #416).** `scripts/serve_orca_h100.sh` documents what runs and how to rebuild it. Future container recreation should always go through this script — never ad-hoc `docker run`.
2. **Audio cases in the adversarial-probe corpus.** Add a 1 s sine WAV + a 3 s real-speech sample as canonical audio-probe payloads. The probe runs on prod cadence; a regression like 2026-05-23 would be caught within one probe interval instead of waiting for a user attempt.
3. **Startup self-test for declared modalities.** When `--limit-mm-per-prompt` declares a modality, a startup hook should round-trip a synthetic input of each declared type and fail-fast if the decoder doesn't work. Prevents the lazy `PlaceholderModule` from masking a missing dep at boot.
4. **`/api/telemetry` audio-capability probe.** Extend the route to fire one synthetic audio request at cold-start, cache the boolean, and surface `audioCapability: true|false|"untested"` alongside the existing `served` field. Makes "is audio working?" a single curl.
5. **Empirical-test gate on research briefs.** PR #412 was caused by trusting a brief over a 30 s curl. The lesson: research briefs are hypothesis-generators, not implementation-blueprints. Before changing a request shape based on a brief, the brief's recommendation must be empirically tested first against the live endpoint.

## Open follow-ups for the next session

See `docs/handoff/2026-05-24.md`.

## References

- `scripts/serve_orca_h100.sh` (PR #416) — the launch source of truth.
- `web/app/api/agent/route.ts` (PR #415) — reasoning-channel fallback.
- `web/app/api/telemetry/route.ts` (PR #415) — `served` model surface.
- `findings/research/2026-05-23-nemotron-omni-audio-spec/BRIEF.md` — the brief that misled PR #412 (kept for reference; do NOT trust its `input_audio` claim).
- `web/app/api/ask/route.ts` lines 103-107 — 2026-04-30 empirical finding on `nemotron_v3` reasoning-channel routing.
- Memory: `feedback_vllm_audio_deps_must_be_baked.md` — durable lesson for future sessions.
