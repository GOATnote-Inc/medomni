# Nemotron-3-Nano-Omni audio input — vLLM request format brief

**Date:** 2026-05-23
**Severity:** P0 production (audio input broken; user-visible)
**Compiled by:** backgrounded `general-purpose` research agent, verified against three NVIDIA-published sources (HF BF16 + NVFP4 model cards, NIM API reference) + vLLM stable docs + the medomni repo's own launch scripts.
**Triggered by:** the regression after PR #412 — audio attached, no model response.

## TL;DR (60 seconds)

**The correct shape is `audio_url` with a `data:` URI — not `input_audio`.** The medomni code at `web/app/api/agent/route.ts:351-355` (post-PR #412) is wrong; the comment "vLLM / Nemotron-Omni's OpenAI-compat endpoint expects `input_audio` … NOT `audio_url`" **inverts reality**. Every authoritative Nemotron-3-Nano-Omni source shows exactly one shape:

```json
{
  "type": "audio_url",
  "audio_url": { "url": "data:audio/wav;base64,<base64-bytes>" }
}
```

PR #414 reverts PR #412 and is the correct first move.

**But there is a bigger problem.** `scripts/launch_b300_prod.sh:71` pins `MODEL_ID=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` — the **text-only base**. No audio encoder. No vision encoder. The Omni variant is `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` (different repo, different weights, +Parakeet audio encoder, +CRADIO vision). **If the served model on `orca` is the text-only base, no request-shape fix will ever make audio work — the model literally cannot consume it.**

## 1. Authoritative request shape

Per the NVFP4 model card (verbatim Python example for the vLLM OpenAI-compatible API):

```python
{"type": "audio_url", "audio_url": {"url": audio_url}},
{"type": "text", "text": "Transcribe this audio."},
```

The cards' examples use `Path(...).as_uri()` (i.e. `file://`) because they run on the serving box, but vLLM's `parse_input_audio` (in `vllm.entrypoints.chat_utils`) internally normalises both `input_audio` and `audio_url` into `data:audio/{format};base64,{data}` — so a browser-encoded `data:` URI is the supported transport for our browser → API → vLLM flow. The vLLM canonical multimodal client at `examples/online_serving/openai_chat_completion_client_for_multimodal.py` (v0.20.0) demonstrates `"url": "data:audio/ogg;base64,{audio_base64}"` explicitly.

## 2. Format requirements (Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 card)

| Param | Value |
| --- | --- |
| Formats | `wav`, `mp3` |
| Sample rate | 8 kHz and higher (Parakeet-TDT-0.6B-v2 encoder resamples internally) |
| Max duration | 1 hour |
| Channels | Unspecified; mono is the safe default for the `AudioRecorder` path |
| Encoding for browser → API → vLLM | Base64 inside `audio_url.url` as a `data:audio/wav;base64,…` URI |
| **Reasoning + audio** | **Mutually exclusive.** Must set `extra_body.chat_template_kwargs.enable_thinking = false` for any audio request. (Already documented in our own `scripts/serve_omni_b300.sh:15-16`.) |
| Recommended sampling | `temperature: 0.2`, `top_k: 1` for ASR-style transcription |

## 3. vLLM version + serve-time requirements

- **vLLM 0.20.0 minimum** (`vllm/vllm-openai:v0.20.0` or `:v0.20.0-cu129` for CUDA 12.9). `serve_omni_b300.sh` already pins this.
- **`pip install "vllm[audio]"` is mandatory** inside the container before `vllm serve` — the stock image does NOT include audio deps. **Our launch scripts do not do this.** Without it, audio blocks are accepted by the chat endpoint but silently dropped at the multimodal preprocessor (no error, model responds to text only — matches the symptom we saw with the prior `audio_url` shape, and is also a plausible explanation for *why* the input_audio shape returned 200-with-empty-stream).
- **`--limit-mm-per-prompt '{"audio": 1, "image": 1, "video": 1}'` should be set.** Default audio limit is 0 in some 0.20.x builds — another silent-drop path. Not currently in our launch.
- `--allowed-local-media-path /` is NOT needed for the `data:` URI path (only for `file://`).
- `VLLM_AUDIO_FETCH_TIMEOUT` (default 10 s) and `VLLM_MAX_AUDIO_CLIP_FILESIZE_MB` (default 25 MB) are the only audio-related env vars; our 4 MB body cap and 60 s recording stay under both.

## 4. Working example (verbatim from the Omni-NVFP4 card)

```bash
curl -sS http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "audio_url",
         "audio_url": {"url": "data:audio/wav;base64,UklGR..."}},
        {"type": "text", "text": "Transcribe the audio and answer..."}
      ]
    }],
    "max_tokens": 1024,
    "temperature": 0.2,
    "top_k": 1,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

(`top_k` and `chat_template_kwargs` ride at the top level in the JSON body — vLLM passes unknown keys through. In the Python SDK they go inside `extra_body={...}`.)

## 5. The bigger problem — wrong model in production

`scripts/launch_b300_prod.sh:71`:

```bash
MODEL_ID="${MODEL_ID:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4}"
```

That is the **text-only base** model. The HF card states verbatim *"Input Type(s): Text. Input Format(s): String"* — no audio encoder, no vision encoder.

The Omni (multimodal) variant is `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` — different repo, different weights, +Parakeet audio encoder, +CRADIO vision encoder. Only `serve_omni_b300.sh` and `swap_to_omni_h200.sh` use the Omni handle.

**Confirmation step (maintainer-action):** `curl <tunnel>/v1/models | jq '.data[0].id'` from a machine that has `MEDOMNI_TUNNEL_URL`. If `id` is `NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`, no request-shape fix will work.

## 6. Decision tree

1. **Confirm which model is actually serving on `orca`** — `curl <tunnel>/v1/models`. (Do **not** restart anything — `nemotron-serve` is hot-path-marked live production in `CLAUDE.md` §0.)
2. **If served model is text-only base:** PR #414 still the right merge (revert removes a wrong-shape change); separately, plan a blue-green swap to the Omni handle using `runbooks/blue-green-pod-replacement.md` + `serve_omni_b300.sh`, OR ship a Parakeet-TDT-0.6B-v2 ASR sidecar (NVIDIA NIM container, OpenAI-compatible `/v1/audio/transcriptions`) and keep the text-only chat model. The Omni swap is the right answer if multimodal is on the roadmap; the ASR sidecar is the right answer if audio is a small slice of traffic.
3. **If served model is Omni:** three edits in `web/app/api/agent/route.ts` after the revert:
   - Keep `ContentBlock` with `audio_url` (post-revert state) — *this is the correct shape*.
   - In `uiMessagesToChat`, push `{ type: "audio_url", audio_url: { url: part.url } }` directly — `part.url` is already a `data:audio/wav;base64,…` URI from `AudioRecorder` (no regex-strip needed).
   - At the upstream-request site, when any block has `type === "audio_url"`, force `chat_template_kwargs: {enable_thinking: false}` and `temperature: 0.2` (else reasoning mode silently disables the audio path on Omni).
   - **Server-side:** redeploy with `pip install "vllm[audio]"` in the container build, plus `--limit-mm-per-prompt '{"audio":1,"image":1,"video":1}'` on the `vllm serve` line. Without these, the corrected shape will still be silently dropped.

## 7. Confidence and caveats

High confidence on the shape (3 independent NVIDIA-published sources agree: HF BF16 card, HF NVFP4 card, NIM API reference). Medium-high on `vllm[audio]` being mandatory (vLLM blog post states it explicitly; vLLM stable docs corroborate). The exact handle currently serving on `orca` was not confirmable from this research thread (no shell access to the pod, can't touch). The reasoning-vs-audio mutual exclusion is baked into our own runbook (`serve_omni_b300.sh:15-16`) and into the NVFP4 card's `extra_body` defaults.

## 8. Sources

- [Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 (HF model card)](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16)
- [Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 (HF model card)](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4)
- [NVIDIA NIM API reference — Nemotron-3-Nano-Omni](https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-nano-omni-30b-a3b-reasoning)
- [vLLM blog — Run Nemotron 3 Nano Omni with vLLM (2026-04-28)](https://vllm.ai/blog/nemotron-omni)
- [vLLM multimodal inputs docs (stable)](https://docs.vllm.ai/en/stable/features/multimodal_inputs/)
- [vLLM canonical multimodal client example (v0.20.0)](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.0/examples/online_serving/openai_chat_completion_client_for_multimodal.py)
- [vLLM chat_utils API — `parse_input_audio` normalisation](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/chat_utils/)
- [NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 (text-only base, HF card)](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4) — confirms the prod-launched model has no audio.

## 9. Relevant files in this repo

- `web/app/api/agent/route.ts` — bug at lines 351-356, 385-398 (post-PR #412; reverted by PR #414).
- `web/app/api/ask/route.ts` — same bug pattern at lines 39-54 (reverted by PR #414).
- `scripts/launch_b300_prod.sh` line 71 — **wrong model handle** (text-only base).
- `scripts/serve_omni_b300.sh` line 35 — right model handle; missing `vllm[audio]` install + `--limit-mm-per-prompt`; correctly documents reasoning↔audio mutex.
- `scripts/swap_to_omni_h200.sh` — the blue-green swap script for moving prod to Omni.
- `runbooks/blue-green-pod-replacement.md` line 411 — warns about `--allowed-local-media-path` 400s (relevant if Omni serve adds it).

---

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
