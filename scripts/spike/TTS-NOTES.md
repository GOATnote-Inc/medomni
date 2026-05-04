# TTS — Tier 0 + Tier 1 (Kokoro WebGPU)

## What this spike adds

Two-tier text-to-speech for the medomni clinical agent:

- **Tier 0** — `useSpeechSynthesis` (existing). Browser-native
  `window.speechSynthesis`. Zero deps, zero network. Default voice is
  whatever the OS ships (often a 1990s formant synth — robotic).
- **Tier 1** — `useKokoroTts` (new). Kokoro.js TTS (82 M params)
  running 100% client-side via WebGPU (or WASM fallback). Audio never
  leaves the device. Sovereign by construction.

The unified `useTts` façade routes between them based on the user's
pick in `<VoicePicker>`.

## Default voices

- **No WebGPU** (Safari < 17, Firefox without `dom.webgpu.enabled`):
  Tier 0 auto-pick. The existing `PREFERRED_VOICE_NAMES` cascade
  (`Google US English` → `Microsoft Aria Online (Natural)` → ... →
  `Samantha`) picks the most natural voice the OS ships.
- **WebGPU available** (Chrome/Edge/Arc/Brave 121+, Opera 110+):
  Tier 0 auto-pick is still the default until the user opens
  `<VoicePicker>` and selects a Kokoro voice. We deliberately do NOT
  auto-upgrade — the 160 MB model download must be opt-in.
- **Default Kokoro voice when picked**: `af_heart` (graded **A** on
  the Kokoro v1.0 model card; en-us female).

## One-time 160 MB model download

The first time a user selects a Kokoro voice and triggers playback,
`KokoroTTS.from_pretrained("onnx-community/Kokoro-82M-v1.0-ONNX")`
fetches:

- ONNX model weights (~160 MB total, dtype-dependent: `fp32` for WebGPU,
  `q8` for WASM)
- Tokenizer config
- The selected voice's style tensor (~520 KB binary)

`<VoicePicker>` surfaces the live progress as
`downloading model… NN%`, switching to `ready · on-device` when done.

## How to switch voices

1. Click the voice toggle to enable voice-out.
2. Click the voice picker pill (`Browser · default ▾`).
3. Pick from:
   - **Browser voices · Tier 0** — every voice the OS exposes via
     `speechSynthesis.getVoices()`, sorted en-US first.
   - **AI voices · Kokoro WebGPU** — 20 graded voices from the
     Kokoro v1.0 catalog, sorted by overall grade.
4. The choice is persisted to `localStorage["medomni:tts:voice"]` as
   either `browser:<voiceURI>` or `kokoro:<voiceId>` and restored on
   next page load.

## Where the model is cached

- **Model weights + tokenizer**: Transformers.js IndexedDB cache
  (DB name `transformers-cache`, store `weights`). Cleared by clearing
  site data.
- **Voice style tensors**: Cache Storage under the cache name
  `kokoro-voices` (set by `kokoro-js` itself; see
  `node_modules/kokoro-js/dist/kokoro.js`). Clear via DevTools →
  Application → Cache Storage → `kokoro-voices`.

## Bundle impact

`kokoro-js` is dynamically imported inside `loadKokoro()` — it is
NOT in the initial JS bundle. The first user who picks a Kokoro voice
pulls the kokoro-js JS chunk + its `@huggingface/transformers`
dependency on demand. Users who never opt in pay zero bytes for it.

## Sovereignty posture

Aligned with `web/CLAUDE.md` §2 ("Sovereignty by construction") and
the medomni repo charter:

- No cloud TTS keys (ElevenLabs, Cloud Google TTS, Azure Neural — all
  forbidden).
- Audio synthesized in-browser via WebGPU. Nothing crosses the network
  except the one-time model fetch from Hugging Face's CDN.
- `.env.example` permits exactly two secrets (`HF_TOKEN`,
  `BREV_PEM_PATH`); this spike adds NONE.

## Known limitations users will hit

1. **First-time download is large.** 160 MB is non-trivial on cellular.
   Picker copy warns "AI voices download a 160MB model on first use
   (cached after)."
2. **Safari < 17 / Firefox without flag**: WebGPU absent → Kokoro falls
   back to WASM (works, ~3-5x slower per utterance). The picker shows a
   `WASM` chip in that mode.
3. **First utterance latency**: ~100 ms once loaded; ~5-30 s on the
   very first call (model load cost).
4. **AudioContext autoplay policy**: the model needs a user gesture
   before audio plays. The voice toggle / picker click satisfies that.
5. **Voice quality grades vary**: Kokoro voices are graded A through F.
   `af_heart` (A), `af_bella` (A-), and `bf_emma` (B-) are the most
   reliable; the F-graded voices (e.g. `am_adam`) can mispronounce
   medical terminology.
