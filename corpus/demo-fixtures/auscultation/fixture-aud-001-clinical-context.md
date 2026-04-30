# Clinical context — fixture-aud-001-crackles-pneumonia.{wav,mp3}

## What the artifact shows

A 14-second recording of stethoscope-bell auscultation over the chest of a patient with documented pneumonia. The dominant adventitious sound is **bilateral fine inspiratory crackles** ("rales") — short, discontinuous, popping sounds heard predominantly during the inspiratory phase, classically described as the sound of velcro being slowly separated. Crackles in this acoustic profile are consistent with sudden equilibration of pressure as collapsed or fluid-filled small airways open during inspiration.

In the source's annotation by James Heilman, MD, the patient was diagnosed with pneumonia; the clinical context for the recording was therefore consolidation-related crackles rather than CHF-related crackles or interstitial-fibrosis crackles (the three most common causes of fine inspiratory crackles).

## Correct interpretation a clinician would expect

A defensible read: **"Bilateral fine inspiratory crackles, consistent with alveolar consolidation. Most likely cause in an acutely-ill patient with fever and cough: bacterial pneumonia. Differential: pulmonary edema (heart-failure-pattern crackles tend to be coarser and basilar with concomitant S3 / JVD), pulmonary fibrosis (chronic, dry, late-inspiratory). Recommend: chest X-ray to confirm consolidation, CBC, blood cultures if febrile, empiric CAP antibiotic per local antibiogram."**

A nurse-persona answer should add: place patient on continuous SpO2; obtain vital signs including temperature and respiratory rate; sit upright; consider supplemental O2 if SpO2 < 92%; report findings to physician.

## What Omni produced (smoke result)

**HTTP 400 — "Invalid or unsupported audio file"** for both WAV (16-bit PCM 16kHz mono) and MP3 (64 kbps mono 16kHz) variants.

Root cause is **server-side, not fixture-side**: the running `vllm-omni-b300` container does not have the `vllm[audio]` extras installed (no `soundfile`, no `pyav`/`av` Python module). The vLLM audio loader at `vllm/multimodal/media/audio.py` falls through both `load_audio_soundfile` and `load_audio_pyav` and raises `ValueError("Invalid or unsupported audio file.")` — this fails at media-decode time, before the fixture ever reaches the C-RADIO / Parakeet encoders.

**Verification**:

```text
$ docker exec vllm-omni-b300 python3 -c "import soundfile"
ModuleNotFoundError: No module named 'soundfile'

$ docker exec vllm-omni-b300 python3 -c "import av"
ModuleNotFoundError: No module named 'av'
```

**Required fix (owner: serve_omni_b300.sh / concurrent agent)**: rebuild or `pip install` `vllm[audio]` (which pulls in `soundfile` + `av`) inside the container, then bounce. Per the task constraints I do **not** modify `serve_*.sh` or the running container. Once the audio path is wired up, no fixture change is needed — this WAV is a clean 16 kHz mono PCM file, which is the canonical format vLLM audio expects.

The Wikimedia Commons Ogg/Vorbis source is also valid input once the path is fixed, but WAV/MP3 are simpler for stage demo packaging.

## Demo runbook scene mapping

- **SPEC §7 / Multimodal scene (3 min)** — implements the "nurse records lung sounds" beat. On-stage prompt: *"Just listened to the right lower lung — what does this sound like to you?"*
- **Pairs with `fixture-cxr-001-rml-pneumonia.png`** for the multimodal-fusion demonstration: audio crackles + chest film consolidation → graph retrieval converges on bacterial CAP node cluster.

⚠ **Demo readiness gate (this fixture only):** blocked on the server-side `vllm[audio]` install. Until that is shipped, this scene must be cut from the on-stage script or replaced with a pre-recorded answer.
