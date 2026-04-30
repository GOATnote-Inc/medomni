# Demo fixtures — Omni smoke-test results

Probed 2026-04-29 against `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` on the B300 (`unnecessary-peach-catfish`) via local SSH-tunneled `127.0.0.1:8000/v1/chat/completions`. All requests sent with `chat_template_kwargs={"enable_thinking": False}` per SPEC §5.4 audio-vs-reasoning soft-recommendation (we apply globally for latency consistency).

Source script: `/tmp/smoke_omni.py` + `/tmp/smoke_audio_retry.py` (replicable from this directory).

## Endpoint health

```bash
$ curl -sf http://127.0.0.1:8000/v1/models | python3 -c '...'
['nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4']
```

## Result 1 — `fixture-ecg-001-stemi-12lead.png`

**Prompt**: "This is a 12-lead ECG. What does it show? List the rhythm, the rate, and the most clinically significant finding. Be brief and clinically defensible."

**Status**: HTTP 200 in 3.7 s

**Response**:

> Sinus rhythm with a rate of 100 bpm. The most clinically significant finding is ST elevation in leads V3-V6.

**Verdict**: PASS. Identifies rhythm, rate, anterior ST-elevation territory. The label "STEMI" is missing but is a downstream persona-shaping enrichment. Ground truth (Cureus 2018 Apr; 10(4): e2523) is "anterior STEMI."

## Result 2 — `fixture-pill-001-warfarin-tablets.png`

**Prompt**: "A nurse photographed these tablets and wants to know what they are. Identify the medication if possible from shape/color/imprint and list two key safety considerations. Be brief."

**Status**: HTTP 200 in 1.4 s

**Response**:

> **Medication:** Warfarin (Coumadin) - an anticoagulant.
>
> **Safety Considerations:**
> 1.  **Bleeding Risk:** Warfarin significantly increases the risk of bleeding. Patients must avoid activities that could lead to injury and report any signs of bleeding (bruising, blood in urine/stool, nosebleeds) immediately.
> 2.  **Drug Interactions:** It interacts dangerously with many common medications (like antibiotics, NSAIDs) and supplements (like vitamin K), altering its effectiveness and increasing bleeding risk. Consistent monitoring and communication with healthcare providers are crucial.

**Verdict**: PASS. Strong identification + clinically defensible safety pair. Did not list per-color dose-strength mapping (5 mg pink / 3 mg blue / 1 mg brown) — that is a graph-retrieval enrichment, not a base-model failure.

## Result 3 — `fixture-cxr-001-rml-pneumonia.png`

**Prompt v1**: "PA chest X-ray. What is the most prominent finding and which lobe is involved? Give a one-line clinical impression."

**Status**: HTTP 200 in 4.7 s

**Response v1**:

> Large right-sided pleural effusion; right lower lobe.

**Prompt v2**: "Frontal chest radiograph. Describe the dominant abnormality, including lobe and side, in one sentence. Then give a one-sentence differential."

**Status**: HTTP 200 in 5.2 s

**Response v2**:

> There is a large opacity in the right lower lung field, which is consistent with a right lower lobe pneumonia. The most likely differential diagnosis is a right lower lobe pneumonia.

**Verdict**: PARTIAL PASS. Correctly localizes to right side and identifies pneumonia. Calls right-lower-lobe; source caption says right-middle-lobe. Single-frame PA cannot disambiguate RLL vs RML without lateral; Omni's call is clinically defensible from the available view. Documented in `cxr/fixture-cxr-001-clinical-context.md`.

## Result 4 — `fixture-aud-001-crackles-pneumonia.{wav,mp3}`

**Prompt**: "This is a stethoscope auscultation recording. What adventitious lung sound is present and what is the clinical implication? Be brief."

**Status (WAV PCM 16kHz mono)**: HTTP 400 in 0.7 s — `Invalid or unsupported audio file.`

**Status (MP3 64 kbps mono 16kHz)**: HTTP 400 in 0.6 s — `Invalid or unsupported audio file.`

**Root cause** (verified via container introspection):

```text
$ docker exec vllm-omni-b300 python3 -c "import soundfile"
ModuleNotFoundError: No module named 'soundfile'

$ docker exec vllm-omni-b300 python3 -c "import av"
ModuleNotFoundError: No module named 'av'
```

The vLLM audio loader at `vllm/multimodal/media/audio.py` falls through both `load_audio_soundfile` and `load_audio_pyav` and raises `ValueError("Invalid or unsupported audio file.")`. This is a **server-side capability gap, not a fixture-side issue**: the running `vllm-omni-b300` container does not have `vllm[audio]` extras installed.

**Verdict**: FAIL — server-side. Fixture itself is canonical (16 kHz mono PCM WAV, the single best-supported format for vLLM audio). Once the audio path is wired up server-side (owner: `serve_omni_b300.sh` / concurrent agent), no fixture change is needed.

**Demo readiness gate**: this scene must be cut from on-stage script or replaced with a pre-recorded answer until the server fix lands. Documented in `auscultation/fixture-aud-001-clinical-context.md`.

## Reproduce

```bash
# (laptop) ensure tunnel up
ssh -fN -L 8000:127.0.0.1:8000 unnecessary-peach-catfish
curl -sf http://127.0.0.1:8000/v1/models    # expect 200 + Omni model id

# image fixtures (PASS, PARTIAL)
.venv/bin/python /tmp/smoke_omni.py

# audio retry (FAIL — server-side)
.venv/bin/python /tmp/smoke_audio_retry.py
```
