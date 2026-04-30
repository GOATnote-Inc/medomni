# MedOmni v1.0 — demo fixture manifest

Fixture set assembled 2026-04-29 to back the SPEC §7 / DEMO-RUNBOOK multimodal demo. All sources are real, public-domain or open-CC-license, and clinically labeled. No fabricated content; no proprietary content; no paywalled atlas images.

## Inventory

| ID | Type | File | Source | License | Bytes | SHA-256 |
|---|---|---|---|---|---|---|
| `fixture-ecg-001` | image (ECG, 12-lead, anterior STEMI) | `ecg/fixture-ecg-001-stemi-12lead.png` | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:ST_elevation_myocardial_infarction_ECG.jpg) (Cureus 2018) | CC BY 4.0 | 2,636,608 | `9229f21123b65e4aed458d90f40cae578a60cf9e758c9fa7d03c7d2c727b81df` |
| `fixture-pill-001` | image (warfarin tablets, 5/3/1 mg trade-dress) | `pill/fixture-pill-001-warfarin-tablets.png` | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Warfarintablets5-3-1.jpg) | CC BY-SA 3.0 / GFDL | 646,082 | `482d290640d41045322617c7f262452a360c17b843ce9214e65fbc9d6834fd06` |
| `fixture-cxr-001` | image (PA chest X-ray, lobar pneumonia) | `cxr/fixture-cxr-001-rml-pneumonia.png` | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:X-ray_of_lobar_pneumonia.jpg) | CC0 1.0 (public domain) | 8,339,427 | `e84b5864ae8fb41f6374523d278501fc336e7d09120ee653e370a211e0f46694` |
| `fixture-aud-001` | audio (auscultation, pneumonia crackles) WAV PCM 16 kHz mono | `auscultation/fixture-aud-001-crackles-pneumonia.wav` | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Crackles_pneumoniaO.ogg) | CC BY-SA 3.0 / GFDL | 446,460 | `c660b9963629df342edf166f86b48640cc08013b12c0019004348856c9baa459` |
| `fixture-aud-001` (alt) | audio MP3 64 kbps mono 16 kHz | `auscultation/fixture-aud-001-crackles-pneumonia.mp3` | derived from above | (license inherited) | 112,653 | `f118848bffec21a3296612d890eb1abdd1dcc54785807d878d754a3de3d8a1ea` |

Per-fixture provenance + attribution lines live in each `*-LICENSE.md`. Per-fixture clinical interpretation + Omni smoke-result lives in each `*-clinical-context.md`.

## Demo scene mapping

The DEMO-RUNBOOK.md (T3, 2026-04-29) split SPEC §7 into 7 scenes. This fixture set serves the multimodal scenes (Scene 2 + Scene 3); ECG and CXR slot in as substitutes / stretch fixtures. Both runbook scenes 2 and 3 carry explicit `[chosen demo prop confirmed before pitch]` placeholders — these fixtures fill them.

| DEMO-RUNBOOK scene | Time | Fixture(s) | On-stage prompt | Status |
|---|---|---|---|---|
| Scene 1 — tamoxifen + Mirena open | 0:00–2:30 | (existing) `corpus/clinical-demo/CLN-DEMO-TAMOXIFEN-MIRENA/case.json` | "Premenopausal woman, BRCA+, on tamoxifen, considering Mirena IUD" | Outside this fixture set |
| **Scene 2 — Pill identification camera** | 2:30–5:00 | `fixture-pill-001` (warfarin tablets) — substitutes for the runbook's atorvastatin "Z 10" placeholder | *"I find an unfamiliar tablet in the drawer. Patient says 'that's my heart pill.' What is this?"* | **Demo-ready**. Omni identifies as warfarin and gives clinically defensible safety pair. |
| **Scene 3 — Auscultation, mobile** | 5:00–7:30 | `fixture-aud-001` (pneumonia crackles, 14 s WAV + MP3) | *"Patient just back from surgery, tachypneic. Want a second listen before I escalate."* | **BLOCKED — server-side `vllm[audio]` install missing.** Fixture is canonical 16 kHz mono PCM; once container is rebuilt with `vllm[audio]`, no fixture change needed. |
| Scene 4 — Ad placement | 7:30–9:00 | (existing) | (ad disclosure UI) | Outside this fixture set |
| Scene 5 — Adversarial + reproducibility | 9:00–10:30 | `fixture-ecg-001` available as the deterministic-replay payload (alongside Scene 1's tamoxifen case) | "Same prompt, same image — SHA256 byte-equal across two runs" | Demo-ready (deterministic decoder gates apply) |
| **Stretch — ECG single-shot** (not in current runbook scene list, but available if Sarah's persona needs a 4th bedside beat) | optional | `fixture-ecg-001` (anterior STEMI 12-lead) | *"Mid-50s patient, sudden chest pressure, this is the strip"* | **Demo-ready**. Omni names rhythm, rate, anterior ST-elevation territory. |
| **Stretch — chest X-ray** | optional | `fixture-cxr-001` (lobar pneumonia) | *"Just back from radiology, what's the read?"* | Demo-ready with documented lobe-localization gap (RLL vs RML) — clinically defensible from single-PA. |
| Scene 6 — Airplane-mode | 10:30–11:30 | (existing) | "Turn off WiFi, run a new clinical query" | Outside this fixture set |
| Scene 7 — Close | 11:30–12:00 | (existing) | "$78K / clinician / yr × 144K reachable = $11B SAM" | Outside this fixture set |

The on-stage nurse beats from the user brief — **"nurse photographs unfamiliar pill / nurse records lung sounds / nurse asks about ECG strip"** — collapse exactly onto runbook Scenes 2, 3, and the optional ECG-stretch beat using `fixture-pill-001`, `fixture-aud-001`, and `fixture-ecg-001` respectively.

## Smoke-test summary (vs. Omni `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` on B300, 2026-04-29)

| Fixture | HTTP | Latency | Verdict | Note |
|---|---|---|---|---|
| `fixture-ecg-001` | 200 | 3.7 s | **PASS** | Sinus rhythm, rate 100 bpm, anterior ST-elevation V3–V6. Clinically defensible. |
| `fixture-pill-001` | 200 | 1.4 s | **PASS** | Identifies "Warfarin (Coumadin)"; lists bleeding-risk + drug-interaction safety. Strong. |
| `fixture-cxr-001` | 200 | 4.7–5.2 s | **PARTIAL PASS** | Calls right-sided pneumonia / RLL; source caption says RML. Defensible from a single PA. |
| `fixture-aud-001` (WAV) | 400 | 0.7 s | **FAIL — server-side** | `vllm[audio]` extras not installed in container; `soundfile` and `av` both ModuleNotFoundError. Fix required in `serve_omni_b300.sh`. |
| `fixture-aud-001` (MP3) | 400 | 0.6 s | **FAIL — server-side** | Same root cause. |

Detailed responses + raw JSON in `SMOKE-RESULTS.md`.

## Limitations + v1.5 wishlist

What we have now is the minimum viable, public-domain-defensible fixture set for a 12-minute on-stage demo. The honest gaps:

1. **ECG is CC BY (not CC0).** Attribution must appear on-screen during the multimodal scene. A CC0 12-lead with a comparable STEMI footprint would be tighter; Häggström's collection has limb-lead diagrams but not a clean 12-lead STEMI strip.
2. **Audio fixture is 14 s of pneumonia crackles only.** v1.5 should add ≥1 wheezing clip (asthma-exacerbation / COPD) and ≥1 heart-sound clip (S3 gallop, systolic murmur). PASCAL Heart Sound Challenge or HF_Lung_V1 (CC BY 4.0) are open candidates once we work out per-clip license attestation.
3. **Pill fixture is a 2004-vintage 800 × 496 photograph.** Modern Pillbox / DailyMed (US-government public-domain) imagery is sharper but the dataset was retired 2021 and the FTP archive (`ftp.nlm.nih.gov/projects/pillbox/pillbox_production_images_full_202008.zip`) needs ingestion + per-image sha-mapping — defer to v1.5.
4. **CXR localization gap.** A fixture with both PA + lateral views in one matrix would let Omni do the silhouette-sign disambiguation directly; single-frame PA limits the model to lower-zone-only call.
5. **No POCUS / ultrasound video, no dermatoscopy, no scanned-discharge-summary OCR fixture** — SPEC §5.4 lists these as Omni's available modalities. v1.5 should add at least one ultrasound clip (under Omni's ≤2 min, 2 FPS × 256-frame budget) and one scanned discharge summary to exercise the OCRBenchV2-EN path.
6. **Real-time hospital-data integration is out of scope** — the v1 fixture set is a static, repeatable demo corpus, not a live FHIR pull. Live FHIR fixtures live in HealthCraft and the OpenEM corpus, not here.

## Reproducibility

`MANIFEST.md` itself + the per-fixture `LICENSE.md` files are tracked in git. The fixture binaries (`*.png`, `*.wav`, `*.mp3`) carry the SHA-256 hashes above; any drift breaks bit-identical demo replay (SPEC §5.6 layer 7) and should be treated as a regression.

Conversion provenance:
- `.jpg → .png` via macOS `sips -s format png` (lossless re-encode of decoded bitmap)
- `.ogg → .wav` via `ffmpeg -ar 16000 -ac 1 -c:a pcm_s16le`
- `.ogg → .mp3` via `ffmpeg -ar 16000 -ac 1 -b:a 64k`
