# Clinical context — fixture-ecg-001-stemi-12lead.png

## What the artifact shows

A standard 12-lead surface ECG with the conventional 6-limb-lead + 6-precordial-lead layout (I, II, III, aVR, aVL, aVF, V1–V6) and a long lead-II rhythm strip across the bottom. Findings consistent with the Cureus source caption ("ST-elevation myocardial infarction in the anterior leads") plus an underlying sinus rhythm at approximately 90–100 bpm. ST-segment elevation is most prominent in the precordial leads (V2–V5 territory) — this is the textbook footprint of an **anterior wall STEMI** secondary to LAD-territory occlusion.

## Correct interpretation a clinician would expect

A defensible read: **"Sinus rhythm, rate ~95 bpm, ST-elevation across V2–V5 (anterior leads) consistent with acute anterior STEMI. This is a STEMI alert: activate cath lab, dual-antiplatelet load, anticoagulate, transport to PCI-capable facility within first-medical-contact-to-balloon < 90 min."**

A nurse-persona answer should add: report finding to physician immediately; do not delay transport for additional testing; place on continuous cardiac monitor; large-bore IV × 2; obtain right-sided leads if inferior changes are suspected.

## What Omni produced (smoke result)

> "Sinus rhythm with a rate of 100 bpm. The most clinically significant finding is ST elevation in leads V3-V6."

Defensible. Identifies rhythm, rate, and the anterior ST-elevation territory. Does not explicitly use the word "STEMI" — that label can be added by the persona-shaping orchestrator (the Colang post-processor or the persona prompt prepend), not by the base model. **No retake needed.**

## Demo runbook scene mapping

- **SPEC §7 / Multimodal scene (3 min)** — "Upload an ECG image (12-lead PNG) plus an audio clip 'I have palpitations.'" This fixture is the ECG image. The nurse-on-stage prompt: *"Patient is mid-50s, sudden chest pressure, this is the strip — what do you see?"*
- Optional re-use as a **single-modality benchmark fixture** for the rerun-and-bit-identical-SHA256 reproducibility scene (§7 / Reproducibility, 2 min).
