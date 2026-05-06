---
name: differential
description: Tree-of-Thoughts differential diagnosis for diagnostic-uncertainty cases
trigger:
  - "differential"
  - "ddx"
  - "what could cause"
  - "could it be"
---

When the user is asking for a differential diagnosis (DDx), follow this protocol instead of single-shot reasoning. Per AMIE-DDx (Nature 2025) + MedKGI (arxiv 2512.24181): top-10 DDx accuracy = 59.1% with structured branching vs 33.6% unassisted.

## Protocol

1. **RETRIEVE** the patient's vitals, active conditions, recent labs, and current meds first. DDx without data = bad DDx.

2. Emit `<ddx-tree>` with **5 branches**, each anchored to a different reasoning frame:
   - Branch A — most-likely / common (think Bayesian prior)
   - Branch B — most-dangerous / can't-miss (think rule-out)
   - Branch C — rare but pattern-matching (think zebra)
   - Branch D — iatrogenic / drug-induced (think recent meds)
   - Branch E — systemic / multi-organ (think autoimmune, paraneoplastic)

3. For each branch, list the **top 2 candidates** + the **discriminating data** that would confirm or reject.

4. Emit a **ranked top-K list** (K=5-10 per condition severity) with:
   - Probability bucket (Most likely | Likely | Possible | Don't miss | Rare)
   - Next test that would change probability the most

5. **Mark uncertainty.** If conformal-prediction set size > 10 OR semantic entropy probe > 0.6, append `<abstain-ranking/>` — your ranking is unreliable; recommend specialist consult.

## Example

User: "30F with sudden left-arm weakness, recent transcontinental flight, swollen R calf"

```
<ddx-tree>
  <branch-A label="most-likely">
    1. Embolic stroke from DVT (paradoxical embolism via PFO)
       discriminating: head CT, echo with bubble study
    2. Functional / conversion disorder
       discriminating: imaging negative + neuro exam pattern
  </branch-A>
  <branch-B label="cant-miss">
    1. Hemorrhagic stroke (less likely given clinical picture but rule out)
       discriminating: head CT
    2. Aortic dissection w/ stroke
       discriminating: CT angio
  </branch-B>
  <branch-C label="zebra">
    1. Multiple sclerosis flare
       discriminating: MRI brain + spine
  </branch-C>
  ...
</ddx-tree>

Ranked DDx (severity × probability):
1. [Most likely, can't-miss] Paradoxical embolic stroke via PFO + DVT
2. [Likely] Hemorrhagic stroke
3. [Don't miss] Aortic dissection
4. [Possible] MS flare
...

Recommended next steps: head CT + CT angio + lower-ext venous Doppler + bubble-study echo.
```
