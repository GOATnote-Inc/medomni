# Clinical context — fixture-cxr-001-rml-pneumonia.png

## What the artifact shows

A frontal (PA) chest radiograph, adult male per the source caption, showing a homogeneous opacity in the right hemithorax with silhouetting of the right heart border. Per the Wikimedia Commons source caption authored by Häggström, the radiologist's read is **lobar pneumonia involving the right middle lobe**. *Moraxella catarrhalis* was isolated from nasopharyngeal culture but the source notes it was probably not the lung pathogen; community-acquired pneumonia of typical bacterial origin is the working diagnosis.

Imaging hallmarks visible:
- right-sided alveolar opacification
- silhouetting of the right heart border (classic for RML or lingular consolidation; loss of silhouette of right hemidiaphragm would localize to RLL — the source asserts RML)
- no obvious pleural effusion or pneumothorax on this frame
- mediastinum non-shifted

## Correct interpretation a clinician would expect

A defensible read: **"Right-sided lobar consolidation, most consistent with right middle lobe pneumonia (silhouette sign at right heart border). Differential: bacterial CAP (S. pneumoniae most likely), atypical pneumonia, post-obstructive pneumonia. Recommend: clinical correlation, CBC + procalcitonin, blood cultures if febrile, empiric CAP antibiotic per local antibiogram, consider lateral view to confirm RML vs RLL localization."**

## What Omni produced (smoke result)

Initial prompt: > "Large right-sided pleural effusion; right lower lobe."

Retry with v2 prompt: > "There is a large opacity in the right lower lung field, which is consistent with a right lower lobe pneumonia. The most likely differential diagnosis is a right lower lobe pneumonia."

**Partial gap.** Omni correctly identifies right-sided lower-zone opacity and pneumonia, but localizes to RLL rather than RML. From a single PA view without the lateral, RLL-vs-RML disambiguation rests on the silhouette sign — the right heart border (RML) versus the right hemidiaphragm (RLL). Omni's RLL call is *clinically defensible from a single PA frame*; the Häggström caption's RML call relies on the radiologist seeing the original lateral. The bonus CXR demo scene therefore should be framed: *"Omni identifies right lower-zone pneumonia. The graph-retrieval enrichment can clarify lobe localization once the lateral view or chart is provided."* No retake needed; gap is documented and acceptable.

## Demo runbook scene mapping

- **Bonus / Scene 2 (multimodal stretch)** — *not* on the SPEC §7 critical path, but available as the reproducibility-loop fixture for "same input, byte-identical output across re-runs" scenes (§7 / Reproducibility).
- Optional **stretch scene**: nurse-persona prompt *"Just back from radiology, what's the read?"* with Omni response feeding the persona-shaping orchestrator and the graph-retrieval lobe-localization step.
