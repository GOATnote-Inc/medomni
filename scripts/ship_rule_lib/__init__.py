"""Ship-rule eval driver helpers — paired-bootstrap stats, manifest hashing,
leakage probes, and benchmark sub-runners shared by `scripts/ship_rule_eval.py`.

The two-stage grading pattern (pod-side gen, laptop-side gpt-4.1 grade) is
canonical per `findings/2026-05-05-v2.5-reasoning-sft/PREREG.yaml` and the
task #69/#70 precedents. The medomni `CLAUDE.md §2` rule "no cloud LLM keys
in any code path" applies to deployed serving on Brev pods, NOT to local
laptop-side eval tooling. The driver enforces that wall by:
    1. The pod-side runners (subcommand `gen-*`) never import openai or
       reference OPENAI_API_KEY. Output is plain JSON with text + provenance.
    2. The laptop-side grader (subcommand `grade`) imports openai lazily
       only inside the function body, after a key-presence check.
"""
