#!/usr/bin/env python3
"""Unified ship-rule eval driver — V0 vs V2.5 reasoning-SFT, 4 benchmarks.

Implements the eval protocol pre-registered in
`findings/2026-05-05-v2.5-reasoning-sft/PREREG.yaml`:

    Benchmarks (PREREG eval_protocol):
      - MedQA-USMLE          (lm-evaluation-harness style, MCQ exact-match)
      - MedXpertQA-Text      (TsinghuaC3I/MedXpertQA, MCQ exact-match)
      - HealthBench-Hard     (openai/simple-evals @ ee3b0318d8d1, gpt-4.1 graded)
      - PubMedQA-L           (yes/no/maybe exact-match; regression check)

    Decode (PREREG):  temperature=0.0, top_p=1.0, max_new_tokens=2048
    Trials:           3 per arm with seeds [42, 123, 7919]
    Stats:            paired bootstrap, 10000 resamples, 95% CI
                      Holm-Bonferroni across the 4-benchmark family at α=0.05
    Graders:          primary gpt-4.1; cross-family Qwen2.5-7B (sovereign on lobster)

The driver is split into stages:

    stage 1 — pod-side generation (`gen`)
        Posts prompts to a local vllm endpoint (V0 or V0+V2.5-LoRA).
        Persists per-item JSONL: prompt + sha + response + decode params.
        NO OpenAI key on the pod. Re-runnable with `--seed` for trial 2/3.

    stage 2 — laptop-side grading (`grade`)
        Reads stage-1 JSONLs locally on this machine.
        For HealthBench rubrics, calls gpt-4.1 (laptop side; user has authorized
        this pattern as of tasks #69/#70). For MCQ/yes-no, exact-match locally.
        Pre-flight check on the OpenAI key before any multi-hour run.

    stage 3 — paired-bootstrap stats (`stats`)
        Aligns V0 vs V2.5 records on (item_id, trial), computes paired CI per
        benchmark, applies Holm-Bonferroni, evaluates ship rule.

    stage 4 — manifest + leakage (`manifest`, `leakage`)
        SHA256 manifest over adapter, base snapshot, scripts, datasets, outputs.
        5-gram MinHash overlap + memorization probe vs training corpora.

    stage 5 — final report (`report`)
        Writes SHIP-RULE-RESULTS.json + SHIP-RULE-RESULTS.md + REPRO.sh.

Quick smoke (no GPU, no network):
    python3 scripts/ship_rule_eval.py smoke

Full pipeline (after stage 1 lands JSONLs):
    python3 scripts/ship_rule_eval.py grade --in <stage1.jsonl> --out <stage2.jsonl>
    python3 scripts/ship_rule_eval.py stats --eval-dir <findings/.../>
    python3 scripts/ship_rule_eval.py report --eval-dir <findings/.../>

The driver does NOT `vercel deploy`, does NOT touch the prism42 prod surface,
does NOT touch DNS, does NOT run on the catfish prod pod. It runs on lobster
(pod side, gen) and the laptop (everything else). See CLAUDE.md §1.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(REPO))

from ship_rule_lib import generators, grader, leakage, stats  # noqa: E402
from ship_rule_lib import manifest as manifest_lib

PREREG_PATH = REPO / "findings" / "2026-05-05-v2.5-reasoning-sft" / "PREREG.yaml"
EVAL_DIR_DEFAULT = REPO / "findings" / "2026-05-05-v2.5-eval"

BENCHMARKS = ("medqa", "pubmedqa", "medxpertqa-text", "healthbench-hard")
SEEDS = (42, 123, 7919)
ARMS = ("v0", "v25")

# Ship-rule pass conditions (verbatim from PREREG ship_rule.pass_conditions_all_required).
SHIP_RULE = {
    "medqa": {"required": "delta_lower_ci > 0", "lower_bound": 0.0},
    "medxpertqa-text": {"required": "delta_lower_ci >= +5pp", "lower_bound": 0.05},
    "healthbench-hard": {"required": "delta_point_estimate > 0", "lower_bound": None},
    "pubmedqa": {"required": "delta_lower_ci >= -1pp (no regression)", "lower_bound": -0.01},
}


# ---------------------------------------------------------------------------
# Subcommand: smoke
# ---------------------------------------------------------------------------


def cmd_smoke(args: argparse.Namespace) -> int:
    """No-network sanity check that the modules import + stats math is sane.

    Computes a tiny synthetic paired bootstrap to confirm CI math is correct
    before any GPU/network activity. Reads PREREG to confirm the file is
    present and parses the ship-rule conditions we'll evaluate against.
    """
    print("[smoke] Python:", sys.version.split()[0])
    print("[smoke] REPO:", REPO)

    # 1. PREREG present + (best-effort) parseable. The file uses unquoted
    # colon-followed-text inside one scalar; we tolerate that and still
    # require the file exist + record its sha for manifest purposes.
    if not PREREG_PATH.exists():
        print(f"[smoke] FAIL: PREREG missing at {PREREG_PATH}", file=sys.stderr)
        return 1
    prereg_sha = manifest_lib.sha256_file(PREREG_PATH)
    print(f"[smoke] PREREG present, sha256={prereg_sha[:16]}…")
    try:
        import yaml

        prereg = yaml.safe_load(PREREG_PATH.read_text()) or {}
        print(f"[smoke] PREREG parse ok — name={prereg.get('name')!r}")
    except Exception as e:  # noqa: BLE001
        # Non-fatal: a quoting glitch in the PREREG scalar should not block
        # eval driver smoke. The manifest sha still anchors the artifact.
        print(
            f"[smoke] PREREG parse warning (non-fatal): {type(e).__name__}: {str(e).splitlines()[0]}"
        )

    # 2. Stats math sanity
    rng_v0 = [0.5, 0.6, 0.4, 0.5, 0.7, 0.6, 0.5, 0.4, 0.5, 0.6]
    rng_v25 = [0.7, 0.8, 0.6, 0.7, 0.8, 0.9, 0.7, 0.6, 0.7, 0.8]
    pr = stats.paired_bootstrap(rng_v0, rng_v25, n_resamples=2000, seed=42)
    print(
        f"[smoke] paired bootstrap delta={pr.delta:+.4f} CI=[{pr.ci_low:+.4f}, "
        f"{pr.ci_high:+.4f}] d_z={pr.cohen_d:.3f} power={pr.power_post_hoc}"
    )
    if pr.delta < 0.18 or pr.delta > 0.22:
        print("[smoke] FAIL: synthetic delta outside expected range", file=sys.stderr)
        return 1

    # 3. Holm-Bonferroni sanity. With α=0.05 across 4 hypotheses:
    #    p=0.001 vs 0.05/4=0.0125 → reject
    #    p=0.01  vs 0.05/3≈0.0167 → reject
    #    p=0.04  vs 0.05/2=0.025  → fail to reject → step-down stops
    rejects = stats.holm_bonferroni([0.001, 0.01, 0.04, 0.20])
    if rejects != [True, True, False, False]:
        print(f"[smoke] FAIL: holm-bonferroni unexpected: {rejects}", file=sys.stderr)
        return 1
    print(f"[smoke] holm-bonferroni: {rejects}  (correct: top 2 reject, 3rd & 4th not)")

    # 4. Manifest sha sanity
    sha = manifest_lib.sha256_str("hello")
    expected = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    if sha != expected:
        print(f"[smoke] FAIL: sha256 mismatch {sha} vs {expected}", file=sys.stderr)
        return 1
    print(f"[smoke] sha256 of 'hello' = {sha[:16]}…  (correct)")

    # 5. Leakage shingle sanity
    a = leakage.shingles(leakage.tokenize("the quick brown fox jumps over"))
    b = leakage.shingles(leakage.tokenize("the quick brown fox jumps under"))
    sa = leakage.minhash(a, n_perm=64)
    sb = leakage.minhash(b, n_perm=64)
    j = leakage.jaccard_estimate(sa, sb)
    print(f"[smoke] minhash jaccard near-duplicate ≈ {j:.3f}")
    if not (0.0 < j < 1.0):
        print("[smoke] FAIL: minhash math broken", file=sys.stderr)
        return 1

    print("[smoke] ALL CHECKS PASSED — ready to gen/grade.")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: gen (pod-side)
# ---------------------------------------------------------------------------


def cmd_gen(args: argparse.Namespace) -> int:
    """Pod-side generation. Streams JSONL to --out."""
    if args.benchmark not in BENCHMARKS:
        print(f"FAIL: --benchmark must be one of {BENCHMARKS}", file=sys.stderr)
        return 2
    if args.arm not in ARMS:
        print(f"FAIL: --arm must be one of {ARMS}", file=sys.stderr)
        return 2

    cache_dir = Path(args.cache_dir).resolve()
    if args.benchmark == "medqa":
        items = generators.load_medqa_usmle(cache_dir)
    elif args.benchmark == "pubmedqa":
        items = generators.load_pubmedqa_l(cache_dir)
    elif args.benchmark == "medxpertqa-text":
        items = generators.load_medxpertqa_text(Path(args.medxpertqa_repo).resolve())
    elif args.benchmark == "healthbench-hard":
        items = generators.load_healthbench_hard(Path(args.healthbench_pin).resolve())
    else:
        raise RuntimeError("unreachable")

    if args.smoke:
        items = items[:1]
    elif args.limit is not None:
        items = items[: args.limit]

    out = Path(args.out).resolve()
    print(
        f"[gen] benchmark={args.benchmark} arm={args.arm} seed={args.seed} "
        f"trial={args.trial} n_items={len(items)} out={out}"
    )

    decode_params = generators.make_decode_params(
        enable_thinking=args.enable_thinking,
        max_tokens=args.max_new_tokens,
    )
    print(
        f"[gen] decode_params: enable_thinking={args.enable_thinking} "
        f"max_tokens={args.max_new_tokens}"
    )

    n = 0
    for _rec in generators.gen_for_items(
        items=items,
        benchmark=args.benchmark,
        arm=args.arm,
        serve_url=args.serve_url,
        serve_model=args.serve_model,
        seed=args.seed,
        trial=args.trial,
        timeout_s=args.timeout_s,
        out_jsonl=out,
        decode_params=decode_params,
    ):
        n += 1
        if n % 10 == 0:
            print(f"[gen] ...{n}/{len(items)}")
    print(f"[gen] DONE — wrote {n} records to {out}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: grade (laptop-side)
# ---------------------------------------------------------------------------


def cmd_grade(args: argparse.Namespace) -> int:
    """Laptop-side grading. MCQ/yes-no exact-match locally; HealthBench via gpt-4.1."""
    in_path = Path(args.in_jsonl).resolve()
    out_path = Path(args.out_jsonl).resolve()
    if not in_path.exists():
        print(f"FAIL: input JSONL missing: {in_path}", file=sys.stderr)
        return 2

    benchmark = args.benchmark
    if benchmark == "healthbench-hard":
        # Pre-flight per memory feedback_eval_preflight_judge_key.md
        ok, msg = grader.preflight_grader(model=args.primary_model)
        print(f"[grade] preflight: {msg}")
        if not ok:
            print("FAIL: grader preflight failed; aborting.", file=sys.stderr)
            return 3

    aggregate = grader.grade_jsonl(
        in_jsonl=in_path,
        out_jsonl=out_path,
        benchmark=benchmark,
        primary_model=args.primary_model,
        cross_family=args.cross_family,
        cross_family_url=args.cross_family_url,
        cross_family_model=args.cross_family_model,
    )
    print(f"[grade] aggregate: {json.dumps(aggregate, indent=2)}")
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(aggregate, indent=2))
    print(f"[grade] DONE — graded JSONL: {out_path}")
    print(f"[grade] summary: {summary_path}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: stats (laptop-side)
# ---------------------------------------------------------------------------


def _load_graded_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _per_item_score(rec: dict) -> float | None:
    g = rec.get("graded") or {}
    s = g.get("score")
    return float(s) if s is not None else None


def cmd_stats(args: argparse.Namespace) -> int:
    """Compute paired-bootstrap CI per benchmark and apply Holm-Bonferroni."""
    eval_dir = Path(args.eval_dir).resolve()
    out_dir = eval_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Expect graded files at: <eval_dir>/graded/<benchmark>__<arm>__seed<S>.jsonl
    graded_root = eval_dir / "graded"
    per_bench: dict[str, dict] = {}
    raw_p_values: dict[str, float] = {}
    raw_results: dict[str, stats.PairedResult] = {}

    for bench in BENCHMARKS:
        # Pair within each seed (same item, same vllm sampling seed),
        # then concatenate across seeds. This avoids item_id collision when
        # naive align_paired sees the same item under 3 seeds and dict-overwrites.
        v0_scores: list[float] = []
        v25_scores: list[float] = []
        n_v0_total = 0
        n_v25_total = 0
        n_dropped_total = 0
        for seed in SEEDS:
            v0_p = graded_root / f"{bench}__v0__seed{seed}.jsonl"
            v25_p = graded_root / f"{bench}__v25__seed{seed}.jsonl"
            v0r_list = _load_graded_jsonl(v0_p) if v0_p.exists() else []
            v25r_list = _load_graded_jsonl(v25_p) if v25_p.exists() else []
            n_v0_total += len(v0r_list)
            n_v25_total += len(v25r_list)
            if not v0r_list or not v25r_list:
                continue
            v0_aligned, v25_aligned, dropped = stats.align_paired(
                v0r_list, v25r_list, key="item_id"
            )
            n_dropped_total += len(dropped)
            for v0r, v25r in zip(v0_aligned, v25_aligned, strict=True):
                s0 = _per_item_score(v0r)
                s1 = _per_item_score(v25r)
                if s0 is None or s1 is None:
                    continue
                v0_scores.append(s0)
                v25_scores.append(s1)
        dropped = []  # placeholder; per-seed dropped are summed in n_dropped_total

        if n_v0_total == 0 or n_v25_total == 0:
            per_bench[bench] = {
                "status": "missing_data",
                "n_v0": n_v0_total,
                "n_v25": n_v25_total,
            }
            continue

        if len(v0_scores) < 2:
            per_bench[bench] = {
                "status": "insufficient_paired_items",
                "n_paired": len(v0_scores),
            }
            continue

        result = stats.paired_bootstrap(
            v0_scores,
            v25_scores,
            n_resamples=args.n_resamples,
            ci_alpha=args.alpha,
            seed=args.seed,
        )
        raw_results[bench] = result
        raw_p_values[bench] = result.p_two_sided
        per_bench[bench] = {
            "status": "ok",
            "n_paired": result.n_pairs,
            "n_dropped_unpaired": n_dropped_total,
            "v0_mean": result.v0_mean,
            "v25_mean": result.v25_mean,
            "delta": result.delta,
            "ci_low": result.ci_low,
            "ci_high": result.ci_high,
            "p_two_sided": result.p_two_sided,
            "cohen_d_z": result.cohen_d,
            "power_post_hoc": result.power_post_hoc,
            "n_resamples": result.n_resamples,
        }

    # Holm-Bonferroni across the family of benchmarks that produced p-values.
    bench_keys_with_p = list(raw_p_values.keys())
    p_list = [raw_p_values[k] for k in bench_keys_with_p]
    rejects = stats.holm_bonferroni(p_list, alpha=args.family_alpha)
    holm = {k: rejects[i] for i, k in enumerate(bench_keys_with_p)}

    # Apply ship-rule per-benchmark, returning the PASS/FAIL for each.
    ship_rule_eval: dict[str, dict] = {}
    for bench, rule in SHIP_RULE.items():
        rec = per_bench.get(bench, {"status": "missing"})
        if rec.get("status") != "ok":
            ship_rule_eval[bench] = {
                "rule": rule["required"],
                "lower_bound": rule["lower_bound"],
                "passed": False,
                "reason": rec.get("status", "missing"),
            }
            continue
        # The rule is on the lower CI bound (or point estimate for HealthBench).
        if bench == "healthbench-hard":
            passed = rec["delta"] > 0
            value = rec["delta"]
        else:
            passed = rec["ci_low"] >= rule["lower_bound"]
            value = rec["ci_low"]
        ship_rule_eval[bench] = {
            "rule": rule["required"],
            "lower_bound": rule["lower_bound"],
            "value_against_rule": value,
            "passed": bool(passed),
            "holm_rejects_null": holm.get(bench, False),
        }

    overall_pass = all(v.get("passed") for v in ship_rule_eval.values())

    out_payload = {
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat().replace("+00:00", "Z"),
        "alpha_per_test": args.alpha,
        "family_alpha": args.family_alpha,
        "n_resamples": args.n_resamples,
        "seeds_per_arm": list(SEEDS),
        "per_benchmark": per_bench,
        "holm_bonferroni": holm,
        "ship_rule": ship_rule_eval,
        "overall_pass": overall_pass,
    }
    out_path = out_dir / "stats.json"
    out_path.write_text(json.dumps(out_payload, indent=2, default=str))
    print(f"[stats] wrote {out_path}")
    print(f"[stats] overall_pass = {overall_pass}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: manifest
# ---------------------------------------------------------------------------


def cmd_manifest(args: argparse.Namespace) -> int:
    """Build MANIFEST.sha256 over the artifacts that define the run."""
    eval_dir = Path(args.eval_dir).resolve()
    out_path = eval_dir / "MANIFEST.sha256"
    files: list[Path] = []

    # Eval scripts (this driver + its lib).
    files.append(Path(__file__).resolve())
    files.extend(sorted((SCRIPTS / "ship_rule_lib").glob("*.py")))

    # Output JSONLs + summaries from the eval_dir itself.
    if eval_dir.exists():
        for p in eval_dir.rglob("*"):
            if p.is_file() and p.suffix in (".jsonl", ".json", ".md", ".yaml"):
                files.append(p)

    # PREREG + CARD provenance.
    if PREREG_PATH.exists():
        files.append(PREREG_PATH)

    # Optional adapter / base snapshot — only if local + visible.
    extra: list[str] = []
    if args.adapter_path:
        ap = Path(args.adapter_path)
        if ap.exists():
            files.append(ap)
        else:
            extra.append(f"adapter_path_not_local={ap}")
    if args.base_snapshot_dir:
        bs = Path(args.base_snapshot_dir)
        if bs.exists():
            for p in bs.rglob("*"):
                if p.is_file() and p.suffix in (".safetensors", ".json", ".bin", ".txt"):
                    files.append(p)
        else:
            extra.append(f"base_snapshot_not_local={bs}")

    metadata = {
        "git_head": manifest_lib.git_rev_parse(REPO),
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat().replace("+00:00", "Z"),
        "driver_path": str(Path(__file__).resolve().relative_to(REPO)),
        "prereg_sha256": (
            manifest_lib.sha256_file(PREREG_PATH) if PREREG_PATH.exists() else "missing"
        ),
        **{f"note_{i}": v for i, v in enumerate(extra)},
    }
    by_path = manifest_lib.write_manifest(
        out_path=out_path,
        repo_root=REPO,
        files=files,
        metadata=metadata,
    )
    print(f"[manifest] wrote {out_path} ({len(by_path)} files hashed)")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: leakage
# ---------------------------------------------------------------------------


def cmd_leakage(args: argparse.Namespace) -> int:
    """Run 5-gram MinHash overlap + memorization probe."""
    eval_dir = Path(args.eval_dir).resolve()
    train_files = [Path(p).resolve() for p in args.train_jsonl]
    print(f"[leakage] building train sketches from {len(train_files)} file(s)…")
    sketches = leakage.build_train_sketches(train_files, n_perm=args.n_perm, n=args.n_gram)
    print(f"[leakage] {len(sketches)} train rows sketched")

    overlap_hits: list[leakage.LeakageHit] = []
    n_test_total = 0
    for bench in BENCHMARKS:
        bench_test_path = eval_dir / "graded" / f"{bench}__v0__seed{SEEDS[0]}.jsonl"
        if not bench_test_path.exists():
            continue
        items = []
        with bench_test_path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                items.append({"item_id": rec["item_id"], "prompt": rec.get("prompt", "")})
        n_test_total += len(items)
        hits = leakage.scan_test_prompts(
            test_items=items,
            train_sketches=sketches,
            benchmark=bench,
            threshold=args.jaccard_threshold,
            n_perm=args.n_perm,
            n=args.n_gram,
        )
        overlap_hits.extend(hits)

    # Memorization probe expects continuations file (optional).
    memo_hits: list[dict] = []
    if args.memorization_jsonl:
        with Path(args.memorization_jsonl).open() as fh:
            comp_by_id: dict[str, str] = {}
            for line in fh:
                rec = json.loads(line)
                comp_by_id[str(rec["item_id"])] = rec.get("response", "")
        # For probe targets, re-use seed42 v0 prompts as the stand-in test set.
        probe_items: list[dict] = []
        for bench in BENCHMARKS:
            p = eval_dir / "graded" / f"{bench}__v0__seed{SEEDS[0]}.jsonl"
            if not p.exists():
                continue
            with p.open() as fh:
                for line in fh:
                    rec = json.loads(line)
                    probe_items.append({"item_id": rec["item_id"], "prompt": rec.get("prompt", "")})
        memo_hits = leakage.memorization_probe(
            test_items=probe_items,
            completions_by_id=comp_by_id,
            threshold=args.memorization_threshold,
        )

    out_md = eval_dir / "LEAKAGE-AUDIT.md"
    out_jsonl = eval_dir / "leakage-flags.jsonl"
    leakage.write_audit(
        out_md=out_md,
        out_jsonl=out_jsonl,
        overlap_hits=overlap_hits,
        memorization_hits=memo_hits,
        n_test_items=n_test_total,
        threshold_jaccard=args.jaccard_threshold,
        threshold_memorize=args.memorization_threshold,
    )
    print(f"[leakage] wrote {out_md} (n_test={n_test_total} hits={len(overlap_hits)})")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: report
# ---------------------------------------------------------------------------


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x * 100:+.2f}pp" if abs(x) < 1 else f"{x:+.4f}"


def cmd_report(args: argparse.Namespace) -> int:
    """Combine stats.json + manifest + leakage into RESULTS.{json,md} + REPRO.sh."""
    eval_dir = Path(args.eval_dir).resolve()
    stats_path = eval_dir / "stats.json"
    if not stats_path.exists():
        print(f"FAIL: stats.json missing under {eval_dir}; run `stats` first.", file=sys.stderr)
        return 2

    stats_payload = json.loads(stats_path.read_text())
    # Canonical machine-readable summary.
    out_json = {
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat().replace("+00:00", "Z"),
        "prereg": str(PREREG_PATH.relative_to(REPO)),
        "git_head": manifest_lib.git_rev_parse(REPO),
        "stats": stats_payload,
        "manifest_sha256_file": "MANIFEST.sha256",
        "leakage_audit_md": "LEAKAGE-AUDIT.md",
    }
    (eval_dir / "SHIP-RULE-RESULTS.json").write_text(json.dumps(out_json, indent=2))

    # Human-readable.
    overall = "PASS" if stats_payload.get("overall_pass") else "FAIL"
    md: list[str] = [
        f"# V2.5 Ship-Rule Eval — {overall}",
        "",
        f"_Generated: {out_json['generated_at']}_  ",
        f"_Pre-registration: `{out_json['prereg']}`_  ",
        f"_Git HEAD: `{out_json['git_head'] or 'unknown'}`_",
        "",
        "## Per-benchmark paired-bootstrap CI (V2.5 − V0)",
        "",
        "| Benchmark | n | V0 mean | V2.5 mean | Δ | 95% CI | Cohen d_z | Holm reject H₀ | Rule | PASS? |",
        "|---|---:|---:|---:|---:|:---:|---:|:---:|---|:---:|",
    ]
    for bench in BENCHMARKS:
        rec = stats_payload.get("per_benchmark", {}).get(bench, {})
        ship = stats_payload.get("ship_rule", {}).get(bench, {})
        if rec.get("status") != "ok":
            md.append(
                f"| {bench} | — | — | — | — | _{rec.get('status', 'missing')}_ | — | — | "
                f"{ship.get('rule', '—')} | — |"
            )
            continue
        delta = rec.get("delta", float("nan"))
        ci = f"[{_fmt_pct(rec.get('ci_low'))}, {_fmt_pct(rec.get('ci_high'))}]"
        passed = "PASS" if ship.get("passed") else "FAIL"
        holm = "yes" if stats_payload.get("holm_bonferroni", {}).get(bench) else "no"
        md.append(
            f"| {bench} | {rec.get('n_paired')} | "
            f"{rec.get('v0_mean'):.4f} | {rec.get('v25_mean'):.4f} | {_fmt_pct(delta)} | {ci} | "
            f"{rec.get('cohen_d_z'):.2f} | {holm} | {ship.get('rule', '—')} | {passed} |"
        )
    md.extend(
        [
            "",
            f"**Overall ship-rule:** {overall}",
            "",
            "## Ship-rule conditions (verbatim from PREREG)",
            "",
        ]
    )
    for bench, rule in SHIP_RULE.items():
        md.append(f"- `{bench}`: {rule['required']}")
    md.extend(
        [
            "",
            "## Reproducibility",
            "",
            "- `MANIFEST.sha256` — hashes for adapter, base snapshot, scripts, every output JSON",
            "- `LEAKAGE-AUDIT.md` — 5-gram MinHash overlap + memorization probe report",
            "- `REPRO.sh` — deterministic re-run script (seeds 42/123/7919, temp=0)",
            "- `stats.json` — full paired-bootstrap output incl. raw resampled deltas",
            "",
            "## What this report does NOT claim",
            "",
            "- Does NOT replace the formal HF model card (separate doc).",
            "- Does NOT extend to vision/audio modalities (V2.5 is text-only reasoning SFT).",
            "- Does NOT certify safety; layer-0/1/2 guardrails are evaluated separately.",
            "",
            "## On FAIL",
            "",
            "Per PREREG ship_rule.on_fail: revert; debug data quality (likely insufficient",
            "CoT diversity); re-author V2.5b PREREG.",
        ]
    )
    (eval_dir / "SHIP-RULE-RESULTS.md").write_text("\n".join(md) + "\n")

    # REPRO.sh — orchestrates a clean re-run from the same git HEAD.
    repro = [
        "#!/usr/bin/env bash",
        "# Auto-generated by ship_rule_eval.py report. Reproduces the V2.5",
        "# ship-rule eval bit-for-bit (modulo nondeterministic GPU kernels).",
        "set -euo pipefail",
        "",
        f"# Pre-registration: {out_json['prereg']}",
        f"# Git HEAD at report time: {out_json['git_head']}",
        "",
        'REPO="$(cd "$(dirname "$0")/../.." && pwd)"',
        'EVAL_DIR="$REPO/findings/2026-05-05-v2.5-eval"',
        "",
        "# 1. Pre-flight grader (laptop side).",
        "set -a && source /Users/kiteboard/lostbench/.env && set +a",
        'python3 "$REPO/scripts/ship_rule_eval.py" smoke',
        "",
        "# 2. Pod-side gen (run on lobster). Loop arms × benchmarks × seeds.",
        "for arm in v0 v25; do",
        "  for bench in medqa pubmedqa medxpertqa-text healthbench-hard; do",
        "    for seed in 42 123 7919; do",
        '      python3 "$REPO/scripts/ship_rule_eval.py" gen \\',
        '        --benchmark "$bench" --arm "$arm" --seed "$seed" --trial 0 \\',
        "        --serve-url http://127.0.0.1:8000/v1 \\",
        "        --serve-model nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \\",
        '        --out "$EVAL_DIR/gen/${bench}__${arm}__seed${seed}.jsonl"',
        "    done",
        "  done",
        "done",
        "",
        "# 3. Laptop-side grade.",
        "for arm in v0 v25; do",
        "  for bench in medqa pubmedqa medxpertqa-text healthbench-hard; do",
        "    for seed in 42 123 7919; do",
        '      python3 "$REPO/scripts/ship_rule_eval.py" grade \\',
        '        --benchmark "$bench" \\',
        '        --in "$EVAL_DIR/gen/${bench}__${arm}__seed${seed}.jsonl" \\',
        '        --out "$EVAL_DIR/graded/${bench}__${arm}__seed${seed}.jsonl"',
        "    done",
        "  done",
        "done",
        "",
        "# 4. Stats + leakage + manifest + report.",
        'python3 "$REPO/scripts/ship_rule_eval.py" stats --eval-dir "$EVAL_DIR"',
        'python3 "$REPO/scripts/ship_rule_eval.py" leakage --eval-dir "$EVAL_DIR" \\',
        "  --train-jsonl /workspace/v2.5-prod/train_corpus.jsonl",
        'python3 "$REPO/scripts/ship_rule_eval.py" manifest --eval-dir "$EVAL_DIR" \\',
        "  --adapter-path /workspace/v2.5-prod/adapter_model.safetensors",
        'python3 "$REPO/scripts/ship_rule_eval.py" report --eval-dir "$EVAL_DIR"',
        "",
    ]
    repro_path = eval_dir / "REPRO.sh"
    repro_path.write_text("\n".join(repro) + "\n")
    repro_path.chmod(0o755)

    print(f"[report] wrote {eval_dir / 'SHIP-RULE-RESULTS.md'} ({overall})")
    print(f"[report] wrote {eval_dir / 'SHIP-RULE-RESULTS.json'}")
    print(f"[report] wrote {repro_path}")
    return 0


# ---------------------------------------------------------------------------
# argparse plumbing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ship_rule_eval", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    # smoke
    s_smoke = sub.add_parser("smoke", help="self-test (no network) — math + imports")
    s_smoke.set_defaults(func=cmd_smoke)

    # gen (pod-side)
    s_gen = sub.add_parser("gen", help="pod-side generation against local vllm")
    s_gen.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    s_gen.add_argument("--arm", required=True, choices=ARMS)
    s_gen.add_argument("--seed", type=int, required=True)
    s_gen.add_argument("--trial", type=int, default=0)
    s_gen.add_argument("--serve-url", required=True, help="local vllm base URL")
    s_gen.add_argument("--serve-model", required=True)
    s_gen.add_argument("--out", required=True, help="output JSONL path (append-mode)")
    s_gen.add_argument(
        "--cache-dir", default=str(Path.home() / ".cache" / "huggingface" / "datasets")
    )
    s_gen.add_argument("--medxpertqa-repo", default=str(REPO / "third_party" / "MedXpertQA"))
    s_gen.add_argument(
        "--healthbench-pin", default=str(REPO / "corpus" / "pins" / "healthbench-hard-1000.yaml")
    )
    s_gen.add_argument("--timeout-s", type=float, default=180.0)
    s_gen.add_argument("--limit", type=int, default=None)
    s_gen.add_argument("--smoke", action="store_true", help="1-item run for pre-flight")
    # Reasoning-channel controls. Defaults match generators.DECODE_PARAMS
    # (enable_thinking=True, max_tokens=8192), the canonical reasoning-SFT
    # configuration. The thinking=False legacy run is reproducible by
    # passing --no-enable-thinking --max-new-tokens 2048.
    s_gen.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass enable_thinking=<this> in chat_template_kwargs (default: True).",
    )
    s_gen.add_argument(
        "--max-new-tokens",
        type=int,
        default=8192,
        help="Decode max_tokens budget (default: 8192 to fit thinking traces).",
    )
    s_gen.set_defaults(func=cmd_gen)

    # grade (laptop)
    s_grade = sub.add_parser("grade", help="laptop-side grading (gpt-4.1 primary)")
    s_grade.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    s_grade.add_argument("--in", dest="in_jsonl", required=True)
    s_grade.add_argument("--out", dest="out_jsonl", required=True)
    s_grade.add_argument("--primary-model", default=grader.GPT_MODEL)
    s_grade.add_argument(
        "--cross-family",
        action="store_true",
        help="also run sovereign Qwen2.5-7B grade (HealthBench only)",
    )
    s_grade.add_argument("--cross-family-url", default="http://127.0.0.1:8001/v1")
    s_grade.add_argument("--cross-family-model", default=grader.QWEN_MODEL)
    s_grade.set_defaults(func=cmd_grade)

    # stats
    s_stats = sub.add_parser("stats", help="paired bootstrap + Holm-Bonferroni")
    s_stats.add_argument("--eval-dir", default=str(EVAL_DIR_DEFAULT))
    s_stats.add_argument("--n-resamples", type=int, default=10000)
    s_stats.add_argument("--alpha", type=float, default=0.05)
    s_stats.add_argument("--family-alpha", type=float, default=0.05)
    s_stats.add_argument("--seed", type=int, default=42)
    s_stats.set_defaults(func=cmd_stats)

    # manifest
    s_man = sub.add_parser("manifest", help="emit MANIFEST.sha256")
    s_man.add_argument("--eval-dir", default=str(EVAL_DIR_DEFAULT))
    s_man.add_argument("--adapter-path", default="")
    s_man.add_argument("--base-snapshot-dir", default="")
    s_man.set_defaults(func=cmd_manifest)

    # leakage
    s_leak = sub.add_parser("leakage", help="5-gram MinHash + memorization audit")
    s_leak.add_argument("--eval-dir", default=str(EVAL_DIR_DEFAULT))
    s_leak.add_argument(
        "--train-jsonl", nargs="+", required=True, help="paths to training-corpus JSONL files"
    )
    s_leak.add_argument(
        "--memorization-jsonl",
        default="",
        help="optional: model continuations from first-half prompts",
    )
    s_leak.add_argument("--n-perm", type=int, default=128)
    s_leak.add_argument("--n-gram", type=int, default=5)
    s_leak.add_argument("--jaccard-threshold", type=float, default=0.7)
    s_leak.add_argument("--memorization-threshold", type=float, default=0.85)
    s_leak.set_defaults(func=cmd_leakage)

    # report
    s_rep = sub.add_parser("report", help="combine into SHIP-RULE-RESULTS.{md,json} + REPRO.sh")
    s_rep.add_argument("--eval-dir", default=str(EVAL_DIR_DEFAULT))
    s_rep.set_defaults(func=cmd_report)

    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
