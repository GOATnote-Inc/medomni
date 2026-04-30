#!/usr/bin/env python3
"""emit_manifest — build the 9-layer reproducibility manifest from a finished
sovereign_bench artifact JSON. Implements SPEC §5.6.

Usage:
    python scripts/emit_manifest.py \\
        --artifact results/ci-medomni-heldout-phase1.5-<ts>/heldout.json \\
        --out      results/ci-medomni-heldout-phase1.5-<ts>/MANIFEST.yaml

Determinism contract: running this twice on the same artifact JSON, with the
same on-disk repo state, produces a byte-identical YAML. To honor that:

  * dict keys are sort-emitted alphabetically (recursive)
  * timestamps are derived only from `--frozen-timestamp` or, by default, from
    the artifact's own `generated_at` field — never from `time.time()`
  * `host` / `cli_user` are intentionally omitted from the manifest body
    (they're invariant to the artifact, not to the run)
  * pod probes (docker inspect, nvidia-smi) are gated behind `--probe-pods`;
    without that flag the script never reaches out, so consecutive runs match
  * HF Hub queries are gated behind `--probe-hf`; without that flag we resolve
    weight SHAs from the local HF cache only

This is the laptop-side emitter. It does NOT execute the bench. It does NOT
mutate any pod state.

Layers (per SPEC §5.6):
  1. Container image digests
  2. Weight SHAs (HF revision / local snapshot)
  3. Corpus SHAs
  4. Config files (sovereign_bench, judges/triton, guardrails YAML, persona)
  5. Random seeds (from artifact)
  6. Hardware foot-gun flags (parsed from serve_*.sh)
  7. Benchmark fixtures (chemoprevention 6 + tamoxifen 1)
  8. Judge model digest
  9. Git SHA + dirty count
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import OrderedDict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# -----------------------------------------------------------------------------
# Static layer config — what the manifest expects to see, regardless of whether
# the corresponding probe currently has a value. This makes the schema visible
# in every emitted manifest, "missing" included, which is itself audit-evidence.
# -----------------------------------------------------------------------------

CONTAINERS = OrderedDict([
    # name, pod-host
    ("vllm-omni-b300", "B300"),
    ("vllm-embed", "B300"),
    ("vllm-rerank", "B300"),
    ("vllm-judge", "B300"),
    ("trtllm-judge", "RunPod prism"),
    ("trtllm-rerank", "RunPod prism"),
])

MODELS = OrderedDict([
    # canonical id, role
    ("nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4", "inference"),
    ("nvidia/llama-nemotron-embed-1b-v2",                   "embedding"),
    ("nvidia/llama-3.2-nv-rerankqa-1b-v2",                  "rerank"),
    ("Qwen/Qwen2.5-7B-Instruct",                            "judge"),
    ("nvidia/NemoGuard-JailbreakDetect",                    "guard_jailbreak"),
    ("nvidia/llama-3.1-nemoguard-8b-content-safety",        "guard_content"),
    ("nvidia/Nemotron-Content-Safety-Reasoning-4B",         "guard_reasoning"),
])

CONFIG_FILES = [
    "scripts/sovereign_bench.py",
    "mla/judges/triton.py",
    "scripts/guardrails_config.yaml",
    "scripts/serve_omni_b300.sh",
    "scripts/serve_judge_b300.sh",
    "scripts/serve_retrieval_b300.sh",
    "scripts/serve_trtllm_judge_prism.sh",
    "scripts/retrieval.py",
]

SERVE_SCRIPTS = [
    "scripts/serve_omni_b300.sh",
    "scripts/serve_retrieval_b300.sh",
    "scripts/serve_judge_b300.sh",
    "scripts/serve_trtllm_judge_prism.sh",
]

CORPUS_FIXED_FILES = [
    "corpus/medical-guidelines/chunks.jsonl",
    "corpus/medical-guidelines/MANIFEST.md",
    "corpus/clinical-demo/CLN-DEMO-TAMOXIFEN-MIRENA/case.json",
    "corpus/clinical-demo/CLN-DEMO-TAMOXIFEN-MIRENA/rubric.json",
    "corpus/clinical-demo/CLN-DEMO-TAMOXIFEN-MIRENA/rubric-v2.json",
    "corpus/clinical-demo/CLN-DEMO-TAMOXIFEN-MIRENA/ideal-answer.md",
    "corpus/clinical-fixtures-heldout/MANIFEST.md",
]

# fixtures used in the held-out bench (chemoprevention 6 + tamoxifen 1 = 7)
HELDOUT_FIXTURES = [
    "corpus/clinical-fixtures-heldout/CLN-HELDOUT-5ARI-PROSTATE",
    "corpus/clinical-fixtures-heldout/CLN-HELDOUT-ASPIRIN-CRC",
    "corpus/clinical-fixtures-heldout/CLN-HELDOUT-BISPHOSPHONATE-AI",
    "corpus/clinical-fixtures-heldout/CLN-HELDOUT-HPV-CATCHUP",
    "corpus/clinical-fixtures-heldout/CLN-HELDOUT-SMOKING-CESSATION-CANCER",
    "corpus/clinical-fixtures-heldout/CLN-HELDOUT-STATIN-CV-CANCER",
]
DEMO_FIXTURE = "corpus/clinical-demo/CLN-DEMO-TAMOXIFEN-MIRENA"

# host -> (ssh-alias, optional ssh-args)
POD_SSH = {
    "B300":          ("unnecessary-peach-catfish", []),
    "RunPod prism":  ("runpod-prism",              ["-F", "configs/ssh_runpod.conf", "-tt"]),
}


# -----------------------------------------------------------------------------
# Probe helpers (best-effort; never fail the whole emitter)
# -----------------------------------------------------------------------------

def _sha256_of_path(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_of_dir_files(dir_path: Path, names: list[str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in sorted(names):
        out[name] = _sha256_of_path(dir_path / name)
    return out


def _git(cwd: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(cwd), *args],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _git_dirty_count(cwd: Path) -> int:
    out = _git(cwd, "status", "--porcelain")
    if out is None:
        return -1  # signals "not a repo" rather than "clean"
    return len([line for line in out.splitlines() if line.strip()])


def _ssh_capture(host: str, cmd: str, ssh_args: list[str], timeout: int = 8) -> str | None:
    args = ["ssh", "-o", f"ConnectTimeout={timeout}", "-o", "BatchMode=yes",
            *ssh_args, host, cmd]
    try:
        out = subprocess.check_output(
            args, stderr=subprocess.DEVNULL, text=True, timeout=timeout + 4,
        )
        return out.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None


def _docker_image_digest(host_label: str, container: str) -> str:
    """Probe a pod for `docker inspect` of one container. Returns 'unreachable'
    rather than raising. Only invoked when --probe-pods is set."""
    if host_label not in POD_SSH:
        return "unknown_pod"
    host, ssh_args = POD_SSH[host_label]
    cmd = (
        f"docker inspect --format '{{{{index .RepoDigests 0}}}}' {container} 2>/dev/null "
        f"|| docker inspect --format '{{{{.Image}}}}' {container} 2>/dev/null "
        f"|| echo 'not_running'"
    )
    out = _ssh_capture(host, cmd, ssh_args)
    if out is None:
        return "unreachable"
    return out.strip() or "not_running"


def _hf_revision_local(model_id: str) -> str | None:
    """Resolve the HF cache snapshot SHA for a model id, locally only.
    Looks under ~/.cache/huggingface/hub/models--<owner>--<name>/snapshots/<sha>.
    Picks the first snapshot dir alphabetically for determinism."""
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    folder = "models--" + model_id.replace("/", "--")
    snap_dir = cache_root / folder / "snapshots"
    if not snap_dir.is_dir():
        return None
    snapshots = sorted([p.name for p in snap_dir.iterdir() if p.is_dir()])
    return snapshots[0] if snapshots else None


def _hf_revision_remote(model_id: str) -> str | None:
    """Resolve via HfApi if huggingface_hub is importable. Best-effort."""
    try:
        from huggingface_hub import HfApi  # type: ignore
    except ImportError:
        return None
    try:
        info = HfApi().model_info(model_id, timeout=8)
        return info.sha
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Layer 6 — parse hardware foot-gun flags from serve scripts
# -----------------------------------------------------------------------------

FOOT_GUN_FLAGS = [
    "--no-async-scheduling",
    "--kv-cache-dtype",
    "--max-model-len",
    "--gpu-memory-utilization",
    "--tensor-parallel-size",
    "--max-num-seqs",
    "--reasoning-parser",
    "--tool-call-parser",
    "--enable-auto-tool-choice",
    "--video-pruning-rate",
    "--trust-remote-code",
    "--dtype",
    "--max_batch_size",
    "--max_input_len",
    "--max_seq_len",
    "--max_num_tokens",
    "--gemm_plugin",
]


def _parse_serve_flags(script_path: Path) -> dict:
    if not script_path.exists():
        return {"_status": "missing"}
    text = script_path.read_text()
    found: dict[str, str | bool] = {}
    for flag in FOOT_GUN_FLAGS:
        # capture either `--flag value` or bare `--flag` (last occurrence wins)
        pattern = rf"({re.escape(flag)})(?:[ \t]+([^\s\\]+))?"
        matches = re.findall(pattern, text)
        if not matches:
            continue
        # take last occurrence, prefer the one with a non-empty value
        non_empty = [m for m in matches if m[1]]
        if non_empty:
            value = non_empty[-1][1]
            # strip surrounding quotes
            if value.startswith('"') and value.endswith('"') and len(value) >= 2:
                value = value[1:-1]
            found[flag] = value
        else:
            found[flag] = True
    return found if found else {"_status": "no_known_flags_found"}


# -----------------------------------------------------------------------------
# Layer 5 — pull seeds + sampling from the artifact JSON
# -----------------------------------------------------------------------------

def _extract_sampling(artifact: dict) -> dict:
    return {
        "clinical_system_prompt": artifact.get("clinical_system_prompt"),
        "max_tokens":              artifact.get("max_tokens"),
        "n_per_trial":             artifact.get("n_per_trial"),
        "retrieval_mode":          artifact.get("retrieval_mode"),
        "retrieval_top_n":         artifact.get("retrieval_top_n"),
        "seed":                    artifact.get("seed"),
        "temperature":             artifact.get("temperature"),
    }


def _extract_models_used(artifact: dict) -> dict:
    return {
        "embed":  {"id": artifact.get("embed_model"),  "url": artifact.get("embed_url")},
        "judge":  {"id": artifact.get("judge_model"),  "url": artifact.get("judge_url")},
        "rerank": {"id": artifact.get("rerank_model"), "url": artifact.get("rerank_url")},
        "serve":  {"id": artifact.get("serve_model"),  "url": artifact.get("serve_url")},
    }


# -----------------------------------------------------------------------------
# Deterministic YAML dump
# -----------------------------------------------------------------------------

def _sort_recursive(obj):
    """Recursively re-order dict keys alphabetically. Lists preserved as-is
    (caller is responsible for any in-list sorting)."""
    if isinstance(obj, dict):
        return OrderedDict((k, _sort_recursive(obj[k])) for k in sorted(obj.keys()))
    if isinstance(obj, list):
        return [_sort_recursive(x) for x in obj]
    return obj


def _yaml_dump(data, out_path: Path, header: str) -> None:
    """Deterministic YAML emit. PyYAML preferred; JSON fallback is also valid YAML."""
    sorted_data = _sort_recursive(data)
    try:
        import yaml  # type: ignore

        # represent OrderedDict as plain mapping; sort_keys=True for determinism
        # (we already sorted, but belt-and-braces).
        class _SortedDumper(yaml.SafeDumper):
            pass

        def _represent_ordered_dict(dumper, data_):
            return dumper.represent_mapping("tag:yaml.org,2002:map", data_.items())

        _SortedDumper.add_representer(OrderedDict, _represent_ordered_dict)

        with out_path.open("w") as f:
            f.write(header)
            yaml.dump(
                sorted_data, f,
                Dumper=_SortedDumper,
                default_flow_style=False,
                sort_keys=True,
                allow_unicode=False,
                width=4096,
            )
        return
    except ImportError:
        pass

    with out_path.open("w") as f:
        f.write(header)
        f.write("# (JSON-as-YAML fallback; pyyaml unavailable)\n")
        json.dump(sorted_data, f, indent=2, default=str, sort_keys=True)
        f.write("\n")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True,
                        help="path to the bench artifact JSON (e.g. heldout.json)")
    parser.add_argument("--out", required=True,
                        help="path to write MANIFEST.yaml")
    parser.add_argument("--probe-pods", action="store_true",
                        help="ssh into pods to capture container image digests "
                             "(non-deterministic; only use when generating a "
                             "specific run's record, not for byte-identical re-runs)")
    parser.add_argument("--probe-hf", action="store_true",
                        help="query HF Hub to resolve model revisions; "
                             "without this flag, only the local HF cache is consulted")
    args = parser.parse_args()

    artifact_path = Path(args.artifact).resolve()
    out_path = Path(args.out).resolve()
    if not artifact_path.is_file():
        print(f"FAIL: artifact JSON not found at {artifact_path}")
        return 2

    artifact = json.loads(artifact_path.read_text())
    artifact_sha = _sha256_of_path(artifact_path)
    artifact_generated_at = artifact.get("generated_at") or "unknown"
    artifact_run_id = artifact.get("run_id") or "unknown"

    # ---- Layer 1: container image digests ---------------------------------
    layer_1_containers: dict = OrderedDict()
    for container, host in CONTAINERS.items():
        if args.probe_pods:
            layer_1_containers[container] = {
                "pod":    host,
                "digest": _docker_image_digest(host, container),
            }
        else:
            layer_1_containers[container] = {
                "pod":    host,
                "digest": "not_probed (re-emit with --probe-pods to capture)",
            }

    # ---- Layer 2: weight SHAs --------------------------------------------
    layer_2_weights: dict = OrderedDict()
    for model_id, role in MODELS.items():
        local = _hf_revision_local(model_id)
        remote = _hf_revision_remote(model_id) if args.probe_hf else None
        layer_2_weights[model_id] = {
            "role":         role,
            "local_cache":  local,
            "hf_revision":  remote,
            "resolved":     remote or local or "unresolved",
        }

    # ---- Layer 3: corpus SHAs --------------------------------------------
    layer_3_corpus: dict = OrderedDict()
    for rel in sorted(CORPUS_FIXED_FILES):
        layer_3_corpus[rel] = _sha256_of_path(REPO / rel)

    # held-out fixtures (case.json + rubric.json each)
    layer_3_heldout: dict = OrderedDict()
    for fix in sorted(HELDOUT_FIXTURES):
        fix_dir = REPO / fix
        layer_3_heldout[fix] = OrderedDict([
            ("case_sha",        _sha256_of_path(fix_dir / "case.json")),
            ("ideal_sha",       _sha256_of_path(fix_dir / "ideal-answer.md")),
            ("rubric_sha",      _sha256_of_path(fix_dir / "rubric.json")),
        ])

    # OpenEM 370 if present
    openem_dir = REPO / "data" / "openem"
    layer_3_openem: dict | None = None
    if openem_dir.is_dir():
        files = sorted([p.name for p in openem_dir.iterdir() if p.is_file()])
        layer_3_openem = OrderedDict()
        for name in files:
            layer_3_openem[name] = _sha256_of_path(openem_dir / name)

    # ---- Layer 4: config files -------------------------------------------
    layer_4_configs: dict = OrderedDict()
    for rel in sorted(CONFIG_FILES):
        layer_4_configs[rel] = _sha256_of_path(REPO / rel)

    # ---- Layer 5: random seeds (from artifact) ---------------------------
    layer_5_sampling = _extract_sampling(artifact)

    # ---- Layer 6: hardware foot-gun flags --------------------------------
    layer_6_serve_flags: dict = OrderedDict()
    for rel in sorted(SERVE_SCRIPTS):
        layer_6_serve_flags[rel] = _parse_serve_flags(REPO / rel)

    # ---- Layer 7: benchmark fixtures (chemoprevention 6 + tamoxifen 1) ---
    layer_7_fixtures: dict = OrderedDict()
    # chemoprevention held-out (6)
    chemoprevention: dict = OrderedDict()
    for fix in sorted(HELDOUT_FIXTURES):
        chemoprevention[fix] = layer_3_heldout.get(fix)
    layer_7_fixtures["chemoprevention_heldout"] = chemoprevention
    # tamoxifen demo (1)
    demo_dir = REPO / DEMO_FIXTURE
    layer_7_fixtures["tamoxifen_demo"] = OrderedDict([
        (DEMO_FIXTURE, OrderedDict([
            ("anchors_sha",     _sha256_of_path(demo_dir / "anchors.json")),
            ("case_sha",        _sha256_of_path(demo_dir / "case.json")),
            ("ideal_sha",       _sha256_of_path(demo_dir / "ideal-answer.md")),
            ("rubric_sha",      _sha256_of_path(demo_dir / "rubric.json")),
            ("rubric_v2_sha",   _sha256_of_path(demo_dir / "rubric-v2.json")),
        ])),
    ])

    # ---- Layer 8: judge model digest -------------------------------------
    judge_id = artifact.get("judge_model") or "Qwen/Qwen2.5-7B-Instruct"
    judge_local = _hf_revision_local(judge_id)
    judge_remote = _hf_revision_remote(judge_id) if args.probe_hf else None
    layer_8_judge: dict = OrderedDict([
        ("id",            judge_id),
        ("url",           artifact.get("judge_url")),
        ("local_cache",   judge_local),
        ("hf_revision",   judge_remote),
        ("resolved",      judge_remote or judge_local or "unresolved"),
    ])

    # ---- Layer 9: git -----------------------------------------------------
    layer_9_git: dict = OrderedDict([
        ("repo_path",     str(REPO.relative_to(REPO.parent))),
        ("git_sha",       _git(REPO, "rev-parse", "HEAD")),
        ("dirty_count",   _git_dirty_count(REPO)),
    ])

    # -----------------------------------------------------------------------
    # Compose
    # -----------------------------------------------------------------------
    manifest = OrderedDict([
        ("schema_version",    "medomni-manifest-v1"),
        ("artifact", OrderedDict([
            ("path",          str(artifact_path.relative_to(REPO))
                              if artifact_path.is_relative_to(REPO)
                              else str(artifact_path)),
            ("sha256",        artifact_sha),
            ("generated_at",  artifact_generated_at),
            ("run_id",        artifact_run_id),
        ])),
        ("models_used_in_run", _extract_models_used(artifact)),
        ("layer_1_containers",     layer_1_containers),
        ("layer_2_weights",        layer_2_weights),
        ("layer_3_corpus", OrderedDict([
            ("fixed_files",        layer_3_corpus),
            ("heldout_fixtures",   layer_3_heldout),
            ("openem_370",         layer_3_openem),
        ])),
        ("layer_4_configs",        layer_4_configs),
        ("layer_5_sampling",       layer_5_sampling),
        ("layer_6_serve_flags",    layer_6_serve_flags),
        ("layer_7_fixtures",       layer_7_fixtures),
        ("layer_8_judge",          layer_8_judge),
        ("layer_9_git",            layer_9_git),
    ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Generated by scripts/emit_manifest.py\n"
        "# 9-layer reproducibility manifest per SPEC.md \xa75.6\n"
        "# Schema: medomni-manifest-v1\n"
        "# Determinism: deterministic given the same on-disk repo state and the\n"
        "#              same artifact JSON. Pod / HF Hub probes are gated.\n"
    )
    _yaml_dump(manifest, out_path, header)

    # Print a one-line summary (stderr only would be cleaner but stdout is fine)
    final_sha = _sha256_of_path(out_path)
    short = final_sha[:12] if final_sha else "?"
    print(f"wrote {out_path}")
    print(f"  manifest sha256: {short}...")
    print(f"  artifact:        {artifact_path}")
    print(f"  run_id:          {artifact_run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
