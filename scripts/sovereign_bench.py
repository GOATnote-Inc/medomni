#!/usr/bin/env python3
"""Sovereign HealthBench sweep — Triton-served Nemotron, Triton-served judge.

Drop-in replacement for the Opus-4.7 path in `healthbench_runner.py` that
keeps every grading invariant (paired-design CI machinery, RECUSED items,
audit logging) but routes both the model-under-test and the judge through
local OpenAI-compatible endpoints exposed via ssh tunnel from the H200 /
H100 Brev pods.

No `anthropic` import. No `openai` SDK. Only `httpx` to the local
endpoints, plus the existing `_healthbench_grader_bridge` for the
GRADER_TEMPLATE/calculate_score.

Usage:
    python scripts/sovereign_bench.py \\
        --manifest corpus/pins/healthbench-hard-1000.yaml \\
        --serve-url http://127.0.0.1:8000/v1 \\
        --serve-model nvidia/Llama-3.1-Nemotron-70B-Instruct-HF \\
        --judge-url http://127.0.0.1:8001/v1 \\
        --judge-model meta-llama/Llama-3.1-Nemotron-70B-Instruct-AWQ-INT4 \\
        --n 30 --trials 3 --seed 42 \\
        --out results/r1-bare/healthbench-hard-n30.json

Both --serve-url and --judge-url MUST be 127.0.0.1/localhost. External
URLs raise immediately — see CLAUDE.md §2.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import httpx
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from healthbench_runner import (  # noqa: E402
    HEALTHBENCH_AXES,
    _aggregate,
    _load_manifest,
    _now_iso,
    _real_grader,
    _write_out,
)
from mla.judges.triton import make_triton_judge  # noqa: E402


def _require_local(url: str, label: str) -> None:
    if not url.startswith(("http://127.0.0.1", "http://localhost")):
        raise ValueError(
            f"{label} must be a 127.0.0.1/localhost URL, got {url!r}. "
            "External URLs defeat the sovereign-stack design — see CLAUDE.md §2."
        )


CLINICAL_SYSTEM_PROMPT = (
    "You are a clinical decision-support system assisting a physician–patient "
    "shared-decision-making conversation. Do NOT include disclaimers like "
    "\"I am not a medical professional\" or \"consult your doctor.\" Do NOT "
    "mention being an AI. Be precise, structured, and evidence-aware. "
    "Structure every answer as: (1) Direct answer, (2) Mechanism, "
    "(3) Risk factors, (4) What we know vs do not know, (5) Practical next steps."
)


def _retrieve_anchors(case_path: Path) -> list[dict]:
    """Slot for the retrieval step. Today: read sibling anchors.json (a
    deterministic curated fact bundle keyed to the case_id). Tomorrow: same
    return shape, sourced from NV-Embed-v2 + nx-cugraph subgraph slice over
    the seed KG + OpenEM corpus. The contract is `case_path → list[{id, fact,
    citation, supports: [rubric_ids]}]`. Returns [] if no anchors file."""
    anchors_path = case_path.parent / "anchors.json"
    if not anchors_path.exists():
        return []
    data = json.loads(anchors_path.read_text())
    return list(data.get("facts", []))


def _format_anchors_block(facts: list[dict]) -> str:
    """Render retrieved facts as a single context block. Production swap-in
    will produce the same shape from a graph-RAG subgraph."""
    if not facts:
        return ""
    lines = [
        "Reference facts retrieved for this case (cite inline where used):",
        "",
    ]
    for f in facts:
        lines.append(f"- [{f.get('id', '?')}] {f.get('fact', '').strip()}")
        cit = f.get("citation")
        if cit:
            lines.append(f"  Source: {cit}")
    return "\n".join(lines)


def _generate(
    *,
    client: httpx.Client,
    model: str,
    messages: list[dict],
    max_tokens: int,
    timeout_s: float,
    temperature: float = 0.0,
    system_prompt: str | None = None,
) -> str:
    """Single chat completion against the local serve endpoint."""
    msgs = list(messages)
    if system_prompt and not (msgs and msgs[0].get("role") == "system"):
        msgs = [{"role": "system", "content": system_prompt}, *msgs]
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": msgs,
        "temperature": temperature,
        # Omni emits chain-of-thought into `reasoning` when thinking is on,
        # leaving `content` null. Variance shrinks materially with thinking
        # off; the rubric scores phrasing not chain-of-thought.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    # base_url ends with /v1; relative path so httpx joins correctly.
    resp = client.post("chat/completions", json=body, timeout=timeout_s)
    resp.raise_for_status()
    payload = resp.json()
    msg = payload["choices"][0]["message"]
    return msg.get("content") or msg.get("reasoning") or ""


def _load_single_case(case_path: Path) -> dict:
    """Adapt a single clinical-demo case + sibling rubric.json into a
    HealthBench-shaped example with embedded `rubrics`. Used by the
    tamoxifen acceptance test: case.json carries the conversation,
    rubric.json carries weighted criteria — we map weight (0.0-1.0
    summing to 1.0) onto HealthBench `points` by *100 so calculate_score
    sees integer-friendly weights and the ratio is preserved.

    Prefers `rubric-v2.json` if present (description-graded, paraphrase-
    tolerant) over `rubric.json` (legacy v1 with hard keyword gates).
    """
    case = json.loads(case_path.read_text())
    rubric_v2 = case_path.parent / "rubric-v2.json"
    rubric_v1 = case_path.parent / "rubric.json"
    if rubric_v2.exists():
        rubric_path = rubric_v2
    elif rubric_v1.exists():
        rubric_path = rubric_v1
    else:
        raise FileNotFoundError(
            f"single-case manifest {case_path} requires sibling rubric.json or rubric-v2.json"
        )
    rubric = json.loads(rubric_path.read_text())
    rubrics: list[dict] = []
    for c in rubric.get("criteria", []):
        rubrics.append(
            {
                "criterion": c["description"],
                "points": float(c.get("weight", 0.0)) * 100.0,
                "tags": list(c.get("tags", [])),
            }
        )
    return {
        "id": case.get("id", case_path.parent.name),
        "messages": case.get("messages", []),
        "rubrics": rubrics,
    }


def _load_examples(manifest_path: Path) -> list[dict]:
    """Return list of HealthBench-shaped examples (each with embedded rubrics).

    Three manifest shapes supported:
      1. HealthBench YAML/JSON pin: top-level `examples: [...]`.
      2. Single clinical-demo case: a case.json + sibling rubric.json
         (or rubric-v2.json — preferred).
      3. Fixtures directory: a path to a directory containing
         `*/case.json`, each with a sibling rubric.json. Each subdir is
         loaded as one example. Used for the held-out fixture set.
    """
    p = manifest_path
    if p.is_dir():
        # Fixtures-directory shape — glob for */case.json
        case_paths = sorted(p.glob("*/case.json"))
        if not case_paths:
            return []
        return [_load_single_case(cp) for cp in case_paths]
    if p.suffix == ".json":
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError:
            data = {}
        if isinstance(data, dict) and "messages" in data and "examples" not in data:
            return [_load_single_case(p)]
    manifest = _load_manifest(p)
    return list(manifest.get("examples", []))


def _example_messages(example: dict) -> list[dict]:
    msgs = example.get("messages") or [
        {"role": "user", "content": example.get("prompt", "")}
    ]
    return [{"role": m["role"], "content": m["content"]} for m in msgs]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="HealthBench Hard YAML manifest")
    parser.add_argument("--serve-url", required=True, help="local serve base URL (e.g. http://127.0.0.1:8000/v1)")
    parser.add_argument("--serve-model", required=True, help="model id to send to /chat/completions")
    parser.add_argument("--judge-url", required=True, help="local judge base URL")
    parser.add_argument("--judge-model", required=True, help="judge model id")
    parser.add_argument("--n", type=int, default=30, help="examples per trial")
    parser.add_argument("--trials", type=int, default=3, help="N trials for paired-design CI")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--clinical-system-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="prepend the clinical-decision-support system message (default on)",
    )
    parser.add_argument(
        "--retrieval",
        choices=["none", "anchors", "bm25", "dense", "hybrid", "primekg-hybrid"],
        default="anchors",
        help=(
            "retrieval mode. `anchors`: deterministic curated bundle (legacy "
            "ablation). `bm25` / `dense` / `hybrid`: real retrieval over the "
            "chunked corpus at --corpus-path. `primekg-hybrid`: hybrid "
            "chunk retrieval PLUS Stage-6 PrimeKG subgraph (Phase 2.4). "
            "`none`: no retrieval."
        ),
    )
    parser.add_argument(
        "--corpus-path",
        default="corpus/medical-guidelines/chunks.jsonl",
        help="path to chunked corpus JSONL (used by bm25/dense/hybrid).",
    )
    parser.add_argument(
        "--embed-url", default="http://127.0.0.1:8001/v1",
        help="NeMo Retriever embedding NIM base URL.",
    )
    parser.add_argument(
        "--embed-model", default="nvidia/llama-nemotron-embed-1b-v2",
        help="embedding model id.",
    )
    parser.add_argument(
        "--rerank-url", default="http://127.0.0.1:8002/v1",
        help="reranker NIM base URL (use empty string '' to disable rerank).",
    )
    parser.add_argument(
        "--rerank-model", default="nvidia/llama-3.2-nv-rerankqa-1b-v2",
        help="reranker model id.",
    )
    parser.add_argument(
        "--retrieval-top-n",
        type=int,
        default=8,
        help="number of chunks to inject into the context block.",
    )
    # ----- Phase 2.1 flags -----
    parser.add_argument(
        "--retrieval-cuvs",
        action="store_true",
        help=(
            "swap the dense-recall stage from numpy cosine to cuVS IVF-PQ / "
            "brute-force (RAPIDS 26.04). Requires cuvs-cu13 / cupy-cuda13x in "
            "the runtime; silently falls back to numpy if unavailable. The "
            "actual backend used is recorded in the artifact JSON as "
            "`dense_backend`."
        ),
    )
    parser.add_argument(
        "--graph-expand",
        action="store_true",
        help=(
            "Stage 6: nx-cugraph 2-hop ego-graph expansion over the persona-"
            "tagged chemoprevention entity graph. Expanded subgraph is "
            "serialized into the system prompt alongside retrieved chunks."
        ),
    )
    # ----- Phase 2.4 PrimeKG flags -----
    parser.add_argument(
        "--primekg-path",
        default="/home/shadeform/medomni/primekg/primekg.gpickle",
        help=(
            "path to the PrimeKG pickle written by "
            "scripts/build_primekg_cugraph.py. Used by --retrieval primekg-hybrid "
            "when --primekg-url is empty (i.e., bench runs on B300 directly)."
        ),
    )
    parser.add_argument(
        "--primekg-url",
        default="",
        help=(
            "HTTP base URL of the B300 PrimeKG service "
            "(scripts/serve_primekg_b300.py). When set, the bench fetches "
            "subgraph blocks via /subgraph instead of loading the pickle "
            "locally — used when the bench runs on the laptop side and the "
            "129K-node graph + RAPIDS dispatch only live on B300. The URL "
            "must be 127.0.0.1/localhost (forwarded by ssh tunnel)."
        ),
    )
    parser.add_argument(
        "--primekg-max-hops",
        type=int,
        default=2,
        help="BFS depth for PrimeKG subgraph expansion (default 2).",
    )
    parser.add_argument(
        "--primekg-max-nodes",
        type=int,
        default=100,
        help="cap on PrimeKG subgraph size (BFS-order; default 100).",
    )
    parser.add_argument(
        "--primekg-max-tokens",
        type=int,
        default=2048,
        help="token budget for the serialized PrimeKG block in the prompt.",
    )
    parser.add_argument(
        "--graph-persona",
        choices=["physician", "nurse", "family", "patient"],
        default="patient",
        help="persona bit for graph expansion's edge filter (default: patient).",
    )
    parser.add_argument(
        "--guardrails",
        action="store_true",
        help=(
            "wrap each Omni call in NeMo Guardrails 0.21.0 input + output "
            "rails. v0 implementation is a SHIM (regex / heuristic "
            "classifier) since NemoGuard models are not yet served on B300; "
            "production swap is a 1-line change to point the rail action "
            "handlers at the NemoGuard / Nemotron-Content-Safety endpoints."
        ),
    )
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--out", required=True, help="output JSON path")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="single-example smoke run; READ the JSON artifact before scaling up",
    )
    # ----- Phase 2.2 host shortcuts -----
    # Convenience flags that map to known tunnel-port conventions:
    #   b300:  judge=8003 / rerank=8002 / embed=8001 (Phase 2.1 baseline)
    #   prism: judge=9003 / rerank=9002             (Phase 2.2 TRT-LLM-FP8 + vllm)
    # If set, override --judge-url / --rerank-url; else --judge-url/--rerank-url stay.
    parser.add_argument(
        "--judge-host",
        choices=["b300", "prism"],
        default=None,
        help=(
            "shortcut for judge endpoint. b300=http://127.0.0.1:8003/v1 (vllm-BF16), "
            "prism=http://127.0.0.1:9003/v1 (TRT-LLM-FP8). Overrides --judge-url."
        ),
    )
    parser.add_argument(
        "--rerank-host",
        choices=["b300", "prism"],
        default=None,
        help=(
            "shortcut for reranker endpoint. b300=http://127.0.0.1:8002/v1, "
            "prism=http://127.0.0.1:9002/v1. Overrides --rerank-url."
        ),
    )
    args = parser.parse_args()

    # Apply Phase 2.2 host shortcuts. The Qwen2.5-7B model id is identical
    # whether served by vllm-BF16 (B300) or TRT-LLM-FP8 (prism); only the
    # tunnel port differs. The reranker model id is similarly stable.
    if args.judge_host == "b300":
        args.judge_url = "http://127.0.0.1:8003/v1"
    elif args.judge_host == "prism":
        args.judge_url = "http://127.0.0.1:9003/v1"
    if args.rerank_host == "b300":
        args.rerank_url = "http://127.0.0.1:8002/v1"
    elif args.rerank_host == "prism":
        args.rerank_url = "http://127.0.0.1:9002/v1"

    _require_local(args.serve_url, "--serve-url")
    _require_local(args.judge_url, "--judge-url")

    manifest_path = Path(args.manifest).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    examples = _load_examples(manifest_path)
    if not examples:
        print(f"FAIL: {manifest_path}: no examples in manifest", file=sys.stderr)
        return 1

    n_per_trial = 1 if args.smoke else args.n
    trials = 1 if args.smoke else args.trials
    examples = examples[:n_per_trial]

    run_id = uuid.uuid4().hex[:8]
    started = time.time()

    audit_dir = out_path.parent / "judge-audit"
    audit_dir.mkdir(exist_ok=True)
    judge_fn = make_triton_judge(
        base_url=args.judge_url,
        model_id=args.judge_model,
        audit_log_path=audit_dir / f"judge-{run_id}.jsonl",
        max_retries=3,
    )

    serve_client = httpx.Client(base_url=args.serve_url, timeout=args.timeout_s)

    retriever = None
    n_corpus_chunks = 0
    if args.retrieval in ("bm25", "dense", "hybrid", "primekg-hybrid"):
        from retrieval import HybridRetriever
        rerank_url = args.rerank_url.strip() or None
        retriever = HybridRetriever.from_jsonl(
            Path(args.corpus_path).resolve(),
            embed_url=args.embed_url,
            embed_model=args.embed_model,
            rerank_url=rerank_url,
            rerank_model=args.rerank_model,
            use_cuvs=args.retrieval_cuvs,
        )
        n_corpus_chunks = len(retriever.chunks)
        if args.retrieval in ("dense", "hybrid"):
            print(
                f"[sovereign_bench] building dense index for "
                f"{n_corpus_chunks} chunks via {args.embed_url}..."
            )
            retriever.build_dense_index()
            print(f"[sovereign_bench] dense index ready (backend={retriever.dense_backend})")

    # Phase 2.1 graph expansion (Stage 6).
    graph = None
    if args.graph_expand:
        try:
            from graph_subgraph_slice import (  # noqa: WPS433
                build_graph,
                expand_subgraph,
                seed_nodes_for,
                serialize_subgraph_for_prompt,
            )
            graph = build_graph()
            print(
                f"[sovereign_bench] graph loaded: "
                f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges; "
                f"persona={args.graph_persona}"
            )
        except Exception as e:  # noqa: BLE001
            print(f"[sovereign_bench] graph expansion DISABLED: {e}")
            graph = None

    # Phase 2.4 PrimeKG load (Stage 6 — factual layer).
    primekg = None
    primekg_client: httpx.Client | None = None
    primekg_meta: dict = {}
    if args.retrieval == "primekg-hybrid":
        if args.primekg_url:
            # Remote service mode: laptop side, B300 hosts the graph.
            _require_local(args.primekg_url, "--primekg-url")
            primekg_client = httpx.Client(
                base_url=args.primekg_url.rstrip("/"),
                timeout=30.0,
            )
            try:
                resp = primekg_client.get("health")
                resp.raise_for_status()
                primekg_meta = resp.json()
                print(
                    f"[sovereign_bench] PrimeKG service OK: "
                    f"{primekg_meta.get('n_nodes', '?'):,} nodes, "
                    f"{primekg_meta.get('n_edges', '?'):,} edges via "
                    f"{args.primekg_url}; hops={args.primekg_max_hops} "
                    f"max_nodes={args.primekg_max_nodes}"
                )
            except Exception as e:  # noqa: BLE001
                print(f"[sovereign_bench] PrimeKG service DISABLED: {e}")
                primekg_client = None
        else:
            try:
                from graph_primekg_subgraph import load_primekg  # noqa: WPS433
                primekg = load_primekg(args.primekg_path)
                print(
                    f"[sovereign_bench] PrimeKG loaded: "
                    f"{primekg.graph.number_of_nodes():,} nodes, "
                    f"{primekg.graph.number_of_edges():,} edges; "
                    f"hops={args.primekg_max_hops} max_nodes={args.primekg_max_nodes}"
                )
            except Exception as e:  # noqa: BLE001
                print(f"[sovereign_bench] PrimeKG DISABLED: {e}")
                primekg = None

    # Phase 2.1 guardrails shim. Production swap: HTTP calls to NemoGuard
    # JailbreakDetect (port 8004) + Nemotron-Content-Safety-Reasoning-4B
    # (port 8005). v0 shim uses lightweight regex / keyword classifiers so
    # we can measure the call-pattern effect on the held-out score before
    # paying the full GPU-memory cost.
    def _guardrail_input_shim(query: str) -> tuple[bool, str]:
        if not args.guardrails:
            return True, ""
        # Layer 0 jailbreak heuristics (NR-Labs-style patterns).
        bad = [
            "ignore previous instructions",
            "system prompt:",
            "you are now",
            "developer mode",
            "DAN mode",
            "jailbreak",
        ]
        ql = query.lower()
        for pat in bad:
            if pat.lower() in ql:
                return False, f"input_rail:jailbreak:{pat}"
        # Direct-identifier (very rough; demo-only).
        import re as _re
        if _re.search(r"\bMRN[-:#]?\s*\d{4,}\b", query) or _re.search(r"\bSSN[-:#]?\s*\d{3}\b", query):
            return False, "input_rail:pii:direct_identifier"
        return True, ""

    def _guardrail_output_shim(response: str) -> tuple[bool, str]:
        if not args.guardrails:
            return True, ""
        rl = response.lower()
        # Hallucinated dosing: catch obvious unsupported numerics on
        # zoledronic acid that are NOT the canonical 4 mg q6mo / 5 mg yearly.
        import re as _re
        if "zoledronic" in rl or "zometa" in rl:
            for m in _re.findall(r"(\d+(?:\.\d+)?\s*mg)", rl):
                if m.replace(" ", "") not in {"4mg", "5mg"}:
                    return False, f"output_rail:dosing_unsupported:{m}"
        # Safety policy: paternalistic prescription register on patient persona.
        forbidden = [
            "you must take",
            "you must not take",
            "i recommend you start",
            "you should start zoledronic",
        ]
        for pat in forbidden:
            if pat in rl:
                return False, f"output_rail:safety:{pat}"
        return True, ""

    # Per-example retrieval blocks (computed lazily inside the trial loop so
    # each example can have its own retrieved context). For `anchors` mode
    # the block is shared (case-keyed), matching the legacy contract.
    legacy_anchor_block = ""
    n_anchors = 0
    if args.retrieval == "anchors":
        facts = _retrieve_anchors(manifest_path)
        legacy_anchor_block = _format_anchors_block(facts)
        n_anchors = len(facts)

    trial_results: list[dict] = []

    print(
        f"[sovereign_bench] run_id={run_id} examples={n_per_trial} trials={trials} "
        f"serve={args.serve_model} judge={args.judge_model}"
    )

    for trial_idx in range(trials):
        per_example: list[dict] = []
        for ex_idx, example in enumerate(examples):
            t0 = time.time()
            messages = _example_messages(example)
            rail_log: list[str] = []
            try:
                # Per-example retrieval. For `bm25`/`dense`/`hybrid` we build
                # the query from the example's last user message and retrieve
                # afresh. For `anchors` we use the case-keyed block built once.
                example_retrieval_block = ""
                graph_block = ""
                user_msgs = [m for m in messages if m.get("role") == "user"]
                query_text = user_msgs[-1]["content"] if user_msgs else ""

                # Input rail (Phase 2.1 shim).
                ok, rail_msg = _guardrail_input_shim(query_text)
                if not ok:
                    rail_log.append(rail_msg)
                    response_text = (
                        "[blocked by input rail; refusing to process. "
                        f"reason={rail_msg}]"
                    )
                    graded = _real_grader(response_text, example, judge_fn=judge_fn)
                    per_example.append(
                        {
                            "example_id": example.get("id", f"ex{ex_idx}"),
                            "trial": trial_idx,
                            "response": response_text,
                            "score": graded.get("score"),
                            "per_axis": graded.get("per_axis", {a: None for a in HEALTHBENCH_AXES}),
                            "judge_incomplete": graded.get("judge_incomplete", 0),
                            "duration_ms": int((time.time() - t0) * 1000),
                            "rail_log": list(rail_log),
                        }
                    )
                    continue

                if retriever is not None:
                    facts = retriever.retrieve(
                        query_text,
                        top_n=args.retrieval_top_n,
                        mode=args.retrieval,
                    )
                    example_retrieval_block = _format_anchors_block(facts)
                elif legacy_anchor_block:
                    example_retrieval_block = legacy_anchor_block

                if graph is not None:
                    seeds = seed_nodes_for(
                        query=query_text,
                        retrieved_chunk_ids=[],
                    )
                    sub = expand_subgraph(
                        graph, seeds=seeds, persona=args.graph_persona, hops=2
                    )
                    graph_block = serialize_subgraph_for_prompt(sub)

                primekg_block = ""
                primekg_seed_names: list[str] = []
                primekg_sub_size: tuple[int, int] | None = None
                if primekg is not None:
                    pk_seeds = primekg.seed_entities_from_query(query_text)
                    pk_sub = primekg.subgraph_slice(
                        pk_seeds,
                        max_hops=args.primekg_max_hops,
                        max_nodes=args.primekg_max_nodes,
                    )
                    primekg_block = primekg.serialize_to_context(
                        pk_sub, max_tokens=args.primekg_max_tokens
                    )
                    primekg_seed_names = [
                        primekg.graph.nodes[s].get("node_name", str(s))
                        for s in pk_seeds
                    ]
                    primekg_sub_size = (
                        pk_sub.number_of_nodes(),
                        pk_sub.number_of_edges(),
                    )
                elif primekg_client is not None:
                    try:
                        r = primekg_client.post(
                            "subgraph",
                            json={
                                "query": query_text,
                                "max_hops": args.primekg_max_hops,
                                "max_nodes": args.primekg_max_nodes,
                                "max_tokens": args.primekg_max_tokens,
                            },
                        )
                        r.raise_for_status()
                        pk = r.json()
                        primekg_block = pk.get("block", "")
                        primekg_seed_names = pk.get("seed_names", [])
                        primekg_sub_size = (
                            int(pk.get("n_nodes", 0)),
                            int(pk.get("n_edges", 0)),
                        )
                    except Exception as e:  # noqa: BLE001
                        print(
                            f"  WARN trial={trial_idx} ex={ex_idx} "
                            f"primekg-svc fail: {e}",
                            file=sys.stderr,
                        )
                        primekg_block = ""

                sys_msg: str | None = None
                if args.clinical_system_prompt:
                    sys_msg = CLINICAL_SYSTEM_PROMPT
                # Order: PrimeKG (factual graph) → corpus chunks (verbatim
                # primary literature) → persona-tagged register graph.
                if primekg_block:
                    sys_msg = (sys_msg + "\n\n" if sys_msg else "") + primekg_block
                if example_retrieval_block:
                    sys_msg = (sys_msg + "\n\n" if sys_msg else "") + example_retrieval_block
                if graph_block:
                    sys_msg = (sys_msg + "\n\n" if sys_msg else "") + graph_block
                response_text = _generate(
                    client=serve_client,
                    model=args.serve_model,
                    messages=messages,
                    max_tokens=args.max_tokens,
                    timeout_s=args.timeout_s,
                    temperature=args.temperature,
                    system_prompt=sys_msg,
                )

                # Output rail (Phase 2.1 shim).
                ok, rail_msg = _guardrail_output_shim(response_text)
                if not ok:
                    rail_log.append(rail_msg)
                    response_text = (
                        f"[blocked by output rail; reason={rail_msg}]\n\n"
                        + response_text
                    )
            except httpx.HTTPError as exc:
                print(
                    f"  trial={trial_idx} ex={ex_idx} SERVE FAIL: {exc}",
                    file=sys.stderr,
                )
                continue

            graded = _real_grader(response_text, example, judge_fn=judge_fn)
            per_example.append(
                {
                    "example_id": example.get("id", f"ex{ex_idx}"),
                    "trial": trial_idx,
                    "response": response_text,
                    "score": graded.get("score"),
                    "per_axis": graded.get("per_axis", {a: None for a in HEALTHBENCH_AXES}),
                    "judge_incomplete": graded.get("judge_incomplete", 0),
                    "duration_ms": int((time.time() - t0) * 1000),
                    "rail_log": list(rail_log),
                    "primekg_seeds": primekg_seed_names,
                    "primekg_sub_n_nodes": (
                        primekg_sub_size[0] if primekg_sub_size else None
                    ),
                    "primekg_sub_n_edges": (
                        primekg_sub_size[1] if primekg_sub_size else None
                    ),
                }
            )
            print(
                f"  trial={trial_idx} ex={ex_idx} "
                f"score={graded.get('score')} "
                f"recused={graded.get('judge_incomplete', 0)}"
            )

        trial_aggregate = _aggregate(per_example)
        trial_results.append(
            {
                "trial": trial_idx,
                "per_example": per_example,
                "aggregate": trial_aggregate,
            }
        )

    payload = {
        "dry_run": False,
        "sovereign": True,
        "run_id": run_id,
        "generated_at": _now_iso(),
        "manifest_path": str(manifest_path),
        "seed": args.seed,
        "serve_url": args.serve_url,
        "serve_model": args.serve_model,
        "judge_url": args.judge_url,
        "judge_model": args.judge_model,
        "judge_host": args.judge_host,
        "rerank_host": args.rerank_host,
        "n_per_trial": n_per_trial,
        "trials": trials,
        "temperature": args.temperature,
        "clinical_system_prompt": args.clinical_system_prompt,
        "retrieval_mode": args.retrieval,
        "retrieval_cuvs": args.retrieval_cuvs,
        "dense_backend": getattr(retriever, "dense_backend", None) if retriever else None,
        "graph_expand": args.graph_expand,
        "graph_persona": args.graph_persona if args.graph_expand else None,
        "primekg_enabled": (primekg is not None or primekg_client is not None),
        "primekg_mode": (
            "local"
            if primekg is not None
            else ("service" if primekg_client is not None else None)
        ),
        "primekg_path": args.primekg_path if primekg is not None else None,
        "primekg_url": args.primekg_url if primekg_client is not None else None,
        "primekg_n_nodes": (
            primekg.graph.number_of_nodes()
            if primekg is not None
            else primekg_meta.get("n_nodes")
        ),
        "primekg_n_edges": (
            primekg.graph.number_of_edges()
            if primekg is not None
            else primekg_meta.get("n_edges")
        ),
        "primekg_max_hops": (
            args.primekg_max_hops
            if (primekg is not None or primekg_client is not None)
            else None
        ),
        "primekg_max_nodes": (
            args.primekg_max_nodes
            if (primekg is not None or primekg_client is not None)
            else None
        ),
        "guardrails_enabled": args.guardrails,
        "n_anchors": n_anchors,
        "n_corpus_chunks": n_corpus_chunks,
        "embed_url": args.embed_url if retriever else None,
        "embed_model": args.embed_model if retriever else None,
        "rerank_url": args.rerank_url if retriever else None,
        "rerank_model": args.rerank_model if retriever else None,
        "retrieval_top_n": args.retrieval_top_n if retriever else None,
        "trial_results": trial_results,
        "wall_time_s": int(time.time() - started),
    }

    _write_out(out_path, payload)
    print(f"[sovereign_bench] artifact: {out_path}")
    print(
        "[sovereign_bench] READ THE ARTIFACT before claiming success: judge-401 "
        "silently produces reward=0 everywhere. Spot-check at least one trial's "
        "per_example[].score for non-zero values."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
