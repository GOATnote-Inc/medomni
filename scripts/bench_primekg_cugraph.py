#!/usr/bin/env python3
"""Phase 2.4 — nx-cugraph speedup benchmark on PrimeKG.

Runs Louvain / PageRank / betweenness / k-hop BFS against:
  (1) full PrimeKG (~129K nodes / ~4M edges)
  (2) a random 5,000-node subgraph

…on both the CPU NetworkX backend (NETWORKX_AUTOMATIC_BACKENDS unset)
and the nx-cugraph backend (NETWORKX_AUTOMATIC_BACKENDS=cugraph).

Reports wall-time + speedup ratio per algorithm. The number lands in
the demo deck and the SPEC §5.3 stage-6 budget.

Sales context (NVIDIA blog, 262K-node graph on A100):
  Louvain ~100x · PageRank ~76x · Betweenness 50-57x ·
  Clustering 130-330x · Degree centrality 2-3x.

B300 (Blackwell Ultra, 8 TB/s HBM3E, 14 PFLOPS FP4 dense) should hit
the high end of these on PrimeKG.

Usage (run on B300):
    ~/medomni-rapids/.venv/bin/python scripts/bench_primekg_cugraph.py \\
        --primekg-path /home/shadeform/medomni/primekg/primekg.gpickle \\
        --out /home/shadeform/medomni/primekg/bench-primekg-cugraph.json
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import time
from pathlib import Path


def _import_nx(backend: str | None):
    """Configure backend env BEFORE importing nx so dispatch wires up.

    Note: this only works if `networkx` was NOT already imported. The
    main() in this script forks subprocesses to guarantee a clean import
    state per backend.
    """
    if backend == "cugraph":
        os.environ["NETWORKX_AUTOMATIC_BACKENDS"] = "cugraph"
    elif backend is None or backend == "cpu":
        os.environ.pop("NETWORKX_AUTOMATIC_BACKENDS", None)
    import networkx as nx  # noqa: WPS433
    return nx


def _bench_one(name: str, fn, *, warmup: int = 1, runs: int = 3) -> dict:
    """Run a 0-arg callable, return min/median/wall."""
    timings: list[float] = []
    for _ in range(warmup):
        try:
            _ = fn()
        except Exception as e:  # noqa: BLE001
            return {"name": name, "error": f"warmup-fail: {e}"}
    for _ in range(runs):
        t0 = time.perf_counter()
        try:
            _ = fn()
        except Exception as e:  # noqa: BLE001
            return {"name": name, "error": f"run-fail: {e}", "timings_s": timings}
        timings.append(time.perf_counter() - t0)
    timings.sort()
    return {
        "name": name,
        "min_s": timings[0],
        "median_s": timings[len(timings) // 2],
        "max_s": timings[-1],
        "runs": runs,
    }


def run_suite(G_full, *, label: str, sample_seed: int = 42) -> dict:
    """Run the algorithm suite. Caller controls which backend is loaded
    (the backend choice happened via env-var before the nx import we
    received in `G_full.__class__.__module__`'s import line)."""
    import networkx as nx  # noqa: WPS433
    print(f"\n=== suite: {label} ({type(G_full).__name__}, "
          f"{G_full.number_of_nodes():,} nodes, "
          f"{G_full.number_of_edges():,} edges) ===")

    # Build a 5K-node subgraph deterministically.
    rng = random.Random(sample_seed)
    sub_nodes = rng.sample(list(G_full.nodes()), k=min(5000, G_full.number_of_nodes()))
    G_sub = G_full.subgraph(sub_nodes).copy()
    print(f"  random 5K subgraph: {G_sub.number_of_nodes():,} nodes, "
          f"{G_sub.number_of_edges():,} edges (seed={sample_seed})")

    results: dict[str, dict] = {}

    # Pick a fixed BFS source for determinism.
    bfs_source = sub_nodes[0]

    # k-hop BFS depth=2 (sub).
    results["khop_bfs_d2_sub5k"] = _bench_one(
        "khop_bfs_d2_sub5k",
        lambda: dict(nx.single_source_shortest_path_length(G_sub, bfs_source, cutoff=2)),
    )
    # k-hop BFS depth=2 (full).
    full_source = list(G_full.nodes())[0]
    results["khop_bfs_d2_full"] = _bench_one(
        "khop_bfs_d2_full",
        lambda: dict(nx.single_source_shortest_path_length(G_full, full_source, cutoff=2)),
    )

    # PageRank — sub (cheap on full would still be fast; do both).
    results["pagerank_sub5k"] = _bench_one(
        "pagerank_sub5k",
        lambda: nx.pagerank(G_sub, alpha=0.85, max_iter=50, tol=1e-4),
        runs=3,
    )
    results["pagerank_full"] = _bench_one(
        "pagerank_full",
        lambda: nx.pagerank(G_full, alpha=0.85, max_iter=30, tol=1e-3),
        runs=2,
        warmup=0,
    )

    # Betweenness centrality — only on the 5K sub (full is O(VE), 1+hr CPU).
    results["betweenness_sub5k_k50"] = _bench_one(
        "betweenness_sub5k_k50",
        lambda: nx.betweenness_centrality(G_sub, k=50, seed=42),
        runs=2,
        warmup=0,
    )

    # Connected components — fast everywhere; useful sanity.
    if not G_full.is_directed():
        results["connected_components_full"] = _bench_one(
            "connected_components_full",
            lambda: list(nx.connected_components(G_full)),
            runs=2,
            warmup=0,
        )
    else:
        results["weakly_connected_components_full"] = _bench_one(
            "weakly_connected_components_full",
            lambda: list(nx.weakly_connected_components(G_full)),
            runs=2,
            warmup=0,
        )

    # Louvain — only when nx_cugraph or python-louvain is wired up.
    # nx 3.x has nx.community.louvain_communities; cuGraph dispatches it.
    try:
        results["louvain_sub5k"] = _bench_one(
            "louvain_sub5k",
            lambda: list(nx.community.louvain_communities(G_sub, seed=42)),
            runs=2,
            warmup=0,
        )
    except Exception as e:  # noqa: BLE001
        results["louvain_sub5k"] = {"name": "louvain_sub5k", "error": f"nx api: {e}"}

    return results


def _run_in_subprocess(backend: str, primekg_path: str) -> dict:
    """Spawn a fresh Python process so NetworkX's dispatch wiring picks
    up the backend env var at first import (not retroactively)."""
    import subprocess
    import sys as _sys
    env = os.environ.copy()
    if backend == "cugraph":
        env["NETWORKX_AUTOMATIC_BACKENDS"] = "cugraph"
    else:
        env.pop("NETWORKX_AUTOMATIC_BACKENDS", None)
    cmd = [
        _sys.executable,
        __file__,
        "--worker",
        "--primekg-path",
        primekg_path,
        "--backend-label",
        backend,
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        return {"error": f"subprocess: rc={proc.returncode}", "stderr": proc.stderr[-2000:]}
    # Last line is JSON.
    try:
        last = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
        return json.loads(last[-1])
    except Exception as e:  # noqa: BLE001
        return {"error": f"parse: {e}", "stdout_tail": proc.stdout[-2000:]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primekg-path",
        default="/home/shadeform/medomni/primekg/primekg.gpickle",
    )
    parser.add_argument(
        "--out", default="/home/shadeform/medomni/primekg/bench-primekg-cugraph.json"
    )
    parser.add_argument("--worker", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--backend-label", default="cpu",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        # Worker mode — env var was set by parent before this process
        # started, so the import below picks up the right backend.
        with open(args.primekg_path, "rb") as f:
            payload = pickle.load(f)  # noqa: S301
        G = payload["graph"]
        suite = run_suite(G, label=f"backend={args.backend_label}")
        # Last stdout line = JSON for parent to collect.
        print(json.dumps(suite))
        return 0

    pickle_path = Path(args.primekg_path)
    out_path = Path(args.out)

    print(f"[bench] dispatching subprocess workers per backend...")
    suites: dict[str, dict] = {}
    for backend in ("cpu", "cugraph"):
        print(f"[bench]  spawning {backend} worker...")
        t0 = time.time()
        suites[backend] = _run_in_subprocess(backend, str(pickle_path))
        print(f"[bench]  {backend} worker done in {time.time() - t0:.1f}s")

    # ---- Speedup table ----
    print("\n=== speedup (cpu vs cugraph; > 1.0x = cugraph faster) ===")
    rows: list[dict] = []
    cpu_keys = set(suites.get("cpu", {}).keys())
    gpu_keys = set(suites.get("cugraph", {}).keys())
    for algo in sorted(cpu_keys | gpu_keys):
        c = suites.get("cpu", {}).get(algo, {})
        g = suites.get("cugraph", {}).get(algo, {})
        if not isinstance(c, dict) or not isinstance(g, dict):
            continue
        if "error" in c or "error" in g:
            print(f"  {algo:32s}  cpu={c.get('error', c.get('median_s'))} "
                  f"gpu={g.get('error', g.get('median_s'))}")
            rows.append({"algo": algo, "cpu": c, "cugraph": g, "speedup": None})
            continue
        cpu_s = c.get("median_s")
        gpu_s = g.get("median_s")
        if cpu_s is None or gpu_s is None or gpu_s <= 0:
            continue
        speedup = cpu_s / gpu_s
        rows.append(
            {"algo": algo, "cpu_s": cpu_s, "cugraph_s": gpu_s, "speedup": speedup}
        )
        print(
            f"  {algo:32s}  cpu={cpu_s:8.3f}s  gpu={gpu_s:8.3f}s  "
            f"speedup={speedup:7.2f}x"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"suites": suites, "speedup_table": rows}, indent=2))
    print(f"\n[bench] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
