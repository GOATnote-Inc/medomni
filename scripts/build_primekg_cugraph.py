#!/usr/bin/env python3
"""Phase 2.4 — Stage 6 PrimeKG cuGraph loader.

Loads Harvard Marinka Zitnik lab PrimeKG (`kg.csv` + `nodes.tab`) into a
cuDF-backed cuGraph DiGraph and persists a NetworkX-shaped pickle for
fast subsequent loads on B300. Built to be re-runnable: if the pickle
exists and is newer than both source files, the load is skipped.

Default paths (designed to run on B300 inside Alpha's RAPIDS venv):

    /home/shadeform/medomni/primekg/
      kg.csv               # ~4M edges (981 MB)
      nodes.tab            # ~129K nodes (9 MB)
      primekg.gpickle      # written by this script (NetworkX DiGraph)
      primekg-stats.json   # node/edge type distribution + load timing

Schema (per the Chandak/Huang/Zitnik 2023 Nature Sci. Data paper):

    nodes.tab columns: node_index node_id node_type node_name node_source
    kg.csv  columns:   relation display_relation x_index x_id x_type x_name x_source
                       y_index y_id y_type y_name y_source

Most biomedical relations are symmetric for retrieval purposes; we build
an UNDIRECTED graph (saved as `nx.Graph`). Directed-relation semantics
(e.g. enzyme→substrate) live on the edge attribute `relation`, but the
BFS expansion is symmetric so a query about `metformin` reaches both
its targets and the diseases it indicates_for in one hop.

Sales context (NVIDIA blog benchmarks, A100, 262K-node graph):
  Louvain ~100x · PageRank ~76x · Betweenness 50-57x · k-hop BFS small but real
B300 (Blackwell Ultra, 8 TB/s HBM3E) should hit the high end on a graph
of PrimeKG's shape (129K nodes / 4M edges).

Usage:
    NETWORKX_AUTOMATIC_BACKENDS=cugraph \\
    ~/medomni-rapids/.venv/bin/python scripts/build_primekg_cugraph.py \\
        --primekg-dir /home/shadeform/medomni/primekg \\
        --force         # rebuild even if pickle is fresh
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path


def _human_mb(n: int) -> str:
    return f"{n / 1024 / 1024:.1f} MB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primekg-dir",
        default="/home/shadeform/medomni/primekg",
        help="directory containing kg.csv and nodes.tab",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild pickle even if cache is fresh",
    )
    parser.add_argument(
        "--undirected",
        action="store_true",
        default=True,
        help=(
            "build symmetric undirected graph (default). Directed relation "
            "semantics live on the edge attribute `relation`."
        ),
    )
    args = parser.parse_args()

    pkdir = Path(args.primekg_dir).resolve()
    kg_csv = pkdir / "kg.csv"
    nodes_tab = pkdir / "nodes.tab"
    pickle_path = pkdir / "primekg.gpickle"
    stats_path = pkdir / "primekg-stats.json"

    if not kg_csv.exists():
        print(f"FAIL: missing {kg_csv}", file=sys.stderr)
        return 1
    if not nodes_tab.exists():
        print(f"FAIL: missing {nodes_tab}", file=sys.stderr)
        return 1

    if (
        pickle_path.exists()
        and not args.force
        and pickle_path.stat().st_mtime > kg_csv.stat().st_mtime
        and pickle_path.stat().st_mtime > nodes_tab.stat().st_mtime
    ):
        print(
            f"[primekg] cache fresh: {pickle_path} "
            f"({_human_mb(pickle_path.stat().st_size)}) — use --force to rebuild"
        )
        return 0

    print(f"[primekg] kg.csv      : {_human_mb(kg_csv.stat().st_size)}")
    print(f"[primekg] nodes.tab   : {_human_mb(nodes_tab.stat().st_size)}")

    # Lazy imports so the script doesn't require RAPIDS just to print --help.
    t_total = time.time()

    print("[primekg] importing cudf + cugraph + networkx...")
    t0 = time.time()
    import cudf  # noqa: WPS433
    import cugraph  # noqa: WPS433
    import networkx as nx  # noqa: WPS433
    print(
        f"[primekg]   cudf {cudf.__version__} cugraph {cugraph.__version__} "
        f"networkx {nx.__version__} ({time.time() - t0:.2f}s)"
    )

    # ---- Load nodes.tab via cuDF (tab-separated) ----
    t0 = time.time()
    nodes_df = cudf.read_csv(str(nodes_tab), sep="\t")
    print(
        f"[primekg] nodes.tab loaded: {len(nodes_df):,} rows, "
        f"cols={list(nodes_df.columns)} ({time.time() - t0:.2f}s)"
    )

    # ---- Load kg.csv via cuDF ----
    t0 = time.time()
    kg_df = cudf.read_csv(str(kg_csv))
    print(
        f"[primekg] kg.csv loaded:    {len(kg_df):,} rows, "
        f"cols={list(kg_df.columns)} ({time.time() - t0:.2f}s)"
    )

    # ---- Distributions (cudf -> pandas to print) ----
    node_type_dist = (
        nodes_df["node_type"].value_counts().to_pandas().to_dict()
    )
    edge_type_dist = (
        kg_df["display_relation"].value_counts().to_pandas().to_dict()
    )
    print(f"[primekg] node types ({len(node_type_dist)}):")
    for k, v in sorted(node_type_dist.items(), key=lambda kv: -kv[1]):
        print(f"           {k:32s} {v:>10,}")
    print(f"[primekg] top-15 edge types (of {len(edge_type_dist)}):")
    for k, v in sorted(edge_type_dist.items(), key=lambda kv: -kv[1])[:15]:
        print(f"           {k:32s} {v:>10,}")

    # ---- Build cuGraph for sanity check (optional speedup demo) ----
    t0 = time.time()
    edges_for_cu = kg_df[["x_index", "y_index"]].rename(
        columns={"x_index": "src", "y_index": "dst"}
    )
    G_cu = cugraph.Graph(directed=not args.undirected)
    G_cu.from_cudf_edgelist(edges_for_cu, source="src", destination="dst")
    print(
        f"[primekg] cugraph.Graph built: V={G_cu.number_of_vertices():,} "
        f"E={G_cu.number_of_edges():,} directed={not args.undirected} "
        f"({time.time() - t0:.2f}s)"
    )

    # ---- Build NetworkX graph carrying full attrs ----
    # We use nx.Graph (undirected) so BFS is symmetric for retrieval; the
    # `relation` attribute on each edge preserves the original directional
    # semantic. nx-cugraph will dispatch BFS / Louvain / PageRank when env
    # NETWORKX_AUTOMATIC_BACKENDS=cugraph is set.
    t0 = time.time()
    GraphCls = nx.Graph if args.undirected else nx.DiGraph
    G_nx = GraphCls()

    # Add nodes with attrs. Convert to pandas for fast iteration.
    nodes_pd = nodes_df.to_pandas()
    for row in nodes_pd.itertuples(index=False):
        G_nx.add_node(
            int(row.node_index),
            node_id=str(row.node_id),
            node_type=str(row.node_type),
            node_name=str(row.node_name),
            node_source=str(row.node_source),
        )
    print(
        f"[primekg] nx nodes added: {G_nx.number_of_nodes():,} "
        f"({time.time() - t0:.2f}s)"
    )

    # Add edges with attrs.
    t0 = time.time()
    kg_pd = kg_df[
        ["relation", "display_relation", "x_index", "y_index"]
    ].to_pandas()
    for row in kg_pd.itertuples(index=False):
        G_nx.add_edge(
            int(row.x_index),
            int(row.y_index),
            relation=str(row.relation),
            display_relation=str(row.display_relation),
        )
    print(
        f"[primekg] nx edges added: {G_nx.number_of_edges():,} "
        f"({time.time() - t0:.2f}s)"
    )

    # ---- Build name + node_id → node_index secondary indices ----
    # These are essential for `seed_entities_from_query` — string match on
    # query → node_index → BFS seed.
    t0 = time.time()
    name_to_index: dict[str, int] = {}
    nodeid_to_index: dict[str, int] = {}
    for idx, attrs in G_nx.nodes(data=True):
        name = attrs.get("node_name", "").lower()
        nid = attrs.get("node_id", "")
        if name:
            # Last writer wins; for duplicate names (rare) we keep the
            # numerically first index.
            name_to_index.setdefault(name, idx)
        if nid:
            nodeid_to_index.setdefault(nid, idx)
    print(
        f"[primekg] secondary indices built: "
        f"{len(name_to_index):,} names, {len(nodeid_to_index):,} ids "
        f"({time.time() - t0:.2f}s)"
    )

    # ---- Persist ----
    t0 = time.time()
    payload = {
        "graph": G_nx,
        "name_to_index": name_to_index,
        "nodeid_to_index": nodeid_to_index,
        "schema": {
            "directed": not args.undirected,
            "n_nodes": G_nx.number_of_nodes(),
            "n_edges": G_nx.number_of_edges(),
            "node_types": node_type_dist,
            "edge_types": edge_type_dist,
        },
    }
    with open(pickle_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"[primekg] pickle written: {pickle_path} "
        f"({_human_mb(pickle_path.stat().st_size)}, {time.time() - t0:.2f}s)"
    )

    # ---- Stats artifact ----
    stats = {
        "kg_csv_bytes": kg_csv.stat().st_size,
        "nodes_tab_bytes": nodes_tab.stat().st_size,
        "n_nodes": G_nx.number_of_nodes(),
        "n_edges": G_nx.number_of_edges(),
        "directed": not args.undirected,
        "node_types": node_type_dist,
        "edge_types_top15": dict(
            sorted(edge_type_dist.items(), key=lambda kv: -kv[1])[:15]
        ),
        "n_edge_types_total": len(edge_type_dist),
        "name_index_size": len(name_to_index),
        "nodeid_index_size": len(nodeid_to_index),
        "total_load_seconds": round(time.time() - t_total, 2),
        "rapids_versions": {
            "cudf": cudf.__version__,
            "cugraph": cugraph.__version__,
            "networkx": nx.__version__,
        },
    }
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True))
    print(f"[primekg] stats written : {stats_path}")
    print(f"[primekg] TOTAL: {time.time() - t_total:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
