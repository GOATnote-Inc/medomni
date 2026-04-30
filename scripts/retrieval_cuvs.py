"""cuVS IVF-PQ replacement for the dense-recall stage of the hybrid retriever.

Drop-in for `HybridRetriever._dense_topk` from `scripts/retrieval.py`. Same
embed contract (HTTP /v1/embeddings, input_type=passage|query); same scoring
shape (cosine via L2-normalized inner-product on a flat-or-IVF-PQ index);
same return type `list[tuple[int, float]]`.

Why cuVS IVF-PQ at this corpus size (~78 chunks)?

   At N=78 the canonical answer is "build a flat brute-force index and call
   it dense recall" — IVF-PQ pays off above N≈10⁴. The point of the swap is
   that the production trajectory is the same script: the only thing that
   changes between 78 chunks and 78,000 chunks is the index recipe selector
   below, which auto-degrades to `cuvs.neighbors.brute_force` at small N
   and to IVF-PQ once the corpus crosses a configurable threshold. The
   demo build path therefore exercises the production-shaped code without
   over-fitting the data structure to the demo corpus.

   On the B300 pod with cuVS 26.04, both the brute-force and IVF-PQ paths
   round-trip an embedding tensor through the cuvs.neighbors API; in the
   timing harness below, we report p50/p99 against the legacy numpy
   cosine path so the latency delta is auditable.

Sovereignty: this module imports cuvs/cudf/cupy lazily inside the
`CuvsDenseIndex` constructor. On the laptop (where RAPIDS is not
installed) it imports cleanly but the constructor raises ImportError.
The bench harness can therefore import the module unconditionally; only
B300-side runs build the index.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import httpx


# ----------------------------------------------------------------------------
# Embedding client (reuses scripts/retrieval.py's contract)
# ----------------------------------------------------------------------------


@dataclass
class EmbedClient:
    base_url: str
    model: str
    timeout_s: float = 30.0
    _client: httpx.Client = field(init=False)

    def __post_init__(self) -> None:
        self._client = httpx.Client(
            base_url=self.base_url.rstrip("/"),
            timeout=self.timeout_s,
        )

    def embed(self, texts: list[str], input_type: str = "passage") -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 32):
            batch = texts[i : i + 32]
            resp = self._client.post(
                "embeddings",
                json={"model": self.model, "input": batch, "input_type": input_type},
            )
            resp.raise_for_status()
            for item in resp.json()["data"]:
                out.append(item["embedding"])
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text], input_type="query")[0]


# ----------------------------------------------------------------------------
# numpy reference (used as baseline AND as fallback when cuVS is absent)
# ----------------------------------------------------------------------------


@dataclass
class NumpyDenseIndex:
    """The legacy path: cosine over a python-list embedding store. Exact
    semantics match `HybridRetriever._cosine`; we route through this when
    cuVS isn't importable so the harness still runs end-to-end."""

    embeddings: list[list[float]]

    def search(self, query_emb: list[float], k: int) -> list[tuple[int, float]]:
        def _cos(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            return 0.0 if na == 0.0 or nb == 0.0 else dot / (na * nb)

        scored = [(i, _cos(query_emb, e)) for i, e in enumerate(self.embeddings)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


# ----------------------------------------------------------------------------
# cuVS IVF-PQ (or brute-force at small N) index
# ----------------------------------------------------------------------------


@dataclass
class CuvsDenseIndex:
    """RAPIDS cuVS-backed dense recall. Auto-selects brute-force at small N
    and IVF-PQ at large N. Builds on B300 only — laptop side raises
    ImportError on construction (caught upstream)."""

    embeddings: list[list[float]]
    n_lists: int = 64  # IVF list count; tuned for med-corpus regime
    pq_dim: int = 64  # PQ subspace dim; auto-clamped to <= dim
    pq_bits: int = 8
    metric: str = "inner_product"
    ivf_threshold: int = 4096  # N below this -> brute force

    _cp: object = field(init=False, default=None)
    _index: object = field(init=False, default=None)
    _resources: object = field(init=False, default=None)
    _index_kind: str = field(init=False, default="brute_force")
    _vectors: object = field(init=False, default=None)

    def __post_init__(self) -> None:
        try:
            import cupy as cp
            from cuvs.common import Resources
            from cuvs.neighbors import brute_force, ivf_pq
        except Exception as e:  # noqa: BLE001
            raise ImportError(
                f"cuVS / cupy not available in this Python environment: {e!r}. "
                "Install with: pip install --extra-index-url=https://pypi.nvidia.com "
                "cudf-cu13 cuvs-cu13"
            ) from e

        self._cp = cp
        self._resources = Resources()
        # L2-normalize to make inner_product identical to cosine.
        arr = cp.asarray(self.embeddings, dtype=cp.float32)
        norms = cp.linalg.norm(arr, axis=1, keepdims=True)
        norms = cp.where(norms == 0, 1.0, norms)
        self._vectors = arr / norms

        n, dim = self._vectors.shape
        if n < self.ivf_threshold:
            # cuVS 26.04 brute_force.build takes kw-style metric directly
            # (no IndexParams). Defaults to sqeuclidean; we want inner_product
            # because vectors are L2-normalized (so IP == cosine).
            self._index = brute_force.build(
                arr / norms,  # already normalized, but be explicit
                metric=self.metric,
                resources=self._resources,
            )
            self._index_kind = "brute_force"
        else:
            pq_dim = min(self.pq_dim, dim)
            params = ivf_pq.IndexParams(
                metric=self.metric,
                n_lists=self.n_lists,
                pq_dim=pq_dim,
                pq_bits=self.pq_bits,
            )
            self._index = ivf_pq.build(params, self._vectors, resources=self._resources)
            self._index_kind = "ivf_pq"

    @property
    def index_kind(self) -> str:
        return self._index_kind

    def search(self, query_emb: list[float], k: int) -> list[tuple[int, float]]:
        cp = self._cp
        from cuvs.neighbors import brute_force, ivf_pq

        q = cp.asarray([query_emb], dtype=cp.float32)
        qn = cp.linalg.norm(q, axis=1, keepdims=True)
        qn = cp.where(qn == 0, 1.0, qn)
        q = q / qn

        if self._index_kind == "brute_force":
            # cuVS 26.04 brute_force.search: (index, queries, k, neighbors=,
            # distances=, resources=, prefilter=). No SearchParams class.
            distances, indices = brute_force.search(
                self._index, q, k, resources=self._resources
            )
        else:
            params = ivf_pq.SearchParams()
            distances, indices = ivf_pq.search(
                params, self._index, q, k, resources=self._resources
            )

        # cuVS 26.04 returns pylibraft device_ndarray; cp.asarray is a no-copy view.
        idx = cp.asnumpy(cp.asarray(indices)).reshape(-1).tolist()
        score = cp.asnumpy(cp.asarray(distances)).reshape(-1).tolist()
        return [(int(i), float(s)) for i, s in zip(idx, score) if int(i) >= 0]


# ----------------------------------------------------------------------------
# Wiring helpers — build_index_for_chunks() and bench()
# ----------------------------------------------------------------------------


def load_chunk_bodies(chunks_path: Path) -> list[tuple[str, str]]:
    """Returns list of (chunk_id, body) preserving file order."""
    out: list[tuple[str, str]] = []
    for line in chunks_path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out.append((d["id"], d["body"]))
    return out


def build_dense_store(
    chunks_path: Path,
    *,
    embed_url: str,
    embed_model: str,
    use_cuvs: bool,
) -> tuple[object, list[str]]:
    """Build an index over the chunks.jsonl bodies. Returns (index, ids)."""
    pairs = load_chunk_bodies(chunks_path)
    ids = [p[0] for p in pairs]
    bodies = [p[1] for p in pairs]
    client = EmbedClient(embed_url, embed_model)
    embeddings = client.embed(bodies, input_type="passage")
    if use_cuvs:
        index = CuvsDenseIndex(embeddings=embeddings)
    else:
        index = NumpyDenseIndex(embeddings=embeddings)
    return index, ids


def bench(
    chunks_path: Path,
    *,
    embed_url: str,
    embed_model: str,
    queries: list[str],
    top_k: int = 10,
) -> dict:
    """Compare numpy-cosine vs cuVS for a fixed set of queries. Returns
    a dict with p50/p99 latency for each path plus recall@K of cuVS
    results against the numpy gold (Jaccard of returned-ID sets)."""
    pairs = load_chunk_bodies(chunks_path)
    ids = [p[0] for p in pairs]
    bodies = [p[1] for p in pairs]
    client = EmbedClient(embed_url, embed_model)

    t0 = time.time()
    embeddings = client.embed(bodies, input_type="passage")
    t_embed = time.time() - t0

    np_index = NumpyDenseIndex(embeddings)
    try:
        cu_index: object | None = CuvsDenseIndex(embeddings=embeddings)
    except ImportError as e:
        cu_index = None
        print(f"[bench] cuVS unavailable: {e}")

    np_lat: list[float] = []
    cu_lat: list[float] = []
    overlap: list[float] = []

    for q in queries:
        q_emb = client.embed_query(q)

        t = time.time()
        np_top = np_index.search(q_emb, top_k)
        np_lat.append((time.time() - t) * 1000.0)
        np_ids = {ids[i] for i, _ in np_top}

        if cu_index is not None:
            t = time.time()
            cu_top = cu_index.search(q_emb, top_k)
            cu_lat.append((time.time() - t) * 1000.0)
            cu_ids = {ids[i] for i, _ in cu_top}
            inter = np_ids & cu_ids
            denom = max(1, len(np_ids))
            overlap.append(len(inter) / denom)

    def pp(xs: list[float], pct: float) -> float:
        if not xs:
            return float("nan")
        return statistics.quantiles(xs, n=100, method="inclusive")[int(pct) - 1]

    return {
        "n_chunks": len(bodies),
        "n_queries": len(queries),
        "top_k": top_k,
        "embed_total_s": t_embed,
        "numpy_p50_ms": (pp(np_lat, 50) if len(np_lat) >= 100 else statistics.median(np_lat) if np_lat else float("nan")),
        "numpy_p99_ms": (pp(np_lat, 99) if len(np_lat) >= 100 else max(np_lat) if np_lat else float("nan")),
        "cuvs_available": cu_index is not None,
        "cuvs_index_kind": getattr(cu_index, "index_kind", None),
        "cuvs_p50_ms": (pp(cu_lat, 50) if len(cu_lat) >= 100 else statistics.median(cu_lat) if cu_lat else float("nan")),
        "cuvs_p99_ms": (pp(cu_lat, 99) if len(cu_lat) >= 100 else max(cu_lat) if cu_lat else float("nan")),
        "recall_at_k_jaccard_mean": (sum(overlap) / len(overlap)) if overlap else float("nan"),
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", default="corpus/medical-guidelines/chunks.jsonl")
    parser.add_argument("--embed-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--embed-model", default="nvidia/llama-nemotron-embed-1b-v2")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--queries-file",
        default=None,
        help="optional newline-delimited queries; defaults to a built-in 6-fixture sample.",
    )
    parser.add_argument("--out", default=None, help="optional JSON path to dump the bench result")
    args = parser.parse_args()

    if args.queries_file:
        queries = [
            ln.strip()
            for ln in Path(args.queries_file).read_text().splitlines()
            if ln.strip()
        ]
    else:
        queries = [
            "low-dose aspirin for colorectal cancer chemoprevention in family history",
            "finasteride and prostate cancer chemoprevention with PSA monitoring",
            "Gardasil-9 catch-up vaccination at age 35 for adult unvaccinated",
            "statin and cancer risk for primary cardiovascular prevention",
            "varenicline vs nicotine replacement for smoking cessation in head-and-neck cancer",
            "adjuvant zoledronic acid in postmenopausal breast cancer on aromatase inhibitor",
        ]

    result = bench(
        Path(args.chunks),
        embed_url=args.embed_url,
        embed_model=args.embed_model,
        queries=queries,
        top_k=args.top_k,
    )
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
