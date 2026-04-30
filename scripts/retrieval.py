"""Hybrid retrieval (BM25 + dense + reciprocal rank fusion + reranker) for the
sovereign medical-LLM bench. NVIDIA-canonical pipeline shape:

    query → embed (NIM) → top-K=50 dense
            BM25 over corpus → top-K=50 sparse
                          ↓
                      RRF fuse → top-N=20 candidates
                                              ↓
                                       rerank (NIM /v1/ranking) → top-M=8
                                                                       ↓
                                                              context block

Drop-in replacement for the `_retrieve_anchors` slot in `sovereign_bench.py`.
The contract is the same: `query_messages → list[{id, fact, citation,
supports}]` (the `supports` key is empty here — corpus chunks are not
rubric-tagged, that's the point).

NIM endpoints (default ports for B300 co-tenant):
- embed:  http://127.0.0.1:8001/v1/embeddings
- rerank: http://127.0.0.1:8002/v1/ranking
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from rank_bm25 import BM25Okapi


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*", re.IGNORECASE)


def _tokenize(text: str) -> list[str]:
    """BM25 tokenizer. Lowercase, alnum + intra-word hyphen
    (preserves drug names like 'levo-norgestrel', acronyms like 'NSABP-P-1')."""
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


@dataclass
class Chunk:
    id: str
    source_doc_id: str
    source_title: str
    source_authors: str
    source_year: int | None
    section: str
    body: str
    n_tokens_estimated: int

    @classmethod
    def from_jsonl_line(cls, line: str) -> "Chunk":
        d = json.loads(line)
        return cls(
            id=d["id"],
            source_doc_id=d["source_doc_id"],
            source_title=d["source_title"],
            source_authors=d.get("source_authors", ""),
            source_year=d.get("source_year"),
            section=d.get("section", ""),
            body=d["body"],
            n_tokens_estimated=d.get("n_tokens_estimated", 0),
        )

    def citation(self) -> str:
        year = f" {self.source_year}" if self.source_year else ""
        return f"{self.source_authors}{year} — {self.source_title} (§ {self.section})"


@dataclass
class HybridRetriever:
    """BM25 + dense + RRF + reranker. Construct once per bench run, then call
    `.retrieve(query, top_n=8)` per example.

    Phase 2.1 add: when `use_cuvs=True`, the dense-recall stage swaps to a
    cuVS IVF-PQ / brute-force GPU index (RAPIDS 26.04). Falls back to
    numpy cosine on environments where `cuvs`/`cupy` cannot import (e.g.
    on the laptop side). The fallback path is silent-by-design: callers
    that care can read `retriever.dense_backend` after construction.
    """

    chunks: list[Chunk]
    embed_url: str
    embed_model: str
    rerank_url: str | None
    rerank_model: str
    top_k_per_channel: int = 50
    rrf_top_n_pre_rerank: int = 20
    rrf_k: int = 60  # standard RRF constant
    embed_timeout_s: float = 30.0
    rerank_timeout_s: float = 30.0
    use_cuvs: bool = False

    _bm25: BM25Okapi = field(init=False)
    _dense_index: list[list[float]] = field(init=False, default_factory=list)
    _cuvs_index: object = field(init=False, default=None)
    _embed_client: httpx.Client = field(init=False)
    _rerank_client: httpx.Client | None = field(init=False, default=None)
    dense_backend: str = field(init=False, default="numpy")

    def __post_init__(self) -> None:
        if not self.chunks:
            raise ValueError("HybridRetriever requires at least one chunk")
        tokenized_corpus = [_tokenize(c.body) for c in self.chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)
        self._embed_client = httpx.Client(
            base_url=self.embed_url.rstrip("/"),
            timeout=self.embed_timeout_s,
        )
        if self.rerank_url:
            self._rerank_client = httpx.Client(
                base_url=self.rerank_url.rstrip("/"),
                timeout=self.rerank_timeout_s,
            )

    @classmethod
    def from_jsonl(
        cls,
        chunks_path: Path,
        *,
        embed_url: str,
        embed_model: str,
        rerank_url: str | None = None,
        rerank_model: str = "nvidia/llama-3.2-nv-rerankqa-1b-v2",
        use_cuvs: bool = False,
        **kwargs,
    ) -> "HybridRetriever":
        chunks = [
            Chunk.from_jsonl_line(line)
            for line in chunks_path.read_text().splitlines()
            if line.strip()
        ]
        return cls(
            chunks=chunks,
            embed_url=embed_url,
            embed_model=embed_model,
            rerank_url=rerank_url,
            rerank_model=rerank_model,
            use_cuvs=use_cuvs,
            **kwargs,
        )

    def build_dense_index(self) -> None:
        """One-time: embed every chunk via the NIM passage-mode endpoint.
        When `use_cuvs=True` (and cuVS is importable) ALSO build a GPU
        index. The numpy `_dense_index` is always populated as a fallback."""
        # NIM /v1/embeddings supports input_type=passage|query; passage for indexing.
        bodies = [c.body for c in self.chunks]
        # Batch in groups of 32 to stay under typical 8192-token NIM batch caps.
        embeddings: list[list[float]] = []
        for i in range(0, len(bodies), 32):
            batch = bodies[i : i + 32]
            resp = self._embed_client.post(
                "embeddings",
                json={
                    "model": self.embed_model,
                    "input": batch,
                    "input_type": "passage",
                },
            )
            resp.raise_for_status()
            payload = resp.json()
            for item in payload["data"]:
                embeddings.append(item["embedding"])
        if len(embeddings) != len(self.chunks):
            raise RuntimeError(
                f"dense index size mismatch: got {len(embeddings)} embeddings "
                f"for {len(self.chunks)} chunks"
            )
        self._dense_index = embeddings

        if self.use_cuvs:
            try:
                # Lazy import; sibling module.
                from retrieval_cuvs import CuvsDenseIndex
                self._cuvs_index = CuvsDenseIndex(embeddings=embeddings)
                self.dense_backend = f"cuvs:{self._cuvs_index.index_kind}"
            except ImportError as e:
                # Quietly fall back; caller can inspect dense_backend.
                self._cuvs_index = None
                self.dense_backend = f"numpy (cuvs-unavailable: {e})"
        else:
            self.dense_backend = "numpy"

    def _embed_query(self, query: str) -> list[float]:
        resp = self._embed_client.post(
            "embeddings",
            json={
                "model": self.embed_model,
                "input": [query],
                "input_type": "query",
            },
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        # numpy-free cosine; corpora are small and we already pay HTTP overhead.
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)

    def _bm25_topk(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(idx, float(score)) for idx, score in ranked[:k] if score > 0.0]

    def _dense_topk(self, query: str, k: int) -> list[tuple[int, float]]:
        if not self._dense_index:
            return []
        q = self._embed_query(query)
        if self._cuvs_index is not None:
            return self._cuvs_index.search(q, k)
        scores = [(i, self._cosine(q, emb)) for i, emb in enumerate(self._dense_index)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    @staticmethod
    def _rrf_fuse(
        rankings: list[list[tuple[int, float]]], k: int, top_n: int
    ) -> list[int]:
        """Standard reciprocal rank fusion. Each ranking is a list of
        (chunk_idx, score) sorted descending. Returns top_n chunk indices."""
        rrf_scores: dict[int, float] = {}
        for ranking in rankings:
            for rank, (idx, _score) in enumerate(ranking):
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in ranked[:top_n]]

    def _rerank(self, query: str, candidate_idxs: list[int], top_m: int) -> list[int]:
        """Optional rerank step. Falls back to identity ordering if no rerank
        client is configured. vllm /v1/score shape (cross-encoder pairwise):
            {model, text_1: <query>, text_2: [<passage>, ...]}
        Response: {data: [{index, score}, ...]}, where index references the
        position in text_2. Higher score = more relevant."""
        if self._rerank_client is None or not candidate_idxs:
            return candidate_idxs[:top_m]
        passages = [self.chunks[i].body for i in candidate_idxs]
        resp = self._rerank_client.post(
            "score",
            json={
                "model": self.rerank_model,
                "text_1": query,
                "text_2": passages,
            },
        )
        resp.raise_for_status()
        scored = resp.json().get("data", [])
        scored.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        ordered = [candidate_idxs[item["index"]] for item in scored]
        return ordered[:top_m]

    def retrieve(
        self, query: str, *, top_n: int = 8, mode: str = "hybrid"
    ) -> list[dict]:
        """Run retrieval. mode ∈ {bm25, dense, hybrid, primekg-hybrid}.

        Note: `primekg-hybrid` runs the same chunk-level retrieval as
        `hybrid` (BM25 + dense + RRF + rerank). The PrimeKG graph block
        is composed and prepended at the bench level, not here — see
        `sovereign_bench.py`'s graph_block path. This keeps the
        retrieval contract unchanged: `retrieve` always returns chunk
        citations only.

        Returns a list of chunk dicts in the same shape
        `_format_anchors_block` expects:
            {id, fact, citation, supports}
        `supports` is empty (corpus chunks are not rubric-tagged)."""
        # primekg-hybrid uses the same chunk-retrieval shape as hybrid.
        chunk_mode = "hybrid" if mode == "primekg-hybrid" else mode
        if chunk_mode == "bm25":
            ranked = self._bm25_topk(query, self.top_k_per_channel)
            candidates = [idx for idx, _ in ranked[: self.rrf_top_n_pre_rerank]]
        elif chunk_mode == "dense":
            ranked = self._dense_topk(query, self.top_k_per_channel)
            candidates = [idx for idx, _ in ranked[: self.rrf_top_n_pre_rerank]]
        elif chunk_mode == "hybrid":
            bm25_ranked = self._bm25_topk(query, self.top_k_per_channel)
            dense_ranked = self._dense_topk(query, self.top_k_per_channel)
            candidates = self._rrf_fuse(
                [bm25_ranked, dense_ranked],
                k=self.rrf_k,
                top_n=self.rrf_top_n_pre_rerank,
            )
        else:
            raise ValueError(f"unknown retrieval mode: {mode!r}")

        final_idxs = self._rerank(query, candidates, top_m=top_n)
        out: list[dict] = []
        for idx in final_idxs:
            c = self.chunks[idx]
            out.append(
                {
                    "id": c.id,
                    "fact": c.body,
                    "citation": c.citation(),
                    "supports": [],
                    "source_doc_id": c.source_doc_id,
                }
            )
        return out
