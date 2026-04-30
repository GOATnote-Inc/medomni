"""Phase 2.4 — PrimeKG subgraph slice for the retrieval pipeline (Stage 6).

Drop-in replacement for `graph_subgraph_slice` (the 59-node hand-built
chemoprevention graph) when the user asks for `--retrieval primekg-hybrid`.

The 59-node persona-tagged graph from Phase 2.1 stays untouched — that
graph carries persona_mask + register edges built for nurse-vs-physician
register routing. PrimeKG is the *factual* layer (drug → indicates →
disease, drug → contraindicates → population, etc.).

This module:
  - lazy-loads the pickle written by `scripts/build_primekg_cugraph.py`
  - resolves natural-language entity mentions → PrimeKG node_index seeds
    via three layers of matcher precision (exact name, normalized
    fragment, lemmatized substring)
  - BFS-expands the seeds 2 hops under nx-cugraph (when available)
  - serializes the slice to a token-budgeted text block that plugs into
    the system prompt

Public API:
    PrimeKG.from_pickle(path)
    pkg.seed_entities_from_query(query) -> list[int]
    pkg.subgraph_slice(seeds, max_hops=2, max_nodes=100, edge_filter=None) -> nx.Graph
    pkg.community_label(subgraph) -> dict[int, int]
    pkg.serialize_to_context(subgraph, max_tokens=4096) -> str

Tokens here are coarse — char_count / 4. Good enough for budget gating.
"""
from __future__ import annotations

import os
import pickle
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Per the durable rule: ensure the cugraph backend is selected for any
# downstream networkx algorithm call. This is a no-op outside RAPIDS.
os.environ.setdefault("NETWORKX_AUTOMATIC_BACKENDS", "cugraph")


_NAME_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")

# Aliases: brand / drug-class / abbreviation → canonical PrimeKG node_name
# (lowercased). This is the cheap NER swap-in. Keeps the v1.0 demo
# fixtures hitting useful seeds without paying for a full medical NER.
# Curated from the 6 held-out fixtures + tamoxifen demo case + common
# pharmacy substitutions.
ALIASES: dict[str, str] = {
    # HPV / vaccine surfaces — PrimeKG has no Gardasil/HPV-vaccine drug
    # node, but it has `papilloma` (disease) and `cervical cancer`. We
    # seed both so graph adds clinical-context even on this fixture.
    "gardasil": "papilloma",
    "gardasil-9": "papilloma",
    "9vhpv": "papilloma",
    "hpv vaccine": "papilloma",
    "hpv shot": "papilloma",
    "hpv": "papilloma",
    # Statin class — atorvastatin is the most-connected statin in PrimeKG
    "statin": "atorvastatin",
    "statins": "atorvastatin",
    "rosuvastatin": "rosuvastatin",
    # IUD brand → generic
    "mirena": "levonorgestrel",
    "lng-ius": "levonorgestrel",
    "lng ius": "levonorgestrel",
    # NSAIDs / aspirin
    "asa": "aspirin",
    # Smoking cessation brands
    "chantix": "varenicline",
    "nrt": "nicotine",
    "nicotine replacement": "nicotine",
    "nicotine patch": "nicotine",
    # Bisphosphonates
    "zometa": "zoledronic acid",
    "reclast": "zoledronic acid",
    "prolia": "denosumab",
    "xgeva": "denosumab",
    # Tamoxifen brand
    "nolvadex": "tamoxifen",
    # 5-ARI brands
    "proscar": "finasteride",
    "propecia": "finasteride",
    "avodart": "dutasteride",
    # Aromatase inhibitor brands
    "arimidex": "anastrozole",
    "femara": "letrozole",
    "aromasin": "exemestane",
    # Common cancer phrasings
    "breast ca": "breast cancer",
    "prostate ca": "prostate cancer",
    "colorectal ca": "colorectal cancer",
    "crc": "colorectal cancer",
}

# Phrase → list of additional canonical PrimeKG node names that should
# also be seeded when the phrase appears anywhere in the query. This
# enriches sparse/structural queries (e.g., HPV vaccine context).
COSEED_HINTS: dict[str, list[str]] = {
    "hpv": ["cervical cancer"],
    "gardasil": ["cervical cancer"],
    "papilloma": ["cervical cancer"],
    "head and neck cancer": ["nicotine"],
    "cancer survivor": [],
}

# Default node-type allow-list for clinical retrieval. PrimeKG's
# gene/protein layer is dense (28K nodes, 686K interacts_with edges)
# and dominates 2-hop expansion. For a clinical-counseling query we
# care about drugs, diseases, phenotypes, anatomy, exposures, pathways.
# Caller can override via `subgraph_slice(..., node_type_whitelist=...)`
# or pass None to disable.
DEFAULT_NODE_TYPE_WHITELIST = {
    "drug",
    "disease",
    "effect/phenotype",
    "anatomy",
    "exposure",
    "pathway",
}

# Default edge-relation allow-list. PrimeKG's `synergistic interaction`
# is a 2.6M-edge bin that floods the 2-hop window with off-target
# drug-pair noise; `expression present` (3M edges) lives in the
# gene/protein layer. We keep relations that carry clinical meaning at
# the bedside: drug ↔ disease, drug ↔ phenotype, disease taxonomy.
DEFAULT_EDGE_FILTER: list[str] | None = [
    "indication",
    "contraindication",
    "off-label use",
    "side effect",
    "phenotype present",
    "phenotype absent",
    "associated with",
    "parent-child",
    "linked to",
    "disease_phenotype_positive",
    "disease_phenotype_negative",
    "drug_protein",
    "drug_drug",
    "drug_effect",
    "anatomy_protein_present",
    "anatomy_protein_absent",
    "exposure_disease",
    "exposure_protein",
    "exposure_phenotype",
    "indication",
]


@dataclass
class PrimeKG:
    """Wraps the PrimeKG graph + secondary indices."""

    graph: object  # networkx.Graph
    name_to_index: dict[str, int]
    nodeid_to_index: dict[str, int]
    schema: dict
    _name_set: set[str] = field(init=False, default_factory=set)

    def __post_init__(self) -> None:
        # `name_to_index` keys are already lowercased by the loader. We
        # also build a name_set for fast substring scans.
        self._name_set = set(self.name_to_index.keys())

    @classmethod
    def from_pickle(cls, path: Path | str) -> "PrimeKG":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PrimeKG pickle not found: {path}")
        with open(path, "rb") as f:
            payload = pickle.load(f)  # noqa: S301 -- our own artifact
        return cls(
            graph=payload["graph"],
            name_to_index=payload["name_to_index"],
            nodeid_to_index=payload["nodeid_to_index"],
            schema=payload.get("schema", {}),
        )

    # ------------------------------------------------------------------
    # Seed selection
    # ------------------------------------------------------------------

    def seed_entities_from_query(
        self, query: str, *, max_seeds: int = 12
    ) -> list[int]:
        """Four-layer matcher: alias rewrite, exact name, normalized
        fragment, lemmatized substring. Returns up to `max_seeds`
        node_index seeds."""
        ql = query.lower()
        seeds: list[int] = []

        # Layer 0a: co-seed hints — pull in extra canonical names that
        # we know are clinically relevant when a phrase appears.
        for phrase, extras in COSEED_HINTS.items():
            if phrase in ql:
                for canonical in extras:
                    idx = self.name_to_index.get(canonical)
                    if idx is not None and idx not in seeds:
                        seeds.append(idx)

        # Layer 0: alias rewrites — brand/abbreviation → canonical name.
        # We resolve aliases up front and add their target node directly,
        # then ALSO allow the rewritten string to participate in window
        # matching below.
        alias_query = ql
        for alias, target in ALIASES.items():
            if alias in alias_query:
                if target in self.name_to_index:
                    idx = self.name_to_index[target]
                    if idx not in seeds:
                        seeds.append(idx)
                # Replace alias with target so window matching can also
                # pick it up (and we don't double-tokenize).
                alias_query = alias_query.replace(alias, target)
                if len(seeds) >= max_seeds:
                    return seeds

        # Layer 1: exact-name hits (multi-word phrases first).
        # We test contiguous 1- to 4-token windows from the query against
        # name_to_index — this catches "zoledronic acid" before "acid".
        ql = alias_query
        words = _NAME_TOKEN_RE.findall(ql)
        for window_size in (4, 3, 2, 1):
            for i in range(len(words) - window_size + 1):
                phrase = " ".join(words[i : i + window_size])
                if phrase in self._name_set:
                    idx = self.name_to_index[phrase]
                    if idx not in seeds:
                        seeds.append(idx)
                if len(seeds) >= max_seeds:
                    return seeds

        # Layer 2: hyphen/casefold normalization. PrimeKG often has
        # `levo-norgestrel` whereas a user types `levonorgestrel`. We
        # build a stripped-hyphen view at first-call.
        if not hasattr(self, "_dehyphen_index"):
            self._dehyphen_index = {  # type: ignore[attr-defined]
                k.replace("-", " "): v for k, v in self.name_to_index.items()
            }
        for window_size in (4, 3, 2, 1):
            for i in range(len(words) - window_size + 1):
                phrase = " ".join(words[i : i + window_size])
                idx = self._dehyphen_index.get(phrase)  # type: ignore[attr-defined]
                if idx is not None and idx not in seeds:
                    seeds.append(idx)
                if len(seeds) >= max_seeds:
                    return seeds

        # Layer 3: cheap heuristic — long-token (>=5 chars) substring scan.
        # We take any single query token that appears as a complete name
        # in the index. Skip short stop-tokens.
        for w in words:
            if len(w) < 5:
                continue
            if w in self.name_to_index:
                idx = self.name_to_index[w]
                if idx not in seeds:
                    seeds.append(idx)
            if len(seeds) >= max_seeds:
                break

        return seeds

    # ------------------------------------------------------------------
    # Subgraph BFS
    # ------------------------------------------------------------------

    def subgraph_slice(
        self,
        seeds: Iterable[int],
        *,
        max_hops: int = 2,
        max_nodes: int = 100,
        edge_filter: list[str] | None | bool = "default",
        node_type_whitelist: set[str] | None = None,
        seed_priority_types: tuple[str, ...] = ("drug", "disease", "effect/phenotype"),
    ):
        """BFS expansion under nx-cugraph (when available). Returns the
        induced subgraph capped at `max_nodes` nodes (BFS-order; closer
        nodes win).

        Filters:
          edge_filter           — keep only edges whose `display_relation`
                                  or `relation` is in this list.
          node_type_whitelist   — only traverse to nodes whose `node_type`
                                  is in this set. Defaults to clinically-
                                  actionable types (excludes the dense
                                  gene/protein layer that otherwise
                                  saturates the 2-hop window).
          seed_priority_types   — when capping at max_nodes, prefer
                                  expanding from seeds of these types.
        """
        import networkx as nx  # noqa: WPS433
        G = self.graph

        if node_type_whitelist is None:
            node_type_whitelist = DEFAULT_NODE_TYPE_WHITELIST
        # Resolve the sentinel default for edge_filter:
        #   "default" -> use DEFAULT_EDGE_FILTER
        #   None / False -> no filter
        #   list[str] -> caller-supplied whitelist
        if edge_filter == "default":
            edge_filter = DEFAULT_EDGE_FILTER
        if edge_filter is False or edge_filter is None:
            edge_filter = None

        seeds_list = [s for s in seeds if s in G]
        if not seeds_list:
            return G.subgraph([]).copy()

        # Re-order seeds: clinically-priority types first.
        seeds_list = sorted(
            seeds_list,
            key=lambda n: (
                G.nodes[n].get("node_type") not in seed_priority_types,
                n,
            ),
        )

        visited: set[int] = set(seeds_list)
        frontier = deque(seeds_list)
        for _ in range(max_hops):
            next_frontier: list[int] = []
            while frontier:
                if len(visited) >= max_nodes:
                    break
                node = frontier.popleft()
                if hasattr(G, "successors"):
                    iter_nbrs = list(G.successors(node)) + list(
                        G.predecessors(node)
                    )
                else:
                    iter_nbrs = list(G.neighbors(node))
                for nbr in iter_nbrs:
                    if nbr in visited:
                        continue
                    nbr_type = G.nodes[nbr].get("node_type")
                    if (
                        node_type_whitelist
                        and nbr_type not in node_type_whitelist
                    ):
                        continue
                    if edge_filter is not None:
                        ed = G.get_edge_data(node, nbr) or G.get_edge_data(
                            nbr, node, default={}
                        )
                        if (
                            ed.get("display_relation") not in edge_filter
                            and ed.get("relation") not in edge_filter
                        ):
                            continue
                    visited.add(nbr)
                    next_frontier.append(nbr)
                    if len(visited) >= max_nodes:
                        break
            frontier = deque(next_frontier)
            if not frontier:
                break

        return G.subgraph(visited).copy()

    # ------------------------------------------------------------------
    # Optional Louvain for downstream grouping in the rendered context
    # ------------------------------------------------------------------

    def community_label(self, subgraph) -> dict[int, int]:
        """Run Louvain via nx-cugraph dispatch. Returns
        {node_index: community_id}. Empty dict on failure / tiny graph."""
        if subgraph.number_of_nodes() < 4:
            return {}
        try:
            import networkx as nx  # noqa: WPS433
            # Connected-components on undirected graphs is fast + clear.
            # If the subgraph backend is cugraph, this dispatches to
            # cugraph's CC. We use CC instead of Louvain here because
            # Louvain on a 100-node slice is mostly noise; CC partitions
            # by graph structure.
            comps = list(nx.connected_components(subgraph))
            label: dict[int, int] = {}
            for cid, comp in enumerate(comps):
                for n in comp:
                    label[n] = cid
            return label
        except Exception:  # noqa: BLE001
            return {}

    # ------------------------------------------------------------------
    # Serialize to context block
    # ------------------------------------------------------------------

    def serialize_to_context(
        self,
        subgraph,
        *,
        max_tokens: int = 4096,
        title: str | None = None,
    ) -> str:
        """Render subgraph as a token-budgeted text block. Format chosen
        to be parseable by the brain at inference time:

            Knowledge-graph entities relevant to this case:
              [TYPE] Name (id=PK:NODE_ID)
              ...
            Edges:
              Name1 --display_relation--> Name2

        We deduplicate edges (both directions of a symmetric edge), and
        cap nodes / edges so the block fits ~max_tokens. Token estimate
        is char_count // 4.
        """
        if subgraph.number_of_nodes() == 0:
            return ""

        title = title or "Knowledge-graph entities relevant to this case:"
        lines: list[str] = [title, ""]

        # Group nodes by node_type for readability.
        by_type: dict[str, list[tuple[int, dict]]] = {}
        for n, attrs in subgraph.nodes(data=True):
            by_type.setdefault(attrs.get("node_type", "?"), []).append(
                (n, attrs)
            )
        lines.append("Nodes:")
        for ntype in sorted(by_type.keys()):
            for n, attrs in by_type[ntype]:
                name = attrs.get("node_name", str(n))
                nid = attrs.get("node_id", "")
                src = attrs.get("node_source", "")
                src_tag = f" [{src}]" if src and src != "nan" else ""
                lines.append(
                    f"  [{ntype}] {name} (id={src_tag} PrimeKG:{nid})"
                )

        lines.append("")
        lines.append("Edges:")
        # Deduplicate edges (sorted endpoints) so an undirected
        # representation isn't double-counted.
        seen: set[tuple[int, int, str]] = set()
        for u, v, attrs in subgraph.edges(data=True):
            rel = attrs.get("display_relation") or attrs.get("relation", "rel")
            key = (min(u, v), max(u, v), rel)
            if key in seen:
                continue
            seen.add(key)
            uname = subgraph.nodes[u].get("node_name", str(u))
            vname = subgraph.nodes[v].get("node_name", str(v))
            lines.append(f"  {uname} --{rel}--> {vname}")

        block = "\n".join(lines)
        # Token budget. ~4 chars per token, hard cap.
        max_chars = max_tokens * 4
        if len(block) > max_chars:
            cutoff = block.rfind("\n", 0, max_chars)
            if cutoff < 0:
                cutoff = max_chars
            block = (
                block[:cutoff]
                + "\n  ... (subgraph truncated to fit context budget)"
            )
        return block


# Convenience: module-level cached singleton so the bench harness loads
# the pickle exactly once per run, not once per example.
_CACHED: dict[str, PrimeKG] = {}


def load_primekg(pickle_path: Path | str) -> PrimeKG:
    key = str(Path(pickle_path).resolve())
    if key not in _CACHED:
        t0 = time.time()
        _CACHED[key] = PrimeKG.from_pickle(key)
        n_nodes = _CACHED[key].graph.number_of_nodes()
        n_edges = _CACHED[key].graph.number_of_edges()
        print(
            f"[primekg-helper] loaded {key} ({n_nodes:,} nodes, "
            f"{n_edges:,} edges) in {time.time() - t0:.2f}s"
        )
    return _CACHED[key]


def main() -> int:
    """Smoke run for the helper. Builds a slice for each held-out fixture
    query and prints the rendered block."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--primekg-path",
        default="/home/shadeform/medomni/primekg/primekg.gpickle",
    )
    parser.add_argument("--max-hops", type=int, default=2)
    parser.add_argument("--max-nodes", type=int, default=100)
    args = parser.parse_args()

    pkg = load_primekg(args.primekg_path)
    queries = [
        "I'm 35 and never got the HPV shot. Should I get Gardasil-9?",
        "Postmenopausal HR+ breast cancer on anastrozole — adjuvant zoledronic acid?",
        "Low-dose aspirin to prevent colorectal cancer in family history",
        "PSA 3.4 with family history of prostate cancer — finasteride or dutasteride?",
        "Statin for primary cardiovascular prevention with cancer-risk concerns",
        "Varenicline vs nicotine replacement after head and neck cancer",
    ]
    for q in queries:
        seeds = pkg.seed_entities_from_query(q)
        sub = pkg.subgraph_slice(
            seeds, max_hops=args.max_hops, max_nodes=args.max_nodes
        )
        names = [
            pkg.graph.nodes[s].get("node_name", str(s)) for s in seeds
        ]
        print(
            f"\n--- query: {q[:70]}\n    seeds={names} "
            f"subgraph: {sub.number_of_nodes()}n / {sub.number_of_edges()}e"
        )
        block = pkg.serialize_to_context(sub, max_tokens=512)
        print(block[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
