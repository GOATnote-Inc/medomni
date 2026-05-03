// PrimeKG subgraph lookup. Calls the sovereign nx-cugraph subgraph service
// running on the B300 (catfish) at port 8005, exposed publicly via the
// MEDOMNI_PRIMEKG_URL env var (a tunnel to host:8005).
//
// The service is a thin stdlib http.server (see prism42-nemotron-med
// scripts/serve_primekg_b300.py): POST /subgraph takes a free-text query,
// runs MeSH-style lookup against the 268MB PrimeKG pickle in cugraph RAM,
// and returns a k-hop neighborhood pre-rendered as a text block plus
// metadata.

const TIMEOUT_MS = 12_000;

export interface PrimeKGSubgraphResult {
  query: string;
  seed_count: number;
  seed_names: string[];
  n_nodes: number;
  n_edges: number;
  block: string;
  elapsed_ms: number;
}

function withTimeout<T>(p: Promise<T>, ms: number, label: string): Promise<T> {
  return new Promise((resolve, reject) => {
    const t = setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms);
    p.then(
      (v) => {
        clearTimeout(t);
        resolve(v);
      },
      (e) => {
        clearTimeout(t);
        reject(e);
      },
    );
  });
}

export async function primekgLookup(args: {
  query: string;
  maxHops?: number;
  maxNodes?: number;
}): Promise<PrimeKGSubgraphResult> {
  const baseUrl = process.env.MEDOMNI_PRIMEKG_URL;
  if (!baseUrl) {
    throw new Error(
      "MEDOMNI_PRIMEKG_URL is not configured on this deployment. " +
        "PrimeKG knowledge-graph lookup is unavailable; answer from the " +
        "model's prior knowledge and PubMed instead.",
    );
  }
  const query = args.query.trim();
  if (!query) {
    throw new Error("query is required");
  }

  const payload = {
    query,
    max_hops: Math.min(Math.max(args.maxHops ?? 2, 1), 3),
    max_nodes: Math.min(Math.max(args.maxNodes ?? 60, 10), 150),
    max_tokens: 1500,
    edge_filter: null,
  };

  const res = await withTimeout(
    fetch(`${baseUrl.replace(/\/$/, "")}/subgraph`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
    TIMEOUT_MS,
    "primekg subgraph",
  );

  if (!res.ok) {
    const txt = await res.text().catch(() => "(no body)");
    throw new Error(`primekg subgraph ${res.status}: ${txt.slice(0, 240)}`);
  }

  const json = (await res.json()) as PrimeKGSubgraphResult;
  return json;
}
