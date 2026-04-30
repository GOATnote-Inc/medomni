#!/usr/bin/env python3
"""Phase 2.4 — thin HTTP service exposing PrimeKG subgraph slicing.

Designed to run on the B300 (where the 268 MB pickle is resident in
RAPIDS-venv RAM and nx-cugraph dispatch is available). The laptop side
of `sovereign_bench` reaches it via the existing ssh tunnel on port
8005 (the next free port after the 8000-8003 inference services).

API
---
GET  /health
POST /subgraph
       {"query": "<text>",
        "max_hops": 2,
        "max_nodes": 100,
        "max_tokens": 2048,
        "edge_filter": null }
     → {"seed_count": int,
        "seed_names": [str, ...],
        "n_nodes": int,
        "n_edges": int,
        "block": "<rendered text block>",
        "elapsed_ms": int }

Run on B300:
    cd ~/medomni-bench
    NETWORKX_AUTOMATIC_BACKENDS=cugraph \\
        ~/medomni-rapids/.venv/bin/python scripts/serve_primekg_b300.py \\
            --primekg-path /home/shadeform/medomni/primekg/primekg.gpickle \\
            --port 8005

Dependencies in Alpha's venv: stdlib http.server only. No fastapi /
uvicorn install needed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Ensure cugraph dispatch.
os.environ.setdefault("NETWORKX_AUTOMATIC_BACKENDS", "cugraph")

# Module-level singleton populated at startup.
_PKG = None  # type: ignore[var-annotated]


def _make_handler():
    pkg = _PKG  # capture in closure

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # noqa: A003,N802
            sys.stderr.write(
                f"[primekg-svc] {self.address_string()} - {fmt % args}\n"
            )

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                body = json.dumps(
                    {
                        "ok": True,
                        "n_nodes": pkg.graph.number_of_nodes(),
                        "n_edges": pkg.graph.number_of_edges(),
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(404)

        def do_POST(self):  # noqa: N802
            if self.path != "/subgraph":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            try:
                req = json.loads(raw.decode())
            except Exception as e:  # noqa: BLE001
                self.send_error(400, f"bad json: {e}")
                return

            t0 = time.time()
            query = req.get("query", "")
            max_hops = int(req.get("max_hops", 2))
            max_nodes = int(req.get("max_nodes", 100))
            max_tokens = int(req.get("max_tokens", 2048))
            edge_filter = req.get("edge_filter") or None

            seeds = pkg.seed_entities_from_query(query)
            sub = pkg.subgraph_slice(
                seeds,
                max_hops=max_hops,
                max_nodes=max_nodes,
                edge_filter=edge_filter,
            )
            block = pkg.serialize_to_context(sub, max_tokens=max_tokens)
            seed_names = [
                pkg.graph.nodes[s].get("node_name", str(s)) for s in seeds
            ]

            body = json.dumps(
                {
                    "seed_count": len(seeds),
                    "seed_names": seed_names,
                    "n_nodes": sub.number_of_nodes(),
                    "n_edges": sub.number_of_edges(),
                    "block": block,
                    "elapsed_ms": int((time.time() - t0) * 1000),
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primekg-path",
        default="/home/shadeform/medomni/primekg/primekg.gpickle",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8005)
    args = parser.parse_args()

    global _PKG
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from graph_primekg_subgraph import load_primekg  # noqa: WPS433
    print(f"[primekg-svc] loading {args.primekg_path}...", file=sys.stderr)
    _PKG = load_primekg(args.primekg_path)
    print(
        f"[primekg-svc] PrimeKG: {_PKG.graph.number_of_nodes():,} nodes, "
        f"{_PKG.graph.number_of_edges():,} edges",
        file=sys.stderr,
    )
    print(
        f"[primekg-svc] listening on http://{args.host}:{args.port}",
        file=sys.stderr,
    )
    server = ThreadingHTTPServer((args.host, args.port), _make_handler())
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
