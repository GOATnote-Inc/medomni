// /api/telemetry — system facts + live vLLM serving metrics.
//
// Phase 1 of the flagship-demo plan ("make the stack visible"): the System
// panel on /4UWHAt shows what is actually running. This route returns two
// things:
//   1. Static system FACTS — model, serving stack, sovereignty — which are
//      true by construction of this deployment (see CLAUDE.md §2).
//   2. LIVE serving metrics, fetched read-only from the vLLM Prometheus
//      `/metrics` endpoint behind the existing tunnel.
//
// Honesty + safety contract (SPEC §2, §6 Phase 1, §8):
//   - GET only. This route never mutates the serving pod and never does
//     anything but a read-only HTTP GET to an endpoint the pod already
//     exposes. It does NOT ssh, does NOT touch pod config.
//   - If `/metrics` is unavailable (tunnel down, endpoint disabled, slow),
//     the route still returns 200 with the facts and
//     `metrics: { available: false }`. It never throws, never 500s — a
//     telemetry hiccup must not look like an outage.
//   - No fabricated numbers: an absent metric is `null`, surfaced by the
//     panel as "unavailable", never a placeholder value.

import {
  extractVllmMetrics,
  parsePrometheus,
  type VllmServingMetrics,
} from "@/lib/4uwhat/prometheus";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Static facts about this deployment. These are not measured — they are
// true by construction (the model we serve, the engine we serve it on, and
// the sovereignty property the whole repo is built around: no third-party
// AI API is on the inference path — CLAUDE.md §2).
const SYSTEM_FACTS = {
  model: "Nemotron-3-Nano-Omni",
  servingStack: "vLLM",
  sovereign: true,
} as const;

// Keep the upstream `/metrics` read snappy — the System panel polls this
// route ~every 5s, so a hung scrape must not pile up. 4s leaves headroom
// under the poll interval; on timeout we degrade to metrics.available=false.
const METRICS_FETCH_TIMEOUT_MS = 4000;

/** Discriminated metrics block: present-and-parsed, or explicitly absent. */
type MetricsResult =
  | ({ available: true; scrapedAtMs: number } & VllmServingMetrics)
  | { available: false; reason: string };

/**
 * Read-only GET to the vLLM Prometheus endpoint, parsed into the panel's
 * metrics shape. Returns an `available:false` block (never throws) on any
 * failure: missing tunnel env, network error, non-200, or timeout.
 */
async function fetchServingMetrics(): Promise<MetricsResult> {
  const tunnelUrl = process.env.MEDOMNI_TUNNEL_URL;
  if (!tunnelUrl) {
    return { available: false, reason: "MEDOMNI_TUNNEL_URL not set" };
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), METRICS_FETCH_TIMEOUT_MS);
  try {
    // `${tunnelUrl}/metrics` — the standard vLLM Prometheus path. GET only.
    const res = await fetch(`${tunnelUrl.replace(/\/$/, "")}/metrics`, {
      method: "GET",
      headers: { Accept: "text/plain" },
      signal: controller.signal,
      cache: "no-store",
    });
    if (!res.ok) {
      return { available: false, reason: `metrics endpoint ${res.status}` };
    }
    const text = await res.text();
    const samples = parsePrometheus(text);
    const metrics = extractVllmMetrics(samples);
    return { available: true, scrapedAtMs: Date.now(), ...metrics };
  } catch (e) {
    const reason =
      (e as Error).name === "AbortError"
        ? `metrics scrape timed out after ${METRICS_FETCH_TIMEOUT_MS}ms`
        : `metrics scrape failed: ${(e as Error).message}`;
    return { available: false, reason };
  } finally {
    clearTimeout(timer);
  }
}

export async function GET() {
  const metrics = await fetchServingMetrics();
  return new Response(JSON.stringify({ ...SYSTEM_FACTS, metrics }), {
    status: 200,
    headers: {
      "Content-Type": "application/json",
      // Telemetry is live state — never serve a cached body. The panel
      // polls this for fresh numbers.
      "Cache-Control": "no-store",
    },
  });
}
