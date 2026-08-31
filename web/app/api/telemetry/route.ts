// /api/telemetry — system facts + live vLLM serving metrics.
//
// Phase 1 of the flagship-demo plan ("make the stack visible"): the System
// panel on /4UWHAt shows what is actually running. This route returns two
// things:
//   1. Static system FACTS — model, serving stack, sovereignty — which are
//      true by construction of this deployment (Claude API via the BFF;
//      the self-hosted GPU service is decommissioned — see CLAUDE.md §0).
//   2. LIVE serving metrics, fetched read-only from the legacy vLLM
//      Prometheus `/metrics` endpoint IF a tunnel is configured; with the
//      GPU service decommissioned this degrades to `available: false`.
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
// true by construction of the current architecture: the self-hosted GPU
// service (Nemotron on a dedicated H100 via vLLM) was decommissioned in
// June 2026 and inference is served by the Anthropic Claude API through
// this BFF (migration branch `feat/claude-opus-migration`). `sovereign`
// is therefore FALSE: user queries are processed by a third-party AI
// service. Do not flip this back to true unless inference genuinely
// returns to hardware we operate.
const SYSTEM_FACTS = {
  model: "Claude Opus (Anthropic API)",
  servingStack: "Anthropic Messages API via web BFF",
  sovereign: false,
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

/**
 * Read-only GET to vLLM's `/v1/models` endpoint. Returns the actual loaded
 * model id + HF root path (e.g. `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8`)
 * so the System panel can show what is REALLY serving — not just the UX
 * label in `SYSTEM_FACTS.model`. Returns `null` on any failure (same posture
 * as `fetchServingMetrics`: a telemetry hiccup must not look like an outage).
 *
 * This exists because "what's actually serving on orca" became an ssh-only
 * question during the 2026-05-23 audio-fix incident; surfacing it through the
 * panel makes the next such question a single curl from any maintainer.
 */
async function fetchServedModel(): Promise<{ id: string; root: string } | null> {
  const tunnelUrl = process.env.MEDOMNI_TUNNEL_URL;
  if (!tunnelUrl) return null;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), METRICS_FETCH_TIMEOUT_MS);
  try {
    const res = await fetch(`${tunnelUrl.replace(/\/$/, "")}/v1/models`, {
      method: "GET",
      headers: { Accept: "application/json" },
      signal: controller.signal,
      cache: "no-store",
    });
    if (!res.ok) return null;
    const data = (await res.json()) as {
      data?: Array<{ id: string; root?: string }>;
    };
    const first = data.data?.[0];
    if (!first) return null;
    return { id: first.id, root: first.root ?? first.id };
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

export async function GET() {
  const [metrics, served] = await Promise.all([
    fetchServingMetrics(),
    fetchServedModel(),
  ]);
  return new Response(
    JSON.stringify({
      ...SYSTEM_FACTS,
      // `served` is what vLLM reports for the actual loaded model. May
      // differ from `SYSTEM_FACTS.model` (which is a UX label) — the System
      // panel can surface both so "what's actually serving" is never a
      // mystery requiring ssh access. `null` when the tunnel is unreachable
      // or `/v1/models` returns non-OK.
      served,
      metrics,
    }),
    {
      status: 200,
      headers: {
        "Content-Type": "application/json",
        // Telemetry is live state — never serve a cached body. The panel
        // polls this for fresh numbers.
        "Cache-Control": "no-store",
      },
    },
  );
}
