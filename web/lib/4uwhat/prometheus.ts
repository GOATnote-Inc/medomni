// 4UWHAt — minimal Prometheus text-exposition parser.
//
// The vLLM server that hosts Nemotron-3-Nano-Omni exposes a Prometheus
// `/metrics` endpoint (vLLM ships this by default). Phase 1's System panel
// surfaces a few live serving metrics from it. We deliberately do NOT pull
// in a Prometheus client library — the exposition format is simple, the
// dependency surface of this repo is intentionally tiny (see package.json),
// and we only need a handful of `vllm:*` series. This is a focused,
// read-only parser for exactly that.
//
// Format reference (Prometheus text exposition, v0.0.4):
//   # HELP <name> <help text>
//   # TYPE <name> <type>
//   <name>{label="v",...} <value> [<timestamp>]
//   <name> <value>
// Histograms/summaries expand to `<name>_bucket`, `<name>_sum`,
// `<name>_count` series. We read counters/gauges and the `_sum`/`_count`
// of histograms; that is all the panel needs.

/** One parsed Prometheus sample (one line of the exposition). */
export interface PromSample {
  /** Metric name, e.g. `vllm:num_requests_running`. */
  name: string;
  /** Label set, e.g. `{ model_name: "nemotron" }`. Empty object if none. */
  labels: Record<string, string>;
  /** Numeric value. `NaN`, `+Inf`, `-Inf` are parsed to JS equivalents. */
  value: number;
}

/**
 * Parse a Prometheus text-exposition body into a flat list of samples.
 *
 * Tolerant by design: comment lines (`# HELP`, `# TYPE`) and blank lines
 * are skipped, malformed lines are skipped rather than throwing. A vLLM
 * `/metrics` response is trusted-but-best-effort input here — the System
 * panel must degrade gracefully, never crash the route.
 */
export function parsePrometheus(body: string): PromSample[] {
  const out: PromSample[] = [];
  if (!body) return out;

  for (const rawLine of body.split("\n")) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;

    // Split metric identifier (name + optional {labels}) from the trailing
    // "value [timestamp]". The identifier ends at the first space that is
    // NOT inside the {...} label block.
    const id = extractIdentifier(line);
    if (!id) continue;
    const rest = line.slice(id.length).trim();
    if (!rest) continue;

    // rest = "<value>" or "<value> <timestamp>"; take the first token.
    const valueToken = rest.split(/\s+/)[0];
    const value = parsePromValue(valueToken);
    if (value === undefined) continue;

    const braceIdx = id.indexOf("{");
    if (braceIdx === -1) {
      out.push({ name: id, labels: {}, value });
    } else {
      const name = id.slice(0, braceIdx);
      const labelBlock = id.slice(braceIdx + 1, id.lastIndexOf("}"));
      out.push({ name, labels: parseLabels(labelBlock), value });
    }
  }
  return out;
}

/**
 * Return the metric-identifier prefix of a sample line: the name plus, if
 * present, the full balanced `{...}` label block. Returns null if the line
 * does not start with a valid metric name.
 */
function extractIdentifier(line: string): string | null {
  // A metric name: [a-zA-Z_:][a-zA-Z0-9_:]*
  const nameMatch = line.match(/^[a-zA-Z_:][a-zA-Z0-9_:]*/);
  if (!nameMatch) return null;
  const name = nameMatch[0];
  const afterName = line.slice(name.length);
  if (!afterName.startsWith("{")) return name;
  const close = afterName.indexOf("}");
  if (close === -1) return null; // unbalanced — skip line
  return name + afterName.slice(0, close + 1);
}

/**
 * Parse a Prometheus label block body (the text between `{` and `}`) into
 * a key/value record. Handles quoted values with escaped quotes/backslashes.
 */
function parseLabels(block: string): Record<string, string> {
  const labels: Record<string, string> = {};
  if (!block.trim()) return labels;
  // label="value", possibly comma-separated. Value may contain commas, so
  // match name="...escaped..." pairs explicitly.
  const re = /([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"((?:[^"\\]|\\.)*)"/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(block)) !== null) {
    labels[m[1]] = m[2].replace(/\\(["\\n])/g, (_, c) => (c === "n" ? "\n" : c));
  }
  return labels;
}

/** Parse a Prometheus numeric value token, including the Inf/NaN literals. */
function parsePromValue(token: string): number | undefined {
  if (token === "+Inf" || token === "Inf") return Number.POSITIVE_INFINITY;
  if (token === "-Inf") return Number.NEGATIVE_INFINITY;
  if (token === "NaN") return Number.NaN;
  const n = Number(token);
  return Number.isNaN(n) ? undefined : n;
}

/**
 * Sum the values of every sample for `name` (across all label sets). vLLM
 * partitions some series by `model_name` / `engine`; for the panel we want
 * the fleet-wide total, so summing is correct. Returns null if no sample
 * with that name is present.
 */
export function sumMetric(samples: PromSample[], name: string): number | null {
  let total = 0;
  let found = false;
  for (const s of samples) {
    if (s.name !== name) continue;
    if (!Number.isFinite(s.value)) continue;
    total += s.value;
    found = true;
  }
  return found ? total : null;
}

/**
 * First finite value for `name`, or null. Use for gauges that are not
 * meaningfully summable (e.g. a cache-usage fraction).
 */
export function firstMetric(samples: PromSample[], name: string): number | null {
  for (const s of samples) {
    if (s.name === name && Number.isFinite(s.value)) return s.value;
  }
  return null;
}

/**
 * The serving-metrics shape the System panel consumes. Every field is
 * nullable — vLLM versions differ in which `vllm:*` series they export,
 * and a field absent from `/metrics` must surface as "unavailable" rather
 * than a fabricated zero.
 */
export interface VllmServingMetrics {
  /** Requests currently generating (gauge `vllm:num_requests_running`). */
  runningRequests: number | null;
  /** Requests queued for a slot (gauge `vllm:num_requests_waiting`). */
  waitingRequests: number | null;
  /**
   * Cumulative generation tokens (counter `vllm:generation_tokens_total`).
   * A counter — a rate must be derived from two reads; the panel shows the
   * raw cumulative count, honestly labeled.
   */
  generationTokensTotal: number | null;
  /** Cumulative prompt tokens (counter `vllm:prompt_tokens_total`). */
  promptTokensTotal: number | null;
  /**
   * KV-cache utilization, 0..1 (gauge `vllm:gpu_cache_usage_perc` — note
   * vLLM exports this as a fraction despite the `_perc` suffix).
   */
  kvCacheUsage: number | null;
  /** Cumulative finished requests (counter `vllm:request_success_total`). */
  requestsFinishedTotal: number | null;
}

/**
 * Project a parsed sample list onto `VllmServingMetrics`. Pure — does no
 * I/O. Any series missing from `samples` becomes `null`.
 */
export function extractVllmMetrics(samples: PromSample[]): VllmServingMetrics {
  return {
    runningRequests: sumMetric(samples, "vllm:num_requests_running"),
    waitingRequests: sumMetric(samples, "vllm:num_requests_waiting"),
    generationTokensTotal: sumMetric(samples, "vllm:generation_tokens_total"),
    promptTokensTotal: sumMetric(samples, "vllm:prompt_tokens_total"),
    kvCacheUsage: firstMetric(samples, "vllm:gpu_cache_usage_perc"),
    requestsFinishedTotal: sumMetric(samples, "vllm:request_success_total"),
  };
}
