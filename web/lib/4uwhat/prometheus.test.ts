// Inline tests for the Prometheus text-exposition parser
// (lib/4uwhat/prometheus.ts).
//
// No test runner is configured in web/package.json, so this file is a
// runnable assertion script intended for `tsx`. Exits non-zero on first
// failure. Same harness shape as SessionProvider.test.ts.
//
// Run:
//   cd web && npx tsx lib/4uwhat/prometheus.test.ts
//
// Covers:
//   1. parsePrometheus — comment/blank skipping, labeled + unlabeled
//      samples, Inf/NaN literals, malformed lines, label values with
//      commas and escapes.
//   2. sumMetric / firstMetric — across-label aggregation, missing -> null,
//      non-finite values excluded.
//   3. extractVllmMetrics — projects a realistic vLLM `/metrics` body onto
//      VllmServingMetrics; absent series become null (no fabricated zeros).

import {
  extractVllmMetrics,
  firstMetric,
  parsePrometheus,
  sumMetric,
} from "./prometheus";

let passed = 0;
let failed = 0;
const failures: string[] = [];

async function test(name: string, fn: () => Promise<void> | void): Promise<void> {
  try {
    await fn();
    passed += 1;
    console.log(`  ok  ${name}`);
  } catch (e) {
    failed += 1;
    const msg = `${name}: ${(e as Error).message}`;
    failures.push(msg);
    console.error(`  FAIL ${msg}`);
  }
}

function assert(cond: unknown, msg: string): void {
  if (!cond) throw new Error(msg);
}

// A realistic slice of a vLLM `/metrics` response. vLLM partitions some
// series by `model_name`; KV-cache usage is a 0..1 fraction despite the
// `_perc` suffix; histograms expand to _sum/_count/_bucket lines.
const VLLM_METRICS_SAMPLE = `# HELP vllm:num_requests_running Number of requests currently running on GPU.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="nemotron"} 2.0
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="nemotron"} 1.0
# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{model_name="nemotron"} 0.42
# HELP vllm:generation_tokens_total Number of generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model_name="nemotron"} 184213.0
# HELP vllm:prompt_tokens_total Number of prefill tokens processed.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{model_name="nemotron"} 905142.0
# HELP vllm:request_success_total Count of successfully processed requests.
# TYPE vllm:request_success_total counter
vllm:request_success_total{finished_reason="stop",model_name="nemotron"} 311.0
vllm:request_success_total{finished_reason="length",model_name="nemotron"} 7.0
# HELP vllm:e2e_request_latency_seconds End to end request latency.
# TYPE vllm:e2e_request_latency_seconds histogram
vllm:e2e_request_latency_seconds_sum{model_name="nemotron"} 1422.5
vllm:e2e_request_latency_seconds_count{model_name="nemotron"} 318.0
`;

async function main(): Promise<void> {
  await test("parsePrometheus: empty body -> empty list", () => {
    assert(parsePrometheus("").length === 0, "empty body must parse to []");
  });

  await test("parsePrometheus: skips comments and blank lines", () => {
    const samples = parsePrometheus(VLLM_METRICS_SAMPLE);
    for (const s of samples) {
      assert(!s.name.startsWith("#"), `comment leaked as a sample: ${s.name}`);
      assert(s.name.length > 0, "empty metric name parsed");
    }
    // 9 data lines in the sample.
    assert(samples.length === 9, `expected 9 samples, got ${samples.length}`);
  });

  await test("parsePrometheus: labeled sample parses name + labels + value", () => {
    const samples = parsePrometheus(
      'vllm:num_requests_running{model_name="nemotron"} 2.0\n',
    );
    assert(samples.length === 1, "expected 1 sample");
    const s = samples[0];
    assert(s.name === "vllm:num_requests_running", `name drift: ${s.name}`);
    assert(s.labels.model_name === "nemotron", `label drift: ${s.labels.model_name}`);
    assert(s.value === 2, `value drift: ${s.value}`);
  });

  await test("parsePrometheus: unlabeled sample parses", () => {
    const samples = parsePrometheus("process_cpu_seconds_total 12.5\n");
    assert(samples.length === 1, "expected 1 sample");
    assert(samples[0].name === "process_cpu_seconds_total", "name drift");
    assert(
      Object.keys(samples[0].labels).length === 0,
      "unlabeled sample must have empty labels",
    );
    assert(samples[0].value === 12.5, `value drift: ${samples[0].value}`);
  });

  await test("parsePrometheus: handles Inf and NaN value literals", () => {
    const samples = parsePrometheus(
      "a_metric +Inf\nb_metric -Inf\nc_metric NaN\n",
    );
    assert(samples.length === 3, `expected 3, got ${samples.length}`);
    assert(samples[0].value === Number.POSITIVE_INFINITY, "+Inf not parsed");
    assert(samples[1].value === Number.NEGATIVE_INFINITY, "-Inf not parsed");
    assert(Number.isNaN(samples[2].value), "NaN not parsed");
  });

  await test("parsePrometheus: skips malformed lines, keeps good ones", () => {
    const samples = parsePrometheus(
      [
        "good_metric 1.0",
        "this is not prometheus at all",
        "unbalanced{label=\"x\" 5.0",
        "another_good 2.0",
        "",
      ].join("\n"),
    );
    const names = samples.map((s) => s.name);
    assert(names.includes("good_metric"), "good_metric dropped");
    assert(names.includes("another_good"), "another_good dropped");
    assert(!names.includes("unbalanced"), "unbalanced-brace line must be skipped");
  });

  await test("parsePrometheus: label value containing a comma", () => {
    const samples = parsePrometheus(
      'm{a="x,y",b="z"} 3.0\n',
    );
    assert(samples.length === 1, "expected 1 sample");
    assert(samples[0].labels.a === "x,y", `comma in value lost: ${samples[0].labels.a}`);
    assert(samples[0].labels.b === "z", "second label lost");
    assert(samples[0].value === 3, "value parsed wrong");
  });

  await test("sumMetric: sums across label sets", () => {
    const samples = parsePrometheus(VLLM_METRICS_SAMPLE);
    // request_success_total has two label sets: 311 + 7 = 318.
    const total = sumMetric(samples, "vllm:request_success_total");
    assert(total === 318, `expected 318, got ${total}`);
  });

  await test("sumMetric: missing metric -> null (not 0)", () => {
    const samples = parsePrometheus(VLLM_METRICS_SAMPLE);
    const missing = sumMetric(samples, "vllm:does_not_exist");
    assert(missing === null, `missing metric must be null, got ${missing}`);
  });

  await test("sumMetric: excludes non-finite values", () => {
    const samples = parsePrometheus("m 5.0\nm +Inf\n");
    // The +Inf sample is excluded; only 5 counts.
    assert(sumMetric(samples, "m") === 5, "non-finite value must be excluded");
  });

  await test("firstMetric: returns the first finite value", () => {
    const samples = parsePrometheus(VLLM_METRICS_SAMPLE);
    const usage = firstMetric(samples, "vllm:gpu_cache_usage_perc");
    assert(usage === 0.42, `expected 0.42, got ${usage}`);
  });

  await test("extractVllmMetrics: projects the realistic sample correctly", () => {
    const m = extractVllmMetrics(parsePrometheus(VLLM_METRICS_SAMPLE));
    assert(m.runningRequests === 2, `runningRequests: ${m.runningRequests}`);
    assert(m.waitingRequests === 1, `waitingRequests: ${m.waitingRequests}`);
    assert(m.kvCacheUsage === 0.42, `kvCacheUsage: ${m.kvCacheUsage}`);
    assert(
      m.generationTokensTotal === 184213,
      `generationTokensTotal: ${m.generationTokensTotal}`,
    );
    assert(
      m.promptTokensTotal === 905142,
      `promptTokensTotal: ${m.promptTokensTotal}`,
    );
    assert(
      m.requestsFinishedTotal === 318,
      `requestsFinishedTotal: ${m.requestsFinishedTotal}`,
    );
  });

  await test("extractVllmMetrics: absent series become null, not fabricated zeros", () => {
    // An empty/older `/metrics` with none of the vllm:* series we read.
    const m = extractVllmMetrics(parsePrometheus("python_gc_objects 100\n"));
    assert(m.runningRequests === null, "missing runningRequests must be null");
    assert(m.waitingRequests === null, "missing waitingRequests must be null");
    assert(m.kvCacheUsage === null, "missing kvCacheUsage must be null");
    assert(
      m.generationTokensTotal === null,
      "missing generationTokensTotal must be null",
    );
    assert(
      m.requestsFinishedTotal === null,
      "missing requestsFinishedTotal must be null",
    );
  });

  console.log("");
  console.log(`prometheus tests: ${passed} passed, ${failed} failed`);
  if (failed > 0) {
    for (const f of failures) console.error(`  - ${f}`);
    process.exit(1);
  }
}

void main();
