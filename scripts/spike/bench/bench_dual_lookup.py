"""S3 — dual-lookup latency benchmark harness for the medomni clinical agent.

Pattern B (dual-lookup, parallel tool dispatch) adds `get_patient_context` as a
fifth tool that fires in parallel with `primekg_lookup`. This harness POSTs
synthetic queries to /api/agent, parses the SSE event stream, and measures
per-phase wall-clock latencies (TTFB, time-to-tool-dispatch, parallel-tool wait,
text-stream duration, end-to-end).

Read-only against production. No PHI; uses Synthea-style synthetic patient IDs
passed via the `patientId` field S2 is adding. Five hardcoded prompts cover
patient-context-heavy, knowledge-only, and mixed cases.

Env vars:
  BENCH_BASE_URL       (default: http://localhost:3000)
  BENCH_PATIENT_ID     (REQUIRED — synthetic only, no PHI)
  BENCH_CONCURRENCY    (default: 1)
  BENCH_N_REQUESTS     (default: 20)

Output:
  Markdown table on stdout with per-phase p50/p95/p99 (ms) plus failure rate.
  Phases:
    - ttfb:               request_sent -> first_byte
    - time_to_dispatch:   request_sent -> first tool-input-available
    - parallel_tool_wait: first tool-input-available -> max(primekg_done,
                          patient_context_done) (i.e. when both lookups done)
    - text_stream:        text-start -> text-end
    - end_to_end:         request_sent -> last event (text-end or stream close)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

try:
    import numpy as _np

    def _percentile(values: list[float], q: float) -> float:
        if not values:
            return float("nan")
        return float(_np.percentile(values, q))

except ImportError:  # pragma: no cover

    def _percentile(values: list[float], q: float) -> float:
        if not values:
            return float("nan")
        s = sorted(values)
        # linear interpolation, equivalent to numpy default
        if len(s) == 1:
            return s[0]
        pos = (q / 100.0) * (len(s) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(s) - 1)
        frac = pos - lo
        return s[lo] + (s[hi] - s[lo]) * frac


PROMPTS: list[str] = [
    "What does this patient's most recent A1C suggest about glycemic control?",
    "Is metformin still first-line for type 2 diabetes per 2024 ADA?",
    "Given this patient's active medications and conditions, what's the highest drug-interaction risk?",
    "What does the GFR trend look like for this patient over the last 6 months?",
    "What's the CHA2DS2-VASc score for a 72yo with HTN, DM, and prior TIA?",
]

AGENT_PATH = "/4UWHAt/api/agent"
REQUEST_TIMEOUT_S = 90.0


@dataclass
class RequestTiming:
    prompt_idx: int
    ok: bool = False
    failure_reason: str | None = None
    t_request_sent: float = 0.0
    t_first_byte: float | None = None
    t_first_tool_call_emitted: float | None = None
    t_primekg_result: float | None = None
    t_patient_context_result: float | None = None
    t_text_start: float | None = None
    t_text_end: float | None = None
    t_stream_close: float | None = None
    tool_outputs_seen: list[str] = field(default_factory=list)


def _parse_sse_lines(buffer: str) -> tuple[list[str], str]:
    """Split a buffer into complete SSE data lines + leftover.

    UI Message Stream format used by Vercel AI SDK is line-delimited
    `data: <json>` events with `\n` separators (the route's iterSseLines uses
    plain newline, not the canonical `\n\n`). We mirror that.
    """
    out: list[str] = []
    while True:
        nl = buffer.find("\n")
        if nl == -1:
            break
        line = buffer[:nl].rstrip("\r")
        buffer = buffer[nl + 1 :]
        if line.startswith("data: "):
            out.append(line[6:])
    return out, buffer


async def _run_one(
    client: httpx.AsyncClient,
    base_url: str,
    patient_id: str | None,
    prompt: str,
    prompt_idx: int,
) -> RequestTiming:
    timing = RequestTiming(prompt_idx=prompt_idx)

    body: dict[str, Any] = {
        "messages": [
            {
                "id": f"msg-{uuid.uuid4().hex[:12]}",
                "role": "user",
                "parts": [{"type": "text", "text": prompt}],
            }
        ]
    }
    if patient_id is not None:
        body["patientId"] = patient_id

    url = base_url.rstrip("/") + AGENT_PATH

    # Tool ID -> tool name (so we can recognize tool-output-available frames,
    # which carry only toolCallId + output).
    tool_call_names: dict[str, str] = {}
    buffer = ""

    timing.t_request_sent = time.perf_counter()
    try:
        async with client.stream(
            "POST",
            url,
            json=body,
            headers={"Accept": "text/event-stream"},
            timeout=REQUEST_TIMEOUT_S,
        ) as resp:
            if resp.status_code != 200:
                timing.failure_reason = f"http_{resp.status_code}"
                # Drain a small body for diagnostics but don't keep it.
                try:
                    _ = await resp.aread()
                except Exception:
                    pass
                return timing

            async for chunk in resp.aiter_text():
                if timing.t_first_byte is None:
                    timing.t_first_byte = time.perf_counter()
                buffer += chunk
                events, buffer = _parse_sse_lines(buffer)
                for raw in events:
                    if raw.strip() == "[DONE]":
                        continue
                    try:
                        ev = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    _record_event(ev, timing, tool_call_names)
            # Drain any final buffered partial line that ends without newline.
            if buffer.startswith("data: "):
                tail = buffer[6:].strip()
                if tail and tail != "[DONE]":
                    try:
                        ev = json.loads(tail)
                        _record_event(ev, timing, tool_call_names)
                    except json.JSONDecodeError:
                        pass
            timing.t_stream_close = time.perf_counter()
    except httpx.HTTPError as e:
        timing.failure_reason = f"httpx_{type(e).__name__}"
        return timing
    except Exception as e:  # pragma: no cover
        timing.failure_reason = f"exc_{type(e).__name__}"
        return timing

    # Success criteria: we saw at least one byte and at least one tool-input
    # event OR a text-end (knowledge-only prompts can short-circuit).
    if timing.t_first_byte is None:
        timing.failure_reason = "no_bytes"
        return timing

    timing.ok = True
    return timing


def _record_event(
    ev: dict[str, Any],
    timing: RequestTiming,
    tool_call_names: dict[str, str],
) -> None:
    et = ev.get("type")
    now = time.perf_counter()

    if et == "tool-input-start":
        tcid = ev.get("toolCallId")
        name = ev.get("toolName")
        if isinstance(tcid, str) and isinstance(name, str):
            tool_call_names[tcid] = name

    if et == "tool-input-available":
        tcid = ev.get("toolCallId")
        name = ev.get("toolName")
        if isinstance(tcid, str) and isinstance(name, str):
            tool_call_names[tcid] = name
        if timing.t_first_tool_call_emitted is None:
            timing.t_first_tool_call_emitted = now

    elif et == "tool-output-available":
        tcid = ev.get("toolCallId")
        name = tool_call_names.get(tcid, "") if isinstance(tcid, str) else ""
        if name:
            timing.tool_outputs_seen.append(name)
        if name == "primekg_lookup" and timing.t_primekg_result is None:
            timing.t_primekg_result = now
        elif name == "get_patient_context" and timing.t_patient_context_result is None:
            timing.t_patient_context_result = now

    elif et == "text-start":
        if timing.t_text_start is None:
            timing.t_text_start = now

    elif et == "text-end":
        timing.t_text_end = now


def _summarize(timings: list[RequestTiming]) -> str:
    total = len(timings)
    successes = [t for t in timings if t.ok]
    failures = total - len(successes)
    failure_rate = (failures / total * 100.0) if total else 0.0

    # Failure-class breakdown
    fail_classes: dict[str, int] = {}
    for t in timings:
        if not t.ok and t.failure_reason:
            fail_classes[t.failure_reason] = fail_classes.get(t.failure_reason, 0) + 1

    def _ms(start: float, end: float | None) -> float | None:
        if end is None or start is None:
            return None
        return (end - start) * 1000.0

    phases: dict[str, list[float]] = {
        "ttfb": [],
        "time_to_dispatch": [],
        "parallel_tool_wait": [],
        "text_stream": [],
        "end_to_end": [],
    }

    for t in successes:
        v = _ms(t.t_request_sent, t.t_first_byte)
        if v is not None:
            phases["ttfb"].append(v)

        v = _ms(t.t_request_sent, t.t_first_tool_call_emitted)
        if v is not None:
            phases["time_to_dispatch"].append(v)

        # parallel_tool_wait = max(primekg_done, patient_context_done)
        # measured from first tool-input-available. If neither tool ran (e.g.
        # knowledge-only prompt), skip — not a Pattern B observation.
        if t.t_first_tool_call_emitted is not None:
            ends = [e for e in (t.t_primekg_result, t.t_patient_context_result) if e is not None]
            if ends:
                phases["parallel_tool_wait"].append(
                    (max(ends) - t.t_first_tool_call_emitted) * 1000.0
                )

        v = _ms(t.t_text_start, t.t_text_end)
        if v is not None:
            phases["text_stream"].append(v)

        end = t.t_text_end or t.t_stream_close
        v = _ms(t.t_request_sent, end)
        if v is not None:
            phases["end_to_end"].append(v)

    lines: list[str] = []
    lines.append(
        f"# bench_dual_lookup — n={total}, ok={len(successes)}, failed={failures} ({failure_rate:.1f}%)"
    )
    if fail_classes:
        lines.append("")
        lines.append(
            "Failure classes: " + ", ".join(f"{k}={v}" for k, v in sorted(fail_classes.items()))
        )
    lines.append("")
    lines.append("| phase | n | p50 (ms) | p95 (ms) | p99 (ms) |")
    lines.append("| --- | --- | --- | --- | --- |")
    for name, vals in phases.items():
        n = len(vals)
        if n == 0:
            lines.append(f"| {name} | 0 | - | - | - |")
            continue
        p50 = _percentile(vals, 50)
        p95 = _percentile(vals, 95)
        p99 = _percentile(vals, 99)
        lines.append(f"| {name} | {n} | {p50:.0f} | {p95:.0f} | {p99:.0f} |")

    # Decision-criteria-relevant call-out at the bottom.
    e2e = phases["end_to_end"]
    pw = phases["parallel_tool_wait"]
    if e2e:
        e2e_p95 = _percentile(e2e, 95)
        verdict = ""
        if e2e_p95 < 1500:
            verdict = "ship (p95 < 1500ms)"
        elif e2e_p95 < 3000:
            verdict = "needs precomputed cache (p95 in [1500, 3000])"
        else:
            verdict = "unviable; fall back to Pattern C (p95 > 3000ms)"
        lines.append("")
        lines.append(f"end_to_end p95 = {e2e_p95:.0f} ms -> {verdict}")
    if pw:
        pw_p95 = _percentile(pw, 95)
        lines.append(f"parallel_tool_wait p95 = {pw_p95:.0f} ms (target < 600 ms)")

    return "\n".join(lines)


async def _main_async(
    base_url: str,
    patient_id: str | None,
    concurrency: int,
    n_requests: int,
) -> int:
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(http2=False) as client:

        async def _bound(idx: int) -> RequestTiming:
            async with sem:
                prompt_idx = idx % len(PROMPTS)
                return await _run_one(client, base_url, patient_id, PROMPTS[prompt_idx], prompt_idx)

        tasks = [asyncio.create_task(_bound(i)) for i in range(n_requests)]
        timings = await asyncio.gather(*tasks)

    print(_summarize(timings))
    return 0


def main(*, require_patient_id: bool = True) -> int:
    base_url = os.environ.get("BENCH_BASE_URL", "http://localhost:3000")
    patient_id_env = os.environ.get("BENCH_PATIENT_ID")
    if require_patient_id and not patient_id_env:
        print("ERROR: BENCH_PATIENT_ID is required.", file=sys.stderr)
        return 2
    patient_id = patient_id_env if require_patient_id else None
    try:
        concurrency = int(os.environ.get("BENCH_CONCURRENCY", "1"))
        n_requests = int(os.environ.get("BENCH_N_REQUESTS", "20"))
    except ValueError:
        print("ERROR: BENCH_CONCURRENCY/BENCH_N_REQUESTS must be integers.", file=sys.stderr)
        return 2

    return asyncio.run(_main_async(base_url, patient_id, concurrency, n_requests))


if __name__ == "__main__":
    raise SystemExit(main())
