// 4UWHAt — Receipts (per-turn audit trail) storage adapter.
//
// Persists a tight summary of every assistant turn to localStorage so the
// /receipts page can render a long-form audit log without depending on
// any new server-side telemetry. The /api/agent route is unchanged; we
// surface what useChat already produces in `messages[]` (text parts +
// tool-invocation parts) plus the same verification numbers the in-line
// badge under each assistant turn computes.
//
// SSR-safe: every read/write guards `typeof window !== "undefined"` so a
// server-render component can import this module without crashing.

export interface ReceiptToolCall {
  /** Tool name as exposed by the agent route's TOOL_SPEC, e.g. "pubmed_search". */
  name: string;
  /** Best-effort argument payload from the streamed tool-invocation part. */
  args: unknown;
}

export interface ReceiptVerification {
  /** Number of distinct tool invocations on this turn. */
  toolsCalled: number;
  /**
   * Spec-monitor checks passed. Today this matches the in-line badge's
   * static demo set; once the production runtime monitor lands, the
   * shape stays the same and the numbers come from there.
   */
  checksPassed: number;
  /** Spec-monitor checks total. */
  checksTotal: number;
  /** Model id reported by the agent route (e.g. "nemotron"). */
  model: string;
}

export interface Receipt {
  /** Stable id (crypto.randomUUID()). */
  id: string;
  /** Turn timestamp, ms since epoch. */
  timestamp: number;
  /** patientId at request time, or null if none was selected. */
  patientId: string | null;
  /** Persona at request time (physician/nurse/family/patient), or null. */
  persona: string | null;
  /** Full user prompt text. */
  prompt: string;
  /** Full assistant response text (joined across text parts). */
  response: string;
  /** Tool calls made on this turn. */
  toolCalls: ReceiptToolCall[];
  /** Verification numbers (mirrors the in-line badge). */
  verification: ReceiptVerification;
  /** Best-effort latency, ms. Null if not computable. */
  latencyMs: number | null;
}

export const RECEIPTS_STORAGE_KEY = "medomni:receipts:v1";

/** Cap so localStorage never balloons past ~a few hundred KB. */
export const RECEIPTS_MAX = 100;

function isBrowser(): boolean {
  return typeof window !== "undefined";
}

function isReceipt(v: unknown): v is Receipt {
  if (!v || typeof v !== "object") return false;
  const r = v as Record<string, unknown>;
  return (
    typeof r.id === "string" &&
    typeof r.timestamp === "number" &&
    (r.patientId === null || typeof r.patientId === "string") &&
    (r.persona === null || typeof r.persona === "string") &&
    typeof r.prompt === "string" &&
    typeof r.response === "string" &&
    Array.isArray(r.toolCalls) &&
    typeof r.verification === "object" &&
    r.verification !== null &&
    (r.latencyMs === null || typeof r.latencyMs === "number")
  );
}

export function loadReceipts(): Receipt[] {
  if (!isBrowser()) return [];
  try {
    const raw = window.localStorage.getItem(RECEIPTS_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isReceipt);
  } catch {
    return [];
  }
}

export function saveReceipt(r: Receipt): void {
  if (!isBrowser()) return;
  try {
    const existing = loadReceipts();
    const next = [...existing, r].slice(-RECEIPTS_MAX);
    window.localStorage.setItem(RECEIPTS_STORAGE_KEY, JSON.stringify(next));
  } catch {
    /* swallow; persistence is best-effort */
  }
}

export function clearReceipts(): void {
  if (!isBrowser()) return;
  try {
    window.localStorage.removeItem(RECEIPTS_STORAGE_KEY);
  } catch {
    /* ignore */
  }
}

function formatTimestamp(ms: number): string {
  try {
    return new Date(ms).toISOString();
  } catch {
    return String(ms);
  }
}

/**
 * Render the full receipt list as a fenced markdown summary. Format is
 * stable enough that a reviewer can paste it into a ticket / email and
 * read every turn without opening the page.
 */
export function exportReceiptsAsMarkdown(rs: Receipt[]): string {
  const date = new Date().toISOString().slice(0, 10);
  const header = `# Audit receipts (medomni · ${date})\n\n`;
  if (rs.length === 0) {
    return header + "_No receipts logged._\n";
  }
  const body = rs
    .map((r) => {
      const lines: string[] = [];
      lines.push(`## ${formatTimestamp(r.timestamp)}`);
      lines.push("");
      lines.push(
        `- patient: \`${r.patientId ?? "(none)"}\` · persona: \`${r.persona ?? "(none)"}\` · model: \`${r.verification.model}\`${
          r.latencyMs != null ? ` · latency: ${r.latencyMs} ms` : ""
        }`,
      );
      lines.push("");
      lines.push("### Prompt");
      lines.push("");
      lines.push("```");
      lines.push(r.prompt || "(empty)");
      lines.push("```");
      lines.push("");
      lines.push(`### Tool calls (${r.toolCalls.length})`);
      lines.push("");
      if (r.toolCalls.length === 0) {
        lines.push("_(none)_");
      } else {
        for (const tc of r.toolCalls) {
          lines.push(`- **${tc.name}**`);
          lines.push("  ```json");
          let argsText: string;
          try {
            argsText = JSON.stringify(tc.args, null, 2);
          } catch {
            argsText = String(tc.args);
          }
          for (const ln of argsText.split("\n")) {
            lines.push(`  ${ln}`);
          }
          lines.push("  ```");
        }
      }
      lines.push("");
      lines.push("### Response");
      lines.push("");
      lines.push("```");
      lines.push(r.response || "(empty)");
      lines.push("```");
      lines.push("");
      lines.push("### Verification");
      lines.push("");
      lines.push(
        `- tools called: ${r.verification.toolsCalled}`,
      );
      lines.push(
        `- spec-monitor checks: ${r.verification.checksPassed}/${r.verification.checksTotal} passed`,
      );
      lines.push(`- model: ${r.verification.model}`);
      lines.push("");
      return lines.join("\n");
    })
    .join("\n---\n\n");
  return header + body + "\n";
}
