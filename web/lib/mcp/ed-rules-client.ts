// HealthCraft ED Decision Rules MCP client.
//
// Speaks the Streamable-HTTP subset of MCP that
// `healthcraft/src/healthcraft/agents_assemble/streamable_http_server.py`
// implements: POST JSON-RPC 2.0 to the configured URL. No SSE on the
// client side (the server returns plain JSON for tools/list and
// tools/call). No SDK dependency — built-in fetch + AbortController.
//
// Cached lookups live in module scope and rebuild on cold start. That's
// safe because the server advertises `tools/listChanged: false`.
//
// Failure mode is FAIL-OPEN: every method returns a typed Result; the
// /api/agent caller folds errors into the unified `runTool` ToolResult
// envelope (`{ error: "..." }`) so the agent keeps streaming with the
// remaining hand-rolled tools.

const DEFAULT_URL = "https://mcp.thegoatnote.com/mcp";
const DEFAULT_TIMEOUT_MS = 10_000;
const TOOLS_LIST_TIMEOUT_MS = 5_000;
const INITIALIZE_TIMEOUT_MS = 5_000;

const PROTOCOL_VERSION = "2025-03-26";
const CLIENT_INFO = { name: "medomni-agent", version: "0.1.0" } as const;

export interface McpToolSchema {
  name: string;
  description: string;
  inputSchema: Record<string, unknown>;
}

export interface McpServerInfo {
  protocolVersion: string;
  serverName: string;
  serverVersion: string;
  capabilities: Record<string, unknown>;
}

export type McpResult<T> =
  | { ok: true; data: T }
  | { ok: false; error: string };

// -------------------------------------------------------------------------
// Wire helpers
// -------------------------------------------------------------------------

interface JsonRpcResponse<T> {
  jsonrpc: "2.0";
  id: number;
  result?: T;
  error?: { code: number; message: string; data?: unknown };
}

let nextId = 1;

async function rpc<T>(
  url: string,
  method: string,
  params: Record<string, unknown>,
  timeoutMs: number,
  fhirHeaders?: Record<string, string>,
): Promise<McpResult<T>> {
  const id = nextId++;
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const resp = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json, text/event-stream",
        ...(fhirHeaders ?? {}),
      },
      body: JSON.stringify({ jsonrpc: "2.0", id, method, params }),
      signal: ctrl.signal,
    });
    if (!resp.ok) {
      return { ok: false, error: `MCP ${resp.status}: ${await resp.text().catch(() => "")}` };
    }
    const body = (await resp.json()) as JsonRpcResponse<T>;
    if (body.error) {
      return { ok: false, error: `MCP rpc error ${body.error.code}: ${body.error.message}` };
    }
    if (body.result === undefined) {
      return { ok: false, error: "MCP rpc returned no result" };
    }
    return { ok: true, data: body.result };
  } catch (e) {
    const msg = (e as Error).message ?? "fetch failed";
    return { ok: false, error: `MCP rpc threw: ${msg}` };
  } finally {
    clearTimeout(timer);
  }
}

// -------------------------------------------------------------------------
// Caching client
// -------------------------------------------------------------------------

interface InitializeResult {
  protocolVersion: string;
  capabilities: Record<string, unknown>;
  serverInfo: { name: string; version: string };
}

interface ToolsListResult {
  tools: McpToolSchema[];
}

interface ToolsCallResult {
  content: Array<{ type: string; text?: string }>;
  structuredContent?: unknown;
  isError?: boolean;
}

let serverInfoCache: Promise<McpResult<McpServerInfo>> | null = null;
let toolsListCache: Promise<McpResult<McpToolSchema[]>> | null = null;

function getUrl(): string {
  return process.env.MCP_ED_RULES_URL?.trim() || DEFAULT_URL;
}

export async function initializeServer(): Promise<McpResult<McpServerInfo>> {
  if (serverInfoCache) return serverInfoCache;
  serverInfoCache = (async (): Promise<McpResult<McpServerInfo>> => {
    const r = await rpc<InitializeResult>(
      getUrl(),
      "initialize",
      {
        protocolVersion: PROTOCOL_VERSION,
        capabilities: {},
        clientInfo: CLIENT_INFO,
      },
      INITIALIZE_TIMEOUT_MS,
    );
    if (!r.ok) return { ok: false, error: r.error };
    return {
      ok: true,
      data: {
        protocolVersion: r.data.protocolVersion,
        serverName: r.data.serverInfo.name,
        serverVersion: r.data.serverInfo.version,
        capabilities: r.data.capabilities,
      },
    };
  })();
  return serverInfoCache;
}

export async function listMcpTools(): Promise<McpResult<McpToolSchema[]>> {
  if (toolsListCache) return toolsListCache;
  toolsListCache = (async (): Promise<McpResult<McpToolSchema[]>> => {
    const init = await initializeServer();
    if (!init.ok) return { ok: false, error: init.error };
    const r = await rpc<ToolsListResult>(
      getUrl(),
      "tools/list",
      {},
      TOOLS_LIST_TIMEOUT_MS,
    );
    if (!r.ok) return { ok: false, error: r.error };
    return { ok: true, data: r.data.tools };
  })();
  return toolsListCache;
}

export interface FhirContext {
  fhirServerUrl?: string;
  fhirAccessToken?: string;
  patientId?: string;
}

function fhirHeaders(ctx?: FhirContext): Record<string, string> | undefined {
  if (!ctx) return undefined;
  const h: Record<string, string> = {};
  if (ctx.fhirServerUrl) h["X-FHIR-Server-URL"] = ctx.fhirServerUrl;
  if (ctx.fhirAccessToken) h["X-FHIR-Access-Token"] = ctx.fhirAccessToken;
  if (ctx.patientId) h["X-Patient-ID"] = ctx.patientId;
  return Object.keys(h).length > 0 ? h : undefined;
}

export async function callMcpTool(
  name: string,
  args: Record<string, unknown>,
  ctx?: FhirContext,
): Promise<McpResult<unknown>> {
  const r = await rpc<ToolsCallResult>(
    getUrl(),
    "tools/call",
    { name, arguments: args },
    DEFAULT_TIMEOUT_MS,
    fhirHeaders(ctx),
  );
  if (!r.ok) return r;
  if (r.data.isError) {
    const errText = r.data.content?.[0]?.text ?? "unknown MCP tool error";
    return { ok: false, error: errText };
  }
  // Prefer structuredContent if present (the SHARP envelope + data).
  // Fall back to parsed text content for older transports.
  if (r.data.structuredContent !== undefined) {
    return { ok: true, data: r.data.structuredContent };
  }
  const text = r.data.content?.find((c) => c.type === "text")?.text;
  if (!text) {
    return { ok: false, error: "MCP tool returned no content" };
  }
  try {
    return { ok: true, data: JSON.parse(text) };
  } catch {
    return { ok: true, data: { _raw: text } };
  }
}

// Reset module-scope caches. Test-only; not called from production paths.
export function _resetCachesForTest(): void {
  serverInfoCache = null;
  toolsListCache = null;
}
