// Convert HealthCraft MCP tool schemas into the OpenAI-style function spec
// that medomni's /api/agent route already emits to vllm. The MCP schemas
// are already JSON Schema (modelcontextprotocol convention) so this is
// almost a passthrough — we just rewrap and prefix the tool name so the
// agent's runTool dispatcher can identify MCP-routed calls without an
// allowlist.

import type { McpToolSchema } from "@/lib/mcp/ed-rules-client";

// Tools coming from the HealthCraft ED Decision Rules MCP are prefixed
// with this string before being shown to the model. Two reasons:
//   1. Namespacing — `applyDecisionRule` is a more recognizable name than
//      `ed_applyDecisionRule`, but inside medomni we also have in-process
//      tools like `clinical_calculate`, so a fixed prefix prevents
//      accidental collisions when HealthCraft adds new tools later.
//   2. Dispatch — runTool() can prefix-route by `mcp_` instead of an
//      explicit name allowlist.
export const MCP_TOOL_PREFIX = "mcp_";

export interface OpenAIFunctionTool {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
}

export function adaptMcpToolToOpenAI(t: McpToolSchema): OpenAIFunctionTool {
  return {
    type: "function",
    function: {
      name: MCP_TOOL_PREFIX + t.name,
      description: t.description,
      parameters: t.inputSchema,
    },
  };
}

export function adaptMcpToolList(tools: McpToolSchema[]): OpenAIFunctionTool[] {
  return tools.map(adaptMcpToolToOpenAI);
}

// Reverse: strip the prefix when dispatching a model-emitted tool name
// back to the MCP server. Returns null if the name isn't MCP-routed.
export function stripMcpPrefix(toolName: string): string | null {
  if (!toolName.startsWith(MCP_TOOL_PREFIX)) return null;
  return toolName.slice(MCP_TOOL_PREFIX.length);
}
