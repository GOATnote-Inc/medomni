// Translate the route's OpenAI-compatible chat shape (used by the vLLM path)
// into Anthropic Messages API shapes. Pure functions, no network, no key.
//
// Used for the INITIAL incoming conversation (user/assistant text + image).
// Tool turns during the agent loop are appended natively as Anthropic
// MessageParams by the route, so the `tool` handling here is defensive.

import type Anthropic from "@anthropic-ai/sdk";

export interface ChatContentBlock {
  type: string;
  text?: string;
  image_url?: { url?: string };
}

export interface ChatMsg {
  role: "system" | "user" | "assistant" | "tool";
  content: string | ChatContentBlock[] | null;
  tool_calls?: Array<{
    id: string;
    type: "function";
    function: { name: string; arguments: string };
  }>;
  tool_call_id?: string;
}

// OpenAI-style function tool: { type:"function", function:{ name, description, parameters } }
export interface OpenAIFunctionToolLike {
  type?: "function";
  function?: {
    name: string;
    description?: string;
    parameters?: Record<string, unknown>;
  };
}

type Base64MediaType = Anthropic.Base64ImageSource["media_type"];

const ALLOWED_IMAGE_MEDIA: ReadonlySet<string> = new Set([
  "image/jpeg",
  "image/png",
  "image/gif",
  "image/webp",
]);

/**
 * Build an Anthropic image block from an OpenAI-style image_url. Handles both
 * `data:` base64 URLs (file uploads from the browser) and real http(s) URLs.
 */
function imageBlock(url: string): Anthropic.ImageBlockParam | null {
  if (url.startsWith("data:")) {
    const m = /^data:([^;]+);base64,([\s\S]*)$/.exec(url);
    if (!m) return null;
    const media = m[1];
    if (!ALLOWED_IMAGE_MEDIA.has(media)) return null;
    return {
      type: "image",
      source: { type: "base64", media_type: media as Base64MediaType, data: m[2] },
    };
  }
  return { type: "image", source: { type: "url", url } };
}

function toUserContent(
  content: string | ChatContentBlock[] | null,
): string | Anthropic.ContentBlockParam[] {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const blocks: Anthropic.ContentBlockParam[] = [];
  for (const b of content) {
    if (b.type === "text" && typeof b.text === "string") {
      blocks.push({ type: "text", text: b.text });
    } else if (b.type === "image_url" && b.image_url?.url) {
      const img = imageBlock(b.image_url.url);
      if (img) blocks.push(img);
    }
    // audio_url is intentionally dropped — Claude has no native audio input.
  }
  if (blocks.length === 0) return "";
  return blocks;
}

function safeParseInput(argsRaw: string): Record<string, unknown> {
  try {
    const v = JSON.parse(argsRaw || "{}");
    return v && typeof v === "object" ? (v as Record<string, unknown>) : {};
  } catch {
    return {};
  }
}

/**
 * Convert OpenAI-shaped chat messages to Anthropic MessageParams. System
 * messages are skipped (the route passes `system` separately). Consecutive
 * `tool` messages are grouped into a single user turn of tool_result blocks.
 */
export function toAnthropicMessages(chat: ChatMsg[]): Anthropic.MessageParam[] {
  const out: Anthropic.MessageParam[] = [];
  let pendingToolResults: Anthropic.ToolResultBlockParam[] = [];

  const flushToolResults = () => {
    if (pendingToolResults.length > 0) {
      out.push({ role: "user", content: pendingToolResults });
      pendingToolResults = [];
    }
  };

  for (const m of chat) {
    if (m.role === "system") continue;

    if (m.role === "tool") {
      pendingToolResults.push({
        type: "tool_result",
        tool_use_id: m.tool_call_id ?? "",
        content: typeof m.content === "string" ? m.content : JSON.stringify(m.content ?? ""),
      });
      continue;
    }

    flushToolResults();

    if (m.role === "user") {
      const content = toUserContent(m.content);
      // Anthropic rejects empty content; skip blank user turns.
      if (typeof content === "string" && !content.trim()) continue;
      if (Array.isArray(content) && content.length === 0) continue;
      out.push({ role: "user", content });
      continue;
    }

    // assistant
    const blocks: Anthropic.ContentBlockParam[] = [];
    if (typeof m.content === "string" && m.content.trim()) {
      blocks.push({ type: "text", text: m.content });
    } else if (Array.isArray(m.content)) {
      for (const b of m.content) {
        if (b.type === "text" && typeof b.text === "string" && b.text.trim()) {
          blocks.push({ type: "text", text: b.text });
        }
      }
    }
    for (const tc of m.tool_calls ?? []) {
      blocks.push({
        type: "tool_use",
        id: tc.id,
        name: tc.function.name,
        input: safeParseInput(tc.function.arguments),
      });
    }
    if (blocks.length > 0) out.push({ role: "assistant", content: blocks });
  }

  flushToolResults();
  return out;
}

/**
 * Convert OpenAI-style function tools (the route's TOOL_SPEC, plus any
 * MCP-adapted tools) to Anthropic tool definitions.
 */
export function toAnthropicTools(spec: readonly unknown[]): Anthropic.Tool[] {
  const tools: Anthropic.Tool[] = [];
  for (const t of spec as OpenAIFunctionToolLike[]) {
    const f = t.function;
    if (!f?.name) continue;
    tools.push({
      name: f.name,
      description: f.description,
      input_schema: (f.parameters ?? { type: "object", properties: {} }) as Anthropic.Tool.InputSchema,
    });
  }
  return tools;
}
