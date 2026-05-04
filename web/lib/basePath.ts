// Mirror of `basePath` in next.config.ts. Single source of truth for
// client code that constructs absolute URLs against the page origin.
//
// Why this exists: Next.js auto-prefixes basePath onto its OWN
// primitives (next/link href, next/image src, router.push, useChat
// when called from inside a Next.js fetch wrapper). It does NOT
// prefix raw browser primitives — `fetch("/api/..." )`,
// `new URL("/asset")`, `audioContext.audioWorklet.addModule("/...")`,
// `<audio src="/...">`, `<video src="/...">`, etc. Those resolve
// against the document origin and bypass basePath entirely. Behind
// the v0-goat-note-landing-page-3c reverse-proxy at
// www.thegoatnote.com/4UWHAt that means the request lands at the
// origin root, falls outside the `/4UWHAt(/*) → medomni` rewrite,
// and gets v0's static 404. The page looks fine; the feature
// silently breaks. Three confirmed instances 2026-05-04:
// AudioWorklet (worklet 404 → "Unable to load a worklet's module"),
// useChat /api/agent (404 → no assistant reply), Composer
// /api/ask (404 → request fails silently).
//
// Use BASE_PATH whenever you need an absolute URL from a non-Next
// browser primitive. Keep this in lockstep with next.config.ts
// `basePath`.
export const BASE_PATH = "/4UWHAt";
