"use client";

// Tiny client wrapper around react-markdown so the /skills page can stay
// a server component. Mirrors the typography classes used by
// MarkdownAnswer in `web/app/agent/page.tsx` so a clinician sees the
// skill body in the same visual register as the agent's answers.

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export function SkillMarkdown({ text }: { text: string }) {
  return (
    <div
      className={[
        "leading-relaxed",
        "[&_p]:my-2 [&_ul]:my-2 [&_ol]:my-2 [&_li]:my-1",
        "[&_h1]:text-xl [&_h1]:font-semibold [&_h1]:mt-4 [&_h1]:mb-2",
        "[&_h2]:text-lg [&_h2]:font-semibold [&_h2]:mt-4 [&_h2]:mb-2",
        "[&_h3]:text-base [&_h3]:font-medium [&_h3]:mt-3 [&_h3]:mb-1",
        "[&_strong]:text-slate-900 [&_strong]:font-semibold",
        "[&_em]:italic",
        "[&_code]:px-1 [&_code]:py-0.5 [&_code]:rounded",
        "[&_code]:bg-slate-100 [&_code]:text-[12px] [&_code]:font-mono",
        "[&_pre]:p-3 [&_pre]:rounded [&_pre]:bg-slate-100 [&_pre]:overflow-x-auto",
        "[&_pre]:text-[12px] [&_pre]:leading-relaxed",
        "[&_a]:text-slate-800 [&_a]:underline [&_a]:decoration-slate-400",
        "[&_a]:underline-offset-2 hover:[&_a]:text-slate-900",
        "[&_blockquote]:border-l-2 [&_blockquote]:border-slate-300",
        "[&_blockquote]:pl-3 [&_blockquote]:text-slate-600",
        "[&_table]:my-2 [&_table]:text-sm",
        "[&_th]:text-left [&_th]:font-medium [&_th]:border-b",
        "[&_th]:border-slate-300 [&_th]:px-2 [&_th]:py-1",
        "[&_td]:px-2 [&_td]:py-1 [&_td]:border-b [&_td]:border-slate-100",
        "[&_hr]:my-3 [&_hr]:border-slate-200",
      ].join(" ")}
    >
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
    </div>
  );
}
