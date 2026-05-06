// Public Skill Registry — /4UWHAt/skills
//
// Trust-through-transparency surface for clinicians: every skill module
// the agent can compose into its system prompt is listed here, with name,
// description, the keyword set that triggers it, and the full markdown
// body. No vendor (Med-PaLM, MedGemma, Hippocratic) ships an inspectable
// skill-router surface like this. The agent answers users see at /4UWHAt
// are composed of these modules; this page is the receipt.
//
// Server component — reads the skill markdown at request time via fs and
// passes the parsed manifest down. The selected skill comes from
// `?active=<slug>`. SkillMarkdown is a thin client wrapper around
// react-markdown so the static body can render with proper typography.
//
// basePath is `/4UWHAt`, so this file at `web/app/skills/page.tsx`
// serves at `https://medomni.vercel.app/4UWHAt/skills`. Anonymous-access
// path, same posture as `/4UWHAt`.

import Link from "next/link";
import { DemoBanner } from "@/components/DemoBanner";
import { Footer } from "@/components/Footer";
import { SkillMarkdown } from "@/components/skills/SkillMarkdown";
import {
  loadSkillRegistry,
  type SkillManifest,
} from "@/lib/agent/skill-registry";

export const dynamic = "force-dynamic";

interface PageProps {
  searchParams: Promise<{ active?: string | string[] }>;
}

export default async function SkillsPage({ searchParams }: PageProps) {
  const params = await searchParams;
  const rawActive = Array.isArray(params.active) ? params.active[0] : params.active;
  const skills = loadSkillRegistry();

  // Resolve the active skill. Defaults to the first routable skill (i.e.
  // the user lands on something interesting, not the always-on system
  // prompt). Falls back to the first skill if the slug doesn't match.
  const firstRoutable = skills.find((s) => s.routable) ?? skills[0];
  const active =
    skills.find((s) => s.slug === rawActive) ?? firstRoutable;

  return (
    <>
      <DemoBanner />
      <main className="flex-1 w-full max-w-6xl mx-auto px-6 py-10 flex flex-col gap-8">
        <header className="flex flex-col gap-3">
          <p className="text-xs tracking-widest uppercase text-slate-500 font-medium">
            GOATnote · MedOmni · Skill registry
          </p>
          <h1 className="text-3xl sm:text-4xl font-semibold tracking-tight text-slate-900">
            What this agent knows how to do.
          </h1>
          <p className="max-w-3xl text-base text-slate-600 leading-relaxed">
            MedOmni answers are composed of skill modules. Each one is
            inspectable below. Click a skill to see exactly what guidance
            it loads into the agent&apos;s reasoning and which keywords
            route a user query to it. No vendor ships this view —
            you&apos;re reading the system prompt the model actually sees.
          </p>
          <p className="max-w-3xl text-sm text-slate-500 leading-relaxed">
            The runtime classifier lives in{" "}
            <code className="px-1 py-0.5 rounded bg-slate-100 text-[12px] font-mono">
              web/lib/agent/skills.ts
            </code>
            ; bodies live in{" "}
            <code className="px-1 py-0.5 rounded bg-slate-100 text-[12px] font-mono">
              web/lib/agent/skills/*.md
            </code>
            . Each query against{" "}
            <code className="px-1 py-0.5 rounded bg-slate-100 text-[12px] font-mono">
              /api/agent?profile=v_final
            </code>{" "}
            emits an{" "}
            <code className="px-1 py-0.5 rounded bg-slate-100 text-[12px] font-mono">
              X-Active-Skill
            </code>{" "}
            response header so the front-end can light up which one fired.
          </p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-[18rem_1fr] gap-8">
          {/* Left rail: skill list */}
          <nav
            aria-label="Skill registry"
            className="flex flex-col gap-2 lg:sticky lg:top-6 lg:self-start"
          >
            <p className="text-xs uppercase tracking-wider text-slate-400 font-medium px-2">
              Skills available
            </p>
            <ul className="flex flex-col gap-1">
              {skills.map((s) => (
                <SkillNavItem
                  key={s.id}
                  skill={s}
                  isActive={active.id === s.id}
                />
              ))}
            </ul>
            <p className="text-xs text-slate-400 mt-3 px-2 leading-relaxed">
              Routable skills fire from a keyword classifier on the user&apos;s
              latest message. The system prompt is always loaded.
            </p>
          </nav>

          {/* Right pane: active skill detail */}
          <article className="min-w-0">
            <SkillDetail skill={active} />
          </article>
        </div>
      </main>
      <Footer />
    </>
  );
}

function SkillNavItem({
  skill,
  isActive,
}: {
  skill: SkillManifest;
  isActive: boolean;
}) {
  const cls = [
    "block w-full text-left rounded-md px-3 py-2 transition-colors",
    "border",
    isActive
      ? "bg-slate-900 text-white border-slate-900"
      : "bg-white text-slate-700 border-slate-200 hover:border-slate-400 hover:bg-slate-50",
  ].join(" ");
  // Use anchor links via Next Link so navigation is client-side fast.
  // Active highlighting is keyed off `?active=<slug>` (read by the server
  // component above). Anonymous-access friendly — no JS required to use.
  return (
    <li>
      <Link href={`/skills?active=${skill.slug}`} className={cls}>
        <div className="flex items-center justify-between gap-2">
          <span className="font-medium text-sm">{skill.title}</span>
          {skill.routable ? (
            <span
              className={[
                "text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded",
                isActive
                  ? "bg-white/15 text-white"
                  : "bg-emerald-50 text-emerald-700 border border-emerald-200",
              ].join(" ")}
              title="Router can dispatch to this skill"
            >
              router
            </span>
          ) : (
            <span
              className={[
                "text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded",
                isActive
                  ? "bg-white/15 text-white"
                  : "bg-slate-100 text-slate-600 border border-slate-200",
              ].join(" ")}
              title="Always loaded into the system prompt"
            >
              always-on
            </span>
          )}
        </div>
        <p
          className={[
            "text-xs mt-1 line-clamp-2",
            isActive ? "text-slate-200" : "text-slate-500",
          ].join(" ")}
        >
          {skill.description || "No description."}
        </p>
      </Link>
    </li>
  );
}

function SkillDetail({ skill }: { skill: SkillManifest }) {
  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-col gap-2 border-b border-slate-200 pb-4">
        <div className="flex flex-wrap items-center gap-2">
          <h2 className="text-2xl font-semibold tracking-tight text-slate-900">
            {skill.title}
          </h2>
          {skill.routable ? (
            <span className="text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded bg-emerald-50 text-emerald-700 border border-emerald-200">
              router
            </span>
          ) : (
            <span className="text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded bg-slate-100 text-slate-600 border border-slate-200">
              always-on
            </span>
          )}
          <span
            className="text-[11px] font-mono text-slate-500"
            title="Source file under web/lib/agent/skills/"
          >
            {skill.fileName}
          </span>
        </div>
        {skill.description && (
          <p className="text-base text-slate-600 leading-relaxed">
            {skill.description}
          </p>
        )}
        {skill.routable && skill.triggers.length > 0 && (
          <div className="flex flex-col gap-1 mt-1">
            <p className="text-xs uppercase tracking-wider text-slate-400 font-medium">
              Activation keywords
            </p>
            <ul className="flex flex-wrap gap-1.5">
              {skill.triggers.map((kw) => (
                <li
                  key={kw}
                  className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-100 text-slate-700 border border-slate-200"
                >
                  {kw}
                </li>
              ))}
            </ul>
            <p className="text-xs text-slate-500 mt-1 leading-relaxed">
              When the user&apos;s latest message contains any of these
              substrings (case-insensitive), the router splices this
              skill&apos;s body into the system prompt before dispatching
              to the model.
            </p>
          </div>
        )}
        {!skill.routable && (
          <p className="text-xs text-slate-500 mt-1 leading-relaxed">
            This module is always loaded into the system prompt under the{" "}
            <code className="px-1 py-0.5 rounded bg-slate-100 text-[11px] font-mono">
              ?profile=v_final
            </code>{" "}
            inference profile, regardless of the user&apos;s query.
          </p>
        )}
      </div>

      <section
        className="bg-white border border-slate-200 rounded-lg p-5 sm:p-6"
        aria-label="Skill markdown body"
      >
        <SkillMarkdown text={skill.markdown} />
      </section>

      <p className="text-xs text-slate-500 leading-relaxed">
        Source on disk:{" "}
        <code className="px-1 py-0.5 rounded bg-slate-100 text-[11px] font-mono">
          web/lib/agent/skills/{skill.fileName}
        </code>{" "}
        in{" "}
        <a
          href="https://github.com/GOATnote-Inc/medomni"
          className="underline decoration-slate-400 underline-offset-2 hover:text-slate-700"
          target="_blank"
          rel="noreferrer"
        >
          GOATnote-Inc/medomni
        </a>
        . Skills ship at markdown-PR cadence, not training-cycle cadence.
      </p>
    </div>
  );
}
