// usePersona — thin hook over SessionContext.
// Returns [persona, setPersona]. If the consumer is rendered outside a
// SessionProvider, throws a descriptive error in development; degrades to
// [DEFAULT_PERSONA, no-op] in production so a missing provider doesn't
// take the page down.

"use client";

import { useContext } from "react";
import {
  DEFAULT_PERSONA,
  SessionContext,
  SESSION_CONTEXT_MISSING,
  type Persona,
} from "@/components/4uwhat/SessionProvider";

type PersonaSetter = (p: Persona) => void;

export function usePersona(): [Persona, PersonaSetter] {
  const ctx = useContext(SessionContext);
  if (ctx === SESSION_CONTEXT_MISSING) {
    if (process.env.NODE_ENV !== "production") {
      throw new Error(
        "usePersona() must be used inside a <SessionProvider>. " +
          "Wrap your app subtree (typically in app/layout.tsx for the " +
          "/4UWHAt routes) with <SessionProvider>.",
      );
    }
    return [DEFAULT_PERSONA, () => {}];
  }
  return [ctx.persona, ctx.setPersona];
}
