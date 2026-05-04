// usePatientId — thin hook over SessionContext.
// Returns [patientId, setPatientId]. If the consumer is rendered outside a
// SessionProvider, throws a descriptive error in development; degrades to
// [null, no-op] in production so a missing provider doesn't take the page
// down.

"use client";

import { useContext } from "react";
import {
  SessionContext,
  SESSION_CONTEXT_MISSING,
} from "@/components/4uwhat/SessionProvider";

type PatientIdSetter = (id: string | null) => void;

export function usePatientId(): [string | null, PatientIdSetter] {
  const ctx = useContext(SessionContext);
  if (ctx === SESSION_CONTEXT_MISSING) {
    if (process.env.NODE_ENV !== "production") {
      throw new Error(
        "usePatientId() must be used inside a <SessionProvider>. " +
          "Wrap your app subtree (typically in app/layout.tsx for the " +
          "/4UWHAt routes) with <SessionProvider>.",
      );
    }
    return [null, () => {}];
  }
  return [ctx.patientId, ctx.setPatientId];
}
