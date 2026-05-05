// 4UWHAt — SessionProvider
// React context for the patient-specific Records OS surface. Holds the
// currently selected `patientId` (FHIR Patient resource id) and the active
// `persona` (physician / nurse / family / patient). Both are persisted to
// localStorage so a hard refresh keeps the user on the same patient under
// the same persona — no re-pick needed.
//
// SSR-safe: window/localStorage reads happen in a useEffect, so the
// initial render is identical on server and first client paint. After
// hydration the effect rehydrates from storage.
//
// Consumed by `usePatientId` and `usePersona` hooks. The "Ask your record"
// command bar reads patientId via the hook and forwards it as the
// top-level `patientId` body field on POST /api/agent (the agent route
// handler then passes it to get_patient_context).

"use client";

import {
  createContext,
  useCallback,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

export type Persona = "physician" | "nurse" | "family" | "patient";

const PERSONAS: readonly Persona[] = [
  "physician",
  "nurse",
  "family",
  "patient",
];

export const PATIENT_ID_STORAGE_KEY = "medomni:patientId";
export const PERSONA_STORAGE_KEY = "medomni:persona";

export const DEFAULT_PERSONA: Persona = "patient";

export interface SessionContextValue {
  patientId: string | null;
  persona: Persona;
  setPatientId: (id: string | null) => void;
  setPersona: (p: Persona) => void;
}

// Sentinel that flags "no provider mounted" — hooks distinguish this from
// a legitimately-default context to throw in dev / fall back in prod.
export const SESSION_CONTEXT_MISSING: SessionContextValue = Object.freeze({
  patientId: null,
  persona: DEFAULT_PERSONA,
  setPatientId: () => {
    /* no-op when used outside a provider */
  },
  setPersona: () => {
    /* no-op when used outside a provider */
  },
});

export const SessionContext = createContext<SessionContextValue>(
  SESSION_CONTEXT_MISSING,
);

function isPersona(v: unknown): v is Persona {
  return typeof v === "string" && (PERSONAS as readonly string[]).includes(v);
}

function safeReadLocalStorage(key: string): string | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage.getItem(key);
  } catch {
    // localStorage can throw in private mode / disabled-storage browsers.
    return null;
  }
}

function safeWriteLocalStorage(key: string, value: string | null): void {
  if (typeof window === "undefined") return;
  try {
    if (value === null) {
      window.localStorage.removeItem(key);
    } else {
      window.localStorage.setItem(key, value);
    }
  } catch {
    /* swallow; persistence is best-effort */
  }
}

interface SessionProviderProps {
  children: ReactNode;
  /**
   * Optional initial patientId override. Useful for tests / Storybook;
   * production code should leave this undefined and let localStorage
   * rehydration drive the value.
   */
  initialPatientId?: string | null;
  /**
   * Optional initial persona override. Defaults to DEFAULT_PERSONA.
   */
  initialPersona?: Persona;
}

export function SessionProvider({
  children,
  initialPatientId = null,
  initialPersona = DEFAULT_PERSONA,
}: SessionProviderProps) {
  // SSR + first client paint render with the initial values; rehydration
  // happens in the useEffect below after mount.
  const [patientId, setPatientIdState] = useState<string | null>(
    initialPatientId,
  );
  const [persona, setPersonaState] = useState<Persona>(initialPersona);

  // One-shot rehydration from localStorage on mount.
  useEffect(() => {
    const storedPatientId = safeReadLocalStorage(PATIENT_ID_STORAGE_KEY);
    if (storedPatientId !== null && storedPatientId !== "") {
      setPatientIdState(storedPatientId);
    }
    const storedPersona = safeReadLocalStorage(PERSONA_STORAGE_KEY);
    if (isPersona(storedPersona)) {
      setPersonaState(storedPersona);
    }
    // We deliberately do not list deps — this effect runs once on mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const setPatientId = useCallback((id: string | null) => {
    setPatientIdState(id);
    safeWriteLocalStorage(PATIENT_ID_STORAGE_KEY, id && id.trim() ? id : null);
  }, []);

  const setPersona = useCallback((p: Persona) => {
    if (!isPersona(p)) return;
    setPersonaState(p);
    safeWriteLocalStorage(PERSONA_STORAGE_KEY, p);
  }, []);

  const value = useMemo<SessionContextValue>(
    () => ({ patientId, persona, setPatientId, setPersona }),
    [patientId, persona, setPatientId, setPersona],
  );

  return (
    <SessionContext.Provider value={value}>{children}</SessionContext.Provider>
  );
}
