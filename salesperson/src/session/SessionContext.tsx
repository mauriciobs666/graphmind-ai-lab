// The one place the two credentials live as React state. `useSession()` is
// one of S12a's named exports (docs/plans/salesperson-ui.md §5.1's S12a row)
// — every other module in this tree reads/writes credentials through it
// rather than touching `./storage` directly, so there is exactly one path
// that can drift from localStorage.
import {
  type ReactNode,
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
} from 'react';
import {
  clearParticipantSession,
  clearPresenterSession,
  loadParticipantSession,
  loadPresenterSession,
  saveParticipantSession,
  savePresenterSession,
} from './storage';
import type { ParticipantSession, PresenterSession } from './types';

interface SessionContextValue {
  participant: ParticipantSession | null;
  presenter: PresenterSession | null;
  setParticipant: (session: ParticipantSession) => void;
  /** C3 — clears only the participant credential. */
  clearParticipant: () => void;
  setPresenter: (session: PresenterSession) => void;
  /** C2 — clears only the presenter credential. */
  clearPresenter: () => void;
  /** C7 — after the participant's own reset, the credential survives and the
   * join screen renders the language step (previous language pre-selected)
   * instead of the full join form. `null` means "no pending language step" —
   * ephemeral UI state, never persisted (a reload with a live participant
   * session goes straight back to the ordinary chat/join render). */
  pendingLanguageStep: string | null;
  setPendingLanguageStep: (language: string | null) => void;
  /** §4.12 (v1.37) — the join response's one-shot `welcome` line (§5.2 *The
   * join greeting*). Mirrors `pendingLanguageStep` exactly: plain React
   * state, never persisted (`storage.ts` never sees it, `ParticipantSession`
   * never carries it), so a reload always starts this `null`. Set once by
   * `api/hooks.ts`'s `useJoin()` `onSuccess`; cleared once, by `ChatView.tsx`
   * after it has rendered the line — the only clearing site. */
  welcomeMessage: string | null;
  setWelcomeMessage: (message: string | null) => void;
}

const SessionContext = createContext<SessionContextValue | null>(null);

export function SessionProvider({ children }: { children: ReactNode }) {
  const [participant, setParticipantState] = useState<ParticipantSession | null>(
    () => loadParticipantSession(),
  );
  const [presenter, setPresenterState] = useState<PresenterSession | null>(
    () => loadPresenterSession(),
  );
  const [pendingLanguageStep, setPendingLanguageStep] = useState<string | null>(null);
  const [welcomeMessage, setWelcomeMessage] = useState<string | null>(null);

  const setParticipant = useCallback((session: ParticipantSession) => {
    saveParticipantSession(session);
    setParticipantState(session);
  }, []);

  const clearParticipant = useCallback(() => {
    clearParticipantSession();
    setParticipantState(null);
  }, []);

  const setPresenter = useCallback((session: PresenterSession) => {
    savePresenterSession(session);
    setPresenterState(session);
  }, []);

  const clearPresenter = useCallback(() => {
    clearPresenterSession();
    setPresenterState(null);
  }, []);

  const value = useMemo<SessionContextValue>(
    () => ({
      participant,
      presenter,
      setParticipant,
      clearParticipant,
      setPresenter,
      clearPresenter,
      pendingLanguageStep,
      setPendingLanguageStep,
      welcomeMessage,
      setWelcomeMessage,
    }),
    [
      participant,
      presenter,
      setParticipant,
      clearParticipant,
      setPresenter,
      clearPresenter,
      pendingLanguageStep,
      welcomeMessage,
    ],
  );

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

export function useSession(): SessionContextValue {
  const ctx = useContext(SessionContext);
  if (!ctx) {
    throw new Error('useSession must be used within a SessionProvider');
  }
  return ctx;
}
