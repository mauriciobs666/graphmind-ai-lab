// Credential persistence — §5.3's "Storage medium" decision: `localStorage`
// for both keys (restart/tab survival for the participant's cart and order;
// one browser, two tabs, no re-authentication for the presenter). Separate
// keys are what let clearing one credential leave the other alone (C2/C3);
// the medium itself is not a C3 requirement, see §5.3.
import type { ParticipantSession, PresenterSession } from './types';

export const SESSION_STORAGE_KEYS = {
  participant: 'salesperson.participant',
  presenter: 'salesperson.presenter',
} as const;

// `localStorage` access can throw (private-mode Safari with cookies off, a
// full quota) — a session that fails to persist is a degraded experience,
// never a crash, so every access is swallowed here and nowhere else in this
// module re-throws.
function readJson<T>(key: string): T | null {
  try {
    const raw = window.localStorage.getItem(key);
    if (!raw) return null;
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

function writeJson(key: string, value: unknown): void {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Session simply does not persist across reloads.
  }
}

function removeKey(key: string): void {
  try {
    window.localStorage.removeItem(key);
  } catch {
    // Nothing to clean up if storage was never reachable.
  }
}

export function loadParticipantSession(): ParticipantSession | null {
  return readJson<ParticipantSession>(SESSION_STORAGE_KEYS.participant);
}

export function saveParticipantSession(session: ParticipantSession): void {
  writeJson(SESSION_STORAGE_KEYS.participant, session);
}

export function clearParticipantSession(): void {
  removeKey(SESSION_STORAGE_KEYS.participant);
}

export function loadPresenterSession(): PresenterSession | null {
  return readJson<PresenterSession>(SESSION_STORAGE_KEYS.presenter);
}

export function savePresenterSession(session: PresenterSession): void {
  writeJson(SESSION_STORAGE_KEYS.presenter, session);
}

export function clearPresenterSession(): void {
  removeKey(SESSION_STORAGE_KEYS.presenter);
}

// The two credential headers (§5.3's credentials table).
export function participantAuthHeader(session: ParticipantSession): string {
  return `Bearer ${session.participantId}.${session.token}`;
}

export function presenterAuthHeader(session: PresenterSession): string {
  return `Bearer presenter.${session.token}`;
}
