// §5.3's client contract, C1–C14, as one pure decision function. Kept apart
// from `./hooks` (the React/TanStack Query wiring) so every rule is testable
// against a plain `ApiError` object — no fetch, no DOM, no React tree — which
// is what makes the mutation-testing requirement in
// docs/plans/salesperson-ui.md §5.1's S12a row (break the rule, watch the
// test go red) actually cheap to run.
//
// **C1 is emergent, not a branch of its own.** "Every 401/403 is dispatched
// per credential, never by one global handler" is what this whole module
// structurally *is*: `resolveErrorAction` keys on `route` (which fixes the
// credential — §5.3's credentials table makes route -> credential a
// function) before it ever looks at `status`, so there is no single
// `status === 401` branch shared by both credentials to regress into one.
import type { ApiError } from './client';

export type Route =
  | 'health'
  | 'join'
  | 'state'
  | 'messagesRead'
  | 'messagesPost'
  | 'catalog'
  | 'orderAdvance'
  | 'resetMine'
  | 'presenterSession'
  | 'presenterParticipants'
  | 'presenterResetAll';

/** Whether a call was fired by the 2 s poll timer or by something the
 * participant/presenter did. §5.3 C9: this is genuinely not derivable from
 * `route` alone ("`/state` takes both branches: the poll and a user-initiated
 * refresh") — a poll-hook tick passes `'poll'`; a C4 recovery re-read or any
 * other call the UI triggers directly passes `'user'`. */
export type Trigger = 'user' | 'poll';

export type RereadTarget =
  | { endpoint: 'state' }
  | { endpoint: 'presenterParticipants' }
  | { endpoint: 'messagesAndState' }
  | { endpoint: 'none' };

export type ErrorAction =
  // C2 — presenter-route 401/403: presenter credential cleared, presenter
  // view returns to key entry. Participant session/view untouched.
  | { kind: 'clearPresenter' }
  // C2's second half — POST /presenter/session's own 403: no credential to
  // clear, already on key entry, report the bad key in place.
  | { kind: 'presenterKeyRejected' }
  // C3 — participant-route 401: participant credential cleared, participant
  // view returns to join. Presenter credential/view untouched.
  | { kind: 'clearParticipant' }
  // C4 — 504 (named token, bodyless proxy, or browser timeout alike): re-read
  // from the surviving credential's own endpoint and report from there.
  | { kind: 'reread'; via: RereadTarget }
  // C6a — 409 TurnInProgress: retain composer text, re-enable on `idle`.
  | { kind: 'turnInProgressRetain' }
  // C6b — 409 unscoped_participant: an alarm, not busy and not success.
  | { kind: 'unscopedAlarm' }
  // C9 — nothing changed; the action splits on who asked, not on the route.
  | { kind: 'nothingChangedRetry'; scope: Trigger }
  // C10 — /order/advance's 404/409 are ordinary stale-button outcomes.
  | { kind: 'staleOrderRefresh' }
  // C11 — 422 dispatches on the field, not the route.
  | { kind: 'fieldError'; field: string; audience: 'user' | 'dev' }
  // C14 — the message posted, no reply is coming, and there is nothing to
  // re-read.
  | { kind: 'deadTurn' }
  // C13 — the guard against a (route, response) no rule above covers.
  | { kind: 'unhandled'; route: Route; status: number };

const PARTICIPANT_ROUTES: ReadonlySet<Route> = new Set([
  'state',
  'messagesRead',
  'messagesPost',
  'catalog',
  'orderAdvance',
  'resetMine',
]);

const PRESENTER_ROUTES: ReadonlySet<Route> = new Set([
  'presenterSession',
  'presenterParticipants',
  'presenterResetAll',
]);

// C11 — "who supplied the value" is the discriminator, not the route. A
// route may bound both kinds (`POST /shop/api/session` bounds `displayName`,
// user-supplied, and `language`, UI-supplied).
const USER_SUPPLIED_FIELDS: ReadonlySet<string> = new Set(['displayName', 'text', 'key']);

// §5.3 C4 — per writing route, the endpoint the *surviving* credential can
// reach. `join` has none: the credential that would read anything was never
// minted. Every non-writing route is absent on purpose — a `504` there has no
// rule and falls through to C13.
const C4_REREAD: Partial<Record<Route, RereadTarget>> = {
  join: { endpoint: 'none' },
  messagesPost: { endpoint: 'messagesAndState' },
  orderAdvance: { endpoint: 'state' },
  resetMine: { endpoint: 'state' },
  presenterResetAll: { endpoint: 'presenterParticipants' },
};

export function resolveErrorAction(route: Route, error: ApiError, trigger: Trigger): ErrorAction {
  const { status, body } = error;
  const token = body?.error;

  // ── C11 — 422 dispatches on the field, ahead of everything else ─────────
  if (status === 422) {
    const field = typeof body?.field === 'string' ? body.field : '';
    return {
      kind: 'fieldError',
      field,
      audience: USER_SUPPLIED_FIELDS.has(field) ? 'user' : 'dev',
    };
  }

  // ── C1/C2/C3 — 401/403, dispatched per credential (i.e. per route) ──────
  if (status === 401 || status === 403) {
    if (route === 'presenterSession') {
      // Special-cased ahead of the generic PRESENTER_ROUTES branch below:
      // presenterSession *mints* the presenter credential rather than
      // checking one, so the server contract says it cannot answer 401 at
      // all (§5.2) — only its own 403 (a wrong key) is a named case. Every
      // other status here, including a hypothetical 401, must fall through
      // to C13 rather than being silently treated as clearPresenter, which
      // has nothing to clear and nowhere useful to navigate.
      if (status === 403) {
        // C2's second half: the key just typed is wrong; there is no
        // credential to clear and no navigation, because the user is
        // already on key entry.
        return { kind: 'presenterKeyRejected' };
      }
      return { kind: 'unhandled', route, status };
    }
    if (PRESENTER_ROUTES.has(route)) {
      return { kind: 'clearPresenter' };
    }
    if (PARTICIPANT_ROUTES.has(route)) {
      // C3 — and, when this fires on `/state` right after a reset-all `504`,
      // C5: the same clear is the correct read of "the sweep committed",
      // not a fresh auth failure to special-case.
      return { kind: 'clearParticipant' };
    }
    // `health` and `join` carry no credential (§5.3) — a 401/403 here has no
    // rule and is C13's.
    return { kind: 'unhandled', route, status };
  }

  // ── C14 — the one 503 that is not "nothing changed", checked before the
  // generic C9 branch below ───────────────────────────────────────────────
  if (status === 503 && route === 'messagesPost' && token === 'turn_not_scheduled') {
    return { kind: 'deadTurn' };
  }

  // ── C9 — 503 means nothing changed; the action keys on who asked ───────
  if (status === 503) {
    return { kind: 'nothingChangedRetry', scope: trigger };
  }

  // ── C4 — 504, any of its three shapes, keyed on status alone ────────────
  if (status === 504) {
    const via = C4_REREAD[route];
    if (via) return { kind: 'reread', via };
    return { kind: 'unhandled', route, status };
  }

  // ── C6a — 409 TurnInProgress on POST /messages ──────────────────────────
  if (status === 409 && route === 'messagesPost' && token === 'turn_in_progress') {
    return { kind: 'turnInProgressRetain' };
  }

  // ── C6b — 409 unscoped_participant on POST /reset ───────────────────────
  if (status === 409 && route === 'resetMine' && token === 'unscoped_participant') {
    return { kind: 'unscopedAlarm' };
  }

  // ── C10 — /order/advance's 404/409 are ordinary stale-button outcomes ──
  if (route === 'orderAdvance' && (status === 404 || status === 409)) {
    return { kind: 'staleOrderRefresh' };
  }

  // ── C13 — anything else is loud, not swallowed ──────────────────────────
  return { kind: 'unhandled', route, status };
}
