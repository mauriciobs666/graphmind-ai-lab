// One test group per C1–C14 (docs/plans/salesperson-ui.md §5.3), each
// exercising `resolveErrorAction` directly — no fetch, no React — so a rule
// that regresses to the wrong mechanism (a global 401 handler, a wrong C4
// re-read endpoint, a silently-swallowed unruled response) goes red here
// rather than merely rendering the right thing for the wrong reason (§5.3's
// own warning about C3/C4).
import { describe, expect, it } from 'vitest';
import type { ApiError } from './client';
import { type RereadTarget, type Route, resolveErrorAction } from './dispatch';

function err(status: number, body: Record<string, unknown> | null = null): ApiError {
  return { status, body, method: 'GET', path: '/x', credential: 'none' } as ApiError;
}

describe('C1 — every 401/403 is dispatched per credential, never by one global handler', () => {
  it('the same status code on a participant route and a presenter route produces different actions', () => {
    const participantOutcome = resolveErrorAction('state', err(401, { error: 'invalid_token' }), 'poll');
    const presenterOutcome = resolveErrorAction(
      'presenterParticipants',
      err(401, { error: 'presenter_session_gone' }),
      'poll',
    );
    expect(participantOutcome).toEqual({ kind: 'clearParticipant' });
    expect(presenterOutcome).toEqual({ kind: 'clearPresenter' });
    // The point of C1: these are not the same action. A single app-wide
    // `401 -> rejoin` handler would make them identical.
    expect(participantOutcome).not.toEqual(presenterOutcome);
  });
});

describe('C2 — a presenter-route 401/403 clears only the presenter credential', () => {
  const authenticatedPresenterRoutes: Route[] = ['presenterParticipants', 'presenterResetAll'];

  for (const route of authenticatedPresenterRoutes) {
    it(`${route}: 401 presenter_session_gone -> clearPresenter`, () => {
      expect(resolveErrorAction(route, err(401, { error: 'presenter_session_gone' }), 'user')).toEqual({
        kind: 'clearPresenter',
      });
    });

    it(`${route}: 403 wrong_credential_type -> clearPresenter`, () => {
      expect(resolveErrorAction(route, err(403, { error: 'wrong_credential_type' }), 'user')).toEqual({
        kind: 'clearPresenter',
      });
    });
  }

  it('POST /presenter/session 403 (bad key) is reported in place instead — nothing to clear', () => {
    const outcome = resolveErrorAction('presenterSession', err(403, { error: 'bad_presenter_key' }), 'user');
    expect(outcome).toEqual({ kind: 'presenterKeyRejected' });
    expect(outcome).not.toEqual({ kind: 'clearPresenter' });
  });

  it('a 401 on presenterSession — which the server contract says cannot occur — falls through to unhandled (C13), not clearPresenter', () => {
    // presenterSession mints the presenter credential rather than checking
    // one, so it is special-cased ahead of the generic PRESENTER_ROUTES
    // branch: only its own 403 is a named case, and everything else on this
    // route (a hypothetical 401 included) must stay loud rather than be
    // silently absorbed by the generic authenticated-presenter-route clear.
    const outcome = resolveErrorAction('presenterSession', err(401, { error: 'unexpected' }), 'user');
    expect(outcome).toEqual({ kind: 'unhandled', route: 'presenterSession', status: 401 });
    expect(outcome).not.toEqual({ kind: 'clearPresenter' });
  });
});

describe('C3 — a participant-route 401 clears only the participant credential', () => {
  const participantRoutes: Route[] = [
    'state',
    'messagesRead',
    'messagesPost',
    'catalog',
    'orderAdvance',
    'resetMine',
  ];

  for (const route of participantRoutes) {
    it(`${route}: 401 invalid_token -> clearParticipant`, () => {
      expect(resolveErrorAction(route, err(401, { error: 'invalid_token' }), 'poll')).toEqual({
        kind: 'clearParticipant',
      });
    });
  }

  it('does not clear the presenter credential or return a presenter action', () => {
    const outcome = resolveErrorAction('state', err(401, { error: 'invalid_token' }), 'poll');
    expect(outcome).not.toEqual({ kind: 'clearPresenter' });
  });
});

describe('C4 — after a 504, re-read from the surviving credential\'s own endpoint — per writing route', () => {
  const cases: Array<{ route: Route; via: RereadTarget }> = [
    { route: 'join', via: { endpoint: 'none' } },
    { route: 'messagesPost', via: { endpoint: 'messagesAndState' } },
    { route: 'orderAdvance', via: { endpoint: 'state' } },
    { route: 'resetMine', via: { endpoint: 'state' } },
    { route: 'presenterResetAll', via: { endpoint: 'presenterParticipants' } },
  ];

  const shapes: Array<{ label: string; body: Record<string, unknown> | null }> = [
    { label: 'named token body', body: { error: 'reset_state_unknown' } },
    { label: 'bodyless proxy 504', body: null },
  ];

  for (const { route, via } of cases) {
    for (const { label, body } of shapes) {
      it(`${route}: 504 (${label}) -> reread ${via.endpoint}`, () => {
        expect(resolveErrorAction(route, err(504, body), 'user')).toEqual({ kind: 'reread', via });
      });
    }

    it(`${route}: a browser fetch timeout (isTimeout, status 504, no body) -> reread ${via.endpoint}`, () => {
      const timeoutErr = { status: 504, body: null, method: 'POST', path: '/x', credential: 'none', isTimeout: true } as ApiError;
      expect(resolveErrorAction(route, timeoutErr, 'user')).toEqual({ kind: 'reread', via });
    });

    it(`${route}: a 504 never reports "nothing changed"`, () => {
      const outcome = resolveErrorAction(route, err(504, { error: 'x' }), 'user');
      expect(outcome.kind).not.toBe('nothingChangedRetry');
    });
  }

  it('a reads-only route producing a 504 (should not happen server-side) still falls through to C13, not a fabricated re-read', () => {
    expect(resolveErrorAction('state', err(504, null), 'user')).toEqual({
      kind: 'unhandled',
      route: 'state',
      status: 504,
    });
  });
});

describe('C5 — a 401 from /state after reset-all\'s 504 is evidence the sweep committed, and still routes through C3', () => {
  it('treats it exactly as an ordinary participant 401 (clearParticipant), not as a special case', () => {
    // Simulates: presenter fires reset-all, gets a 504, re-reads
    // /presenter/participants (C4); meanwhile the participant's own poll of
    // /state comes back 401 because the sweep did commit.
    expect(resolveErrorAction('state', err(401, { error: 'invalid_token' }), 'poll')).toEqual({
      kind: 'clearParticipant',
    });
  });
});

describe('C6a — 409 TurnInProgress retains the composer and re-enables on idle', () => {
  it('POST /messages 409 turn_in_progress -> turnInProgressRetain', () => {
    expect(resolveErrorAction('messagesPost', err(409, { error: 'turn_in_progress' }), 'user')).toEqual({
      kind: 'turnInProgressRetain',
    });
  });

  it('is not confused with a 409 on a different route or a different body (dispatches on the body, not the code)', () => {
    const resetAllOnDifferentRoute = resolveErrorAction(
      'resetMine',
      err(409, { error: 'unscoped_participant' }),
      'user',
    );
    expect(resetAllOnDifferentRoute).not.toEqual({ kind: 'turnInProgressRetain' });
  });
});

describe('C6b — 409 unscoped_participant is an alarm, neither success nor busy', () => {
  it('POST /reset 409 unscoped_participant -> unscopedAlarm', () => {
    expect(resolveErrorAction('resetMine', err(409, { error: 'unscoped_participant' }), 'user')).toEqual({
      kind: 'unscopedAlarm',
    });
  });

  it('is not treated as C6a\'s turn-in-progress retention', () => {
    const outcome = resolveErrorAction('resetMine', err(409, { error: 'unscoped_participant' }), 'user');
    expect(outcome).not.toEqual({ kind: 'turnInProgressRetain' });
  });
});

describe('C9 — a 503 means nothing changed; the action keys on who asked, not on the route or the source', () => {
  it('a user-initiated call gets a retry control (scope "user")', () => {
    expect(resolveErrorAction('resetMine', err(503, { error: 'quiesce_timeout' }), 'user')).toEqual({
      kind: 'nothingChangedRetry',
      scope: 'user',
    });
  });

  it('a background poll tick gets a staleness indicator and no control (scope "poll")', () => {
    expect(resolveErrorAction('state', err(503, { error: 'graph_unavailable' }), 'poll')).toEqual({
      kind: 'nothingChangedRetry',
      scope: 'poll',
    });
  });

  it('the SAME route (/state) takes both branches depending on the trigger, not the route', () => {
    const userTriggered = resolveErrorAction('state', err(503, { error: 'graph_unavailable' }), 'user');
    const pollTriggered = resolveErrorAction('state', err(503, { error: 'graph_unavailable' }), 'poll');
    expect(userTriggered).toEqual({ kind: 'nothingChangedRetry', scope: 'user' });
    expect(pollTriggered).toEqual({ kind: 'nothingChangedRetry', scope: 'poll' });
    expect(userTriggered).not.toEqual(pollTriggered);
  });

  it('all four sharing sources collapse to the same one meaning/action (quiesce_timeout, graph_unavailable, graph_read_timeout, demo_not_seeded)', () => {
    const sources = [
      { route: 'resetMine' as const, token: 'quiesce_timeout' },
      { route: 'state' as const, token: 'graph_unavailable' },
      { route: 'messagesRead' as const, token: 'graph_read_timeout' },
      { route: 'join' as const, token: 'demo_not_seeded' },
      { route: 'messagesPost' as const, token: 'demo_not_seeded' },
    ];
    for (const { route, token } of sources) {
      expect(resolveErrorAction(route, err(503, { error: token }), 'user')).toEqual({
        kind: 'nothingChangedRetry',
        scope: 'user',
      });
    }
  });
});

describe('C10 — /order/advance 404/409 are ordinary order outcomes, not auth failures and not alarms', () => {
  it('404 no_current_order -> staleOrderRefresh', () => {
    expect(resolveErrorAction('orderAdvance', err(404, { error: 'no_current_order' }), 'user')).toEqual({
      kind: 'staleOrderRefresh',
    });
  });

  it('409 order_transition_refused -> staleOrderRefresh', () => {
    expect(
      resolveErrorAction('orderAdvance', err(409, { error: 'order_transition_refused' }), 'user'),
    ).toEqual({ kind: 'staleOrderRefresh' });
  });

  it('routes neither through C3 (a stale button must not log the participant out)', () => {
    const outcome = resolveErrorAction('orderAdvance', err(404, { error: 'no_current_order' }), 'user');
    expect(outcome).not.toEqual({ kind: 'clearParticipant' });
  });

  it('routes neither through C6a/C6b (the 409 here is not a turn-progress code)', () => {
    const outcome = resolveErrorAction('orderAdvance', err(409, { error: 'order_transition_refused' }), 'user');
    expect(outcome).not.toEqual({ kind: 'turnInProgressRetain' });
    expect(outcome).not.toEqual({ kind: 'unscopedAlarm' });
  });
});

describe('C11 — 422 dispatches on the field, not the route — all six (route, field) cells', () => {
  const cells: Array<{ route: Route; field: string; audience: 'user' | 'dev' }> = [
    { route: 'join', field: 'displayName', audience: 'user' },
    { route: 'join', field: 'language', audience: 'dev' },
    { route: 'messagesPost', field: 'text', audience: 'user' },
    { route: 'messagesRead', field: 'limit', audience: 'dev' },
    { route: 'orderAdvance', field: 'transition', audience: 'dev' },
    { route: 'presenterSession', field: 'key', audience: 'user' },
  ];

  for (const { route, field, audience } of cells) {
    it(`${route} / field=${field} -> audience=${audience}`, () => {
      expect(resolveErrorAction(route, err(422, { error: 'validation_failed', field }), 'user')).toEqual({
        kind: 'fieldError',
        field,
        audience,
      });
    });
  }

  it('the route held constant, only the field varying, changes the outcome (a route-keyed dispatcher would fail this)', () => {
    const displayNameOutcome = resolveErrorAction(
      'join',
      err(422, { error: 'validation_failed', field: 'displayName' }),
      'user',
    );
    const languageOutcome = resolveErrorAction(
      'join',
      err(422, { error: 'validation_failed', field: 'language' }),
      'user',
    );
    expect(displayNameOutcome.kind).toBe('fieldError');
    expect(languageOutcome.kind).toBe('fieldError');
    expect(displayNameOutcome).not.toEqual(languageOutcome);
  });
});

describe('C13 — anything with no rule fails loudly, naming route and status', () => {
  it('an injected response no rule covers (418 on /state) renders as unhandled, clearing nothing', () => {
    expect(resolveErrorAction('state', err(418, { error: 'teapot' }), 'poll')).toEqual({
      kind: 'unhandled',
      route: 'state',
      status: 418,
    });
  });

  it('a bare 500 with no mapping on a route that carries no credential is unhandled, not silently swallowed', () => {
    expect(resolveErrorAction('health', err(500, null), 'user')).toEqual({
      kind: 'unhandled',
      route: 'health',
      status: 500,
    });
  });

  it('a 401 on a no-credential route (join) has no rule and is unhandled, not misrouted through C3', () => {
    const outcome = resolveErrorAction('join', err(401, { error: 'unexpected' }), 'user');
    expect(outcome).toEqual({ kind: 'unhandled', route: 'join', status: 401 });
    expect(outcome).not.toEqual({ kind: 'clearParticipant' });
  });
});

describe('C14 — 503 turn_not_scheduled on POST /messages is the one 503 that is not "nothing changed"', () => {
  it('POST /messages 503 turn_not_scheduled -> deadTurn', () => {
    expect(
      resolveErrorAction('messagesPost', err(503, { error: 'turn_not_scheduled' }), 'user'),
    ).toEqual({ kind: 'deadTurn' });
  });

  it('does not route through C9 (its action is a retry, which would duplicate the transcript line)', () => {
    const outcome = resolveErrorAction('messagesPost', err(503, { error: 'turn_not_scheduled' }), 'user');
    expect(outcome).not.toEqual({ kind: 'nothingChangedRetry', scope: 'user' });
  });

  it('a different 503 token on the same route (demo_not_seeded) still takes C9\'s path, not C14\'s', () => {
    expect(
      resolveErrorAction('messagesPost', err(503, { error: 'demo_not_seeded' }), 'user'),
    ).toEqual({ kind: 'nothingChangedRetry', scope: 'user' });
  });
});
