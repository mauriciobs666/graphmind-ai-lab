// AC-5's participant reset control, mounted the way it actually is in the
// running app as of v1.36/§4.11: `LayoutShell` is the router's own top-level
// pathless layout route (`element: <LayoutShell />, children: [...]`), with
// `QueryClientProvider > SessionProvider > RouterProvider` wrapping it from
// outside (App.tsx's composition — see `layout/Shell.tsx`'s top comment).
// This is what lets the third test below assert the reset lands the client
// on the language step with the previous language pre-selected by reading
// `SessionContext`'s own rendered state (via `Probe`) rather than inspecting
// the `fetch` call. `ResetControl` now calls `api/hooks.ts`'s
// `useResetMine()` directly (no more hand-rolled dispatch or session-bridge
// splice) — S12a's own `hooks.test.tsx` (`C3`/`C4`/`C7 (hook level)`)
// already covers that hook's own fetch-level contract; this file covers
// this component's own wiring: which phase/copy each `.action.kind` renders,
// and that a `clearParticipant` action closes the sheet the same way success
// does.
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { createMemoryRouter, RouterProvider } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { LayoutShell } from '../../layout/Shell';
import { SessionProvider, useSession } from '../../session/SessionContext';
import { SESSION_STORAGE_KEYS, saveParticipantSession } from '../../session/storage';

function Probe() {
  const { pendingLanguageStep, participant } = useSession();
  return (
    <div>
      <div data-testid="pending-language">{pendingLanguageStep ?? 'none'}</div>
      <div data-testid="participant">{participant ? 'present' : 'absent'}</div>
    </div>
  );
}

function jsonResponse(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    text: async () => JSON.stringify(body),
  } as Response;
}

function renderApp() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const router = createMemoryRouter(
    [
      {
        element: <LayoutShell />,
        children: [{ path: '/', element: <Probe /> }],
      },
    ],
    { initialEntries: ['/'] },
  );
  return render(
    <QueryClientProvider client={client}>
      <SessionProvider>
        <RouterProvider router={router} />
      </SessionProvider>
    </QueryClientProvider>,
  );
}

async function openProfileSheet() {
  const user = userEvent.setup();
  await user.click(screen.getByRole('button', { name: 'Profile' }));
  return user;
}

beforeEach(() => {
  window.localStorage.clear();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('ResetControl', () => {
  it('shows a join prompt instead of a reset control when no participant session exists', async () => {
    renderApp();
    await openProfileSheet();
    expect(screen.getByText(/join the store to manage your session/i)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Reset my session' })).not.toBeInTheDocument();
  });

  it('requires a confirm step, and Cancel backs out without calling the API', async () => {
    saveParticipantSession({ participantId: 'p-1', token: 'tok', displayName: 'Ada', language: 'pt-BR' });
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    renderApp();
    const user = await openProfileSheet();

    expect(screen.queryByText(/cannot be undone/i)).not.toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Reset my session' }));
    expect(screen.getByText(/cannot be undone/i)).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(screen.queryByText(/cannot be undone/i)).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Reset my session' })).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('confirming calls POST /shop/api/reset and lands the client on the language step with the previous language, asserted on rendered state — not the fetch', async () => {
    saveParticipantSession({ participantId: 'p-1', token: 'tok', displayName: 'Ada', language: 'pt-BR' });
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      expect(String(input)).toMatch(/\/shop\/api\/reset$/);
      expect((init?.method ?? 'GET').toUpperCase()).toBe('POST');
      return jsonResponse(200, { threadId: 't1', language: 'pt-BR' });
    });
    vi.stubGlobal('fetch', fetchMock);
    renderApp();
    const user = await openProfileSheet();

    await user.click(screen.getByRole('button', { name: 'Reset my session' }));
    await user.click(screen.getByRole('button', { name: 'Yes, reset' }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    // Rendered state, not the fetch call: SessionContext's own
    // pendingLanguageStep flips to the previous language...
    await waitFor(() =>
      expect(screen.getByTestId('pending-language')).toHaveTextContent('pt-BR'),
    );
    // ...the credential survives the reset (C7) — never cleared...
    expect(screen.getByTestId('participant')).toHaveTextContent('present');
    // ...and the sheet itself closes.
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  it('shows an inline error and stays on the confirm step when the reset call fails', async () => {
    saveParticipantSession({ participantId: 'p-1', token: 'tok', displayName: 'Ada', language: 'en' });
    const fetchMock = vi.fn(async () => jsonResponse(503, { error: 'quiesce_failed' }));
    vi.stubGlobal('fetch', fetchMock);
    renderApp();
    const user = await openProfileSheet();

    await user.click(screen.getByRole('button', { name: 'Reset my session' }));
    await user.click(screen.getByRole('button', { name: 'Yes, reset' }));

    await waitFor(() =>
      expect(screen.getByRole('alert')).toHaveTextContent(/nothing was reset/i),
    );
    // Still on the confirm step, credential untouched, nothing navigated.
    expect(screen.getByRole('button', { name: 'Cancel' })).toBeInTheDocument();
    expect(screen.getByTestId('pending-language')).toHaveTextContent('none');
  });

  it('clears the participant credential and closes the sheet on a 401 (C3), via useResetMine()\'s own dispatch', async () => {
    saveParticipantSession({ participantId: 'p-1', token: 'tok', displayName: 'Ada', language: 'en' });
    const fetchMock = vi.fn(async () => jsonResponse(401, { error: 'invalid_token' }));
    vi.stubGlobal('fetch', fetchMock);
    renderApp();
    const user = await openProfileSheet();

    await user.click(screen.getByRole('button', { name: 'Reset my session' }));
    await user.click(screen.getByRole('button', { name: 'Yes, reset' }));

    // Rendered/persisted state, not a mock call: the credential is actually
    // gone from SessionContext (which mirrors storage — SessionContext.tsx)
    // and from storage itself, and the sheet — now stale — closes.
    await waitFor(() => expect(screen.getByTestId('participant')).toHaveTextContent('absent'));
    expect(window.localStorage.getItem(SESSION_STORAGE_KEYS.participant)).toBeNull();
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  it('reports the reset as unconfirmed (not failed) on a 504, per C4/F8', async () => {
    saveParticipantSession({ participantId: 'p-1', token: 'tok', displayName: 'Ada', language: 'en' });
    const fetchMock = vi.fn(async () => jsonResponse(504, null));
    vi.stubGlobal('fetch', fetchMock);
    renderApp();
    const user = await openProfileSheet();

    await user.click(screen.getByRole('button', { name: 'Reset my session' }));
    await user.click(screen.getByRole('button', { name: 'Yes, reset' }));

    await waitFor(() =>
      expect(screen.getByRole('alert')).toHaveTextContent(/couldn.t confirm whether that worked/i),
    );
    // Distinct from the 503 copy, and still on the confirm step — untouched
    // credential, nothing navigated.
    expect(screen.getByRole('alert')).not.toHaveTextContent(/nothing was reset/i);
    expect(screen.getByRole('button', { name: 'Cancel' })).toBeInTheDocument();
    expect(screen.getByTestId('participant')).toHaveTextContent('present');
  });

  it('shows a distinct alarm, with no implied retry, on a 409 unscoped_participant (C6b)', async () => {
    saveParticipantSession({ participantId: 'p-1', token: 'tok', displayName: 'Ada', language: 'en' });
    const fetchMock = vi.fn(async () => jsonResponse(409, { error: 'unscoped_participant' }));
    vi.stubGlobal('fetch', fetchMock);
    renderApp();
    const user = await openProfileSheet();

    await user.click(screen.getByRole('button', { name: 'Reset my session' }));
    await user.click(screen.getByRole('button', { name: 'Yes, reset' }));

    await waitFor(() =>
      expect(screen.getByRole('alert')).toHaveTextContent(/no longer scoped to this store/i),
    );
    expect(screen.getByRole('alert')).not.toHaveTextContent(/try again/i);
  });

  it('shows a visible unexpected-response message for a status resolveErrorAction leaves unhandled (C13)', async () => {
    saveParticipantSession({ participantId: 'p-1', token: 'tok', displayName: 'Ada', language: 'en' });
    const fetchMock = vi.fn(async () => jsonResponse(500, { error: 'boom' }));
    vi.stubGlobal('fetch', fetchMock);
    renderApp();
    const user = await openProfileSheet();

    await user.click(screen.getByRole('button', { name: 'Reset my session' }));
    await user.click(screen.getByRole('button', { name: 'Yes, reset' }));

    await waitFor(() =>
      expect(screen.getByRole('alert')).toHaveTextContent(/unexpected response/i),
    );
    expect(screen.getByRole('alert')).toHaveTextContent(/500/);
  });

  it('clears the previous attempt\'s stale error the instant a retry starts, before the new response resolves (analyst Pass 3 minor)', async () => {
    saveParticipantSession({ participantId: 'p-1', token: 'tok', displayName: 'Ada', language: 'en' });
    let resolveSecond: (response: Response) => void = () => {};
    const fetchMock = vi
      .fn()
      .mockImplementationOnce(async () => jsonResponse(503, { error: 'quiesce_failed' }))
      .mockImplementationOnce(
        () => new Promise<Response>((resolve) => { resolveSecond = resolve; }),
      );
    vi.stubGlobal('fetch', fetchMock);
    renderApp();
    const user = await openProfileSheet();

    await user.click(screen.getByRole('button', { name: 'Reset my session' }));
    await user.click(screen.getByRole('button', { name: 'Yes, reset' }));
    await waitFor(() =>
      expect(screen.getByRole('alert')).toHaveTextContent(/nothing was reset/i),
    );

    // Retry, without closing the sheet: the second `fetch` deliberately
    // never resolves during this assertion window — the first attempt's
    // error must be gone the instant the retry starts (isPending flips
    // synchronously on `mutate()`), not still rendered until the slow
    // second response lands.
    await user.click(screen.getByRole('button', { name: 'Yes, reset' }));
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();

    // Let the slow response settle so the mutation doesn't leak into the
    // next test.
    resolveSecond(jsonResponse(503, { error: 'quiesce_failed' }));
    await waitFor(() => expect(screen.getByRole('alert')).toBeInTheDocument());
  });
});
