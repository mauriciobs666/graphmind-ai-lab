// AC-5's roster table (docs/plans/salesperson-ui.md §5.1's S12d row):
// `GET /shop/api/presenter/participants` renders as one row per
// participant carrying only `displayName`/`language` — §5.2's four-key
// projection, no activity data (see S10) — plus the explicit empty state.
// The reset-everyone control's own dispatch/rendering contract is covered
// by `./PresenterResetAllControl.test.tsx`; this file only asserts it is
// mounted (by heading).
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import i18n from '../i18n/config';
import { SessionProvider } from '../session/SessionContext';
import { savePresenterSession } from '../session/storage';
import { PresenterRoster } from './PresenterRoster';

function jsonResponse(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    text: async () => JSON.stringify(body),
  } as Response;
}

function renderRoster() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <SessionProvider>
          <PresenterRoster />
        </SessionProvider>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  window.localStorage.clear();
  savePresenterSession({ token: 'ptok-1' });
});

afterEach(async () => {
  vi.unstubAllGlobals();
  await i18n.changeLanguage('en');
});

describe('PresenterRoster', () => {
  it('shows a loading state before the first response resolves', () => {
    vi.stubGlobal('fetch', vi.fn(() => new Promise(() => {})));
    renderRoster();
    expect(screen.getByText(/loading participants/i)).toBeInTheDocument();
  });

  it('shows an explicit empty state when the roster has no participants', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse(200, [])));
    renderRoster();

    await waitFor(() => expect(screen.getByText('No one has joined yet.')).toBeInTheDocument());
    expect(screen.queryByRole('table')).not.toBeInTheDocument();
  });

  it('shows a load-error state on a documented 503 (C9 — graph_unavailable) before any data has arrived', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse(503, { error: 'graph_unavailable' })));
    renderRoster();

    await waitFor(() =>
      expect(screen.getByRole('alert')).toHaveTextContent(/couldn't load the participant roster/i),
    );
    // C9's generic copy, not C13's — no route/status named.
    expect(screen.getByRole('alert')).not.toHaveTextContent(/unexpected response/i);
  });

  it('shows a load-error state on the other documented 503 (C9 — graph_read_timeout) before any data has arrived', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse(503, { error: 'graph_read_timeout' })));
    renderRoster();

    await waitFor(() =>
      expect(screen.getByRole('alert')).toHaveTextContent(/couldn't load the participant roster/i),
    );
  });

  it('shows a distinct, status-named unhandled-response message on a genuinely unmapped status (C13) — never the generic C9 staleness copy', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse(500, { error: 'boom' })));
    renderRoster();

    await waitFor(() =>
      expect(screen.getByRole('alert')).toHaveTextContent(/unexpected response/i),
    );
    expect(screen.getByRole('alert')).toHaveTextContent('500');
    // Must not read as C9's ordinary "nothing changed" outcome.
    expect(screen.getByRole('alert')).not.toHaveTextContent(/couldn't load the participant roster/i);
  });

  it("renders exactly name and language per participant — the four-key contract's negative control: a participant carrying cart/order activity in the fixture still shows nothing beyond name and language", async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        jsonResponse(200, [
          { participantId: 'p-1', displayName: 'Ada', language: 'en', joinedAt: 1000 },
          { participantId: 'p-2', displayName: 'Grace', language: 'pt-BR', joinedAt: 2000 },
          // The negative control: this participant is understood (server-side)
          // to hold cart items and a placed order — the roster projection
          // carries none of that (§5.2), so nothing about it can leak here
          // even if a future response widened the JSON body, which is what
          // the extra raw keys below stand in for.
          {
            participantId: 'p-3',
            displayName: 'Marge',
            language: 'es',
            joinedAt: 3000,
            cartItemCount: 2,
            orderStatus: 'placed',
          },
        ]),
      ),
    );
    renderRoster();

    await waitFor(() => expect(screen.getAllByRole('row')).toHaveLength(4)); // header + 3
    expect(screen.getByRole('columnheader', { name: 'Name' })).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: 'Language' })).toBeInTheDocument();
    expect(screen.getByText('Ada')).toBeInTheDocument();
    expect(screen.getByText('en')).toBeInTheDocument();
    expect(screen.getByText('Grace')).toBeInTheDocument();
    expect(screen.getByText('pt-BR')).toBeInTheDocument();
    expect(screen.getByText('Marge')).toBeInTheDocument();
    expect(screen.getByText('es')).toBeInTheDocument();
    // No activity data of any kind — the contract's actual point.
    expect(screen.queryByText(/2/)).not.toBeInTheDocument();
    expect(screen.queryByText(/placed/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/cart/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/order/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/1000|2000|3000/)).not.toBeInTheDocument(); // joinedAt, also not rendered
  });

  it('mounts the reset-everyone control alongside the roster', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse(200, [])));
    renderRoster();

    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Reset everyone' })).toBeInTheDocument(),
    );
  });

  it('routes its own chrome through t() — switches to pt-BR, English literals gone', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        jsonResponse(200, [{ participantId: 'p-1', displayName: 'Ada', language: 'en', joinedAt: 1 }]),
      ),
    );
    renderRoster();
    await waitFor(() => expect(screen.getByText('Participants')).toBeInTheDocument());

    await i18n.changeLanguage('pt-BR');

    expect(await screen.findByText('Participantes')).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: 'Nome' })).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: 'Idioma' })).toBeInTheDocument();
    expect(screen.queryByText('Participants')).not.toBeInTheDocument();
  });
});
