// FR-10/AC-8 coverage: name + delivery address render from `/shop/api/
// state`'s `profile` block, with an em-dash placeholder for whichever field
// has no value (docs/plans/salesperson-ui.md §2.4's parity table, S14 row).
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import i18n from '../i18n/config';
import { SessionProvider } from '../session/SessionContext';
import { saveParticipantSession } from '../session/storage';
import { ProfilePanel } from './ProfilePanel';

function jsonResponse(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    text: async () => JSON.stringify(body),
  } as Response;
}

function stateWith(profile: { name: string; deliveryAddress: string | null }) {
  return {
    profile,
    cart: { items: [], total: 0 },
    order: null,
    turn: { state: 'idle', queuePosition: 0, lastTurn: null },
  };
}

function renderPanel() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <SessionProvider>
          <ProfilePanel />
        </SessionProvider>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  window.localStorage.clear();
  saveParticipantSession({ participantId: 'p-1', token: 'tok', displayName: 'Ada', language: 'en' });
});

afterEach(async () => {
  vi.unstubAllGlobals();
  await i18n.changeLanguage('en');
});

describe('ProfilePanel', () => {
  it('renders the name and delivery address when both are set', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => jsonResponse(200, stateWith({ name: 'Ada Lovelace', deliveryAddress: '1 Analytical Engine Way' }))),
    );
    renderPanel();

    await waitFor(() => expect(screen.getByText('Ada Lovelace')).toBeInTheDocument());
    expect(screen.getByText('1 Analytical Engine Way')).toBeInTheDocument();
    expect(screen.queryByText('—')).not.toBeInTheDocument();
  });

  it('shows an em-dash for the delivery address when it is null', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => jsonResponse(200, stateWith({ name: 'Ada Lovelace', deliveryAddress: null }))),
    );
    renderPanel();

    await waitFor(() => expect(screen.getByText('Ada Lovelace')).toBeInTheDocument());
    expect(screen.getByText('—')).toBeInTheDocument();
  });

  it('shows an em-dash for the name when it is empty', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => jsonResponse(200, stateWith({ name: '', deliveryAddress: null }))),
    );
    renderPanel();

    await waitFor(() => expect(screen.getAllByText('—')).toHaveLength(2));
  });

  it('routes its own labels through t() — switches to pt-BR, English literals gone', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => jsonResponse(200, stateWith({ name: 'Ada Lovelace', deliveryAddress: '1 Analytical Engine Way' }))),
    );
    renderPanel();
    await waitFor(() => expect(screen.getByText('Name')).toBeInTheDocument());
    expect(screen.getByText('Delivery address')).toBeInTheDocument();

    await i18n.changeLanguage('pt-BR');

    expect(await screen.findByText('Nome')).toBeInTheDocument();
    expect(screen.getByText('Endereço de entrega')).toBeInTheDocument();
    expect(screen.queryByText('Name')).not.toBeInTheDocument();
    expect(screen.queryByText('Delivery address')).not.toBeInTheDocument();
  });

  it('shows the pt-BR loading copy — English literal gone', async () => {
    await i18n.changeLanguage('pt-BR');
    vi.stubGlobal('fetch', vi.fn(() => new Promise(() => {})));
    renderPanel();
    expect(screen.getByText('Carregando seu perfil…')).toBeInTheDocument();
    expect(screen.queryByText(/loading your profile/i)).not.toBeInTheDocument();
  });

  it('shows the pt-BR load-error copy on a rejected fetch — English literal gone', async () => {
    await i18n.changeLanguage('pt-BR');
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse(500, { error: 'boom' })));
    renderPanel();
    await waitFor(() =>
      expect(screen.getByRole('alert')).toHaveTextContent(
        'Não foi possível carregar seu perfil. Tentando novamente automaticamente…',
      ),
    );
    expect(screen.queryByText(/couldn't load your profile/i)).not.toBeInTheDocument();
  });

  // `profile.staleNotice` has no pre-existing dynamic test driving its
  // branch (a background-refetch race) even before the sweep — no new
  // behavioural harness is added for it here (S17 is a pure string-
  // extraction sweep). Covered by `i18n/locales.test.ts`'s key-coverage
  // check (Layer 1) and a direct, literal `t('profile.staleNotice')` call in
  // `ProfilePanel.tsx`.
});
