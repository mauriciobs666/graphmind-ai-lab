// FR-10/AC-8 coverage: name + delivery address render from `/shop/api/
// state`'s `profile` block, with an em-dash placeholder for whichever field
// has no value (docs/plans/salesperson-ui.md §2.4's parity table, S14 row).
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
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

afterEach(() => {
  vi.unstubAllGlobals();
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
});
