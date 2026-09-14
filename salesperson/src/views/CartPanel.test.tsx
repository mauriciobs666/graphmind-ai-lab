// FR-8/AC-6 coverage: cart lines + running total render from `/shop/api/
// state`'s `cart` block, and the explicit empty state (docs/plans/
// salesperson-ui.md §2.4's parity table, S14 row).
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { SessionProvider } from '../session/SessionContext';
import { saveParticipantSession } from '../session/storage';
import { CartPanel } from './CartPanel';

function jsonResponse(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    text: async () => JSON.stringify(body),
  } as Response;
}

function stateWith(cart: { items: unknown[]; total: number }) {
  return {
    profile: { name: 'Ada', deliveryAddress: null },
    cart,
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
          <CartPanel />
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

describe('CartPanel', () => {
  it('shows the explicit empty state when the cart has no items', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => jsonResponse(200, stateWith({ items: [], total: 0 }))),
    );
    renderPanel();

    await waitFor(() => expect(screen.getByText('Your cart is empty.')).toBeInTheDocument());
    expect(screen.queryByRole('list')).not.toBeInTheDocument();
  });

  it('renders one line per item with quantity/name/price and a running total', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        jsonResponse(
          200,
          stateWith({
            items: [
              { productId: 'wireless-mouse-pro', name: 'Wireless Mouse Pro', quantity: 2, unitPrice: 29.99 },
              { productId: 'usb-c-hub-7-in-1', name: 'USB-C Hub 7-in-1', quantity: 1, unitPrice: 39.99 },
            ],
            total: 99.97,
          }),
        ),
      ),
    );
    renderPanel();

    await waitFor(() => expect(screen.getAllByRole('listitem')).toHaveLength(2));
    expect(screen.getByText(/2×/)).toBeInTheDocument();
    expect(screen.getByText(/Wireless Mouse Pro/)).toBeInTheDocument();
    expect(screen.getByText(/USB-C Hub 7-in-1/)).toBeInTheDocument();
    expect(screen.getByText('$59.98')).toBeInTheDocument(); // 2 × 29.99
    expect(screen.getByText('$39.99')).toBeInTheDocument();
    expect(screen.getByText('Total')).toBeInTheDocument();
    expect(screen.getByText('$99.97')).toBeInTheDocument();
  });

  it('shows a loading state before the first response resolves', () => {
    vi.stubGlobal('fetch', vi.fn(() => new Promise(() => {})));
    renderPanel();
    expect(screen.getByText(/loading your cart/i)).toBeInTheDocument();
  });
});
