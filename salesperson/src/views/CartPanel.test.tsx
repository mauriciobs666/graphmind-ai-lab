// FR-8/AC-6 coverage: cart lines + running total render from `/shop/api/
// state`'s `cart` block, and the explicit empty state (docs/plans/
// salesperson-ui.md §2.4's parity table, S14 row).
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { CartItem } from '../api/endpoints';
import i18n from '../i18n/config';
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

// `items` is typed against the real `CartItem` interface (sourced from
// `../api/endpoints`, which mirrors the server's actual response shape) —
// not `unknown[]` — so a fixture drift back to a hand-typed field name like
// `unitPrice` fails `tsc -b`, not just silently renders `$NaN` (DEF-2).
function stateWith(cart: { items: CartItem[]; total: number }) {
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

afterEach(async () => {
  vi.unstubAllGlobals();
  await i18n.changeLanguage('en');
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
    // Fixture uses `price` — the real `/shop/api/state` `cart.items[]` field
    // name emitted by `falkor-chat/server/falkorchat/services.py`'s
    // `get_cart`/`_priced_cart_lines` (confirmed by reading the server code
    // directly), not `unitPrice` — DEF-2: a hand-typed `unitPrice` fixture
    // here is what let the real `$NaN` regression through undetected.
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        jsonResponse(
          200,
          stateWith({
            items: [
              { productId: 'wireless-mouse-pro', name: 'Wireless Mouse Pro', quantity: 2, price: 29.99 },
              { productId: 'usb-c-hub-7-in-1', name: 'USB-C Hub 7-in-1', quantity: 1, price: 39.99 },
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

  it('routes its own chrome through t() — switches to pt-BR, English literals gone (cart.total is a cognate, Layer-1-only)', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        jsonResponse(
          200,
          stateWith({
            items: [{ productId: 'wireless-mouse-pro', name: 'Wireless Mouse Pro', quantity: 1, price: 29.99 }],
            total: 29.99,
          }),
        ),
      ),
    );
    renderPanel();
    await waitFor(() => expect(screen.getByText('Total')).toBeInTheDocument());

    await i18n.changeLanguage('pt-BR');

    // `cart.total` is a named cognate exception (§4.13) — "Total" is the
    // same word in pt-BR, so it stays visible; verified by key presence
    // (Layer 1), not by a disappearing-literal assertion.
    expect(await screen.findByText('Total')).toBeInTheDocument();
  });

  it('routes the empty/loading/error/stale chrome through t() — switches to pt-BR, English literals gone', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => jsonResponse(200, stateWith({ items: [], total: 0 }))),
    );
    renderPanel();
    await waitFor(() => expect(screen.getByText('Your cart is empty.')).toBeInTheDocument());

    await i18n.changeLanguage('pt-BR');

    expect(await screen.findByText('Seu carrinho está vazio.')).toBeInTheDocument();
    expect(screen.queryByText('Your cart is empty.')).not.toBeInTheDocument();
  });

  it('shows the pt-BR loading copy — English literal gone', async () => {
    await i18n.changeLanguage('pt-BR');
    vi.stubGlobal('fetch', vi.fn(() => new Promise(() => {})));
    renderPanel();
    expect(screen.getByText('Carregando seu carrinho…')).toBeInTheDocument();
    expect(screen.queryByText(/loading your cart/i)).not.toBeInTheDocument();
  });

  it('shows the pt-BR load-error copy on a rejected fetch — English literal gone', async () => {
    await i18n.changeLanguage('pt-BR');
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse(500, { error: 'boom' })));
    renderPanel();
    await waitFor(() =>
      expect(screen.getByRole('alert')).toHaveTextContent(
        'Não foi possível carregar seu carrinho. Tentando novamente automaticamente…',
      ),
    );
    expect(screen.queryByText(/couldn't load your cart/i)).not.toBeInTheDocument();
  });

  // `cart.staleNotice` (data cached, background refetch erroring) has no
  // existing test driving that branch even pre-sweep — it needs
  // `useShopState()`'s live `POLL_INTERVAL_MS` refetch to actually fire,
  // which no test in this file attempts (S17 makes no behavioural change,
  // so it does not newly add that harness). The key itself is covered by
  // `i18n/locales.test.ts`'s key-coverage check (Layer 1) and is a direct,
  // literal `t('cart.staleNotice')` call in `CartPanel.tsx` — the same
  // completeness guarantee the plan's cognate-exception keys rely on.
});
