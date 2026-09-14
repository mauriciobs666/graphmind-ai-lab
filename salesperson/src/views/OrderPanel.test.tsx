// FR-9/AC-7 coverage: status chip + lines + total, `cancel` as an ordinary
// customer action, and `fulfill`/`deliver` boxed inside a visually distinct,
// explicitly-labelled "demo controls" warehouse-simulation affordance
// (docs/plans/salesperson-ui.md §4.6, §5.1's S14 row done-condition).
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { SessionProvider } from '../session/SessionContext';
import { saveParticipantSession } from '../session/storage';
import { OrderPanel } from './OrderPanel';

function jsonResponse(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    text: async () => JSON.stringify(body),
  } as Response;
}

function orderWith(status: string) {
  return {
    orderId: 'ord-1',
    status,
    lines: [
      { productId: 'wireless-mouse-pro', name: 'Wireless Mouse Pro', quantity: 2, unitPrice: 29.99, lineTotal: 59.98 },
    ],
    total: 59.98,
  };
}

function stateWith(order: unknown) {
  return {
    profile: { name: 'Ada', deliveryAddress: null },
    cart: { items: [], total: 0 },
    order,
    turn: { state: 'idle', queuePosition: 0, lastTurn: null },
  };
}

function renderPanel() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <SessionProvider>
          <OrderPanel />
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

describe('OrderPanel', () => {
  it('shows the explicit empty state when there is no order', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse(200, stateWith(null))));
    renderPanel();

    await waitFor(() =>
      expect(screen.getByText(/don't have an order yet/i)).toBeInTheDocument(),
    );
    expect(screen.queryByRole('button', { name: 'Cancel order' })).not.toBeInTheDocument();
  });

  it('renders the status chip, order lines and running total for a placed order', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse(200, stateWith(orderWith('placed')))));
    renderPanel();

    await waitFor(() => expect(screen.getByText('Placed')).toBeInTheDocument());
    expect(screen.getByText(/2×/)).toBeInTheDocument();
    expect(screen.getByText(/Wireless Mouse Pro/)).toBeInTheDocument();
    expect(screen.getByText('Total')).toBeInTheDocument();
    // The single line's total and the order's own total happen to coincide
    // in this fixture (one line, no extra charges) — both render.
    expect(screen.getAllByText('$59.98')).toHaveLength(2);
  });

  it('presents cancel as an ordinary action, separate from the demo controls box', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse(200, stateWith(orderWith('placed')))));
    renderPanel();

    const cancelButton = await screen.findByRole('button', { name: 'Cancel order' });
    const demoControls = screen.getByTestId('order-demo-controls');
    expect(demoControls).not.toContainElement(cancelButton);
  });

  it('labels the fulfil/deliver box explicitly as a warehouse simulation, visually distinct from the rest of the card', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse(200, stateWith(orderWith('placed')))));
    renderPanel();

    const demoControls = await screen.findByTestId('order-demo-controls');
    expect(demoControls).toHaveTextContent(/demo controls/i);
    expect(demoControls).toHaveTextContent(/warehouse simulation/i);
    // Visually distinct: a dashed border, not the plain card chrome the rest
    // of the panel uses.
    expect(demoControls.className).toMatch(/border-dashed/);
    expect(
      screen.getByRole('button', { name: 'Simulate: mark fulfilled' }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: 'Simulate: mark delivered' }),
    ).toBeInTheDocument();
  });

  it('enables only "mark fulfilled" for a placed order, and only "mark delivered" for a fulfilled one', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse(200, stateWith(orderWith('placed')))));
    renderPanel();

    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Simulate: mark fulfilled' })).toBeEnabled(),
    );
    expect(screen.getByRole('button', { name: 'Simulate: mark delivered' })).toBeDisabled();
  });

  it('disables "mark fulfilled" and enables "mark delivered" for a fulfilled order, and hides Cancel (AC-8)', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse(200, stateWith(orderWith('fulfilled')))));
    renderPanel();

    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Simulate: mark delivered' })).toBeEnabled(),
    );
    expect(screen.getByRole('button', { name: 'Simulate: mark fulfilled' })).toBeDisabled();
    expect(screen.queryByRole('button', { name: 'Cancel order' })).not.toBeInTheDocument();
  });

  it('disables both simulation buttons and hides Cancel for a delivered order', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse(200, stateWith(orderWith('delivered')))));
    renderPanel();

    await waitFor(() => expect(screen.getByText('Delivered')).toBeInTheDocument());
    expect(screen.getByRole('button', { name: 'Simulate: mark fulfilled' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Simulate: mark delivered' })).toBeDisabled();
    expect(screen.queryByRole('button', { name: 'Cancel order' })).not.toBeInTheDocument();
  });

  it('clicking Cancel order posts POST /shop/api/order/advance with transition "cancel"', async () => {
    const fetchMock = vi
      .fn()
      .mockImplementationOnce(async () => jsonResponse(200, stateWith(orderWith('placed'))))
      .mockImplementationOnce(async (input: RequestInfo | URL, init?: RequestInit) => {
        expect(String(input)).toMatch(/\/shop\/api\/order\/advance$/);
        expect(JSON.parse(String(init?.body))).toEqual({ transition: 'cancel' });
        return jsonResponse(200, { orderId: 'ord-1', status: 'cancelled' });
      })
      .mockImplementation(async () => jsonResponse(200, stateWith(orderWith('cancelled'))));
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    renderPanel();

    await user.click(await screen.findByRole('button', { name: 'Cancel order' }));

    await waitFor(() => expect(fetchMock.mock.calls.length).toBeGreaterThanOrEqual(2));
  });

  it('clicking "Simulate: mark fulfilled" posts transition "fulfill"', async () => {
    const fetchMock = vi
      .fn()
      .mockImplementationOnce(async () => jsonResponse(200, stateWith(orderWith('placed'))))
      .mockImplementationOnce(async (input: RequestInfo | URL, init?: RequestInit) => {
        expect(String(input)).toMatch(/\/shop\/api\/order\/advance$/);
        expect(JSON.parse(String(init?.body))).toEqual({ transition: 'fulfill' });
        return jsonResponse(200, { orderId: 'ord-1', status: 'fulfilled' });
      })
      .mockImplementation(async () => jsonResponse(200, stateWith(orderWith('fulfilled'))));
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    renderPanel();

    await user.click(await screen.findByRole('button', { name: 'Simulate: mark fulfilled' }));

    await waitFor(() => expect(fetchMock.mock.calls.length).toBeGreaterThanOrEqual(2));
  });
});
