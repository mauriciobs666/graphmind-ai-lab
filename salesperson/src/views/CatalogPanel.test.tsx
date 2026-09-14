// FR-11/AC-11 coverage: a product with an image asset renders an `<img>`;
// one without renders text-only, with **no** `<img>` in the DOM at all for
// that card (docs/plans/salesperson-ui.md §5.1's S14 row done-condition —
// both cases asserted explicitly, not just the happy path).
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { SessionProvider } from '../session/SessionContext';
import { saveParticipantSession } from '../session/storage';
import { CatalogPanel } from './CatalogPanel';

function jsonResponse(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    text: async () => JSON.stringify(body),
  } as Response;
}

const WITH_IMAGE = {
  productId: 'wireless-mouse-pro',
  name: 'Wireless Mouse Pro',
  category: 'Peripherals',
  price: 29.99,
  imageUrl: '/shop/products/wireless-mouse-pro.jpg',
};

const WITHOUT_IMAGE = {
  productId: 'smart-home-hub',
  name: 'Smart Home Hub',
  category: 'Smart Home',
  price: 89.99,
  imageUrl: null,
};

function renderPanel() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <SessionProvider>
          <CatalogPanel />
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

describe('CatalogPanel', () => {
  it('renders an <img> for a product with an image asset, sourced from its imageUrl', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse(200, [WITH_IMAGE])));
    renderPanel();

    const img = await screen.findByRole('img', { name: 'Wireless Mouse Pro' });
    expect(img).toHaveAttribute('src', '/shop/products/wireless-mouse-pro.jpg');
    expect(screen.getByText('Wireless Mouse Pro')).toBeInTheDocument();
    expect(screen.getByText('$29.99')).toBeInTheDocument();
  });

  it('renders text-only for a product with no image asset, with no <img> anywhere in the DOM', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse(200, [WITHOUT_IMAGE])));
    renderPanel();

    await waitFor(() => expect(screen.getByText('Smart Home Hub')).toBeInTheDocument());
    expect(screen.getByText('$89.99')).toBeInTheDocument();
    expect(screen.getByText(/no photo available/i)).toBeInTheDocument();
    expect(document.querySelectorAll('img')).toHaveLength(0);
    expect(screen.queryByRole('img')).not.toBeInTheDocument();
  });

  it('renders both card shapes correctly side by side, in one grid', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse(200, [WITH_IMAGE, WITHOUT_IMAGE])));
    renderPanel();

    await waitFor(() => expect(screen.getAllByRole('listitem')).toHaveLength(2));
    // Exactly one <img> in the whole grid — the image-less card contributes none.
    expect(screen.getAllByRole('img')).toHaveLength(1);
    expect(screen.getByRole('img', { name: 'Wireless Mouse Pro' })).toBeInTheDocument();
    expect(screen.getByText('Smart Home Hub')).toBeInTheDocument();
  });

  it('shows an explicit empty state for an empty catalog', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse(200, [])));
    renderPanel();

    await waitFor(() => expect(screen.getByText(/no products to show yet/i)).toBeInTheDocument());
  });
});
