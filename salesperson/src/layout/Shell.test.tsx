// Integration coverage for the mobile shell mounted the way App.tsx
// actually mounts it as of v1.36/§4.11: `LayoutShell` is the router's own
// top-level pathless layout route (`element: <LayoutShell />, children:
// [...]`), with a real `QueryClientProvider > SessionProvider >
// RouterProvider` stack wrapping the router from outside — matching
// App.tsx's actual composition exactly (see `./Shell.tsx`'s top comment),
// not a simplified stand-in.
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { createMemoryRouter, RouterProvider } from 'react-router-dom';
import { describe, expect, it } from 'vitest';
import { SessionProvider } from '../session/SessionContext';
import { LayoutShell } from './Shell';

function renderShell(routeElement: React.ReactNode = <div>join screen</div>) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const router = createMemoryRouter(
    [
      {
        element: <LayoutShell />,
        children: [{ path: '/', element: routeElement }],
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

describe('LayoutShell', () => {
  it('renders the header and the routed content, with no sheet open initially', () => {
    renderShell();
    expect(screen.getByText('join screen')).toBeInTheDocument();
    expect(screen.getByText('Storefront')).toBeInTheDocument();
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  it('opens the matching sheet when a header icon is clicked, and marks it pressed', async () => {
    const user = userEvent.setup();
    renderShell();
    const cartButton = screen.getByRole('button', { name: 'Cart' });
    expect(cartButton).toHaveAttribute('aria-pressed', 'false');

    await user.click(cartButton);
    expect(screen.getByRole('dialog', { name: 'Cart' })).toBeInTheDocument();
    expect(cartButton).toHaveAttribute('aria-pressed', 'true');
  });

  it('closes the open sheet without affecting the routed content underneath', async () => {
    const user = userEvent.setup();
    renderShell();
    await user.click(screen.getByRole('button', { name: 'Cart' }));
    const closers = screen.getAllByRole('button', { name: 'Close' });
    await user.click(closers[closers.length - 1]);
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(screen.getByText('join screen')).toBeInTheDocument();
  });

  it('wires each of the four header buttons into its own sheet, with that sheet\'s own real panel mounted inside', async () => {
    // S14 replaced S12b's seed placeholders with real, data-fetching panels
    // (src/views/{Cart,Order,Profile,Catalog}Panel.tsx) — this test no
    // longer has literal placeholder copy to assert on. `renderShell()` has
    // no participant session, so every panel's query is `enabled: false`
    // (see `api/hooks.ts`) and each renders its own permanent, panel-
    // specific "Loading …" copy — that text is real production output, not
    // a placeholder invented for this test, and it differs per panel, so it
    // still proves *which* panel mounted, not just that a dialog opened.
    // Combined with the dialog's accessible name (its title, set by the
    // matching `*Sheet.tsx`), each assertion below proves both halves of
    // "wired to its own sheet": the right title AND the right panel inside.
    const user = userEvent.setup();
    renderShell();

    await user.click(screen.getByRole('button', { name: 'Browse catalog' }));
    const catalogDialog = screen.getByRole('dialog', { name: 'Catalog' });
    expect(within(catalogDialog).getByText(/loading the catalog/i)).toBeInTheDocument();
    await user.click(screen.getAllByRole('button', { name: 'Close' }).at(-1)!);

    await user.click(screen.getByRole('button', { name: 'Cart' }));
    const cartDialog = screen.getByRole('dialog', { name: 'Cart' });
    expect(within(cartDialog).getByText(/loading your cart/i)).toBeInTheDocument();
    await user.click(screen.getAllByRole('button', { name: 'Close' }).at(-1)!);

    await user.click(screen.getByRole('button', { name: 'Order status' }));
    const orderDialog = screen.getByRole('dialog', { name: 'Order status' });
    expect(within(orderDialog).getByText(/loading your order/i)).toBeInTheDocument();
    await user.click(screen.getAllByRole('button', { name: 'Close' }).at(-1)!);

    await user.click(screen.getByRole('button', { name: 'Profile' }));
    const profileDialog = screen.getByRole('dialog', { name: 'Profile' });
    expect(within(profileDialog).getByText(/loading your profile/i)).toBeInTheDocument();
  });
});
