// Integration coverage for the mobile shell mounted the way App.tsx
// actually mounts it as of v1.36/§4.11: `LayoutShell` is the router's own
// top-level pathless layout route (`element: <LayoutShell />, children:
// [...]`), with a real `QueryClientProvider > SessionProvider >
// RouterProvider` stack wrapping the router from outside — matching
// App.tsx's actual composition exactly (see `./Shell.tsx`'s top comment),
// not a simplified stand-in.
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen } from '@testing-library/react';
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

  it('wires each of the four seed placeholders into its own sheet', async () => {
    const user = userEvent.setup();
    renderShell();

    await user.click(screen.getByRole('button', { name: 'Browse catalog' }));
    expect(screen.getByText(/catalog will appear here/i)).toBeInTheDocument();
    await user.click(screen.getAllByRole('button', { name: 'Close' }).at(-1)!);

    await user.click(screen.getByRole('button', { name: 'Cart' }));
    expect(screen.getByText(/cart will appear here/i)).toBeInTheDocument();
    await user.click(screen.getAllByRole('button', { name: 'Close' }).at(-1)!);

    await user.click(screen.getByRole('button', { name: 'Order status' }));
    expect(screen.getByText(/order status will appear here/i)).toBeInTheDocument();
    await user.click(screen.getAllByRole('button', { name: 'Close' }).at(-1)!);

    await user.click(screen.getByRole('button', { name: 'Profile' }));
    expect(screen.getByText(/profile will appear here/i)).toBeInTheDocument();
  });
});
