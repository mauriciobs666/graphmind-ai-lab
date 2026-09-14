// §4.13 (v1.39) Layer 2 — `Header.tsx`'s own chrome (brand mark, nav
// aria-label, the four icon aria-labels) now routes through `t()`. No
// dedicated test file existed for `Header` before the sweep (it was covered
// only indirectly via `./Shell.test.tsx`'s English-only assertions); this
// file is the per-component locale-switch proof §4.13 requires for every
// swept file.
import { render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import i18n from '../i18n/config';
import { Header } from './Header';
import { SheetStateProvider } from './SheetContext';

function renderHeader() {
  return render(
    <SheetStateProvider>
      <Header />
    </SheetStateProvider>,
  );
}

afterEach(async () => {
  await i18n.changeLanguage('en');
});

describe('Header', () => {
  it('renders the brand mark and the four icon buttons with their English labels', () => {
    renderHeader();
    expect(screen.getByText('Storefront')).toBeInTheDocument();
    expect(screen.getByRole('navigation', { name: 'Shop' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Browse catalog' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Cart' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Order status' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Profile' })).toBeInTheDocument();
  });

  it('routes its own chrome through t() — switches to pt-BR, English literals gone, brand stays (cognate exception)', async () => {
    renderHeader();

    await i18n.changeLanguage('pt-BR');

    // Brand name is a cognate exception (§4.13) — Layer-1-only, verified by
    // key/call-site presence, not by a disappearing-literal assertion: it is
    // correctly left as "Storefront" in every bundle.
    expect(await screen.findByText('Storefront')).toBeInTheDocument();

    expect(screen.getByRole('navigation', { name: 'Loja' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Ver catálogo' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Carrinho' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Status do pedido' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Perfil' })).toBeInTheDocument();

    expect(screen.queryByRole('navigation', { name: 'Shop' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Browse catalog' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Cart' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Order status' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Profile' })).not.toBeInTheDocument();
  });
});
