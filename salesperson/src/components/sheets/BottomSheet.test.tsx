import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import i18n from '../../i18n/config';
import { BottomSheet } from './BottomSheet';

afterEach(async () => {
  await i18n.changeLanguage('en');
});

describe('BottomSheet', () => {
  it('renders nothing when closed', () => {
    render(
      <BottomSheet open={false} title="Cart" onClose={() => {}}>
        content
      </BottomSheet>,
    );
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  it('renders as an accessible, labelled dialog when open', () => {
    render(
      <BottomSheet open title="Cart" onClose={() => {}}>
        content here
      </BottomSheet>,
    );
    const dialog = screen.getByRole('dialog', { name: 'Cart' });
    expect(dialog).toBeInTheDocument();
    expect(screen.getByText('content here')).toBeInTheDocument();
    // Focus-on-open: the panel itself (tabIndex={-1}) takes focus so
    // keyboard/AT users land inside the sheet rather than on the page body.
    expect(document.activeElement).toBe(dialog);
  });

  it('calls onClose when the close (×) button is clicked', async () => {
    const onClose = vi.fn();
    const user = userEvent.setup();
    render(
      <BottomSheet open title="Cart" onClose={onClose}>
        content
      </BottomSheet>,
    );
    const closers = screen.getAllByRole('button', { name: 'Close' });
    await user.click(closers[closers.length - 1]);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('calls onClose when the backdrop is clicked', async () => {
    const onClose = vi.fn();
    const user = userEvent.setup();
    render(
      <BottomSheet open title="Cart" onClose={onClose}>
        content
      </BottomSheet>,
    );
    const closers = screen.getAllByRole('button', { name: 'Close' });
    await user.click(closers[0]);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('calls onClose on Escape', async () => {
    const onClose = vi.fn();
    const user = userEvent.setup();
    render(
      <BottomSheet open title="Cart" onClose={onClose}>
        content
      </BottomSheet>,
    );
    await user.keyboard('{Escape}');
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('routes its own "Close" chrome through t() — switches to pt-BR, English literal gone', async () => {
    render(
      <BottomSheet open title="Cart" onClose={() => {}}>
        content
      </BottomSheet>,
    );
    expect(screen.getAllByRole('button', { name: 'Close' })).toHaveLength(2);

    await i18n.changeLanguage('pt-BR');

    expect(await screen.findAllByRole('button', { name: 'Fechar' })).toHaveLength(2);
    expect(screen.queryByRole('button', { name: 'Close' })).not.toBeInTheDocument();
  });
});
