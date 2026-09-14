import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { BottomSheet } from './BottomSheet';

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
});
