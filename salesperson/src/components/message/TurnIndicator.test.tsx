import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { TurnIndicator } from './TurnIndicator';

describe('TurnIndicator', () => {
  it('renders nothing when turn is undefined', () => {
    render(<TurnIndicator turn={undefined} />);
    expect(screen.queryByRole('status')).not.toBeInTheDocument();
  });

  it('renders nothing when idle', () => {
    render(<TurnIndicator turn={{ state: 'idle', queuePosition: 0, lastTurn: null }} />);
    expect(screen.queryByRole('status')).not.toBeInTheDocument();
  });

  it('shows a thinking indicator when state is thinking', () => {
    render(<TurnIndicator turn={{ state: 'thinking', queuePosition: 0, lastTurn: null }} />);
    expect(screen.getByRole('status')).toHaveTextContent(/thinking/i);
  });

  // §5.2 *The queue position* — `queuePosition: 0` on a `queued` turn is
  // "first in line", never "no queue"; this is the row's own named done-
  // condition (docs/plans/salesperson-ui.md §5.1's S13 row).
  it('queuePosition: 0 on a queued turn renders as first in line, never as no queue', () => {
    render(<TurnIndicator turn={{ state: 'queued', queuePosition: 0, lastTurn: null }} />);
    const status = screen.getByRole('status');
    expect(status).toHaveTextContent(/first in line/i);
  });

  it('queuePosition: 1 on a queued turn renders the singular count', () => {
    render(<TurnIndicator turn={{ state: 'queued', queuePosition: 1, lastTurn: null }} />);
    expect(screen.getByRole('status')).toHaveTextContent('1 person ahead of you.');
  });

  it('queuePosition: 3 on a queued turn renders the plural count', () => {
    render(<TurnIndicator turn={{ state: 'queued', queuePosition: 3, lastTurn: null }} />);
    expect(screen.getByRole('status')).toHaveTextContent('3 people ahead of you.');
  });
});
