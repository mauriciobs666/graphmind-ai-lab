import { render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import i18n from '../../i18n/config';
import { TurnIndicator } from './TurnIndicator';

afterEach(async () => {
  await i18n.changeLanguage('en');
});

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

  it('routes its own chrome through t() — switches to pt-BR, English literals gone', async () => {
    const { rerender } = render(
      <TurnIndicator turn={{ state: 'thinking', queuePosition: 0, lastTurn: null }} />,
    );
    expect(screen.getByRole('status')).toHaveTextContent('Thinking…');

    await i18n.changeLanguage('pt-BR');
    expect(await screen.findByRole('status')).toHaveTextContent('Pensando…');
    expect(screen.queryByText('Thinking…')).not.toBeInTheDocument();

    rerender(<TurnIndicator turn={{ state: 'queued', queuePosition: 0, lastTurn: null }} />);
    expect(screen.getByRole('status')).toHaveTextContent(
      'Você é o primeiro da fila — o processamento começa em breve.',
    );
    expect(screen.queryByText(/first in line/i)).not.toBeInTheDocument();

    rerender(<TurnIndicator turn={{ state: 'queued', queuePosition: 1, lastTurn: null }} />);
    expect(screen.getByRole('status')).toHaveTextContent('1 pessoa na sua frente.');
    expect(screen.queryByText('1 person ahead of you.')).not.toBeInTheDocument();

    rerender(<TurnIndicator turn={{ state: 'queued', queuePosition: 3, lastTurn: null }} />);
    expect(screen.getByRole('status')).toHaveTextContent('3 pessoas na sua frente.');
    expect(screen.queryByText('3 people ahead of you.')).not.toBeInTheDocument();
  });
});
