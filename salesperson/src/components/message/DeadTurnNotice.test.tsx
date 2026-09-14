import { render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import i18n from '../../i18n/config';
import { DeadTurnNotice } from './DeadTurnNotice';

afterEach(async () => {
  await i18n.changeLanguage('en');
});

describe('DeadTurnNotice', () => {
  it('renders nothing when not visible', () => {
    render(<DeadTurnNotice visible={false} />);
    expect(screen.queryByRole('status')).not.toBeInTheDocument();
  });

  it('renders the recovery copy when visible, as a status (not alert)', () => {
    render(<DeadTurnNotice visible />);
    expect(screen.getByRole('status')).toHaveTextContent('The reply never arrived — send again.');
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('routes its own chrome through t() — switches to pt-BR, English literal gone', async () => {
    render(<DeadTurnNotice visible />);
    expect(screen.getByText('The reply never arrived — send again.')).toBeInTheDocument();

    await i18n.changeLanguage('pt-BR');

    expect(await screen.findByText('A resposta nunca chegou — envie novamente.')).toBeInTheDocument();
    expect(screen.queryByText('The reply never arrived — send again.')).not.toBeInTheDocument();
  });
});
