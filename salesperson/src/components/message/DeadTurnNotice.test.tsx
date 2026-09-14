import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { DeadTurnNotice } from './DeadTurnNotice';

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
});
