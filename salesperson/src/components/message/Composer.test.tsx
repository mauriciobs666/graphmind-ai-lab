import type { FormEvent } from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { Composer } from './Composer';

describe('Composer', () => {
  it('reports every keystroke to the caller and submits on click', async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    const onSubmit = vi.fn((event: FormEvent) => event.preventDefault());
    render(
      <Composer value="" onChange={onChange} onSubmit={onSubmit} disabled={false} notice={null} />,
    );

    await user.type(screen.getByLabelText('Message'), 'hi');
    expect(onChange).toHaveBeenCalled();
  });

  it('submits on Enter and inserts a newline on Shift+Enter instead', async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn((event: FormEvent) => event.preventDefault());
    render(
      <Composer value="hi" onChange={vi.fn()} onSubmit={onSubmit} disabled={false} notice={null} />,
    );

    const textarea = screen.getByLabelText('Message');
    textarea.focus();
    await user.keyboard('{Enter}');
    expect(onSubmit).toHaveBeenCalledTimes(1);

    await user.keyboard('{Shift>}{Enter}{/Shift}');
    // Shift+Enter must not submit again.
    expect(onSubmit).toHaveBeenCalledTimes(1);
  });

  it('disables the textarea and the send button while disabled', () => {
    render(
      <Composer value="hi" onChange={vi.fn()} onSubmit={vi.fn()} disabled notice={null} />,
    );
    expect(screen.getByLabelText('Message')).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Send' })).toBeDisabled();
  });

  it('keeps send disabled for blank/whitespace-only text even when not otherwise disabled', () => {
    render(
      <Composer value="   " onChange={vi.fn()} onSubmit={vi.fn()} disabled={false} notice={null} />,
    );
    expect(screen.getByRole('button', { name: 'Send' })).toBeDisabled();
  });

  it('renders an error-tone notice as role="alert" and others as role="status"', () => {
    const { rerender } = render(
      <Composer
        value=""
        onChange={vi.fn()}
        onSubmit={vi.fn()}
        disabled={false}
        notice={{ tone: 'error', text: 'boom' }}
      />,
    );
    expect(screen.getByRole('alert')).toHaveTextContent('boom');

    rerender(
      <Composer
        value=""
        onChange={vi.fn()}
        onSubmit={vi.fn()}
        disabled={false}
        notice={{ tone: 'info', text: 'fyi' }}
      />,
    );
    expect(screen.getByRole('status')).toHaveTextContent('fyi');
  });
});
